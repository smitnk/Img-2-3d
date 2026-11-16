"""Hunyuan3D-2 Paint (turbo) on the phone: untextured GLB + reference photos in, textured GLB out, inside headless Blender 4.2
(bundled Python + numpy) with onnxruntime only (no torch / PIL / scipy / cv2 / GPU / Cycles).

  blender -b --factory-startup --python paint_phone.py -- --mesh in.glb --refs a.png b.png --out out.glb --models DIR [--threads 2]
          [--pylibs DIR] [--work DIR] [--steps 10] [--faces 40000] [--tex 1024] [--rot-y180 1] [--seed 0] [--nice 10]

DIR = the ONNX pack (export_onnx.py): ref_unet / gen_unet / vae_enc / vae_dec (.onnx + .onnx.data) and meta.json.
Pipeline (same maths as the torch C2 variant of tools/painttest/run_paint.py, one model in memory at a time):
  bpy: import, merge, Decimate to --faces, Smart UV Project
  numpy: 6 orthographic views -> normal + position maps (software rasteriser, paint_np.py)
  vae_enc (320 px tiles): references, normal, position maps -> latents          [session freed]
  ref_seg0..3 (once per reference) -> 16 features per reference, kept as fp16  [one segment mapped at a time]
  gen_seg0..3 (the generation UNet cut in 4), ONE view per call, segment by segment over the 6 views,
    10 LCM steps, other views' features cached; one segment mapped at a time                          [session freed]
  vae_dec (24x24 latent tiles, 8 latent overlap, blended) -> 6 views of 512 px                   [session freed]
  numpy gather bake (cos^4 weights, depth test) + UV fill -> texture, bpy material + GLB export.
Memory: onnxruntime memory-maps the .onnx.data weights, arena and mem-pattern off, ConstantFolding disabled (the int8 weights are
dequantised op by op inside the graph; folding them would make fp32 copies). "MEM" lines = RSS (anon + file-backed) per step,
"PEAK" lines = VmHWM per stage; stats.json is written next to the output.
"""
import argparse, gc, json, os, struct, sys, threading, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--refs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--pylibs", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--work", default=None)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--faces", type=int, default=40000)
    p.add_argument("--tex", type=int, default=1024)
    p.add_argument("--render", type=int, default=1024)
    p.add_argument("--rot-y180", type=int, default=1, help="1 for meshes from Hunyuan3D shape generation (as the torch test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweeps0", type=int, default=2)
    p.add_argument("--nice", type=int, default=0)
    p.add_argument("--mmap-threshold", type=int, default=1048576, help="glibc M_MMAP_THRESHOLD (0 = leave malloc alone)")
    p.add_argument("--phase", choices=("all", "prep", "gen", "export"), default="all",
                   help="all = prep and export each in their own Blender process (their memory goes back to the OS), gen in this one")
    p.add_argument("--save-views", type=int, default=1)
    return p.parse_args(argv)


# ------------------------------------------------------------------ memory monitor
def status():
    d = {"VmRSS": 0, "VmHWM": 0, "RssAnon": 0, "RssFile": 0}
    with open("/proc/self/status") as f:
        for l in f:
            k = l.split(":")[0]
            if k in d:
                d[k] = int(l.split(":")[1].split()[0]) / 1048576.0
    return d


class Mon:
    def __init__(self):
        self.cur, self.st, self.t0, self.stop = None, {}, {}, False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            cur = self.cur
            if cur:
                s, m = status(), self.st[cur]
                m["rss"] = max(m["rss"], s["VmRSS"])
                m["anon"] = max(m["anon"], s["RssAnon"])
                m["file"] = max(m["file"], s["RssFile"])
            time.sleep(0.1)

    def begin(self, name):
        self.end()
        try:
            open("/proc/self/clear_refs", "w").write("5")  # reset VmHWM
        except Exception:
            pass
        self.st[name] = {"rss": 0.0, "anon": 0.0, "file": 0.0}
        self.t0[name] = time.time()
        self.cur = name
        s = status()
        print(f"[MEM] >> {name} rss={s['VmRSS']:.2f} anon={s['RssAnon']:.2f} file={s['RssFile']:.2f} GB", flush=True)

    def end(self):
        if not self.cur:
            return
        s, m = status(), self.st[self.cur]
        m["hwm"] = round(s["VmHWM"], 3)
        m["time_s"] = round(time.time() - self.t0[self.cur], 1)
        for k in ("rss", "anon", "file"):
            m[k] = round(m[k], 3)
        print(f"PEAK {self.cur}: hwm={m['hwm']}GB sampled_rss={m['rss']}GB anon={m['anon']}GB file={m['file']}GB t={m['time_s']}s", flush=True)
        self.cur = None


# ------------------------------------------------------------------ png writer (no PIL)
def write_png(path, arr):
    import numpy as np
    a = np.ascontiguousarray(arr, np.uint8)
    h, w, c = a.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * c)], 1).tobytes()

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, {3: 2, 4: 6}[c], 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def to_u8(x):
    import numpy as np
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ Blender side
def load_image_np(path):
    """image file -> float32 (H,W,C) 0..1, row 0 = top. Raw file values (Non-Color)."""
    import bpy
    import numpy as np
    im = bpy.data.images.load(path)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    c = im.channels
    px = np.empty(w * h * c, np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    px = px.reshape(h, w, c)[::-1]
    if c == 4 and px[..., 3].min() > 0.999:  # opaque photo saved with an alpha channel
        px = px[..., :3]
    return px.copy()


def prepare_mesh(path, faces):
    """Import GLB, bake world transforms, merge, decimate, smart UV. Returns the Blender object, V (glTF Y-up), F (triangles),
    corner UVs (3F, 2)."""
    import bpy
    import numpy as np
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    bpy.context.view_layer.update()
    print("imported objects:", [(o.name, o.type, len(o.data.vertices) if o.type == "MESH" else 0, o.name in bpy.context.scene.objects) for o in bpy.data.objects], flush=True)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.vertices) > 100]
    for o in meshes:
        if o.parent:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
    for o in [o for o in bpy.data.objects if o not in meshes]:  # empties, armature bone shapes ...
        bpy.data.objects.remove(o)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
        o.modifiers.clear()
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.make_single_user(object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    me = ob.data
    for k in list(me.uv_layers):
        me.uv_layers.remove(k)
    me.materials.clear()
    n0 = len(me.polygons)
    print(f"mesh in: {len(me.vertices)} verts, {n0} faces", flush=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")
    if len(me.polygons) > faces:
        md = ob.modifiers.new("dec", "DECIMATE")
        md.decimate_type, md.ratio, md.use_collapse_triangulate = "COLLAPSE", faces / len(me.polygons), True
        bpy.ops.object.modifier_apply(modifier=md.name)
    me = ob.data
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.004)  # 66 deg
    bpy.ops.object.mode_set(mode="OBJECT")
    me.calc_loop_triangles()
    nv, nt = len(me.vertices), len(me.loop_triangles)
    v = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    t = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", t)
    lo = np.empty(nt * 3, np.int64)
    me.loop_triangles.foreach_get("loops", lo)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uvc = uv.reshape(-1, 2)[lo]
    Vg = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)  # Blender Z-up -> glTF Y-up (what trimesh / the torch test see)
    print(f"mesh used: {nv} verts, {nt} faces (from {n0})", flush=True)
    return ob, Vg, t.reshape(-1, 3), uvc


def save_prep_glb(ob, path):
    import bpy
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_yup=True, export_normals=True)


def load_prep_glb(path):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.vertices) > 100][0]


def export_glb(ob, tex_png, out):
    import bpy
    mat = bpy.data.materials.new("paint")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    img = bpy.data.images.load(tex_png)
    img.colorspace_settings.name = "sRGB"
    img.pack()
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = img
    nt.links.new(tn.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True, export_image_format="AUTO",
                              export_yup=True)


# ------------------------------------------------------------------ onnxruntime
class ORT:
    def __init__(self, models, threads, pylibs=None):
        for p in ([pylibs] if pylibs else []) + [os.path.join(models, "pylibs")]:
            if p and os.path.isdir(p):
                sys.path.insert(0, p)
        import onnxruntime as ort
        self.ort, self.models, self.threads = ort, models, threads

    def session(self, name):
        ort = self.ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = self.threads, 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = False  # free every intermediate at once instead of pooling it
        so.enable_mem_pattern = False
        so.log_severity_level = 3
        so.add_session_config_entry("optimization.disable_specified_optimizers", "ConstantFolding")
        return ort.InferenceSession(os.path.join(self.models, name), so, providers=["CPUExecutionProvider"])


_LIBC = None


def tune_malloc(mmap_threshold):
    """glibc keeps freed heap and raises its mmap threshold while running, so RSS stays at the peak of the previous call;
    a fixed threshold makes big temporaries mmap'd (returned to the OS on free)."""
    global _LIBC
    try:
        import ctypes
        _LIBC = ctypes.CDLL("libc.so.6")
        if mmap_threshold:
            _LIBC.mallopt(-3, int(mmap_threshold))  # M_MMAP_THRESHOLD (also stops its dynamic growth)
            _LIBC.mallopt(-1, 131072)  # M_TRIM_THRESHOLD
    except Exception as e:  # non-glibc libc: harmless
        print("malloc tuning unavailable:", e, flush=True)


def trim():
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def log_mem(tag):
    s = status()
    print(f"MEM {tag}: rss={s['VmRSS']:.2f} anon={s['RssAnon']:.2f} file={s['RssFile']:.2f} hwm={s['VmHWM']:.2f} GB", flush=True)


# ------------------------------------------------------------------ main
def main(a, io=None):
    import numpy as np
    sys.path.insert(0, HERE)
    import paint_np as N
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError:
            pass
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), ".paint_work")
    os.makedirs(work, exist_ok=True)
    outdir = os.path.dirname(os.path.abspath(a.out))
    mon, stats, T0 = Mon(), {"threads": a.threads, "steps": a.steps}, time.time()
    io = io or sys.modules[__name__]
    prep_glb, prep_npz = os.path.join(work, "prep.glb"), os.path.join(work, "prep.npz")
    argv_tail = [x for x in sys.argv[sys.argv.index("--") + 1:]] if "--" in sys.argv else sys.argv[1:]

    def sub(phase):  # a fresh Blender process for the mesh / export phases: its memory is returned to the OS afterwards
        import subprocess
        base = [x for i, x in enumerate(argv_tail) if x != "--phase" and (i == 0 or argv_tail[i - 1] != "--phase")]
        cmd = [__import__("bpy").app.binary_path, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
               os.path.abspath(__file__), "--"] + base + ["--phase", phase]
        r = subprocess.run(cmd)
        if r.returncode:
            raise SystemExit(f"phase {phase} failed ({r.returncode})")

    if a.phase == "prep":
        mon.begin("mesh_bpy")
        ob, Vg, F, uvc = io.prepare_mesh(a.mesh, a.faces)
        np.savez(prep_npz, Vg=Vg, F=F, uvc=uvc)
        io.save_prep_glb(ob, prep_glb)
        mon.end()
        print(f"PEAK prep hwm={status()['VmHWM']:.2f}GB", flush=True)
        json.dump({"hwm": status()["VmHWM"], "anon": status()["RssAnon"]}, open(os.path.join(work, "peak_prep.json"), "w"))
        return
    if a.phase == "export":
        mon.begin("export_glb")
        io.export_glb(io.load_prep_glb(prep_glb), os.path.join(outdir, "texture.png"), a.out)
        mon.end()
        print(f"PEAK export hwm={status()['VmHWM']:.2f}GB", flush=True)
        json.dump({"hwm": status()["VmHWM"], "anon": status()["RssAnon"]}, open(os.path.join(work, "peak_export.json"), "w"))
        return
    meta = json.load(open(os.path.join(a.models, "meta.json")))
    layers = meta["layers"]
    san = lambda n: n.replace(".", "_")  # noqa: E731
    R = ORT(a.models, a.threads, a.pylibs)
    tune_malloc(a.mmap_threshold)
    rng = np.random.RandomState(a.seed)

    # ---- mesh (own Blender process in --phase all) + views
    if a.phase == "all":
        sub("prep")
    z = np.load(prep_npz)
    Vg, F, uvc = z["Vg"], z["F"], z["uvc"]
    mon.begin("views")
    Vr, _ = N.to_render_frame(Vg, bool(a.rot_y180))
    views = N.Views(Vr, F, a.render, log=lambda *x: None)
    normals, poss = [], []
    for i in range(6):
        n, p = views.maps(i, 512)
        normals.append(n)
        poss.append(p)
    np.savez_compressed(os.path.join(outdir, "mesh_uv.npz"), Vr=Vr, F=F, uv=uvc.astype(np.float32))
    for i in range(6):
        write_png(os.path.join(outdir, f"cond_normal_{i}_{N.VNAMES[i]}.png"), to_u8(normals[i]))
        write_png(os.path.join(outdir, f"cond_position_{i}_{N.VNAMES[i]}.png"), to_u8(poss[i]))
    refs = []
    for pth in a.refs:
        refs.append(N.prep_reference(io.load_image_np(pth), 512))
    for i, r in enumerate(refs):
        write_png(os.path.join(outdir, f"ref_{i}.png"), to_u8(r))
    log_mem("after views")

    # ---- VAE encoder (tiled)
    mon.begin("vae_encode")
    sc = meta["vae_scaling"]
    sess = R.session("vae_enc.onnx")
    ET = meta["vae_enc_tile"]

    def encode(img):  # (512,512,3) 0..1 -> (1,4,64,64)
        x = ((img.transpose(2, 0, 1)[None] - 0.5) * 2.0).astype(np.float32)
        m = N.tiled(x, lambda t: sess.run(None, {"x": t})[0], ET, 192, 1 / 8, 8, 16)
        mean, logv = m[:, :4], np.clip(m[:, 4:], -30, 20)
        return ((mean + np.exp(0.5 * logv) * rng.standard_normal(mean.shape)) * sc).astype(np.float32)
    ref_lat = [encode(r) for r in refs]
    nrm_lat = [encode(x) for x in normals]
    pos_lat = [encode(x) for x in poss]
    del sess
    gc.collect()
    del normals, poss

    # ---- reference UNet (4 segment graphs, one mapped at a time) -> features (fp16), all refs concatenated per layer
    mon.begin("ref_unet")
    feats = {n: [None] * len(ref_lat) for n in layers}
    states = [None] * len(ref_lat)
    for sm in meta["ref_segments"]:
        sess = R.session(sm["file"])
        for r in range(len(ref_lat)):
            o = sess.run(None, {"lat": ref_lat[r]} if sm["first"] else dict(zip(sm["in_state"], states[r])))
            ns = len(sm["out_state"])
            states[r] = o[:ns]
            for i, n in enumerate(sm["layers"]):
                feats[n][r] = o[ns + i]
            del o
        del sess
        trim()
    ref_feats = [np.concatenate(feats[n], 1) for n in layers]
    del feats, states, ref_lat
    gc.collect()
    log_mem("ref features")

    # ---- generation: segment by segment over all 6 views (one gen_seg graph mapped at a time)
    mon.begin("denoise")
    segs = meta["segments"]
    sched = N.LCM(a.steps)
    cams = N.camera_info()
    lat = rng.standard_normal((6, 4, 64, 64)).astype(np.float32)
    shapes = meta["ref_shapes"]
    cache = {n: np.zeros((6,) + tuple(shapes[n]), np.float16) for n in layers}  # [view, L, C]
    have = [[False] * 6 for _ in segs]
    in_ref = {f"ref_{san(n)}": ref_feats[i] for i, n in enumerate(layers)}
    step_t, seg_t = [], []
    for si, t in enumerate(sched.ts):
        t0 = time.time()
        eps = np.zeros_like(lat)
        for sw in range(a.sweeps0 if si == 0 else 1):
            states = [None] * 6
            for k, sm in enumerate(segs):
                ts_ = time.time()
                sess = R.session(sm["file"])
                for v in range(6):
                    others = [j for j in range(6) if j != v and have[k][j]]
                    if sm["first"]:
                        feed = {"lat": lat[v:v + 1], "nrm": nrm_lat[v], "pos": pos_lat[v], "t": np.array([t], np.int64),
                                "cam": np.array([cams[v]], np.int64)}
                    else:
                        feed = dict(zip(sm["in_state"], states[v]))
                    for n in sm["layers"]:
                        feed[f"ref_{san(n)}"] = in_ref[f"ref_{san(n)}"]
                        feed[f"mv_{san(n)}"] = (np.concatenate([cache[n][j][None] for j in others], 1) if others
                                                else np.zeros((1, 0, shapes[n][1]), np.float16))
                    o = sess.run(None, feed)
                    del feed
                    ns = len(sm["out_state"])
                    states[v] = o[:ns]
                    for i, n in enumerate(sm["layers"]):
                        cache[n][v] = o[ns + i][0]
                    have[k][v] = True
                    del o
                del sess
                trim()
                seg_t.append(round(time.time() - ts_, 1))
            for v in range(6):
                eps[v] = states[v][0][0]
            del states
        lat = sched.step(eps, si, lat, rng)
        step_t.append(round(time.time() - t0, 1))
        s = status()
        print(f"[step {si + 1}/{len(sched.ts)}] t={t} {step_t[-1]}s rss={s['VmRSS']:.2f} anon={s['RssAnon']:.2f} file={s['RssFile']:.2f} hwm={s['VmHWM']:.2f}", flush=True)
    stats["step_times_s"], stats["segment_times_s"] = step_t, seg_t
    del cache, ref_feats, in_ref
    gc.collect()
    trim()

    # ---- VAE decode (tiled, blended)
    mon.begin("vae_decode")
    sess = R.session("vae_dec.onnx")
    DT = meta["vae_dec_tile"]
    gen = []
    for v in range(6):
        z = (lat[v:v + 1] / sc).astype(np.float32)
        img = N.tiled(z, lambda t: sess.run(None, {"z": t})[0], DT, DT - 8, 8, 3, 64)
        gen.append(np.clip(img[0].transpose(1, 2, 0) / 2 + 0.5, 0, 1).astype(np.float32))
        if a.save_views:
            write_png(os.path.join(outdir, f"view_{v}_{N.VNAMES[v]}.png"), to_u8(gen[-1]))
    del sess
    gc.collect()

    # ---- bake
    mon.begin("bake")
    T = a.tex
    nc = len(F)
    Vuv = uvc.astype(np.float32)
    Fuv = np.arange(nc * 3).reshape(-1, 3)
    corner_v = F.reshape(-1)
    vn = views.vn
    (Pt, Nt), cover = N.uv_raster(Vuv, Fuv, [Vr[corner_v], vn[corner_v]], T)
    tex, trust = N.bake(views, gen, Pt, Nt, cover, log=lambda *x: print(*x, flush=True))
    del Pt, Nt
    tex, known = N.fill(tex, trust, 64)
    stats["texels_covered"], stats["texels_painted"] = int(cover.sum()), int(trust.sum())
    png = os.path.join(outdir, "texture.png")
    write_png(png, to_u8(tex[::-1]))  # PNG row 0 = top = v 1
    mon.end()
    mon.stop = True
    stats["stages"] = mon.st
    stats["peak_hwm_gb"] = max(m.get("hwm", 0) for m in mon.st.values())
    stats["peak_anon_gb"] = max(m["anon"] for m in mon.st.values())
    stats["total_time_s"] = round(time.time() - T0, 1)
    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    if a.phase == "all":
        sub("export")
    for ph in ("prep", "export"):
        pf = os.path.join(work, f"peak_{ph}.json")
        if os.path.exists(pf):
            stats[f"peak_{ph}_hwm_gb"] = round(json.load(open(pf))["hwm"], 3)
            stats["peak_hwm_gb"] = max(stats["peak_hwm_gb"], stats[f"peak_{ph}_hwm_gb"])
    stats["total_time_s"] = round(time.time() - T0, 1)
    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    print(f"PEAK total hwm={stats['peak_hwm_gb']}GB anon={stats['peak_anon_gb']}GB time={stats['total_time_s']}s", flush=True)


if __name__ == "__main__":
    main(parse_args())
