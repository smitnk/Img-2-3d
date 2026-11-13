"""UniRig on the phone: GLB in, rigged GLB out, inside headless Blender (its bundled Python + numpy) with onnxruntime + scipy only.

  blender -b --factory-startup --python unirig_phone.py -- --in in.glb --out out.glb --models DIR [--threads 2] [--work DIR] [--pylibs DIR]

DIR holds the ONNX pack (tools/unirig/phone/make_pack.py): skel_encoder.onnx, skel_decoder.onnx, then skin_a / skin_pt / skin_head.onnx (or
the single skin.onnx), each with its weights in <name>.onnx.data (memory-mapped by onnxruntime), embed.npy, info.json, skin_meta.json. --pylibs: unpacked onnxruntime + scipy wheels (default DIR/pylibs). Needs next to this file (or one level up):
serialization_np.py, tokenizer_np.py, prep_np.py, onnx_skeleton.py, onnx_skin.py.

Pipeline = UniRig @6793c66 (extract.sh -> generate_skeleton.sh -> generate_skin.sh -> merge.sh), each step from numpy:
  glTF import, meshes baked to world space -> trimesh-style vertex merge, Blender Decimate to <= 50000 faces (fast_simplification stand-in)
  -> normals -> affine into [-1,1]^3 -> 65536 samples (RandomState(seed), UniRig's draw order) -> skel_encoder -> 15-beam sampling over
  skel_decoder -> detokenize (bone_i names) -> tails, affine again (mesh + joints) -> voxel skin (ray-cast depth images, scipy Dijkstra)
  -> 32768 samples -> PTv3 serialization (numpy) -> skin.onnx -> reskin onto the mesh vertices -> nearest vertex of every ORIGINAL
  vertex, top-4 weights -> armature (bone_i, roll 0, unconnected) + vertex groups -> GLB.
Memory/CPU policy for a phone: --phase all runs the skeleton (mesh -> skeleton file <work>/<input hash>/skel.npz) in its OWN Blender process, so
everything it used is returned to the OS before the skin phase starts in this one; a skel.npz of the same input (hash of the GLB + seed + faces)
is reused, so a skin crash costs only the skin phase. Each ONNX session is freed before the next; "MEM" lines give RSS per step, "PEAK" lines the
peak of each process. --threads (default 2) = onnxruntime intra-op threads (inter-op 1), BLAS/OpenMP threads; --nice 10 lowers the priority of
this process tree so the phone stays usable.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

PIPE = "1"  # bump when the skeleton file's content changes meaning

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--pylibs", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--phase", choices=("all", "skeleton", "skin"), default="all")
    p.add_argument("--work", default=None, help="directory of the resumable skeleton files (default <out dir>/.unirig_work)")
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--beams", type=int, default=15, help="skeleton beam count (fewer = less memory/time, lower quality)")
    p.add_argument("--max-faces", type=int, default=50000)
    p.add_argument("--voxel-grid", type=int, default=196)
    return p.parse_args(argv)


def setup_paths(models, pylibs):
    pylibs = pylibs or os.path.join(models, "pylibs")
    sys.path[:0] = [p for p in (pylibs, HERE, os.path.dirname(HERE)) if os.path.isdir(p)]


# ---------------------------------------------------------------- Blender side
def load_meshes(path):
    """Import the glTF, bake every mesh object into world space (identity transform, no parent), drop everything else.
    Returns the objects, their world vertices (N, 3) float32 and triangles (F, 3) over the concatenated vertices."""
    import bpy
    import numpy as np
    from mathutils import Matrix
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    bpy.context.view_layer.update()
    world = [(ob, ob.matrix_world.copy()) for ob in bpy.data.objects if ob.type == "MESH"]  # before parents are deleted
    for ob in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(ob)
    objs, vs, fs, off = [], [], [], 0
    for ob, mw in world:
        me = ob.data if ob.data.users == 1 else ob.data.copy()
        ob.data = me
        ob.modifiers.clear()
        ob.vertex_groups.clear()
        me.transform(mw)
        if mw.determinant() < 0:
            me.flip_normals()
        ob.parent = None
        ob.matrix_basis = Matrix.Identity(4)
        me.calc_loop_triangles()
        v = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", v)
        t = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", t)
        objs.append(ob)
        vs.append(v.reshape(-1, 3))
        fs.append(t.reshape(-1, 3) + off)
        off += len(v) // 3
    assert objs and off, "no mesh in the input"
    bpy.context.view_layer.update()
    return objs, np.concatenate(vs), np.concatenate(fs)


def decimate(V, F, target):
    """Blender's collapse Decimate down to `target` triangles (UniRig uses fast_simplification for this)."""
    import bpy
    import numpy as np
    me = bpy.data.meshes.new("dec")
    me.from_pydata(V.tolist(), [], F.tolist())
    ob = bpy.data.objects.new("dec", me)
    bpy.context.scene.collection.objects.link(ob)
    m = ob.modifiers.new("d", "DECIMATE")
    m.decimate_type, m.ratio, m.use_collapse_triangulate = "COLLAPSE", target / len(F), True
    bpy.context.view_layer.update()
    ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    out = ev.to_mesh()
    out.calc_loop_triangles()
    v = np.empty(len(out.vertices) * 3, np.float32)
    out.vertices.foreach_get("co", v)
    t = np.empty(len(out.loop_triangles) * 3, np.int64)
    out.loop_triangles.foreach_get("vertices", t)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return v.reshape(-1, 3), t.reshape(-1, 3)


def bvh_cast_factory(V, F):
    """make_cast for prep_np.voxel_skin_weights: nearest hit distance along rays, through Blender's BVH tree."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)

    def cast(o, d):
        out = np.full(len(o), np.inf)
        for n, (a, b) in enumerate(zip(o.tolist(), d.tolist())):
            h = tree.ray_cast(Vector(a), Vector(b))
            if h[0] is not None:
                out[n] = h[3]
        return out
    return cast


def make_rig(objs, slices, names, heads, tails, parents, idx, wts, out):
    """Armature (head/tail from numpy, no roll, not connected, parent links) + vertex groups per object (top-4 weights per vertex),
    then GLB export. idx/wts (Nverts, 4) over the concatenated vertices of all objects, slices their ranges."""
    import bpy
    bpy.ops.object.add(type="ARMATURE", location=(0, 0, 0))
    arm = bpy.context.object
    arm.name = arm.data.name = "Armature"
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for n, h, t in zip(names, heads.tolist(), tails.tolist()):
        b = eb.new(n)
        b.head, b.tail = h, t
    for i, p in enumerate(parents):
        if p is not None:
            eb[names[i]].parent = eb[names[p]]
            eb[names[i]].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    assert len(arm.data.bones) == len(names), f"Blender dropped {len(names) - len(arm.data.bones)} zero-length bones"
    for ob, (a, b) in zip(objs, slices):
        vg = [ob.vertex_groups.new(name=n) for n in names]
        for v, (ii, ww) in enumerate(zip(idx[a:b].tolist(), wts[a:b].tolist())):
            for j, w in zip(ii, ww):
                if w > 0:
                    vg[j].add([v], w, "REPLACE")
        ob.parent = arm
        ob.modifiers.new("Armature", "ARMATURE").object = arm
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")


# ---------------------------------------------------------------- the pipeline
def peak_gb():
    try:
        return round(next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM:")) / 1048576, 2)
    except Exception:
        return -1.0


def job_dir(a):
    h = hashlib.sha1(open(a.inp, "rb").read())
    h.update(f"{a.seed}|{a.max_faces}|{PIPE}".encode())
    return os.path.join(a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), ".unirig_work"), h.hexdigest()[:16])


def mods():
    import numpy as np
    import onnx_skeleton as S
    import onnx_skin as K
    import prep_np as P
    import serialization_np as Z
    import tokenizer_np as T
    return np, S, K, P, Z, T


def timer(S, phases):
    def phase(name, t0):
        phases[name] = round(time.time() - t0, 1)
        print(f"PHASE {name}: {phases[name]}s", flush=True)
        S.log_rss(name)
    return phase


def skeleton_phase(a, skel_file):
    """GLB -> simplified mesh -> skeleton; writes skel_file (atomically). Run in its own process: its memory dies with it."""
    np, S, K, P, Z, T = mods()
    phases = {}
    phase = timer(S, phases)
    info = json.load(open(f"{a.models}/info.json"))
    info["num_beams"] = a.beams
    embed = np.load(f"{a.models}/embed.npy")
    S.log_rss("skeleton start")

    # ---- 1. mesh (extract.py): merge, simplify, normals
    t0 = time.time()
    objs, V0, F0 = load_meshes(a.inp)
    print(f"INPUT {len(objs)} mesh object(s), {len(V0)} vertices, {len(F0)} triangles", flush=True)
    V, F = P.merge_vertices(V0, F0)
    del objs, V0, F0
    if len(F) > a.max_faces:
        V, F = P.merge_vertices(*decimate(V, F, a.max_faces))
    VN, FN = P.mesh_normals(V, F)
    print(f"MESH {len(V)} vertices, {len(F)} faces after merge/simplify", flush=True)
    phase("mesh", t0)

    # ---- 2. skeleton: normalise, sample 65536, encoder, beam search, detokenize
    t0 = time.time()
    rs = np.random.RandomState(a.seed)
    V1, _, _, A1 = P.affine(rs, V)
    pts, nrm, _ = P.sample_mix(rs, V1, VN, F, FN, 65536, 8192)
    cond = S.encode(a.models, {"vertices": pts.astype(np.float32), "normals": nrm.astype(np.float32)}, a.threads)
    del pts, nrm
    gc.collect()
    phase("skeleton encoder", t0)
    t0 = time.time()
    tok, start = T.Tokenizer(), [int(x) for x in info["start"]]
    dec = S.Decoder(f"{a.models}/skel_decoder.onnx", info["layers"], a.threads)
    ids = S.beam_sample(dec, embed, S.prompt_of(cond, embed, start), tok, start, info, S.noise_numpy(a.seed))
    del dec
    gc.collect()
    sk = tok.detokenize(ids)
    J = len(sk.parents)
    print(f"SKELETON {len(ids)} tokens, {J} joints", flush=True)
    assert J >= 2, "the model produced a single bone"
    phase("skeleton decoder", t0)
    print(f"PEAK skeleton {peak_gb()}GB", flush=True)
    os.makedirs(os.path.dirname(skel_file), exist_ok=True)
    tmp = skel_file + ".tmp.npz"
    np.savez(tmp, V=V, F=F, VN=VN, FN=FN, V1=V1, A1=A1, names=np.array(sk.names), joints=np.asarray(sk.joints), tails=np.asarray(sk.tails),
             parents=np.array([-1 if p is None else p for p in sk.parents]), tokens=len(ids),
             meta=json.dumps({"phases": phases, "peak_gb": peak_gb()}))
    os.replace(tmp, skel_file)


def spawn_skeleton(a, skel_file):
    """The skeleton phase in a fresh Blender process (same binary, same libs); its output lines go straight to ours."""
    import bpy
    cmd = [bpy.app.binary_path, "-b", "--factory-startup", "--python-exit-code", "1", "-t", str(a.threads), "--python", os.path.abspath(__file__),
           "--", "--in", a.inp, "--out", a.out, "--models", a.models, "--threads", str(a.threads), "--phase", "skeleton", "--nice", "0",
           "--work", os.path.dirname(os.path.dirname(skel_file)), "--seed", str(a.seed), "--max-faces", str(a.max_faces)]
    if a.pylibs:
        cmd += ["--pylibs", a.pylibs]
    sys.stdout.flush()
    r = subprocess.run(cmd)
    assert r.returncode == 0 and os.path.isfile(skel_file), f"skeleton process failed (exit {r.returncode})"


def main():
    a = parse_args()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # numpy/BLAS in this process tree must not use every core either
        os.environ[k] = str(a.threads)
    if a.nice:
        try:
            os.nice(a.nice)  # threads and child processes created from here on inherit it
        except OSError:
            pass
    setup_paths(a.models, a.pylibs)
    skel_file = os.path.join(job_dir(a), "skel.npz")
    t_all = time.time()
    if os.path.isfile(skel_file):
        print(f"RESUME skeleton file {skel_file}", flush=True)
    elif a.phase == "skeleton":
        return skeleton_phase(a, skel_file)
    else:
        # in this process: Android's phantom-process limit killed a second Blender process mid-beam on the phone.
        # Each ONNX session is freed before the next, so the peak stays the same; skel.npz still allows resume.
        skeleton_phase(a, skel_file)
        gc.collect()
    if a.phase == "skeleton":
        return
    np, S, K, P, Z, T = mods()
    phases = {}
    phase = timer(S, phases)
    meta = json.load(open(f"{a.models}/skin_meta.json"))
    S.log_rss("skin start")
    z = np.load(skel_file)
    zm = json.loads(str(z["meta"]))
    V, F, VN, FN, V1, A1 = (z[k] for k in ("V", "F", "VN", "FN", "V1", "A1"))
    sk_names, sk_parents = [str(n) for n in z["names"]], [None if p < 0 else int(p) for p in z["parents"]]
    sk_joints, sk_tails, n_tokens = z["joints"], z["tails"], int(z["tokens"])
    J = len(sk_parents)
    print(f"SKELETON {n_tokens} tokens, {J} joints (file {os.path.basename(os.path.dirname(skel_file))})", flush=True)

    # ---- 1. the original meshes again (cheap; the skeleton file holds the simplified mesh)
    t0 = time.time()
    objs, V0, _ = load_meshes(a.inp)
    slices, o = [], 0
    for ob in objs:
        slices.append((o, o + len(ob.data.vertices)))
        o += len(ob.data.vertices)
    phase("mesh", t0)

    # ---- 3. skin inputs: tails, second affine (mesh + joints), voxel skin, 32768 samples, serialization
    t0 = time.time()
    joints, tails = sk_joints.astype(np.float32), P.connect_tails(sk_joints, sk_tails, sk_parents).astype(np.float32)
    rs = np.random.RandomState(a.seed)
    V2, joints2, tails2, A2 = P.affine(rs, V1, joints, tails)
    vskin = P.voxel_skin_weights(V2, F, joints2, bvh_cast_factory, grid=a.voxel_grid).astype(np.float32)
    phase("voxel skin", t0)
    t0 = time.time()
    Vs, Ns, Ss = (x.astype(np.float32) for x in P.sample_mix(rs, V2, VN, F, FN, 32768, 8192, vskin))
    ser = Z.serialize(Vs)
    perm = np.random.RandomState(a.seed).permutation(4)  # PTv3 shuffle_orders
    ser["order"], ser["inverse"] = ser["order"][perm], ser["inverse"][perm]
    parents = np.array([-1 if p is None else p for p in sk_parents])
    inp = K.prepare(Vs, Ns, joints2, Ss, parents, serial=ser, tie=meta["tie"],
                    offs={3: np.array(meta["offs3"]), 5: np.array(meta["offs5"])}, sc=meta["sc"])
    del ser, vskin
    phase("skin inputs", t0)

    # ---- 4. skin network: three graphs one after the other (or the single skin.onnx), weights memory-mapped
    t0 = time.time()
    if K.has_parts(a.models):
        skin = K.run_parts(a.models, inp, a.threads, S.log_rss)
    else:
        sess = K.session(f"{a.models}/skin.onnx", a.threads)
        S.log_rss("session skin.onnx")
        skin = K.run(sess, inp)[0]
        del sess
    del inp
    gc.collect()
    phase("skin network", t0)

    # ---- 5. reskin to the (simplified) mesh, then to the original vertices; rig + export
    t0 = time.time()
    dec_skin = P.reskin(Vs, V2, sk_parents, F, skin)
    Ainv = np.linalg.inv(A2.astype(np.float64) @ A1.astype(np.float64))  # skin-space -> Blender world
    back = lambda x: x.astype(np.float64) @ Ainv[:3, :3].T + Ainv[:3, 3]
    heads, tl = back(joints2), back(tails2)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    short = np.linalg.norm(tl - heads, axis=1) < 1e-4 * diag  # Blender deletes zero-length bones on leaving edit mode
    tl[short] = heads[short] + [0, 0, 1e-3 * diag]
    from scipy.spatial import cKDTree
    near = cKDTree(V).query(V0)[1]
    ws = dec_skin[near]
    idx = np.argsort(-ws, axis=1)[:, :4]
    top = np.take_along_axis(ws, idx, 1)
    wts = np.nan_to_num(top / top.sum(1, keepdims=True))
    make_rig(objs, slices, sk_names, heads, tl, sk_parents, idx, wts, a.out)
    phase("rig + export", t0)
    print(f"PEAK skin {peak_gb()}GB", flush=True)
    rep = dict(input=os.path.basename(a.inp), mesh_objects=len(objs), vertices=len(V0), simplified_faces=len(F), joints=J, tokens=n_tokens,
               threads=a.threads, phases={**zm["phases"], **phases}, peak_rss_gb={"skeleton": zm["peak_gb"], "skin": peak_gb()},
               skin_graphs="parts" if K.has_parts(a.models) else "single", total_seconds=round(time.time() - t_all, 1),
               zero_length_bones_nudged=int(short.sum()))
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
    print("RESULT", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
