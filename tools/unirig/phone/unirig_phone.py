"""UniRig on the phone: GLB in, rigged GLB out, inside headless Blender (its bundled Python + numpy) with onnxruntime + scipy only.

  blender -b --factory-startup --python unirig_phone.py -- --in in.glb --out out.glb --models DIR [--threads 4] [--pylibs DIR]

DIR holds the ONNX pack (tools/unirig/phone/make_pack.py): skel_encoder.onnx, skel_decoder.onnx (+ external data), skin.onnx, embed.npy,
info.json, skin_meta.json. --pylibs: unpacked onnxruntime + scipy wheels (default DIR/pylibs). Needs next to this file (or one level up):
serialization_np.py, tokenizer_np.py, prep_np.py, onnx_skeleton.py, onnx_skin.py.

Pipeline = UniRig @6793c66 (extract.sh -> generate_skeleton.sh -> generate_skin.sh -> merge.sh), each step from numpy:
  glTF import, meshes baked to world space -> trimesh-style vertex merge, Blender Decimate to <= 50000 faces (fast_simplification stand-in)
  -> normals -> affine into [-1,1]^3 -> 65536 samples (RandomState(seed), UniRig's draw order) -> skel_encoder -> 15-beam sampling over
  skel_decoder -> detokenize (bone_i names) -> tails, affine again (mesh + joints) -> voxel skin (ray-cast depth images, scipy Dijkstra)
  -> 32768 samples -> PTv3 serialization (numpy) -> skin.onnx -> reskin onto the mesh vertices -> nearest vertex of every ORIGINAL
  vertex, top-4 weights -> armature (bone_i, roll 0, unconnected) + vertex groups -> GLB.
Each ONNX session is freed before the next phase; "MEM" lines give RSS per phase.
"""
import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--pylibs", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
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
def main():
    a = parse_args()
    setup_paths(a.models, a.pylibs)
    import numpy as np
    import onnx_skeleton as S
    import onnx_skin as K
    import prep_np as P
    import serialization_np as Z
    import tokenizer_np as T

    t_all, phases = time.time(), {}

    def phase(name, t0):
        phases[name] = round(time.time() - t0, 1)
        print(f"PHASE {name}: {phases[name]}s", flush=True)
        S.log_rss(name)

    def ort_session(path, **extra):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = a.threads
        o.log_severity_level = 3
        o.enable_cpu_mem_arena = False
        o.enable_mem_pattern = False
        for k, v in extra.items():
            o.add_session_config_entry(k, v)
        o.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
        S.log_rss(f"session {os.path.basename(path)}")
        return s

    info = json.load(open(f"{a.models}/info.json"))
    meta = json.load(open(f"{a.models}/skin_meta.json"))
    embed = np.load(f"{a.models}/embed.npy")
    S.log_rss("start")

    # ---- 1. mesh (extract.py): merge, simplify, normals
    t0 = time.time()
    objs, V0, F0 = load_meshes(a.inp)
    slices, o = [], 0
    for ob in objs:
        slices.append((o, o + len(ob.data.vertices)))
        o += len(ob.data.vertices)
    print(f"INPUT {len(objs)} mesh object(s), {len(V0)} vertices, {len(F0)} triangles", flush=True)
    V, F = P.merge_vertices(V0, F0)
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

    # ---- 3. skin inputs: tails, second affine (mesh + joints), voxel skin, 32768 samples, serialization
    t0 = time.time()
    joints, tails = sk.joints.astype(np.float32), P.connect_tails(sk.joints, sk.tails, sk.parents).astype(np.float32)
    rs = np.random.RandomState(a.seed)
    V2, joints2, tails2, A2 = P.affine(rs, V1, joints, tails)
    vskin = P.voxel_skin_weights(V2, F, joints2, bvh_cast_factory, grid=a.voxel_grid).astype(np.float32)
    phase("voxel skin", t0)
    t0 = time.time()
    Vs, Ns, Ss = (x.astype(np.float32) for x in P.sample_mix(rs, V2, VN, F, FN, 32768, 8192, vskin))
    ser = Z.serialize(Vs)
    perm = np.random.RandomState(a.seed).permutation(4)  # PTv3 shuffle_orders
    ser["order"], ser["inverse"] = ser["order"][perm], ser["inverse"][perm]
    parents = np.array([-1 if p is None else p for p in sk.parents])
    inp = K.prepare(Vs, Ns, joints2, Ss, parents, serial=ser, tie=meta["tie"],
                    offs={3: np.array(meta["offs3"]), 5: np.array(meta["offs5"])}, sc=meta["sc"])
    del ser, vskin
    phase("skin inputs", t0)

    # ---- 4. skin network
    t0 = time.time()
    sess = ort_session(f"{a.models}/skin.onnx", **{"session.disable_prepacking": "1"})
    skin = K.run(sess, inp)[0]
    del sess, inp
    gc.collect()
    phase("skin network", t0)

    # ---- 5. reskin to the (simplified) mesh, then to the original vertices; rig + export
    t0 = time.time()
    dec_skin = P.reskin(Vs, V2, sk.parents, F, skin)
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
    make_rig(objs, slices, sk.names, heads, tl, sk.parents, idx, wts, a.out)
    phase("rig + export", t0)
    rep = dict(input=os.path.basename(a.inp), mesh_objects=len(objs), vertices=len(V0), simplified_faces=len(F), joints=J, tokens=len(ids),
               threads=a.threads, phases=phases, total_seconds=round(time.time() - t_all, 1), zero_length_bones_nudged=int(short.sum()))
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
    print("RESULT", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
