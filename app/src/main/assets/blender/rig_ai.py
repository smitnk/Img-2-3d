"""Rig with AI: SkinTokens / TokenRig (VAST-AI-Research, MIT) on the phone, inside headless Blender.

blender -b --python rig_ai.py -- --in model.glb --pack /path/skintokens --out /out [--threads 4] [--seed 0]
Mirrors the original predict path (src/model/tokenrig.py, src/data/*) without torch/scipy:
  mesh -> trim/affine into [-1,1] -> 54000 area-weighted surface samples (+normals) -> mesh_cond.onnx + vae_cond.onnx
  -> int8 LLM greedy decode (repetition penalty + vocab switching, as tools/skintokens/export_skintokens.py checked
  against torch) -> skeleton tokens -> vendored TokenizerPart.detokenize -> bones; skin tokens -> skin_dec.onnx
  -> per-sample weights -> 8-nearest inverse-distance onto every vertex (Asset.from_data) -> top-4 -> armature + GLB.
Writes generation.json like the other Blender jobs.
"""
import argparse
import json
import os
import sys
import time
import zipfile

import bpy
import numpy as np
from mathutils import Vector, kdtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "skintok"), os.path.join(HERE, "bx")]
import vcbake  # noqa: E402

N_SAMPLES = 54000
N_SKIN = 6000
T_START = time.time()


def mark(msg):
    import resource
    print(f"RIG [{time.time() - T_START:6.0f}s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="AI rig")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yaw", type=float, default=0.0, help="turn the model about the up axis before the networks (degrees)")
    p.add_argument("--skeleton-only", action="store_true", help="stop after the skeleton (diagnostics)")
    return p.parse_args(argv)


def onnxruntime_from_pack(pack):
    """The pack ships onnxruntime's aarch64 wheel for Blender's Python 3.12 (a zip); unpack once and import from there."""
    dst = os.path.join(pack, "ort_py")
    if not os.path.isdir(os.path.join(dst, "onnxruntime")):
        with zipfile.ZipFile(os.path.join(pack, "ort_aarch64.zip")) as z:
            z.extractall(dst)
    sys.path.insert(0, dst)
    import onnxruntime as ort
    return ort


# ---------------------------------------------------------------- sampling (src/data/augment.py, sampler.py)

def mesh_arrays(objs):
    vs, fs, ns, off = [], [], [], 0
    for o in objs:
        me = o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        me.calc_loop_triangles()
        mw = np.array(o.matrix_world)
        v = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        n = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("normal", n)
        n = n.reshape(-1, 3) @ np.linalg.inv(mw[:3, :3])
        t = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", t)
        vs.append(v); ns.append(n); fs.append(t.reshape(-1, 3) + off)
        off += len(v)
        o.to_mesh_clear()
    return np.concatenate(vs), np.concatenate(fs), np.concatenate(ns)


def affine(v):
    """AugmentAffine(normalize_into=[-1,1]): centre the bbox, divide by half its largest side."""
    lo, hi = v.min(0), v.max(0)
    centre = (lo + hi) / 2
    scale = float(np.max((hi - lo) / 2.0))
    return centre, scale


def sample_surface(v, f, n, k, rng):
    """Area-weighted random points with their FACE normals, like sample_vertex_groups (vertex normals broke Hunyuan meshes)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(f), k, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(k)), rng.random(k)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pts = (w[:, :, None] * v[f[tri]]).sum(1)
    nr = np.cross(b - a, c - a)[tri]
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-10)  # AugmentNormalize
    return pts.astype(np.float32), nr.astype(np.float32)


# ---------------------------------------------------------------- decoding (tools/skintokens/export_skintokens.py)

def fps(points, k):
    idx = np.zeros(k, np.int64)
    d = np.full(len(points), np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((points - points[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(np.argmax(d))
    return idx


def order(points, t, seed=0):
    n = len(points)
    pick = np.random.default_rng(seed).choice(n, 4 * t, replace=4 * t > n)
    f = fps(points[pick], t)
    first = pick[f]
    return np.concatenate([first, np.delete(pick, f), np.setdiff1d(np.arange(n), pick)])


def greedy(step, embed, prefix, tok, eos, k_skin, start, max_len=2048, penalty=2.0, log=print):
    for p, e in enumerate(prefix[:-1]):
        step(e[None, None], p)
    logits = step(prefix[-1][None, None], len(prefix) - 1)
    out = []
    t0 = time.time()
    while len(out) < max_len:
        s = logits.astype(np.float64).copy()
        if out:
            u = np.unique(out)
            s[u] = np.where(s[u] < 0, s[u] * penalty, s[u] / penalty)
        seq = np.array(list(start) + out)
        mask = np.full_like(s, -np.inf)
        if tok.eos in seq:
            mask[tok.eos:] = 0
            where = int(np.nonzero(seq == tok.eos)[0][0])
            if len(seq) - where == tok.bones_in_sequence(ids=seq) * k_skin:
                mask[:] = -np.inf
                mask[eos] = 0
            else:
                mask[eos] = -np.inf
        else:
            mask[tok.next_posible_token(ids=seq)] = 0
        nxt = int(np.argmax(s + mask))
        out.append(nxt)
        if nxt == eos:
            break
        if len(out) % 50 == 0:
            log(f"rig tokens: {len(out)} ({len(out) / max(time.time() - t0, 1e-9):.1f}/s)")
        logits = step(embed[nxt][None, None], len(prefix) + len(out) - 1)
    return np.array(list(start) + out)


def main():
    a = parse_args()
    t_all = time.time()
    try:  # background priority + leave 2 cores: at full priority on all cores the phone's System UI stalled
        os.nice(10)
    except OSError:
        pass
    a.threads = max(1, min(a.threads, (os.cpu_count() or 4) - 2))
    ort = onnxruntime_from_pack(a.pack)
    from tokenizer_part import TokenizerPart
    info = json.load(open(os.path.join(a.pack, "skintokens_info.json")))
    tc = {k: v for k, v in info["tokenizer_config"].items() if k not in ("__target__", "order_config")}
    tok = TokenizerPart.parse(**tc)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.inp)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise SystemExit("no mesh in the input")
    for o in [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]:  # replace any old rig
        bpy.data.objects.remove(o)
    v, f, n = mesh_arrays(objs)
    # free Blender's copy of a big mesh (Hunyuan: ~750k tris) while the networks run; it is re-imported for skinning
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import gc
    gc.collect()
    c_, s_ = np.cos(np.radians(a.yaw)), np.sin(np.radians(a.yaw))
    R = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])  # Blender is Z-up
    v, n = v @ R.T, n @ R.T
    centre, scale = affine(v)
    vn = (v - centre) / scale
    rng = np.random.default_rng(a.seed)
    pts, nrm = sample_surface(vn, f, n, N_SAMPLES, rng)
    print(f"RIG mesh {len(v)} verts, {len(f)} tris; sampled {len(pts)}", flush=True)

    T1, T2, K = info["T1"], info["T2"], info["K"]
    o1, o2 = order(pts, T1), order(pts, T2)
    cond = np.concatenate([pts, nrm], 1).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.enable_cpu_mem_arena = False  # the arena kept the decode buffers: 15 GB peak, phone OOM
    so.enable_mem_pattern = False
    t0 = time.time()
    mc = ort.InferenceSession(os.path.join(a.pack, "mesh_cond.onnx"), so).run(
        None, {"pc": pts[o1[:4 * T1]][None], "feats": nrm[o1[:4 * T1]][None]})[0]
    cl = ort.InferenceSession(os.path.join(a.pack, "vae_cond.onnx"), so).run(None, {"cond": cond[o2][None]})[0]
    mark("encoders")
    print(f"RIG encoders {time.time() - t0:.1f}s", flush=True)

    embed = np.fromfile(os.path.join(a.pack, "embed_f16.bin"), np.float16).reshape(info["vocab"], info["hidden"]).astype(np.float32)
    s = ort.InferenceSession(os.path.join(a.pack, "llm_q.onnx"), so)
    past = [np.zeros((1, info["kv_heads"], 0, info["head_dim"]), np.float32) for _ in range(2 * info["layers"])]

    def step(e, pos):
        feeds = {"embeds": e.astype(np.float32), "position": np.array([[pos]], np.int64)}
        feeds.update({f"past_{i}": p for i, p in enumerate(past)})
        r = s.run(None, feeds)
        past[:] = r[1:]
        return r[0][0, -1]

    start = [tok.bos] + tok.make_cls_head(cls=None)
    t0 = time.time()
    ids = greedy(step, embed, np.concatenate([mc[0], embed[start]], 0), tok, info["eos"], K, start)
    del s, past
    mark("llm")
    print(f"RIG llm {len(ids)} tokens in {time.time() - t0:.0f}s", flush=True)

    where = int(np.nonzero(ids == tok.eos)[0][0])
    det = tok.detokenize(ids=ids[:where + 1])
    J = det.joints.shape[0]
    lo_, hi_ = vn.min(0), vn.max(0)
    pad = 0.05 * (hi_ - lo_)
    inside = float(np.all((det.joints >= lo_ - pad) & (det.joints <= hi_ + pad), 1).mean())
    print(f"RIG skeleton yaw {a.yaw:g}: {J} joints, {inside:.0%} inside the model box", flush=True)
    if a.skeleton_only:
        return
    skin_tok = ids[where + 1:where + 1 + J * K]
    if skin_tok.shape[0] != J * K:
        raise SystemExit(f"rig model stopped early ({skin_tok.shape[0]} of {J * K} skin tokens)")
    skin_tok = skin_tok.reshape(J, K) - tok.vocab_size
    dec = ort.InferenceSession(os.path.join(a.pack, "skin_dec.onnx"), so)
    idx = skin_tok.astype(np.int64)

    def run(j, c):  # every query point decodes independently from the latents
        return dec.run(None, {"indices": idx[j:j + JB], "cond": c[None], "cond_latents": cl})[0]
    # cost and memory grow with joints x points (~100 KB per pair): decode 12k well-spread samples, 8k pairs a call
    sel = o2[:N_SKIN]
    qc = cond[sel]
    JB, PC = 2, 2000
    a_, b_ = run(0, qc[:PC // 2]), run(0, qc[:PC])[:, :PC // 2]
    assert np.abs(a_ - b_).max() < 1e-3, "skin decoder points are not independent"
    skin = np.concatenate([np.concatenate([run(j, qc[i:i + PC]) for i in range(0, len(qc), PC)], 1)
                           for j in range(0, J, JB)], 0).T  # [N_SKIN samples, J]
    pts = pts[sel]
    mark("skin decoded")
    del dec
    skin = np.clip(skin, 0, None)

    # back to the model's own space
    heads = (det.joints * scale + centre) @ R  # undo the yaw
    parents = list(det.parents)
    samples_world = (pts * scale + centre) @ R
    print(f"RIG {J} bones", flush=True)

    del v, f, n, embed
    gc.collect()

    mark("re-import")
    # re-import the model, one object
    bpy.ops.import_scene.gltf(filepath=a.inp)
    for o in [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]:
        bpy.data.objects.remove(o)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    size = float(scale * 2)

    mark("armature")
    arm_data = bpy.data.armatures.new("AIRig")
    arm = bpy.data.objects.new("AIRig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    names = [f"bone_{i}" for i in range(J)]
    children = {i: [c for c, p in enumerate(parents) if p == i] for i in range(J)}
    ebs = []
    for i in range(J):
        eb = arm_data.edit_bones.new(names[i])
        h = Vector(heads[i])
        if children[i]:
            t = Vector(np.mean([heads[c] for c in children[i]], 0))
        elif parents[i] >= 0:
            t = h + (h - Vector(heads[parents[i]])) * 0.5
        else:
            t = h + Vector((0, 0, size * 0.05))
        if (t - h).length < size * 1e-3:
            t = h + Vector((0, 0, size * 0.02))
        eb.head, eb.tail = h, t
        ebs.append(eb)
    for i, p in enumerate(parents):
        if p >= 0:
            ebs[i].parent = ebs[p]
            ebs[i].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    # Weights live on the 54k samples; Blender's Data Transfer (C code) copies them to every vertex.
    # A per-vertex Python loop took ~15 min on a 375k-vertex Hunyuan mesh.
    top4 = np.argsort(-skin, axis=1)[:, :4]
    w4 = np.take_along_axis(skin, top4, 1)
    w4 = w4 / np.maximum(w4.sum(1, keepdims=True), 1e-12)
    mark("sample cloud groups")
    cloud_me = bpy.data.meshes.new("rig_samples")
    cloud_me.from_pydata(samples_world.tolist(), [], [])
    cloud = bpy.data.objects.new("rig_samples", cloud_me)
    bpy.context.scene.collection.objects.link(cloud)
    q = np.round(w4 * 63).astype(np.int64)  # 64 weight levels: J*64 bulk adds instead of 54k*4 single ones
    for j in range(J):
        g = cloud.vertex_groups.new(name=names[j])
        rows, cols = np.nonzero(top4 == j)
        for lvl in np.unique(q[rows, cols]):
            if lvl == 0:
                continue
            idx = rows[q[rows, cols] == lvl]
            g.add(idx.tolist(), float(lvl) / 63.0, "REPLACE")
    for nm in names:
        mesh.vertex_groups.new(name=nm)
    mark("data transfer")
    dt = mesh.modifiers.new("rig_transfer", "DATA_TRANSFER")
    dt.object = cloud
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.modifier_apply(modifier=dt.name)
    bpy.data.objects.remove(cloud)
    mark("limit 4")
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    mod = mesh.modifiers.new("AIRig", "ARMATURE")
    mod.object = arm
    mesh.parent = arm

    mark("export")
    os.makedirs(a.out, exist_ok=True)
    glb = vcbake.export_glb(os.path.join(a.out, "rigged.glb"), deform_only=False)
    tris = sum(len(p.vertices) - 2 for p in mesh.data.polygons)
    info_out = {"generator": "rig_ai", "blender": bpy.app.version_string, "name": a.name, "slug": "rigged",
                "triangles": tris, "vertices": len(mesh.data.vertices),
                "rig": {"bones": J, "skinning": "SkinTokens (AI, int8 ONNX)", "seconds": round(time.time() - t_all)},
                "files": {"glb": glb}}
    with open(os.path.join(a.out, "generation.json"), "w") as fo:
        json.dump(info_out, fo, indent=2)
    print("RIG_DONE", json.dumps(info_out["rig"]), flush=True)


if __name__ == "__main__":
    main()
