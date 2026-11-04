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


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="AI rig")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def onnxruntime_from_pack(pack):
    """The pack ships onnxruntime's aarch64 wheel (a zip); unpack once next to it and import from there."""
    dst = os.path.join(pack, "ort_py")
    if not os.path.isdir(os.path.join(dst, "onnxruntime")):
        with zipfile.ZipFile(os.path.join(pack, "ort_py311_aarch64.zip")) as z:
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
    """Area-weighted random points with barycentric normals (sample_vertex_groups without vertex groups)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(f), k, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(k)), rng.random(k)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pts = (w[:, :, None] * v[f[tri]]).sum(1)
    nr = (w[:, :, None] * n[f[tri]]).sum(1)
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
    t0 = time.time()
    mc = ort.InferenceSession(os.path.join(a.pack, "mesh_cond.onnx"), so).run(
        None, {"pc": pts[o1[:4 * T1]][None], "feats": nrm[o1[:4 * T1]][None]})[0]
    cl = ort.InferenceSession(os.path.join(a.pack, "vae_cond.onnx"), so).run(None, {"cond": cond[o2][None]})[0]
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
    print(f"RIG llm {len(ids)} tokens in {time.time() - t0:.0f}s", flush=True)

    where = int(np.nonzero(ids == tok.eos)[0][0])
    det = tok.detokenize(ids=ids[:where + 1])
    J = det.joints.shape[0]
    skin_tok = ids[where + 1:where + 1 + J * K]
    if skin_tok.shape[0] != J * K:
        raise SystemExit(f"rig model stopped early ({skin_tok.shape[0]} of {J * K} skin tokens)")
    skin_tok = skin_tok.reshape(J, K) - tok.vocab_size
    dec = ort.InferenceSession(os.path.join(a.pack, "skin_dec.onnx"), so)
    skin = np.concatenate([dec.run(None, {"indices": skin_tok[i:i + 2].astype(np.int64), "cond": cond[None], "cond_latents": cl})[0]
                           for i in range(0, J, 2)], 0).T  # [N samples, J]
    del dec
    skin = np.clip(skin, 0, None)

    # back to the model's own space
    heads = det.joints * scale + centre
    parents = list(det.parents)
    samples_world = pts * scale + centre
    print(f"RIG {J} bones", flush=True)

    # 8-nearest inverse-distance weights per vertex (Asset.from_data), then top 4 (group_per_vertex=4)
    kd = kdtree.KDTree(len(samples_world))
    for i, p in enumerate(samples_world):
        kd.insert(p, i)
    kd.balance()

    # one mesh object, an armature, vertex groups
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    mesh = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    arm_data = bpy.data.armatures.new("AIRig")
    arm = bpy.data.objects.new("AIRig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    names = [f"bone_{i}" for i in range(J)]
    children = {i: [c for c, p in enumerate(parents) if p == i] for i in range(J)}
    size = float(np.max(v.max(0) - v.min(0)))
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

    groups = [mesh.vertex_groups.new(name=nm) for nm in names]
    for vi, vert in enumerate(mesh.data.vertices):
        near = kd.find_n(vert.co, 8)
        w = np.zeros(J)
        tot = 0.0
        for _, idx, dist in near:
            iw = 1.0 / (dist + 1e-8)
            w += iw * skin[idx]
            tot += iw
        w /= max(tot, 1e-12)
        top = np.argsort(-w)[:4]
        top = top[w[top] > 1e-4]
        sw = w[top].sum()
        if sw <= 0:
            top = [int(np.argmin([(Vector(heads[j]) - vert.co).length for j in range(J)]))]
            ws = [1.0]
        else:
            ws = w[top] / sw
        for j, ww in zip(top, ws):
            groups[int(j)].add([vi], float(ww), "REPLACE")
    mod = mesh.modifiers.new("AIRig", "ARMATURE")
    mod.object = arm
    mesh.parent = arm

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
