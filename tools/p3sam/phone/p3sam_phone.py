"""Auto Split on the phone: mesh -> P3-SAM masks -> parts -> closed per-part GLBs (+ pegs) -> colour-per-part preview GLB.

Pure numpy / scipy / onnxruntime / manifold3d (no torch). Used by the app's Blender script p3sam_split.py and, with --npz, by CI.

  python p3sam_phone.py --npz mesh.npz --models DIR --out DIR [--target 0] [--pegs 1] [--peg-size 0.4] [--points 16384] [--seed 1] [--threads 0]
mesh.npz: v (n,3) float, f (m,3) int, glTF Y-up. Writes OUT/parts.zip (part_NN.glb), OUT/parts_preview.glb, OUT/generation.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3s_np as N  # noqa: E402
import p3s_post as PP  # noqa: E402
import p3s_split as SP  # noqa: E402

PROMPTS = 128  # prompts for the heads
DENSE = 100000  # surface points through the encoder (dense enough that the voxel surface is closed, like the 100k the model was built for)
BUDGET = 16000  # at most this many voxels: the voxel size grows from 0.005 (training size) until the surface fits
HEADS = 16384  # points the prompt heads see (they cost points x prompts)


def run(v, f, models, out, target=0, pegs=True, peg_size=0.4, points=HEADS, seed=1, threads=0, name="parts", log=print, dense=DENSE, budget=BUDGET):
    t0 = time.time()
    N.tame_malloc()
    os.makedirs(out, exist_ok=True)
    v, f = SP.clean(v, f)
    log(f"P3S mesh {len(v)} vertices, {len(f)} triangles")
    pts, nrm, _ = N.sample_surface(v, f, dense, seed)
    pn = N.normalize_pc(pts).astype(np.float32)
    sub = np.unique(np.linspace(0, len(pn) - 1, points).astype(np.int64))
    model = N.P3Sam(os.path.join(models, "p3sam_enc.onnx"), os.path.join(models, "p3sam_heads.onnx"), threads, log=log)
    t = time.time()
    feats = model.features(pn, nrm.astype(np.float32), sub, budget=budget)
    t_enc = time.time() - t
    pts, pn = pts[sub], pn[sub]
    sel = N.fps(pn, PROMPTS)
    t = time.time()
    logits, iou = model.masks(feats, pn, pn[sel])
    t_heads = time.time() - t
    del feats
    j = iou.argmax(1)
    masks = np.take_along_axis(logits, j[None, :, None], 2)[:, :, 0] > 0
    biou = iou[np.arange(len(j)), j]
    del logits
    t = time.time()
    G = PP.FaceGraph(v, f)
    fl, info = PP.segment(G, pts, masks, biou, target=target if target >= 2 else None)
    t_post = time.time() - t
    log(f"P3S {info['parts']} parts ({info}); post {t_post:.1f}s")
    t = time.time()
    res = SP.split_parts(G, fl, pegs=pegs, peg_size=peg_size, log=log)
    files = []
    for l, p in sorted(res["parts"].items()):
        files.append((f"part_{l:02d}.glb", SP.glb_bytes(p["v"], p["f"], None, f"part_{l:02d}")))
    SP.write_zip(os.path.join(out, "parts.zip"), files)
    prev = SP.colored_parts_glb(G.v, G.f, fl, name)
    open(os.path.join(out, "parts_preview.glb"), "wb").write(prev)
    t_split = time.time() - t
    watertight = []
    for p in res["parts"].values():
        e = np.sort(np.concatenate([p["f"][:, [0, 1]], p["f"][:, [1, 2]], p["f"][:, [2, 0]]]), 1)
        _, c = np.unique(e, axis=0, return_counts=True)
        watertight.append(bool((c == 2).all()))
    stats = dict(script="p3sam_split", name=name, vertices=int(len(G.v)), triangles=int(len(G.f)), parts=int(info["parts"]), pegs_ok=int(sum(1 for q in res["pegs"] if q.get("ok"))),
                 pegs_total=int(len(res["pegs"])), grid=round(float(model.grid_size), 5), voxels=model.voxels, closed_parts=int(sum(watertight)), manifold_parts=int(sum(p["ok"] for p in res["parts"].values())),
                 seconds=dict(encoder=round(t_enc, 1), heads=round(t_heads, 1), post=round(t_post, 1), split=round(t_split, 1), total=round(time.time() - t0, 1)),
                 peak_rss_mb=round(N.hwm_mb()), zip_bytes=os.path.getsize(os.path.join(out, "parts.zip")))
    json.dump(stats, open(os.path.join(out, "generation.json"), "w"))
    np.save(os.path.join(out, "face_labels.npy"), fl)
    log("P3S done " + json.dumps(stats))
    return stats, fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--pegs", type=int, default=1)
    ap.add_argument("--peg-size", type=float, default=0.4)
    ap.add_argument("--points", type=int, default=HEADS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default="parts")
    a = ap.parse_args()
    z = np.load(a.npz)
    run(z["v"].astype(np.float64), z["f"].astype(np.int64), a.models, a.out, a.target, bool(a.pegs), a.peg_size, a.points, a.seed, a.threads, a.name)


if __name__ == "__main__":
    main()
