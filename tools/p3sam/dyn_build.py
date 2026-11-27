"""Export the dynamic P3-SAM graphs and prove the numpy side equals the stock torch model.

usage: python dyn_build.py --src src --weights p3sam.safetensors --meshes a.glb b.glb --work work [--points 16384]

  1. stock model with shuffle_orders=False on mesh[0]; from the stock voxels: numpy tables == recorded stock tables (exact), torch EncDyn == stock features
  2. export enc (p3sam_enc.onnx) and heads (p3sam_heads.onnx), weights as external data
  3. ORT on the stock tables == stock; ORT on the numpy-only path (own voxelization) on every mesh vs stock features (cos), with other point counts
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "phone"))
import p3 as P  # noqa: E402
import dyn as D  # noqa: E402
import p3s_np as N  # noqa: E402
from load_mesh import load_mesh  # noqa: E402


def cosv(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--points", type=int, default=16384)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    res = {}
    p3, sonata, sm, cfg = P.build_p3(a.src, shuffle=False)
    P.load_weights(p3, a.weights)
    p3.eval()
    transform = sonata.transform.default()
    meshes = [load_mesh(m) for m in a.meshes]

    # ---- 1. tables + features vs the stock model (no shuffle, so the stock is deterministic given its voxels)
    v, f = meshes[0][:2]
    pts, nrm, _ = N.sample_surface(v, f, a.points, 42)
    pn = N.normalize_pc(pts).astype(np.float32)
    rec = P.Rec()
    hooks = P.install_recording(p3, sm, rec)
    np.random.seed(0)
    feat_ref, data = P.stock_features(p3, transform, pn, nrm.astype(np.float32))
    for h in hooks:
        h.remove()
    feat_ref = feat_ref.numpy()
    grid_stock = data["grid_coord"].numpy().astype(np.int64)
    T, ns = N.build_tables(grid_stock)
    ok = {}
    ok["n"] = [int(rec.n[s]) for s in range(5)] == ns
    ok["nbr"] = all(np.array_equal(T[f"nbr{s}"], rec.nbr[s]) for s in range(5))
    ok["cluster"] = all(np.array_equal(T[f"pinv{s}"], rec.cluster[s]) for s in range(1, 5))
    ok["pgroup"] = all(np.array_equal(T[f"pg{s}"][:, :P.pool_group_table(rec.cluster[s], rec.n[s]).shape[1]], P.pool_group_table(rec.cluster[s], rec.n[s])) for s in range(1, 5))
    oi_ok = {}
    for (s, oi), (o, i) in sorted(rec.ordinv.items()):
        oi_ok[f"{s}_{oi}"] = bool(np.array_equal(T[f"ord{s}_{oi}"].reshape(-1), o) and np.array_equal(T[f"inv{s}_{oi}"], i))
    ok["order_inverse"] = all(oi_ok.values())
    ok["order_detail"] = oi_ok
    print("TABLES vs stock:", json.dumps(ok), flush=True)
    res["tables_vs_stock"] = ok
    # features: own voxelization vs the stock one (same grid cells; mean instead of random point)
    gc, f0, inv = N.voxelize(pn, nrm.astype(np.float32))
    res["voxels_own_vs_stock"] = [int(len(gc)), int(len(grid_stock))]
    print("VOXELS own", len(gc), "stock", len(grid_stock), flush=True)

    names = N.input_names()
    T["feat0"] = data["feat"].numpy().astype(np.float32)
    enc = D.EncDyn(p3, names).eval()
    args = tuple(torch.from_numpy(np.ascontiguousarray(T[n])) for n in names)
    with torch.no_grad():
        t = time.time()
        f_t = enc(*args).numpy()
    c = cosv(f_t[data["inverse"].numpy()], feat_ref)
    res["torch_dyn_vs_stock"] = dict(min_cos=float(c.min()), max_abs=float(np.abs(f_t[data["inverse"].numpy()] - feat_ref).max()))
    print("EncDyn (torch) vs stock features:", res["torch_dyn_vs_stock"], f"{time.time() - t:.1f}s", flush=True)

    # ---- 2. export
    torch.nn.Linear.forward = lambda self, x: torch.matmul(x, self.weight.t()) + self.bias
    P.export_onnx(enc, args, names, ["feat"], D.dyn_axes(names), f"{a.work}/p3sam_enc.onnx")
    heads = P.HeadsStatic(p3).eval()
    prompts = pn[N.fps(pn, 8)]
    P.export_onnx(heads, (torch.from_numpy(feat_ref), torch.from_numpy(pn), torch.from_numpy(prompts[:4])), ["feats", "points", "prompts"],
                  ["logits", "iou"], {"feats": {0: "N"}, "points": {0: "N"}, "prompts": {0: "K"}, "logits": {0: "N", 1: "K"}, "iou": {0: "K"}},
                  f"{a.work}/p3sam_heads.onnx")

    # ---- 3. ORT: stock tables, then the numpy-only path on all meshes / point counts
    s = N.session(f"{a.work}/p3sam_enc.onnx")
    f_o = s.run(None, {n: T[n] for n in names})[0]
    c = cosv(f_o[data["inverse"].numpy()], feat_ref)
    res["ort_vs_stock"] = dict(min_cos=float(c.min()), max_abs=float(np.abs(f_o[data["inverse"].numpy()] - feat_ref).max()))
    print("ORT dyn enc vs stock features (stock tables):", res["ort_vs_stock"], flush=True)
    sh = N.session(f"{a.work}/p3sam_heads.onnx")
    lg, io = sh.run(None, {"feats": feat_ref, "points": pn, "prompts": prompts[:4]})
    lr, ir = P.stock_heads(p3, feat_ref, pn, prompts[:4], bs=4)
    res["heads_vs_stock"] = dict(max_logit=float(np.abs(lg - lr.numpy()).max()), max_iou=float(np.abs(io - ir.numpy()).max()))
    print("ORT dyn heads vs stock:", res["heads_vs_stock"], flush=True)
    del sh
    res["numpy_path"] = []
    for mi, (mv, mf, name) in enumerate(meshes):
        for npts in (a.points, 24576):
            pts, nrm, _ = N.sample_surface(mv, mf, npts, 7)
            pn = N.normalize_pc(pts).astype(np.float32)
            t = time.time()
            gc, f0, inv = N.voxelize(pn, nrm.astype(np.float32))
            T2, ns2 = N.build_tables(gc)
            T2["feat0"] = f0
            tt = time.time() - t
            t = time.time()
            fo = s.run(None, {n: T2[n] for n in names})[0][inv]
            te = time.time() - t
            np.random.seed(0)
            fs_, _ = P.stock_features(p3, transform, pn, nrm.astype(np.float32))
            c = cosv(fo, fs_.numpy())
            r = dict(mesh=name, points=npts, voxels=ns2, tables_s=round(tt, 2), enc_s=round(te, 1), cos_mean=float(c.mean()), cos_min=float(c.min()), cos_p1=float(np.percentile(c, 1)))
            print("NUMPY PATH vs stock:", json.dumps(r), flush=True)
            res["numpy_path"].append(r)
    json.dump(res, open(f"{a.work}/dyn_build.json", "w"), indent=1)


if __name__ == "__main__":
    main()
