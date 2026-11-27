"""Sweep the post-processing parameters on the raw masks of tune_run.py (local, numpy/scipy only) and report part counts + agreement.

usage: python tune_post.py RAW_DIR [--sweep] [--out result.json]
For every mesh: face labels of every (points, seed) run with the given params; table of part counts and the mean / min pairwise agreement.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "phone"))
import p3s_post as PP  # noqa: E402
import p3s_split as SP  # noqa: E402


def load(raw):
    runs = json.load(open(f"{raw}/runs.json"))
    meshes, graphs = {}, {}
    for r in runs:
        if r["mesh"] not in meshes:
            z = np.load(f"{raw}/{r['mesh']}.npz")
            meshes[r["mesh"]] = SP.clean(z["v"].astype(np.float64), z["f"].astype(np.int64))  # weld like the phone does
            graphs[r["mesh"]] = PP.FaceGraph(*meshes[r["mesh"]])
    data = {}
    for r in runs:
        z = np.load(f"{raw}/{r['tag']}.npz")
        n = int(z["n"])
        masks = PP.unpack_masks(z["masks"], n)
        iou = z["iou"][np.arange(len(z["best"])), z["best"]]
        data[r["tag"]] = dict(r, pts=z["pts"].astype(np.float64), masks=masks, iou=iou)
    return runs, graphs, data


def evaluate(runs, graphs, data, params, target=None):
    out = {}
    for mesh, G in graphs.items():
        labs = {}
        for r in runs:
            if r["mesh"] != mesh:
                continue
            d = data[r["tag"]]
            fl, info = PP.segment(G, d["pts"], d["masks"], d["iou"], params, target, log=lambda *a: None)
            labs[r["tag"]] = fl
        tags = list(labs)
        ag = [PP.agreement(labs[a], labs[b], G.area) for a, b in itertools.combinations(tags, 2)]
        cnt = [int(labs[t].max()) + 1 for t in tags]
        out[mesh] = dict(counts=dict(zip(tags, cnt)), min_count=min(cnt), max_count=max(cnt), agree_mean=float(np.mean(ag)), agree_min=float(np.min(ag)))
    return out, None


def score(res):
    spread = np.mean([r["max_count"] - r["min_count"] for r in res.values()])
    ag = np.mean([r["agree_mean"] for r in res.values()])
    cnt = np.mean([np.mean(list(r["counts"].values())) for r in res.values()])
    return spread, ag, cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs, graphs, data = load(a.raw)
    if not a.sweep:
        res, _ = evaluate(runs, graphs, data, {})
        print(json.dumps(res, indent=1))
        sys.exit()
    rows = []
    grid = dict(nms_iou=[0.6, 0.75, 0.9], min_support=[1, 2, 3], min_iou_pred=[0.0, 0.7], vote=[0.0, 0.5], min_frac=[0.005, 0.01, 0.02], merge_bbox=[0.0, 0.5])
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        res, _ = evaluate(runs, graphs, data, p)
        sp_, ag, cnt = score(res)
        rows.append(dict(params=p, spread=sp_, agree=ag, count=cnt, per_mesh=res))
        print(f"{p} spread {sp_:.2f} agree {ag:.3f} count {cnt:.1f}", flush=True)
    rows.sort(key=lambda r: (r["spread"], -r["agree"]))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    print("BEST", json.dumps(rows[:5], default=float)[:3000])
