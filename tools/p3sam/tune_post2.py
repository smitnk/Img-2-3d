"""Post-processing sweep on the raw masks of tune_run2.py (configs x seeds, plus the 100k-point reference).

usage: python tune_post2.py TUNED2_DIR [--sweep] [--out result.json] [--cfgs v16k v24k]
Per (mesh, config): part counts over seeds, mean / min agreement between seeds, mean agreement with the reference labelling of the same mesh.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "phone"))
import p3s_post as PP  # noqa: E402


def load(root):
    runs, meshes, graphs, data = [], {}, {}, {}
    for d in sorted(glob.glob(f"{root}/*")):
        if not os.path.isdir(d) or not os.path.exists(f"{d}/runs.json"):
            continue
        for r in json.load(open(f"{d}/runs.json")):
            runs.append(r)
            if r["mesh"] not in graphs:
                z = np.load(f"{d}/{r['mesh']}.npz")
                graphs[r["mesh"]] = PP.FaceGraph(z["v"].astype(np.float64), z["f"].astype(np.int64))
            z = np.load(f"{d}/{r['tag']}.npz")
            n = int(z["n"])
            data[r["tag"]] = dict(r, pts=z["pts"].astype(np.float64), masks=PP.unpack_masks(z["masks"], n), iou=z["iou"][np.arange(len(z["best"])), z["best"]])
    return runs, graphs, data


def label_all(runs, graphs, data, params, target=None, cfgs=None):
    out = {}
    for r in runs:
        if cfgs and r["cfg"] not in cfgs and r["cfg"] != "ref":
            continue
        d = data[r["tag"]]
        p = dict(params)
        if r["cfg"] == "ref":  # reference: the recipe the authors ship (400 prompts)
            p = {**p, "min_support": max(p.get("min_support", 2), 3)}
        out[r["tag"]] = PP.segment(graphs[r["mesh"]], d["pts"], d["masks"], d["iou"], p, target, log=lambda *a: None)[0]
    return out


def report(runs, graphs, labs, verbose=True):
    rows, agg = [], []
    meshes = sorted({r["mesh"] for r in runs})
    for m in meshes:
        G = graphs[m]
        ref = [labs[r["tag"]] for r in runs if r["mesh"] == m and r["cfg"] == "ref" and r["tag"] in labs]
        for c in sorted({r["cfg"] for r in runs if r["mesh"] == m and r["tag"] in labs}):
            tags = [r["tag"] for r in runs if r["mesh"] == m and r["cfg"] == c and r["tag"] in labs]
            cnt = [int(labs[t].max()) + 1 for t in tags]
            ag = [PP.agreement(labs[a], labs[b], G.area) for a, b in itertools.combinations(tags, 2)]
            ar = [PP.agreement(ref[0], labs[t], G.area) for t in tags] if ref and c != "ref" else []
            row = dict(mesh=m, cfg=c, counts=cnt, seed_agree_mean=float(np.mean(ag)) if ag else None, seed_agree_min=float(np.min(ag)) if ag else None,
                       ref_agree_mean=float(np.mean(ar)) if ar else None)
            rows.append(row)
            if verbose:
                print(f"{m:15s} {c:6s} parts {cnt} seed-agree {row['seed_agree_mean'] and round(row['seed_agree_mean'], 3)} (min {row['seed_agree_min'] and round(row['seed_agree_min'], 3)}) vs-ref {row['ref_agree_mean'] and round(row['ref_agree_mean'], 3)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cfgs", nargs="*", default=None)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--target", type=int, default=None)
    a = ap.parse_args()
    runs, graphs, data = load(a.root)
    if not a.sweep:
        report(runs, graphs, label_all(runs, graphs, data, json.loads(a.params), a.target, a.cfgs))
        sys.exit()
    res = []
    grid = json.loads(os.environ["P3_GRID"]) if os.environ.get("P3_GRID") else dict(nms_iou=[0.6, 0.75, 0.9], min_support=[1, 2, 3], min_iou_pred=[0.0, 0.6], vote=[0.0, 0.5], min_frac=[0.005, 0.01, 0.02], merge_bbox=[0.0, 0.5])
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        rows = report(runs, graphs, label_all(runs, graphs, data, p, None, a.cfgs), verbose=False)
        cand = [r for r in rows if r["cfg"] != "ref"]
        spread = float(np.mean([(max(r["counts"]) - min(r["counts"])) for r in cand]))
        sa = float(np.mean([r["seed_agree_mean"] for r in cand if r["seed_agree_mean"] is not None]))
        ra = float(np.mean([r["ref_agree_mean"] for r in cand if r["ref_agree_mean"] is not None]))
        res.append(dict(params=p, spread=spread, seed_agree=sa, ref_agree=ra, rows=rows))
        print(json.dumps(p), f"spread {spread:.2f} seed-agree {sa:.3f} ref-agree {ra:.3f}", flush=True)
    res.sort(key=lambda r: -(r["ref_agree"] + r["seed_agree"] - 0.05 * r["spread"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    for r in res[:5]:
        print("BEST", json.dumps(r["params"]), round(r["spread"], 2), round(r["seed_agree"], 3), round(r["ref_agree"], 3))
