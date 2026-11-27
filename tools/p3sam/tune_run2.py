"""Raw P3-SAM masks for configs (dense sample -> voxel budget / grid size, heads points, prompts) x seeds on given meshes.

usage: python tune_run2.py --work work --meshes a.glb --out raw --configs '[{"name":"b16","dense":100000,"budget":16000,"heads":16384,"prompts":128}]' --seeds 1 2 3
config keys: name, dense (points through the encoder), budget (max voxels, adaptive grid size) or grid, heads (points the prompt heads see), prompts
writes raw/<mesh>__<cfg>__<seed>.npz + raw/<mesh>.npz + raw/runs.json (appended)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "phone"))
import p3s_np as N  # noqa: E402
import p3s_split as SP  # noqa: E402
from load_mesh import load_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfgs = json.loads(a.configs)
    model = N.P3Sam(f"{a.work}/p3sam_enc.onnx", f"{a.work}/p3sam_heads.onnx", a.threads)
    runs_path = f"{a.out}/runs{a.tag}.json"
    runs = json.load(open(runs_path)) if os.path.exists(runs_path) else []
    for mp in a.meshes:
        v, f, name = load_mesh(mp)
        v, f = SP.clean(v, f)
        np.savez_compressed(f"{a.out}/{name}.npz", v=v.astype(np.float32), f=f.astype(np.int32))
        for c in cfgs:
            for seed in (c.get("seeds") or a.seeds):
                t = time.time()
                pts, nrm, fi = N.sample_surface(v, f, c["dense"], seed)
                pn = N.normalize_pc(pts).astype(np.float32)
                sub = np.unique(np.linspace(0, len(pn) - 1, c["heads"]).astype(np.int64)) if c["heads"] < len(pn) else np.arange(len(pn))
                feats = model.features(pn, nrm.astype(np.float32), sub, budget=c.get("budget"), grid_size=c.get("grid", N.GRID))
                te = time.time() - t
                ph = pn[sub]
                sel = N.fps(ph, c["prompts"])
                logits, iou = model.masks(feats, ph, ph[sel])
                j = iou.argmax(1)
                m = np.take_along_axis(logits, j[None, :, None], 2)[:, :, 0] > 0
                tag = f"{name}__{c['name']}__{seed}"
                np.savez_compressed(f"{a.out}/{tag}.npz", pts=pts[sub].astype(np.float32), n=len(ph), masks=np.packbits(m, axis=0), iou=iou, best=j, prompts=sel)
                runs.append(dict(tag=tag, mesh=name, cfg=c["name"], seed=seed, grid=round(float(model.grid_size), 5), voxels=model.voxels, enc_s=round(te, 1), total_s=round(time.time() - t, 1)))
                print("RAW", json.dumps(runs[-1]), flush=True)
                json.dump(runs, open(runs_path, "w"), indent=1)


if __name__ == "__main__":
    main()
