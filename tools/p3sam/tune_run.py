"""Raw P3-SAM masks for meshes x seeds x configs through the numpy + ONNX path (deterministic), dumped for post-processing tuning.

usage: python tune_run.py --work work --meshes a.glb b.glb --out raw [--seeds 1 2 3] [--points 16384 24576] [--prompts 128]
writes raw/<mesh>_<points>_<seed>.npz (points, normals, masks packed, iou, prompts) and raw/<mesh>.npz (mesh) + raw/runs.json
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
from load_mesh import load_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--meshes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--points", type=int, nargs="+", default=[16384, 24576])
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = N.P3Sam(f"{a.work}/p3sam_enc.onnx", f"{a.work}/p3sam_heads.onnx", a.threads)
    runs = []
    for mp in a.meshes:
        v, f, name = load_mesh(mp)
        np.savez_compressed(f"{a.out}/{name}.npz", v=v.astype(np.float32), f=f.astype(np.int32))
        for npts in a.points:
            for seed in a.seeds:
                t = time.time()
                pts, nrm, fi = N.sample_surface(v, f, npts, seed)
                pn = N.normalize_pc(pts).astype(np.float32)
                feats = model.features(pn, nrm.astype(np.float32))
                te = time.time() - t
                sel = N.fps(pn, a.prompts)
                logits, iou = model.masks(feats, pn, pn[sel])
                j = iou.argmax(1)
                m = np.take_along_axis(logits, j[None, :, None], 2)[:, :, 0] > 0
                tag = f"{name}_{npts}_{seed}"
                np.savez_compressed(f"{a.out}/{tag}.npz", pts=pts.astype(np.float32), pn=pn, nrm=nrm.astype(np.float32), fi=fi.astype(np.int32),
                                    masks=np.packbits(m, axis=0), n=len(pn), iou=iou, best=j, prompts=sel)
                runs.append(dict(tag=tag, mesh=name, points=npts, seed=seed, enc_s=round(te, 1), total_s=round(time.time() - t, 1)))
                print("RAW", json.dumps(runs[-1]), flush=True)
                json.dump(runs, open(f"{a.out}/runs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
