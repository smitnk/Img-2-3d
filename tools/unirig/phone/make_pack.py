"""Assemble the phone's ONNX pack from the artifacts of unirig-onnx-skel.yml (onnx-skel) and unirig-onnx-skin.yml (unirig-skin-onnx).

  python make_pack.py --skel DIR --skin DIR --out DIR
Copies the graphs (+ external data), embed.npy, info.json and writes skin_meta.json (the spconv neighbour policy read off the real
layer: tie / self-centre flag / kernel offsets, identical for every mesh) from one of the captured inputs_*.npz.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--skel", required=True)
ap.add_argument("--skin", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
skip = ("inputs_", "report_", "ref_", "cond_", "beam_ids_")
for src in (a.skel, a.skin):
    for f in sorted(os.listdir(src)):
        if f.startswith(skip) or not os.path.isfile(f"{src}/{f}"):
            continue
        assert not os.path.exists(f"{a.out}/{f}"), f"{f} exists in both artifacts"
        shutil.copy2(f"{src}/{f}", f"{a.out}/{f}")  # not os.link: onnx refuses external data files with several hard links
cap = np.load(sorted(glob.glob(f"{a.skin}/inputs_*.npz"))[0])
json.dump({"tie": int(cap["tie"]), "sc": bool(cap["sc"]), "offs3": cap["offs3"].tolist(), "offs5": cap["offs5"].tolist()},
          open(f"{a.out}/skin_meta.json", "w"))
need = ["skel_encoder.onnx", "skel_decoder.onnx", "skin.onnx", "embed.npy", "info.json", "skin_meta.json"]
missing = [f for f in need if not os.path.exists(f"{a.out}/{f}")]
assert not missing, f"pack incomplete: {missing}"
print("pack:", sum(os.path.getsize(f"{a.out}/{f}") for f in os.listdir(a.out)) / 2 ** 30, "GB,", len(os.listdir(a.out)), "files")
