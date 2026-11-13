"""Assemble the phone's ONNX pack from the artifacts of unirig-onnx-skel.yml (onnx-skel) and unirig-onnx-skin.yml (unirig-skin-onnx).

  python make_pack.py --skel DIR --skin DIR --out DIR
Copies the graphs, embed.npy, info.json and writes skin_meta.json (the spconv neighbour policy read off the real
layer: tie / self-centre flag / kernel offsets, identical for every mesh) from one of the captured inputs_*.npz.
skel_decoder.onnx is first dynamically quantized to int8 (MatMul/Gemm weights; needs onnxruntime). Every graph is rewritten with its weights as ONE external file <graph>.onnx.data (onnxruntime memory-maps those instead of copying them
into RAM; loose per-tensor files of the legacy exporter are merged). The skin network ships as skin_a / skin_pt / skin_head.onnx when the
export made them (the single skin.onnx is then left out), else as skin.onnx. Asset names must match [A-Za-z0-9_.-]+ and stay under 2 GB.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import onnx_skin as K  # noqa: E402  (externalize, PARTS)

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
if K.has_parts(a.out):  # the parts replace the monolith (same weights, split): do not ship both
    for f in [f for f in os.listdir(a.out) if f.startswith("skin.onnx")]:
        os.remove(f"{a.out}/{f}")
# int8 weights for the decoder (fp32 + beam KV cache got Blender killed on the phone); encoder / skin stay fp32; embed.npy is separate
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
dec, tmp = f"{a.out}/skel_decoder.onnx", f"{a.out}/_q.onnx"
quantize_dynamic(dec, tmp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"], per_channel=False)
os.remove(dec)
if os.path.exists(dec + ".data"):
    os.remove(dec + ".data")
os.replace(tmp, dec)
for g in sorted(f for f in os.listdir(a.out) if f.endswith(".onnx")):
    K.externalize(f"{a.out}/{g}", f"{a.out}/{g}")  # in place: the whole graph is in memory before the old files are replaced
    print("externalized", g, round(os.path.getsize(f"{a.out}/{g}.data") / 2 ** 30, 2), "GB")
for f in os.listdir(a.out):  # loose per-tensor files of the legacy exporter, now merged
    if not (f.endswith((".onnx", ".onnx.data")) or f in ("embed.npy", "info.json")):
        os.remove(f"{a.out}/{f}")
cap = np.load(sorted(glob.glob(f"{a.skin}/inputs_*.npz"))[0])
json.dump({"tie": int(cap["tie"]), "sc": bool(cap["sc"]), "offs3": cap["offs3"].tolist(), "offs5": cap["offs5"].tolist()},
          open(f"{a.out}/skin_meta.json", "w"))
need = ["skel_encoder.onnx", "skel_decoder.onnx", "embed.npy", "info.json", "skin_meta.json"] + ([] if K.has_parts(a.out) else ["skin.onnx"])
missing = [f for f in need if not os.path.exists(f"{a.out}/{f}")]
assert not missing, f"pack incomplete: {missing}"
print("pack:", sum(os.path.getsize(f"{a.out}/{f}") for f in os.listdir(a.out)) / 2 ** 30, "GB,", len(os.listdir(a.out)), "files")
