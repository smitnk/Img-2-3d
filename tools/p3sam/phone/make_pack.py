"""Assemble the p3sam-v1 pack: ONNX graphs (weights as one external .data each), scripts + wheels zip, licence, SOURCE.txt, p3sam_info.json.

  python make_pack.py --onnx DIR (p3sam_enc.onnx[.data], p3sam_heads.onnx[.data]) --whl DIR (cp312 aarch64 wheels) --license LICENSE.txt --out DIR [--version 1]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["p3sam_phone.py", "p3s_np.py", "p3s_post.py", "p3s_split.py"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--whl", required=True)
    ap.add_argument("--license", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--source", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in ("p3sam_enc.onnx", "p3sam_enc.onnx.data", "p3sam_heads.onnx", "p3sam_heads.onnx.data"):
        shutil.copy2(os.path.join(a.onnx, n), os.path.join(a.out, n))
    shutil.copy2(a.license, os.path.join(a.out, "LICENSE.txt"))
    with zipfile.ZipFile(os.path.join(a.out, "p3sam_py.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for s in SCRIPTS:
            z.write(os.path.join(HERE, s), s)
        for w in sorted(glob.glob(os.path.join(a.whl, "*.whl"))):  # wheels unpacked into pylibs/ (put on sys.path by p3sam_split.py)
            with zipfile.ZipFile(w) as wz:
                for info in wz.infolist():
                    if not info.is_dir():
                        z.writestr("pylibs/" + info.filename, wz.read(info))
    open(os.path.join(a.out, "SOURCE.txt"), "w").write(
        "P3-SAM from Tencent Hunyuan3D-Part (https://huggingface.co/tencent/Hunyuan3D-Part, weights p3sam/p3sam.safetensors; licence in LICENSE.txt:\n"
        "NOT licensed in the European Union, the United Kingdom and South Korea). The ONNX graphs are exported from those weights by tools/p3sam/ in the\n"
        "Img-2-3d repository (dyn_build.py); p3sam_py.zip holds the numpy port tools/p3sam/phone/ and cp312 aarch64 wheels (onnxruntime, scipy, manifold3d), unpacked.\n" + a.source + "\n")
    files = {}
    for f in sorted(os.listdir(a.out)):
        assert f.replace("_", "").replace(".", "").replace("-", "").isalnum(), f
        assert os.path.getsize(os.path.join(a.out, f)) < 2 ** 31, f
        files[f] = {"bytes": os.path.getsize(os.path.join(a.out, f)), "sha256": sha(os.path.join(a.out, f))}
    files.pop("p3sam_info.json", None)
    json.dump({"version": a.version, "model": "P3-SAM (Hunyuan3D-Part)", "precision": a.precision, "license": "Tencent Hunyuan3D-Part (not EU, UK, South Korea)", "files": files},
              open(os.path.join(a.out, "p3sam_info.json"), "w"), indent=1)
    print(sum(x["bytes"] for x in files.values()) / 2 ** 20, "MB total;", *files, sep="\n  ")


if __name__ == "__main__":
    main()
