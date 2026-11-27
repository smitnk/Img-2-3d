"""Builds the esrgan-v1 engine pack: the official Real-ESRGAN ncnn model files + esrgan_info.json (version, files, sha256)."""
import hashlib, json, os, shutil, sys

src, out = sys.argv[1], sys.argv[2]
NAME = "realesrgan-x4plus-anime"
os.makedirs(out, exist_ok=True)
files = {}
for ext in ("param", "bin"):
    f = f"{NAME}.{ext}"
    shutil.copy(os.path.join(src, f), os.path.join(out, f))
    data = open(os.path.join(out, f), "rb").read()
    assert len(data) > 1000, f
    files[f] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
assert open(os.path.join(out, NAME + ".param")).readline().strip() == "7767517", "not an ncnn param file"
info = {"version": 1, "model": NAME, "scale": 4, "prepadding": 10, "license": "BSD-3-Clause (Real-ESRGAN, xinntao)", "files": files}
json.dump(info, open(os.path.join(out, "esrgan_info.json"), "w"), indent=1)
print(json.dumps(info, indent=1), sum(v["bytes"] for v in files.values()), "bytes total")
