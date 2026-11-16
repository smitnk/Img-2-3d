"""Hunyuan Paint: all-sides AI texture from 1-4 photos, on the phone (headless Blender 4.2, onnxruntime, no torch).

The pack (/opt/hunyuanpaint, release hunyuanpaint-v1) holds the ONNX graphs plus hunyuanpaint_py.zip (paint_phone.py, paint_np.py and
onnxruntime in pylibs/). This wrapper unpacks the zip (again whenever the pack was updated) and runs paint_phone.py in this Blender.

usage: blender -b --factory-startup --python hunyuan_paint.py -- --mesh in.glb --refs a.png [b.png ...] --out DIR [--name N] [--threads 2]
"""
import argparse
import os
import runpy
import shutil
import sys
import zipfile

PACK = "/opt/hunyuanpaint"


def unpack():
    root, zp = os.path.join(PACK, "hunyuanpaint_py"), os.path.join(PACK, "hunyuanpaint_py.zip")
    stamp = os.path.join(root, ".stamp")
    want = f"{os.path.getsize(zp)}-{int(os.path.getmtime(zp))}"
    if not (os.path.isfile(os.path.join(root, "paint_phone.py")) and os.path.isfile(stamp) and open(stamp).read() == want):
        shutil.rmtree(root, ignore_errors=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        open(stamp, "w").write(want)
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--refs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Painted")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = unpack()
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(PACK, "work")  # in the pack dir (app storage), not in --out which the app clears for every run
    shutil.rmtree(work, ignore_errors=True)
    out = os.path.join(a.out, "painted.glb")
    argv = ["--mesh", a.mesh, "--refs", *a.refs, "--out", out, "--models", PACK, "--pylibs", os.path.join(root, "pylibs"),
            "--work", work, "--threads", str(a.threads), "--nice", "10"]
    # paint_phone re-launches Blender on this file (prep / export phases) with sys.argv after "--": give it its own arguments
    sys.argv = [sys.argv[0], "--", *argv]
    try:
        runpy.run_path(os.path.join(root, "paint_phone.py"), run_name="__main__")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("Hunyuan Paint done:", out, flush=True)


main()
