"""Hunyuan Paint: all-sides AI texture from 1-4 photos, on the phone (headless Blender 4.2, onnxruntime, no torch).

The pack (/opt/hunyuanpaint, release hunyuanpaint-v1) holds the ONNX graphs plus hunyuanpaint_py.zip (paint_phone.py, paint_np.py and
onnxruntime in pylibs/). This wrapper unpacks the zip (again whenever the pack was updated) and runs paint_phone.py in this Blender.

usage: blender -b --factory-startup --python hunyuan_paint.py -- --mesh in.glb --refs a.png [b.png ...] [--ref-views front back ...]
           --out DIR [--name N] [--threads N]    (threads 0 = auto: one per big core, pinned to them, normal priority)
Writes DIR/painted.glb and DIR/generation.json (vertices / triangles, which the app shows on the library card).
"""
import json
import argparse
import os
import runpy
import shutil
import struct
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


def glb_counts(path):
    """(vertices, triangles) of a GLB from its JSON chunk (fallback when the run's stats.json is missing)."""
    with open(path, "rb") as f:
        f.read(12)
        n, _ = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(n))
    acc, v, t = j.get("accessors", []), 0, 0
    for m in j.get("meshes", []):
        for pr in m.get("primitives", []):
            v += acc[pr["attributes"]["POSITION"]]["count"]
            t += acc[pr["indices"]]["count"] // 3 if "indices" in pr else acc[pr["attributes"]["POSITION"]]["count"] // 3
    return v, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--refs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Painted")
    p.add_argument("--ref-views", nargs="*", default=None, help="front/left/back/right label of each --refs photo")
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = unpack()
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(PACK, "work")  # in the pack dir (app storage), not in --out which the app clears for every run
    shutil.rmtree(work, ignore_errors=True)
    out = os.path.join(a.out, "painted.glb")
    argv = ["--mesh", a.mesh, "--refs", *a.refs, "--out", out, "--models", PACK, "--pylibs", os.path.join(root, "pylibs"),
            "--work", work, "--threads", str(a.threads)]
    if a.ref_views:
        argv += ["--ref-views", *a.ref_views]
    # paint_phone re-launches Blender on this file (prep / export phases) with sys.argv after "--": give it its own arguments
    sys.argv = [sys.argv[0], "--", *argv]
    try:
        runpy.run_path(os.path.join(root, "paint_phone.py"), run_name="__main__")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    st = os.path.join(a.out, "stats.json")
    try:
        d = json.load(open(st))
        counts = (d["vertices"], d["triangles"])
    except Exception:
        counts = glb_counts(out)
    json.dump({"script": "hunyuan_paint", "name": a.name, "vertices": counts[0], "triangles": counts[1], "photos": len(a.refs)},
              open(os.path.join(a.out, "generation.json"), "w"))
    print(f"Hunyuan Paint done: {out} ({counts[0]} vertices, {counts[1]} triangles)", flush=True)


main()
