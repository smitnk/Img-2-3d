"""Auto Split into parts: P3-SAM (Tencent Hunyuan3D-Part) part segmentation on the phone, inside headless Blender 4.2 (onnxruntime, numpy, scipy, manifold3d; no torch).

The pack (/opt/p3sam, release p3sam-v1) holds p3sam_enc.onnx(.data), p3sam_heads.onnx(.data) and p3sam_py.zip (p3sam_phone.py, p3s_np.py,
p3s_post.py, p3s_split.py and the aarch64 wheels unpacked in pylibs/). This wrapper unpacks the zip (again whenever the pack was updated),
reads the GLB with Blender's importer and runs p3sam_phone.run.

usage: blender -b --factory-startup --python p3sam_split.py -- --mesh in.glb --out DIR [--name N] [--target 0] [--pegs 1] [--peg-size 0.4] [--threads 0]
Writes DIR/parts_preview.glb (one colour per part), DIR/parts.zip (one closed GLB per part), DIR/generation.json (vertices, triangles, parts, timings).
"""
import argparse
import os
import shutil
import sys
import zipfile

PACK = "/opt/p3sam"


def unpack():
    root, zp = os.path.join(PACK, "p3sam_py"), os.path.join(PACK, "p3sam_py.zip")
    stamp = os.path.join(root, ".stamp")
    want = f"{os.path.getsize(zp)}-{int(os.path.getmtime(zp))}"
    if not (os.path.isfile(os.path.join(root, "p3sam_phone.py")) and os.path.isfile(stamp) and open(stamp).read() == want):
        shutil.rmtree(root, ignore_errors=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        open(stamp, "w").write(want)
    return root


def read_mesh(path):
    """All mesh objects of a GLB, evaluated (modifiers, pose) and in world space, as glTF Y-up (vertices (n,3) float64, triangles (m,3) int64)."""
    import bpy
    import numpy as np
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    deps = bpy.context.evaluated_depsgraph_get()
    vs, fs, base = [], [], 0
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(deps)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        M = np.array(ob.matrix_world)
        co = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        tri = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        vs.append(co)
        fs.append(tri.reshape(-1, 3).astype(np.int64) + base)
        base += n
        ev.to_mesh_clear()
    if not vs:
        raise SystemExit("no mesh in " + path)
    v = np.concatenate(vs)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1), np.concatenate(fs)  # Blender Z-up -> glTF Y-up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="parts")
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--pegs", type=int, default=1)
    p.add_argument("--peg-size", type=float, default=0.4)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--budget", type=int, default=0, help="max voxels (0 = the pack default)")
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = unpack()
    sys.path.insert(0, os.path.join(root, "pylibs"))
    sys.path.insert(0, root)
    import p3sam_phone
    v, f = read_mesh(a.mesh)
    print(f"Auto Split: {len(v)} vertices, {len(f)} triangles read", flush=True)
    threads = a.threads if a.threads > 0 else min(4, os.cpu_count() or 4)
    p3sam_phone.run(v, f, PACK, a.out, target=a.target, pegs=bool(a.pegs), peg_size=a.peg_size, threads=threads, name=a.name,
                    log=lambda s: print(s, flush=True), budget=a.budget or p3sam_phone.BUDGET)
    print("Auto Split done", flush=True)


main()
