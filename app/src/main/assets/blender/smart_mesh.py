"""Smart Mesh: turn a dense AI mesh into a clean all-quad game / animation mesh with CurioMesh (runs in headless Blender 4.2).

The raw Hunyuan mesh is too noisy for retopology (CurioMesh split it into 540 parts and left holes), so it is
voxel-remeshed and smoothed first, then CurioMesh (CurioCrafter/CurioMesh, GPL-3, from the curiomesh pack)
reduces it part by part and checks the result against the original from six sides.

usage: blender -b --factory-startup --python smart_mesh.py -- --in in.glb --out DIR --use GAME --faces 20000 [--name N]
"""
import argparse
import json
import os
import runpy
import sys
import zipfile

import bpy

PACK = "/opt/curiomesh"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Smart Mesh")
    p.add_argument("--use", default="GAME", choices=["LOW_POLY", "GAME", "ANIMATION"])
    p.add_argument("--faces", type=int, default=20000)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:])


def addon():
    """Unpack the add-on once (the pack ships it as one zip; the app's downloader only handles flat files)."""
    root = os.path.join(PACK, "curiomesh")
    if not os.path.isfile(os.path.join(root, "__init__.py")):
        with zipfile.ZipFile(os.path.join(PACK, "curiomesh.zip")) as z:
            z.extractall(PACK)
    return root


def clean(src, dst):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    print("Smart Mesh: cleaning", len(o.data.polygons), "faces", flush=True)
    m = o.modifiers.new("r", "REMESH"); m.mode = "VOXEL"; m.voxel_size = max(o.dimensions) / 220; m.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier="r")
    s = o.modifiers.new("s", "CORRECTIVE_SMOOTH"); s.iterations = 8; s.factor = 0.6
    bpy.ops.object.modifier_apply(modifier="s")
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True)


def main():
    a = args()
    os.makedirs(a.out, exist_ok=True)
    tmp = "/tmp/smart_clean.glb"
    clean(a.inp, tmp)
    os.environ.update(CURIOMESH_GPU="0", CURIOMESH_HELPERS="0")  # no Vulkan on the phone; one Blender process at a time
    root = addon()
    out, rep = os.path.join(a.out, "smart.glb"), "/tmp/smart_report.json"
    runner = os.path.join(root, "skills", "curiomesh-retopo", "scripts", "curiomesh_run.py")
    sys.argv = [sys.argv[0], "--", tmp, "--use", a.use, "--reduce-to", str(a.faces), "--out", out, "--report", rep, "--addon", root]
    print("Smart Mesh: CurioMesh", a.use, a.faces, "faces", flush=True)
    code = runpy.run_path(runner, run_name="curiomesh_run")["main"]()
    r = json.load(open(rep))
    if code != 0 or not r.get("ok") or not os.path.isfile(out):
        raise SystemExit("CurioMesh failed: " + str(r.get("error") or r.get("verdict")))
    o = bpy.data.objects.get(r.get("out_object"))
    info = {"name": a.name, "vertices": len(o.data.vertices) if o else 0, "triangles": r.get("faces", 0),
            "smart_mesh": {k: r.get(k) for k in ("use", "faces", "quads", "tris", "quad_pct", "view_changed_pct", "parts", "seconds", "verdict")}}
    json.dump(info, open(os.path.join(a.out, "generation.json"), "w"), indent=1)
    print("Smart Mesh:", json.dumps(info["smart_mesh"]), flush=True)


main()
