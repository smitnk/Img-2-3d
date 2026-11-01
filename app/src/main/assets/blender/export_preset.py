"""Re-export a model for a target engine with bx_export presets (scenario-labs, MIT).

blender -b --factory-startup --python export_preset.py -- --in model.glb --preset unity --out DIR
Presets: unity, unity_skeletal, unreal, unreal_skeletal, roblox, godot, gltf, web, print. Writes DIR/<name>.<ext>.
"""
import argparse
import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "bx")]
import bx_export  # noqa: E402
import vcbake  # noqa: E402

PRESETS = ("unity", "unity_skeletal", "unreal", "unreal_skeletal", "roblox", "godot", "gltf", "web", "print")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--preset", required=True, choices=PRESETS)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="model")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    vcbake.import_glb(a.inp)
    ext = bx_export.PRESETS[a.preset][0]
    skeletal = a.preset.endswith("_skeletal") or a.preset == "roblox"
    objs = [o for o in bpy.context.scene.objects
            if o.type == "MESH" or (o.type == "ARMATURE" and skeletal) or (o.type == "EMPTY" and ext != ".stl")]
    path = os.path.join(a.out, (bpy.path.clean_name(a.name) or "model") + ext)
    r = bx_export.export(path, a.preset, objs)
    info = {"file": os.path.basename(path), "preset": a.preset, "bytes": r["bytes"], "objects": len(r["objects"])}
    with open(os.path.join(a.out, "export.json"), "w") as f:
        json.dump(info, f)
    print("EXPORT", json.dumps(info))


if __name__ == "__main__":
    main()
