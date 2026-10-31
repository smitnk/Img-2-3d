"""Run a user-supplied Blender Python generator and export the result (runs inside headless Blender 4.2).

Contract for custom scripts (uploaded .py files or text-to-model scripts):
  * Build the character with bpy/bmesh in the current (empty) scene. Z is up, front faces -Y,
    feet on z=0, T-pose for humanoids.
  * Optionally read FORGE_PARAMS (dict) and write to FORGE_OUT_DIR; Model Forge's helpers are importable
    (`import generate_model as gm`, `import export_model`).
  * Do NOT export: this runner exports every mesh/armature to GLB afterwards. If the script already wrote
    a .glb into FORGE_OUT_DIR, that file is used instead.

usage: blender -b --factory-startup --python run_custom.py -- --script user.py --out DIR [--name N] [--params JSON]
"""
import argparse
import glob
import json
import os
import runpy
import sys
import traceback

import bpy

sys.path.insert(0, "/opt/forge/scripts")
import export_model  # noqa: E402
import generate_model as gm  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--script", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Custom Model")
    p.add_argument("--params", default="{}")
    p.add_argument("--blend", action="store_true")
    return p.parse_args(argv)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    gm.reset_scene()
    params = json.loads(a.params or "{}")
    os.environ["FORGE_OUT_DIR"] = a.out
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.script)))
    # The user script sees a clean argv (no runner flags) and the documented globals.
    saved_argv = sys.argv
    sys.argv = [a.script]
    try:
        runpy.run_path(a.script, init_globals={"FORGE_PARAMS": params, "FORGE_OUT_DIR": a.out}, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    except Exception:
        traceback.print_exc()
        raise SystemExit("Custom script raised an exception (see traceback above)")
    finally:
        sys.argv = saved_argv

    slug = gm.slugify(a.name) or "custom"
    own = sorted(glob.glob(os.path.join(a.out, "*.glb")))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if own:
        glb = os.path.basename(own[0])
    else:
        if not meshes:
            raise SystemExit("Custom script created no mesh objects")
        for o in bpy.data.objects:  # objects created but never linked would be silently dropped
            if o.users_collection == () and o.type in ("MESH", "ARMATURE"):
                bpy.context.scene.collection.objects.link(o)
        glb = export_model.export_glb(os.path.join(a.out, f"{slug}.glb"))
    files = {"glb": glb}
    if a.blend:
        files["blend"] = export_model.save_blend(os.path.join(a.out, f"{slug}.blend"))
    tris = 0
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        ev = o.evaluated_get(dg)
        tris += sum(len(p.vertices) - 2 for p in ev.data.polygons)
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    info = {"generator": "custom_script", "script": os.path.basename(a.script), "blender": bpy.app.version_string,
            "name": a.name, "slug": slug, "parts": len(meshes), "triangles": tris,
            "rig": {"bones": len(arm[0].data.bones)} if arm else None, "files": files}
    with open(os.path.join(a.out, "generation.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: info[k] for k in ("script", "parts", "triangles")}))


if __name__ == "__main__":
    main()
