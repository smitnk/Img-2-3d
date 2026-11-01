"""Which scenario-labs Blender expert modules work on our Blender 4.2.3 runtime? (CI probe, not shipped)"""
import importlib, json, os, sys, traceback
import bpy

root = "/opt/scen/skills/dcc/blender"
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "scripts")
    if os.path.isdir(p):
        sys.path.insert(0, p)
res = {}

def step(name, fn):
    try:
        out = fn()
        res[name] = "OK " + (json.dumps(out, default=str)[:160] if out is not None else "")
    except Exception as e:
        res[name] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
        traceback.print_exc()
    print("BXPROBE", name, res[name], flush=True)

mods = ["bx_audit", "bx_review", "bx_gui", "bx_export", "bx_uvbake", "bx_retopo", "bx_rig", "bx_anim",
        "bx_materials", "bx_light", "bx_sculpt", "bx_hardsurface", "bx_gn", "bx_hair", "bx_gp", "bx_previs", "bx_blockout", "bx_kits"]
M = {}
for m in mods:
    step("import " + m, lambda m=m: M.__setitem__(m, importlib.import_module(m)))

def builder_glb():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath="/opt/test.glb")
    return [o for o in bpy.data.objects if o.type == "MESH"]

def audit():
    objs = builder_glb()
    r = M["bx_audit"].deliverable(objs, profile="game")
    return {"pass": r["pass"], "errors": r["errors"][:3], "warnings": r["warnings"][:3]}
step("audit.deliverable(builder glb)", audit)

def review():
    objs = builder_glb()
    os.makedirs("/tmp/rev", exist_ok=True)
    return M["bx_review"].review(objs, "/tmp/rev")
step("review.review(builder glb)", review)

def export_presets():
    return sorted(getattr(M["bx_export"], "PRESETS", {}).keys())
step("export.PRESETS", export_presets)

def export_glb():
    builder_glb()
    return M["bx_export"].round_trip("/tmp/rt.glb", preset="godot") if "godot" in getattr(M["bx_export"], "PRESETS", {}) else "no godot preset"
step("export.round_trip glb", export_glb)

def mannequin_rig():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    R = M["bx_rig"]
    body = R.mannequin()
    body = body[0] if isinstance(body, (list, tuple)) else body
    lm = R.landmarks(body)
    R.enable_rigify()
    meta = R.add_metarig("basic")
    R.fit_metarig(meta, lm)
    rig = R.generate(meta)
    return {"rig": getattr(rig, "name", str(rig)), "bones": len(rig.data.bones) if hasattr(rig, "data") else None}
step("rig.mannequin+rigify", mannequin_rig)

def uv():
    objs = builder_glb()
    o = max(objs, key=lambda o: len(o.data.polygons))
    M["bx_uvbake"].unwrap(o)
    return M["bx_uvbake"].uv_verdict(M["bx_uvbake"].uv_qa(o))
step("uvbake.unwrap+qa", uv)

print("BXPROBE_SUMMARY " + json.dumps(res))
