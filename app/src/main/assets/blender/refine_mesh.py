"""Refine a TripoSR mesh into a clean character asset (runs inside headless Blender 4.2).

Same idea as Model Forge's lizard_refined generator (TripoSR template -> Blender clean-up):
import GLB -> join -> upright + scale to target height, feet on the ground -> voxel remesh (watertight,
even quads) -> smooth -> transfer TripoSR's vertex colours back -> optional decimate -> optional rig
-> GLB export.

usage: blender -b --factory-startup --python refine_mesh.py -- --in in.glb --out DIR [options]
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, "/opt/forge/scripts")
import generate_model as gm  # noqa: E402
import export_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "bx")]
import vcbake  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Refined Model")
    p.add_argument("--height", type=float, default=1.0, help="target height in metres")
    p.add_argument("--voxel", type=float, default=0.008, help="remesh voxel size relative to height")
    p.add_argument("--smooth", type=int, default=4, help="smoothing iterations")
    p.add_argument("--decimate", type=float, default=1.0, help="keep this fraction of faces (0.05-1)")
    p.add_argument("--keep-colors", action="store_true", default=True)
    p.add_argument("--no-colors", dest="keep_colors", action="store_false")
    p.add_argument("--rig", action="store_true", help="fit the 17-bone humanoid rig (best for T/A-posed figures)")
    p.add_argument("--texture", type=int, default=0, help="bake vertex colours into a UV texture of this size (0 = off)")
    p.add_argument("--blend", action="store_true")
    return p.parse_args(argv)


def bake(obj, *mods):
    for kind, props in mods:
        m = obj.modifiers.new(kind.lower(), kind)
        for k, v in props.items():
            setattr(m, k, v)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)


def rigify(body):
    """Rigify rig fitted from mesh landmarks + verified weights (bx_rig). (None, why) if the mesh is not a
    T/A-posed humanoid or binding fails; the caller then falls back to the simple distance-weighted rig."""
    import bx_rig as R
    meta = rig = None
    try:
        R.model_check([body], fix=True)
        meta = R.add_metarig("basic")
        R.fit_metarig(meta, R.landmarks(body))
        problems = R.check_metarig(meta, body)
        if problems:
            raise RuntimeError("; ".join(problems[:3]))
        rig, err = R.generate(meta)
        if rig is None:
            raise RuntimeError(err)
        rep = R.bind(rig, body)
        if not rep.get("ok"):
            raise RuntimeError(f"weights not usable ({rep.get('route')})")
        bpy.data.objects.remove(meta, do_unlink=True)
        deform = sum(1 for b in rig.data.bones if b.use_deform)
        print(f"rigify rig: {deform} deform bones, route {rep.get('route')}")
        return rig, {"bones": deform, "skinning": f"Rigify, bone heat ({rep.get('route')})", "route": rep.get("route")}
    except Exception as e:
        print("rigify fallback:", e)
        for o in (rig, meta):
            if o is not None and o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        body.parent = None
        body.modifiers.clear()
        body.vertex_groups.clear()
        bpy.context.view_layer.objects.active = body
        return None, {"rigify_error": str(e)[:200]}


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    gm.reset_scene()
    bpy.ops.import_scene.gltf(filepath=a.inp)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("Input GLB contains no mesh")
    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.context.scene.objects):
        if o is not src:
            bpy.data.objects.remove(o, do_unlink=True)

    # Scale to target height, centre on X/Y, feet at z=0 (glTF import already converts Y-up to Z-up).
    pts = [v.co for v in src.data.vertices]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    s = a.height / max(hi.z - lo.z, 1e-6)
    c = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for v in src.data.vertices:
        v.co = (v.co - c) * s
    faces_in = len(src.data.polygons)

    out = src.copy()
    out.data = src.data.copy()
    out.name = a.name
    bpy.context.scene.collection.objects.link(out)
    bake(out, ("REMESH", {"mode": "VOXEL", "voxel_size": max(0.002, a.voxel * a.height), "use_smooth_shade": True}),
         ("SMOOTH", {"factor": 0.6, "iterations": max(0, a.smooth)}))

    colors = src.data.color_attributes
    if a.keep_colors and len(colors):
        # Voxel remesh drops attributes; copy TripoSR's per-vertex colour back from the original surface.
        name = colors.active_color.name if colors.active_color else colors[0].name
        out.data.color_attributes.new(name, colors[name].data_type, "POINT")
        dt = out.modifiers.new("colors", "DATA_TRANSFER")
        dt.object = src
        dt.use_vert_data = True
        dt.data_types_verts = {"COLOR_VERTEX"}
        dt.vert_mapping = "POLYINTERP_NEAREST"
        # defaults: all source colour layers -> matching layer names on the target
        bake(out)
        out.data.color_attributes.active_color = out.data.color_attributes[name]
        # Vertex colours only reach the GLB through a material that uses them.
        mat = bpy.data.materials.new("VertexColor")
        mat.use_nodes = True
        nt = mat.node_tree
        attr = nt.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = name
        nt.links.new(attr.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
        out.data.materials.clear()
        out.data.materials.append(mat)
    if 0.0 < a.decimate < 1.0:
        bake(out, ("DECIMATE", {"ratio": max(0.02, a.decimate)}))
    bpy.data.objects.remove(src, do_unlink=True)
    for poly in out.data.polygons:
        poly.use_smooth = True

    texture = None
    if a.texture and len(out.data.color_attributes):
        try:
            texture = vcbake.bake(out, a.texture)
        except Exception as e:  # GPU/EGL unavailable etc: vertex colours still carry the look
            print("texture bake skipped:", e)

    rig, rig_info = None, None
    if a.rig:
        rig, rig_info = rigify(out)
    if a.rig and rig is None:
        H = a.height
        z = lambda f: f * H  # noqa: E731
        lo2 = Vector((min(v.co.x for v in out.data.vertices), 0, 0))
        hi2 = Vector((max(v.co.x for v in out.data.vertices), 0, 0))
        reach = min(-lo2.x, hi2.x)
        bones = [("root", (0, 0, 0), (0, 0, z(0.08)), None), ("pelvis", (0, 0, z(0.48)), (0, 0, z(0.55)), "root"),
                 ("spine", (0, 0, z(0.55)), (0, 0, z(0.78)), "pelvis"), ("neck", (0, 0, z(0.78)), (0, 0, z(0.83)), "spine"),
                 ("head", (0, 0, z(0.83)), (0, 0, z(1.0)), "neck")]
        for sd, x in (("L", 1), ("R", -1)):
            bones += [(f"upper_arm.{sd}", (x * reach * 0.18, 0, z(0.76)), (x * reach * 0.5, 0, z(0.76)), "spine"),
                      (f"forearm.{sd}", (x * reach * 0.5, 0, z(0.76)), (x * reach * 0.82, 0, z(0.76)), f"upper_arm.{sd}"),
                      (f"hand.{sd}", (x * reach * 0.82, 0, z(0.76)), (x * reach, 0, z(0.76)), f"forearm.{sd}"),
                      (f"thigh.{sd}", (x * 0.09 * H, 0, z(0.48)), (x * 0.09 * H, 0, z(0.26)), "pelvis"),
                      (f"shin.{sd}", (x * 0.09 * H, 0, z(0.26)), (x * 0.09 * H, 0, z(0.05)), f"thigh.{sd}"),
                      (f"foot.{sd}", (x * 0.09 * H, 0, z(0.05)), (x * 0.09 * H, -0.1 * H, z(0.01)), f"shin.{sd}")]
        out["bone"] = ""  # blended distance weights (Model Forge auto_weights)
        rig = gm.build_rig([out], bones)
        rig_info = dict(rig_info or {}, bones=17, skinning="blended distance weights")

    slug = gm.slugify(a.name) or "refined"
    files = {"glb": vcbake.export_glb(os.path.join(a.out, f"{slug}.glb"), deform_only=bool(rig_info and rig_info.get("route")))}
    if a.blend:
        files["blend"] = export_model.save_blend(os.path.join(a.out, f"{slug}.blend"))
    info = {"generator": "refine_mesh", "blender": bpy.app.version_string, "name": a.name, "slug": slug,
            "faces_in": faces_in, "triangles": sum(len(p.vertices) - 2 for p in out.data.polygons),
            "vertices": len(out.data.vertices), "colors": bool(len(out.data.color_attributes)),
            "texture": texture, "rig": rig_info if rig else None, "files": files,
            "params": {k: v for k, v in vars(a).items() if k not in ("out", "inp")}}
    with open(os.path.join(a.out, "generation.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: info[k] for k in ("faces_in", "triangles", "colors")}))


if __name__ == "__main__":
    main()
