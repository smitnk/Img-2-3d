"""After every job: quality gate (bx_audit.deliverable) + a 3/4 thumbnail, from the finished GLB.

blender -b --factory-startup --python postprocess.py -- --in model.glb --out DIR
Writes DIR/quality.json and DIR/thumb.png. Each part is best-effort: a failure is recorded, never fatal.
"""
import argparse
import json
import math
import os
import sys
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "bx")]
import vcbake  # noqa: E402


def audit(meshes, profile):
    import bx_audit
    r = bx_audit.deliverable(meshes, profile=profile)
    return {"pass": bool(r["pass"]), "profile": profile, "errors": r["errors"][:20], "warnings": r["warnings"][:20],
            "triangles": sum(v.get("tris", 0) for v in r["per_object"].values()), "parts": len(meshes)}


def thumbnail(meshes, path, res, view="threequarter", region="full"):
    import bx_review as V
    textured = any(n.type == "TEX_IMAGE" and n.image for o in meshes for m in o.data.materials if m and m.use_nodes
                   for n in m.node_tree.nodes)
    vcol = any(len(o.data.color_attributes) for o in meshes)
    sc, cam, world = V._setup_scene(meshes, res)
    try:
        sc.render.film_transparent = True
        sh = sc.display.shading
        sh.light, sh.studio_light = "STUDIO", "Default"
        sh.color_type = "TEXTURE" if textured else "VERTEX" if vcol else "MATERIAL"
        sh.show_cavity = True
        sh.cavity_type = "WORLD"
        world.color = (0.05, 0.05, 0.06)
        lo, hi = V._world_bbox(meshes)
        center = (lo + hi) / 2
        pts = V._sample_points(meshes)
        if region != "full":  # frame part of the figure: head = top 28 %, feet = bottom 18 %
            zs = sorted(p.z for p in pts)
            lo_z, hi_z = zs[0], zs[-1]
            cut = (lambda p: p.z > hi_z - 0.28 * (hi_z - lo_z)) if region == "head" else (lambda p: p.z < lo_z + 0.18 * (hi_z - lo_z))
            pts = [p for p in pts if cut(p)] or pts
            center = sum(pts, pts[0] * 0) / len(pts)
        from mathutils import Vector
        d = Vector((0.7, 1.0, 0.35)).normalized() if view == "back" else V.VIEW_DIRS["threequarter"]
        cam.data.type, cam.data.lens = "PERSP", 50
        fov = 2 * math.atan(cam.data.sensor_width / (2 * cam.data.lens))
        dist = V._persp_distance(pts, center, d, math.tan(fov / 2) / 1.08)
        cam.data.clip_start, cam.data.clip_end = dist * 0.01, dist * 10
        V._look_at(cam, center, d, dist)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True, scene=sc.name)
    finally:
        V._cleanup(sc, cam, world)
    return os.path.basename(path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="game", choices=("game", "render", "print", "roblox"))
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--view", default="threequarter", choices=("threequarter", "back"))
    p.add_argument("--region", default="full", choices=("full", "head", "feet"))
    p.add_argument("--thumb-only", action="store_true")
    a = p.parse_args(argv)
    vcbake.import_glb(a.inp)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.polygons)]
    report = {}
    steps = [("thumbnail", lambda: thumbnail(meshes, os.path.join(a.out, "thumb.png"), a.res, a.view, a.region))]
    if not a.thumb_only:
        steps.insert(0, ("quality", lambda: audit(meshes, a.profile)))
    for key, fn in steps:
        try:
            report[key] = fn()
        except Exception as e:
            traceback.print_exc()
            report[key] = None
            report[key + "_error"] = f"{type(e).__name__}: {e}"[:300]
    with open(os.path.join(a.out, "quality.json"), "w") as f:
        json.dump(report, f, indent=2)
    q = report.get("quality") or {}
    print(f"POSTPROCESS pass={q.get('pass')} errors={len(q.get('errors', []))} warnings={len(q.get('warnings', []))} "
          f"thumb={report.get('thumbnail')}")


if __name__ == "__main__":
    main()
