"""CI test for rig_cleanup.py: bend each elbow 90 deg, twist each wrist 90 deg, measure the forearm cross-section and render previews (Workbench).

blender -b --factory-startup --python rig_test.py -- --glb in.glb --label after --out DIR [--enforce 0.60]
Prints "RATIO forearm.L twist=.. bend=.." (mid-forearm slice: min caliper width deformed / rest) and "MIN forearm.L twist=.. (at s=..) bend=.. (at s=..)"
(the worst of 7 slices along the forearm, informational). With --enforce, exits 1 if any RATIO is below the limit.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app", "src", "main", "assets", "blender"))
import rig_cleanup as rc  # noqa: E402  (its main() only runs as __main__)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--enforce", type=float, default=0.0)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:])


def min_width(Q, axis):
    """smallest caliper width of points Q projected on the plane perpendicular to axis"""
    axis = axis / np.linalg.norm(axis)
    e1 = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(axis, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    X = np.stack([Q @ e1, Q @ e2], 1)
    best = 1e9
    for th in np.linspace(0, math.pi, 180, endpoint=False):
        p = X @ np.array([math.cos(th), math.sin(th)])
        best = min(best, p.max() - p.min())
    return best


def set_pose(arm, ops):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    for name, q in ops:
        arm.pose.bones[name].rotation_quaternion = q
    bpy.context.view_layer.update()


def render(arm, label, name, closeup_bone, out, top=False):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x, sc.render.resolution_y = 900, 700
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "SINGLE"
    sc.display.shading.single_color = (0.8, 0.75, 0.7)
    sc.display.shading.show_cavity = True
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.color = (0.25, 0.27, 0.3)
    cam = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    sc.camera = cam
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    pts = []
    for o in meshes:
        ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        m = ev.to_mesh()
        pts += [o.matrix_world @ v.co for v in list(m.vertices)[::20]]
        ev.to_mesh_clear()
    P = np.array([tuple(p) for p in pts])
    lo, hi = P.min(0), P.max(0)
    ctr = (lo + hi) / 2
    views = {"full": (ctr, max(hi[0] - lo[0], hi[2] - lo[2]) * 1.1)}
    pb = arm.pose.bones[closeup_bone]
    views["arm"] = (np.array(arm.matrix_world @ ((pb.head + pb.tail) / 2)), 0.55)
    for vn, (c, scale) in views.items():
        cam.data.ortho_scale = scale
        if top:  # bend: the forearm swings forward (-Y), so look down from above
            cam.location, cam.rotation_euler = (c[0], c[1], c[2] + 5), (0, 0, 0)
        else:  # in front of the character (it faces -Y), looking along +Y
            cam.location, cam.rotation_euler = (c[0], c[1] - 5, c[2]), (math.radians(90), 0, 0)
        sc.render.filepath = os.path.join(out, f"{label}_{name}_{vn}.png")
        bpy.ops.render.render(write_still=True)


def main():
    a = args()
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.glb)
    arm, meshes = rc.find_rig()
    print(f"{a.label}: restored {rc.restore_twist_constraints(arm)} twist constraints from node extras")
    obj = max(meshes, key=lambda o: len(o.data.vertices))
    A = rc.to_arm_matrix(arm, obj)
    A3, At = rc.np_mat(A)[:3, :3], rc.np_mat(A)[:3, 3]
    P = np.array([tuple(A @ v.co) for v in obj.data.vertices])
    W = rc.read_weights(obj)
    gnames = [g.name for g in obj.vertex_groups]
    bad = []
    for side in rc.SIDES:
        fore, hand, upper = f"forearm.{side}", f"hand.{side}", f"upper_arm.{side}"
        if fore not in arm.data.bones or hand not in arm.data.bones:
            continue
        fam = [i for i, n in enumerate(gnames) if n == fore or n.startswith(f"forearm_twist") and n.endswith(f".{side}")]
        famw = W[:, fam].sum(1)
        h, t = rc.limb_ends(arm, fore)
        ax = t - h
        s = (P - h) @ ax / (ax @ ax)
        slices = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
        sel = {c: np.nonzero((famw >= 0.5) & (abs(s - c) < 0.04))[0] for c in slices}
        rest = {c: min_width(P[i], ax) for c, i in sel.items() if len(i) > 8}
        res = {}
        for pose, ops in (("bend", [(fore, rc.flex_quat(arm, upper, fore, (0, -1, 0)))]),
                          ("twist", [(hand, Quaternion((0, 1, 0), math.radians(90)))])):
            set_pose(arm, ops)
            sk = obj.data.shape_keys  # drivers are not in the GLB: a runtime sets the corrective weight from the bend angle (1.0 at 90 deg)
            for kb in (sk.key_blocks if sk else []):
                kb.value = 1.0 if (pose == "bend" and kb.name == f"corr_elbow.{side}") else 0.0
            bpy.context.view_layer.update()
            ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            m = ev.to_mesh()
            co = np.empty(len(m.vertices) * 3, np.float32)
            m.vertices.foreach_get("co", co)
            ev.to_mesh_clear()
            D = co.reshape(-1, 3).astype(np.float64) @ A3.T + At
            ph, pt = rc.limb_ends(arm, fore, pose=True)
            pax = pt - ph
            res[pose] = {c: min_width(D[i], pax) / rest[c] for c, i in sel.items() if c in rest}
            if side == "L":
                render(arm, a.label, pose, fore, a.out, top=(pose == "bend"))
        set_pose(arm, [])
        mid = {p: res[p][0.5] for p in res}
        worst = {p: min(res[p].items(), key=lambda kv: kv[1]) for p in res}
        print(f"RATIO {fore} twist={mid['twist']:.3f} bend={mid['bend']:.3f}")
        print(f"MIN {fore} twist={worst['twist'][1]:.3f} (at s={worst['twist'][0]}) bend={worst['bend'][1]:.3f} (at s={worst['bend'][0]})")
        bad += [(fore, p, v) for p, v in mid.items() if v < a.enforce]
    if bad:
        print("FAIL below", a.enforce, bad)
        sys.exit(1)


main()
