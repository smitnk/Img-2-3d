"""Pose a rigged GLB and render rest vs posed views (bpy module). python pose_test.py in.glb out_prefix
Arms (bones well off the centre line, above the hips) swing 50 deg down/forward; legs (below the hips) bend 35 deg.
Also reports how far vertices that should stay still (the opposite side) moved: weight bleeding."""
import math
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]  # bpy module or blender -b --python
src, out = args[0], args[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
dg = bpy.context.evaluated_depsgraph_get()


def verts():
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return np.array([mesh.matrix_world @ v.co for v in ev.data.vertices])


rest = verts()
lo, hi = rest.min(0), rest.max(0)
H = hi[2] - lo[2]
cx = (lo[0] + hi[0]) / 2
print("bones:", [b.name for b in rig.data.bones])
moved_side = {}
for pb in rig.pose.bones:
    h = rig.matrix_world @ pb.bone.head_local
    rel = (h.z - lo[2]) / H
    side = math.copysign(1, h.x - cx)
    pb.rotation_mode = "QUATERNION"
    if abs(h.x - cx) > 0.12 * H and rel > 0.45 and pb.parent and abs((rig.matrix_world @ pb.parent.bone.head_local).x - cx) < abs(h.x - cx):
        if not any(abs((rig.matrix_world @ p.bone.head_local).x - cx) > 0.12 * H for p in pb.parent_recursive):  # shoulder only
            axis = pb.bone.matrix_local.to_3x3().inverted() @ Vector((0, 1, 0))
            pb.rotation_quaternion = Quaternion(axis, math.radians(50) * side)
            moved_side.setdefault("arm", []).append(pb.name)
    elif rel < 0.45 and abs(h.x - cx) > 0.03 * H and pb.parent and not any((rig.matrix_world @ p.bone.head_local).z - lo[2] < 0.45 * H for p in pb.parent_recursive):
        axis = pb.bone.matrix_local.to_3x3().inverted() @ Vector((1, 0, 0))  # hip: swing the leg forward
        pb.rotation_quaternion = Quaternion(axis, math.radians(35) * (1 if side > 0 else -1))
        moved_side.setdefault("leg", []).append(pb.name)
bpy.context.view_layer.update()
posed = verts()
d = np.linalg.norm(posed - rest, axis=1)
head = rest[:, 2] > lo[2] + 0.85 * H
print("POSE driven bones:", moved_side)
print(f"POSE mean move {d.mean() / H:.3f}H, max {d.max() / H:.3f}H, head moved {d[head].max() / H:.3f}H (should be ~0)")

sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x = sc.render.resolution_y = 512
sc.display.shading.light = "STUDIO"
sc.display.shading.color_type = "OBJECT"
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type, cam.data.ortho_scale = "ORTHO", 1.3 * max(hi - lo)
c = (lo + hi) / 2
for name, (pos, rot) in {"front": ((c[0], c[1] - 5 * H, c[2]), (90, 0, 0)), "back": ((c[0], c[1] + 5 * H, c[2]), (90, 0, 180)),
                         "side": ((c[0] + 5 * H, c[1], c[2]), (90, 0, 90))}.items():
    cam.location = pos
    cam.rotation_euler = [math.radians(a) for a in rot]
    cam.data.clip_end = 20 * H
    for state in ("posed", "rest"):
        rig.data.pose_position = "POSE" if state == "posed" else "REST"
        sc.render.filepath = f"{out}_{state}_{name}.png"
        bpy.ops.render.render(write_still=True)
