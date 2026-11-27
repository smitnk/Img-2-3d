"""CI helper: char2.glb + a 10-frame animation (both elbows bend 90 deg, both wrists twist 90 deg) -> anim_in.glb
blender -b --factory-startup --python make_anim.py -- IN.glb OUT.glb"""
import math
import sys

import bpy
from mathutils import Quaternion

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for side in "LR":
    for name, q in ((f"forearm.{side}", Quaternion((0, 0, 1), math.radians(90))), (f"hand.{side}", Quaternion((0, 1, 0), math.radians(90)))):
        pb = arm.pose.bones[name]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.keyframe_insert("rotation_quaternion", frame=1)
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=10)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_animations=True)
