# 5 s action scene: stance -> crouch -> flying kick -> land -> punch combo -> victory. blender -b -P action.py -- in.glb outdir [preview]
import bpy, sys, math
from mathutils import Vector, Matrix, Quaternion

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
preview = len(argv) > 2
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.data.objects if o.type == "MESH")
sc = bpy.context.scene
bpy.context.view_layer.objects.active = arm
for o in bpy.data.objects: o.select_set(o == arm)
bpy.ops.object.mode_set(mode="POSE")
B = arm.data.bones
print("BONES", [(b.name, tuple(round(v, 2) for v in b.head_local)) for b in B])
# character axes in armature space: right = side of hand.R, up = +Z, front = cross(up... ) chosen so right x up x front is right-handed
rx = 1 if B["hand.R"].head_local.x > 0 else -1
RIGHT, UP = Vector((rx, 0, 0)), Vector((0, 0, 1))
FRONT = Vector((0, -1, 0))
def v(r, u, f): return (RIGHT * r + UP * u + FRONT * f).normalized()
def M(r, u, f):  # mirror for left side
    return v(-r, u, f)

ORDER = ["spine", "upper_arm.R", "forearm.R", "upper_arm.L", "forearm.L", "thigh.R", "shin.R", "thigh.L", "shin.L"]
def guard(d=1):
    return {"upper_arm.R": v(.35, -.6, .5), "forearm.R": v(-.25, .7, .6), "upper_arm.L": M(.35, -.6, .5), "forearm.L": M(-.25, .7, .6)}
def legs(spread=.18, bend=.1):
    return {"thigh.R": v(spread, -1, bend), "shin.R": v(spread * .5, -1, -bend), "thigh.L": M(spread, -1, bend), "shin.L": M(spread * .5, -1, -bend)}
POSES = {  # frame: (bone directions, root offset (r,u,f), yaw deg)
    1:   ({**guard(), **legs(), "spine": v(0, 1, .05)}, (0, 0, 0), 0),
    18:  ({**guard(), **legs(.22, .5), "thigh.R": v(.2, -.5, .75), "thigh.L": M(.2, -.5, .75), "shin.R": v(.1, -1, -.5), "shin.L": M(.1, -1, -.5), "spine": v(0, 1, .35)}, (0, -.22, 0), 0),
    30:  ({**guard(), "thigh.R": v(.1, -.2, .9), "shin.R": v(.05, -.9, -.4), "thigh.L": M(.15, -.6, .6), "shin.L": M(.1, -1, -.7), "spine": v(0, 1, .1),
           "upper_arm.R": v(.8, .4, -.3), "upper_arm.L": M(.8, .4, -.3)}, (0, .45, .15), -10),
    44:  ({"thigh.R": v(.05, .15, 1), "shin.R": v(.05, .2, 1), "thigh.L": M(.2, -.3, .6), "shin.L": M(.1, -1, -.8), "spine": v(0, 1, -.25),
           "upper_arm.R": v(.9, .3, -.4), "forearm.R": v(.9, .3, -.4), "upper_arm.L": M(.9, .5, .2), "forearm.L": M(.4, .9, .2)}, (0, .6, .35), -35),
    56:  ({"thigh.R": v(.05, .1, 1), "shin.R": v(.05, .15, 1), "thigh.L": M(.2, -.3, .6), "shin.L": M(.1, -1, -.8), "spine": v(0, 1, -.25),
           "upper_arm.R": v(.9, .3, -.4), "forearm.R": v(.9, .3, -.4), "upper_arm.L": M(.9, .5, .2), "forearm.L": M(.4, .9, .2)}, (0, .35, .5), -40),
    66:  ({**guard(), **legs(.25, .55), "thigh.R": v(.25, -.5, .8), "thigh.L": M(.25, -.5, .8), "shin.R": v(.1, -1, -.5), "shin.L": M(.1, -1, -.5), "spine": v(0, 1, .4)}, (0, -.25, .55), -10),
    76:  ({**guard(), **legs(), "spine": v(0, 1, .1)}, (0, 0, .55), 0),
    84:  ({**guard(), **legs(.22), "thigh.R": v(.2, -.9, .4), "thigh.L": M(.2, -.9, -.35), "upper_arm.R": v(.05, .15, 1), "forearm.R": v(0, .12, 1), "spine": v(.05, 1, .2)}, (0, -.04, .62), 25),
    92:  ({**guard(), **legs(), "spine": v(0, 1, .1)}, (0, 0, .62), 0),
    99:  ({**guard(), **legs(.22), "thigh.L": M(.2, -.9, .4), "thigh.R": v(.2, -.9, -.35), "upper_arm.L": M(.05, .15, 1), "forearm.L": M(0, .12, 1), "spine": v(-.05, 1, .2)}, (0, -.04, .68), -25),
    108: ({**legs(), "upper_arm.R": v(.25, 1, .1), "forearm.R": v(0, 1, .05), "upper_arm.L": M(.6, -.8, .1), "forearm.L": M(.3, -1, .3), "spine": v(0, 1, -.05)}, (0, .08, .68), 15),
    120: ({**legs(), "upper_arm.R": v(.2, 1, .05), "forearm.R": v(0, 1, 0), "upper_arm.L": M(.6, -.8, .1), "forearm.L": M(.3, -1, .3), "spine": v(0, 1, -.05)}, (0, .0, .68), 15),
}
pb = arm.pose.bones
for p in pb: p.rotation_mode = "QUATERNION"
base_loc, base_rot = arm.location.copy(), arm.rotation_euler.copy()
arm.rotation_mode = "XYZ"
for f, (dirs, (r, u, fw), yaw) in POSES.items():
    for p in pb: p.rotation_quaternion = Quaternion()
    bpy.context.view_layer.update()
    for name in ORDER:
        if name not in dirs or name not in pb: continue
        p = pb[name]
        cur = (p.tail - p.head).normalized()
        rot = cur.rotation_difference(dirs[name]).to_matrix().to_4x4()
        h = p.head.copy()
        p.matrix = Matrix.Translation(h) @ rot @ Matrix.Translation(-h) @ p.matrix
        bpy.context.view_layer.update()
    for p in pb: p.keyframe_insert("rotation_quaternion", frame=f)
    arm.location = base_loc + Vector((rx * r * 0, 0, u)) + Vector((0, -fw, 0))
    arm.rotation_euler = (base_rot.x, base_rot.y, base_rot.z + math.radians(yaw))
    arm.keyframe_insert("location", frame=f); arm.keyframe_insert("rotation_euler", frame=f)
bpy.ops.object.mode_set(mode="OBJECT")

# scene: floor, lights, camera, clay material
mat = bpy.data.materials.new("clay"); mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (.85, .78, .7, 1); bsdf.inputs["Roughness"].default_value = .6
mesh.data.materials.clear(); mesh.data.materials.append(mat)
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
floor = bpy.context.object
fm = bpy.data.materials.new("floor"); fm.use_nodes = True
fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.32, .34, .4, 1)
floor.data.materials.append(fm)
sc.frame_set(1)
zmin = min((mesh.matrix_world @ Vector(c)).z for c in mesh.bound_box)
floor.location.z = zmin
bpy.ops.object.light_add(type="SUN", rotation=(math.radians(50), 0, math.radians(30))); bpy.context.object.data.energy = 3.5; bpy.context.object.data.angle = math.radians(8)
bpy.ops.object.light_add(type="AREA", location=(-3, -3, 3)); a = bpy.context.object; a.data.energy = 400; a.data.size = 4
a.rotation_euler = (Vector((0, 0, 1)) - a.location).to_track_quat("-Z", "Y").to_euler()
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (.12, .14, .2, 1); world.node_tree.nodes["Background"].inputs[1].default_value = .6
bpy.ops.object.camera_add(location=(3.2, -6.2, 1.6 + zmin + 1)); cam = bpy.context.object; sc.camera = cam
cam.data.lens = 40
tgt = Vector((0, -0.6, zmin + 1.0))
cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.frame_start, sc.frame_end = 1, 120
sc.render.fps = 24
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
sc.cycles.samples = 8 if preview else 48; sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = 540 if preview else 1080
if preview:
    for f in (1, 18, 30, 44, 56, 66, 84, 99, 108, 120):
        sc.frame_set(f); sc.render.filepath = f"{out}/p_{f:03d}.png"; bpy.ops.render.render(write_still=True)
else:
    sc.render.filepath = f"{out}/f_"; sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
print("DONE")
