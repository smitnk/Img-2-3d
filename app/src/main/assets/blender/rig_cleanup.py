"""Anti candy-wrapper rig cleanup: twist bones + re-split skin weights + corrective elbow/knee shape keys (Blender 4.2, headless, on the phone).

For a rigged GLB (UniRig bone names pelvis, spine, neck, head, upper_arm.L/R, forearm.L/R, hand.L/R, thigh.L/R, shin.L/R, foot.L/R, or the app's own
humanoid auto-rig, same names) it
  1. adds twist bones (2 per upper arm, 2 per forearm, 1 per thigh, 1 per shin): children of the limb bone on the same head/tail line cut into equal
     segments, Copy Rotation (Y only, local space, ZXY euler) from the next bone down the chain (forearm / hand / shin / foot) with influence
     k/(n+1) = 0.33 / 0.66 (single twist 0.5), deform on;
  2. re-splits the limb bone's skin weights over main bone + twist bones by the vertex position along the bone (linear blend between segment centres);
  3. smooths the affected groups (vertex_group_smooth), limits to 4 bones per vertex, normalizes;
  4. adds corrective shape keys "corr_elbow.L/R", "corr_knee.L/R" (pose at 90 degrees, push the collapsed inner crease back out to the rest-pose distance
     to the bone axis), driven by the bend angle (0 at 0 deg, 1 at 90 deg, swing part of the bone's local quaternion, so wrist/forearm twist does not fire it).
Preserve Volume (DQS) is NOT used: the armature modifier stays linear blend skinning.
Exports <out>/rigcleanup.glb (skins + morph targets) and generation.json.

LIMITATIONS (glTF): neither Blender drivers nor bone constraints are exported. Therefore
  - the shape keys are exported as morph targets (export_morph=True); a runtime sets their weights itself from the elbow/knee bend angle
    (key name corr_<joint>.<side>, weight = bend/90 deg). The twist bones carry custom properties twist_target / twist_influence (export_extras) so a
    runtime can replicate the Copy Rotation;
  - if the GLB has animations, the twist bones' constraint result and the driver values are BAKED per frame (visual keying of the twist bones into the
    armature action, shape key values keyframed from the evaluated drivers), so the exported animation is already correct without any runtime logic.
    A static GLB (no animation) keeps the twist bones at rest pose: the benefit then comes from the re-split weights, and from the runtime applying the above.

usage: blender -b --factory-startup --python rig_cleanup.py -- --in in.glb --out DIR [--name N]
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

# (limb bone, child it copies the twist from, number of twist bones)
LIMBS = [("upper_arm", "forearm", 2), ("forearm", "hand", 2), ("thigh", "shin", 1), ("shin", "foot", 1)]
# (joint name, upper bone, lower bone, direction the lower bone's tip swings when bending, in armature space: Blender frame, character faces -Y)
JOINTS = [("elbow", "upper_arm", "forearm", (0, -1, 0)), ("knee", "thigh", "shin", (0, 1, 0))]
SIDES = ("L", "R")
MAX_DEG = 90.0


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Rig cleanup")
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


CHILD = {"upper_arm": "forearm", "forearm": "hand", "thigh": "shin", "shin": "foot"}


def limb_ends(arm, name, pose=False):
    """(head, tail) of a limb bone in armature space as numpy arrays. The tail is the HEAD of the next bone down the chain: the glTF importer
    invents bone lengths (they do not exist in glTF), so bone.tail is unreliable, the chain is not."""
    base, side = name.rsplit(".", 1)
    ch = f"{CHILD[base]}.{side}" if base in CHILD else None
    if pose:
        pb = arm.pose.bones[name]
        return np.array(pb.head), np.array(arm.pose.bones[ch].head if ch in arm.pose.bones else pb.tail)
    b = arm.data.bones[name]
    return np.array(b.head_local), np.array(arm.data.bones[ch].head_local if ch in arm.data.bones else b.tail_local)


def twist_names(base, side, n):
    return [f"{base}_twist{k}.{side}" for k in range(1, n + 1)]


def flex_quat(arm, upper, lower, swing_dir, deg=MAX_DEG):
    """Local-space quaternion for pose bone `lower` that bends it by deg towards swing_dir (armature space); also used by the CI test."""
    lb = arm.data.bones[lower]
    lh, lt = limb_ends(arm, lower)
    d = Vector(lt - lh).normalized()
    ax = d.cross(Vector(swing_dir))
    if ax.length < 1e-6:
        ax = Vector((1, 0, 0))
    ax_local = (lb.matrix_local.to_3x3().inverted() @ ax.normalized()).normalized()
    return Quaternion(ax_local, math.radians(deg))


# ------------------------------------------------------------------ scene helpers
def find_rig():
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit("rig_cleanup: no armature in the input GLB (rig the model first)")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"
              and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
    if not meshes:
        raise SystemExit("rig_cleanup: no skinned mesh bound to the armature")
    for o in meshes:
        for m in o.modifiers:
            if m.type == "ARMATURE":
                m.use_deform_preserve_volume = False  # NO Preserve Volume (DQS)
    return arm, meshes


def set_mode(obj, mode):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.ops.object.mode_set(mode=mode)


def read_weights(obj):
    """dense float32 [nverts, ngroups]"""
    n, g = len(obj.data.vertices), len(obj.vertex_groups)
    W = np.zeros((n, g), np.float32)
    for v in obj.data.vertices:
        for ge in v.groups:
            W[v.index, ge.group] = ge.weight
    return W


def to_arm_matrix(arm, obj):
    return arm.matrix_world.inverted() @ obj.matrix_world


def np_mat(m):
    return np.array([list(r) for r in m], np.float64)


# ------------------------------------------------------------------ 1. twist bones
def add_twist_constraint(arm, name, target, influence):
    pb = arm.pose.bones[name]
    for c in list(pb.constraints):
        pb.constraints.remove(c)
    c = pb.constraints.new("COPY_ROTATION")
    c.name = "TwistCopyY"
    c.target, c.subtarget = arm, target
    c.use_x = c.use_z = False
    c.use_y = True
    c.euler_order = "ZXY"  # Y outermost, so Y is the twist and not a gimbal artefact of a bend
    c.mix_mode = "REPLACE"
    c.target_space = c.owner_space = "LOCAL"
    c.influence = influence
    for holder in (pb, pb.bone):  # which of the two the glTF exporter reads as node extras depends on the version
        holder["twist_target"] = target
        holder["twist_influence"] = float(influence)


def restore_twist_constraints(arm):
    """Re-creates the Copy Rotation constraints on a re-imported cleaned GLB from the twist_target / twist_influence node extras (what a runtime
    would do); returns how many were restored. Used by the CI test."""
    n = 0
    for pb in arm.pose.bones:
        h = pb if "twist_target" in pb.keys() else pb.bone if "twist_target" in pb.bone.keys() else None
        if h is not None and h["twist_target"] in arm.pose.bones:
            add_twist_constraint(arm, pb.name, h["twist_target"], float(h["twist_influence"]))
            n += 1
    return n



def make_twist_bones(arm):
    """Returns {limb bone name: [twist bone names]} of the limbs that exist (child bone present)."""
    names = {b.name for b in arm.data.bones}
    plan = {}
    set_mode(arm, "EDIT")
    eb = arm.data.edit_bones
    for base, child, n in LIMBS:
        for side in SIDES:
            limb, ch = f"{base}.{side}", f"{child}.{side}"
            if limb not in names or ch not in names:
                continue
            tw = twist_names(base, side, n)
            plan[limb] = (ch, tw)
            if all(t in names for t in tw):
                continue  # already cleaned
            b = eb[limb]
            h, t = b.head.copy(), eb[ch].head.copy()
            b.tail = t  # the limb bone itself spans head -> next bone's head (a bone tail the importer made up is no use)
            for k, name in enumerate(tw, start=1):
                nb = eb.new(name)
                nb.head = h + (t - h) * (k / (n + 1))
                nb.tail = h + (t - h) * ((k + 1) / (n + 1))
                nb.roll = b.roll
                nb.parent = b
                nb.use_connect = False
                nb.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")
    set_mode(arm, "POSE")
    for limb, (ch, tw) in plan.items():
        n = len(tw)
        for k, name in enumerate(tw, start=1):
            add_twist_constraint(arm, name, ch, int(k / (n + 1) * 100) / 100)  # 0.33 / 0.66, single 0.5
    bpy.ops.object.mode_set(mode="OBJECT")
    return plan


# ------------------------------------------------------------------ 2. weight re-split
def resplit_weights(arm, obj, plan):
    A = to_arm_matrix(arm, obj)
    me = obj.data
    P = np.array([tuple(A @ v.co) for v in me.vertices], np.float64)
    touched = set()
    for limb, (ch, tw) in plan.items():
        main = obj.vertex_groups.get(limb)
        if main is None:
            continue
        groups = [main] + [obj.vertex_groups.get(t) or obj.vertex_groups.new(name=t) for t in tw]
        h, t = limb_ends(arm, limb)
        axis = t - h
        L2 = float(axis @ axis)
        if L2 < 1e-12:
            continue
        n = len(tw)
        centres = (np.arange(n + 1) + 0.5) / (n + 1)
        W = read_weights(obj)
        w = W[:, main.index]
        idx = np.nonzero(w > 1e-5)[0]
        s = np.clip((P[idx] - h) @ axis / L2, 0.0, 1.0)
        for j, g in enumerate(groups):
            frac = np.interp(s, centres, np.eye(n + 1)[j])  # linear blend between segment centres, clamped at the ends
            for vi, f, wi in zip(idx.tolist(), frac.tolist(), w[idx].tolist()):
                val = wi * f
                if val > 1e-5:
                    g.add([vi], val, "REPLACE")
                elif g.index == main.index:
                    g.remove([vi])
        touched.update(g.name for g in groups)
    return touched


# ------------------------------------------------------------------ 3. smooth / limit / normalize
def smooth_limit_normalize(obj, affected):
    me = obj.data
    set_mode(obj, "OBJECT")
    me.vertices.foreach_set("select", [True] * len(me.vertices))
    me.update()
    for g in obj.vertex_groups:
        g.lock_weight = g.name not in affected  # smooth only the affected groups
    try:
        bpy.ops.object.mode_set(mode="WEIGHT_PAINT")  # the operator's poll wants weight-paint (or edit) mode
        bpy.ops.object.vertex_group_smooth(group_select_mode="ALL", factor=0.5, repeat=3, expand=0.0)
    except Exception as e:  # keep going: the re-split weights are already usable
        print("rig_cleanup: vertex_group_smooth failed:", e, flush=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    for g in obj.vertex_groups:
        g.lock_weight = False
    bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)


# ------------------------------------------------------------------ 4. corrective shape keys
def seg_dist(P, a, b):
    """distance of points P[n,3] to segment ab, and the closest points"""
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    C = a + t[:, None] * ab
    return np.linalg.norm(P - C, axis=1), C


def skin_matrices(arm, names):
    out = []
    for nm in names:
        pb = arm.pose.bones.get(nm)
        out.append(np_mat(pb.matrix @ pb.bone.matrix_local.inverted()) if pb else np.eye(4))
    return np.array(out)


def evaluated_coords(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    m = ev.to_mesh()
    c = np.empty(len(m.vertices) * 3, np.float32)
    m.vertices.foreach_get("co", c)
    ev.to_mesh_clear()
    return c.reshape(-1, 3).astype(np.float64)


def add_driver(arm, obj, key, bone):
    fc = obj.data.shape_keys.driver_add(f'key_blocks["{key}"].value')
    d = fc.driver
    d.type = "SCRIPTED"
    for nm, i in (("w", 0), ("y", 2)):
        v = d.variables.new()
        v.name, v.type = nm, "SINGLE_PROP"
        v.targets[0].id = arm
        v.targets[0].data_path = f'pose.bones["{bone}"].rotation_quaternion[{i}]'
    # swing angle of the local rotation (twist about Y removed): 2*acos(sqrt(w^2+y^2)); 0 at rest, 1 at 90 degrees
    d.expression = "min(1.0, 2*acos(min(1.0, sqrt(w*w+y*y)))/(pi/2))"
    return fc


def corrective_keys(arm, obj, plan):
    me = obj.data
    A = to_arm_matrix(arm, obj)
    A3 = np_mat(A)[:3, :3]
    A3inv = np.linalg.inv(A3)
    P = np.array([tuple(A @ v.co) for v in me.vertices], np.float64)
    if me.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    basis = np.array([tuple(k.co) for k in me.shape_keys.key_blocks[0].data], np.float64)
    W = read_weights(obj)
    gnames = [g.name for g in obj.vertex_groups]
    made = []
    for jname, up, lo, swing in JOINTS:
        for side in SIDES:
            U, Lw = f"{up}.{side}", f"{lo}.{side}"
            if U not in arm.data.bones or Lw not in arm.data.bones:
                continue
            fam = [U, Lw] + (plan.get(U, (0, []))[1]) + (plan.get(Lw, (0, []))[1])
            fidx = [gnames.index(n) for n in fam if n in gnames]
            if not fidx:
                continue
            uh, ut = limb_ends(arm, U)
            lh, lt = limb_ends(arm, Lw)
            joint = lh
            R = 0.5 * min(np.linalg.norm(ut - uh), np.linalg.norm(lt - lh))
            famw = W[:, fidx].sum(axis=1)
            dj = np.linalg.norm(P - joint, axis=1)
            sel = np.nonzero((famw >= 0.4) & (dj <= R))[0]
            if len(sel) == 0:
                continue
            # pose 90 degrees
            pb = arm.pose.bones[Lw]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = flex_quat(arm, U, Lw, swing)
            bpy.context.view_layer.update()
            D = evaluated_coords(obj) @ A3.T + np_mat(A)[:3, 3]
            pu_h, pu_t = limb_ends(arm, U, pose=True)
            pl_h, pl_t = limb_ends(arm, Lw, pose=True)
            Mb = skin_matrices(arm, gnames)
            pb.rotation_quaternion = (1, 0, 0, 0)
            bpy.context.view_layer.update()
            Ps, Ds = P[sel], D[sel]
            r0 = np.minimum(seg_dist(Ps, uh, ut)[0], seg_dist(Ps, lh, lt)[0])
            du, cu = seg_dist(Ds, pu_h, pu_t)
            dl, cl = seg_dist(Ds, pl_h, pl_t)
            use_u = du < dl
            rd = np.where(use_u, du, dl)
            C = np.where(use_u[:, None], cu, cl)
            fall = 1.0 - np.clip((dj[sel] - 0.4 * R) / (0.6 * R), 0, 1) ** 2 * (3 - 2 * np.clip((dj[sel] - 0.4 * R) / (0.6 * R), 0, 1))
            push = np.clip(r0 - rd, 0, 0.6 * r0) * fall
            dirv = Ds - C
            nrm = np.linalg.norm(dirv, axis=1)
            ok = (push > 1e-6) & (nrm > 1e-7)
            push[~ok] = 0
            dirv = dirv / np.maximum(nrm, 1e-7)[:, None]
            delta_arm = dirv * push[:, None]
            Mv3 = np.einsum("ng,gij->nij", W[sel].astype(np.float64), Mb[:, :3, :3])
            rest_delta = np.einsum("nij,nj->ni", np.linalg.pinv(Mv3), delta_arm)  # the offset in rest space that skinning turns into delta_arm
            delta_obj = rest_delta @ A3inv.T
            kname = f"corr_{jname}.{side}"
            kb = me.shape_keys.key_blocks.get(kname) or obj.shape_key_add(name=kname, from_mix=False)
            kb.slider_min, kb.slider_max = 0.0, 1.0
            full = basis.copy()
            full[sel] += delta_obj
            kb.data.foreach_set("co", full.reshape(-1).astype(np.float32))
            add_driver(arm, obj, kname, Lw)
            made.append((kname, Lw, int((push > 0).sum()), float(push.max())))
    me.update()
    return made


# ------------------------------------------------------------------ animation bake (GLB has no drivers / constraints)
def bake_animation(arm, obj, plan, keys):
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        return False
    scene = bpy.context.scene
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    twists = [t for ch, tw in plan.values() for t in tw]
    vals = {k: [] for k, _ in keys}
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        for k, _ in keys:
            vals[k].append(obj.data.shape_keys.key_blocks[k].value)  # evaluated driver value
    try:  # twist bones: visual keying of the constraint result into the armature action
        set_mode(arm, "POSE")
        for pb in arm.pose.bones:
            (pb.bone if hasattr(pb.bone, "select") else pb).select = pb.name in twists  # Bone.select (4.2) / PoseBone.select (newer)
        bpy.ops.nla.bake(frame_start=f0, frame_end=f1, only_selected=True, visual_keying=True, clear_constraints=True,
                         use_current_action=True, bake_types={"POSE"})
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as e:
        print("rig_cleanup: twist bake failed:", e, flush=True)
    sk = obj.data.shape_keys
    for k, _ in keys:
        sk.driver_remove(f'key_blocks["{k}"].value')
    for k, _ in keys:
        kb = sk.key_blocks[k]
        for f, v in zip(range(f0, f1 + 1), vals[k]):
            kb.value = v
            kb.keyframe_insert("value", frame=f)
    scene.frame_set(f0)
    return True


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.inp)
    arm, meshes = find_rig()
    n0 = len(arm.data.bones)
    plan = make_twist_bones(arm)
    print("rig_cleanup: twist bones for", {k: v[1] for k, v in plan.items()}, flush=True)
    baked = False
    for obj in meshes:
        affected = resplit_weights(arm, obj, plan)
        smooth_limit_normalize(obj, affected)
        made = corrective_keys(arm, obj, plan)
        print("rig_cleanup:", obj.name, "affected groups", len(affected), "corrective keys", made, flush=True)
        baked = bake_animation(arm, obj, plan, [(m[0], m[1]) for m in made]) or baked
    set_mode(arm, "OBJECT")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "rigcleanup.glb")
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_skins=True, export_morph=True, export_animations=True,
                              export_extras=True, export_apply=False)
    info = {"name": a.name, "vertices": sum(len(o.data.vertices) for o in meshes),
            "triangles": sum(len(p.vertices) - 2 for o in meshes for p in o.data.polygons),
            "rig": {"bones": len(arm.data.bones), "added_bones": len(arm.data.bones) - n0, "skinning": "UniRig + twist bones", "baked_animation": baked}}
    json.dump(info, open(os.path.join(a.out, "generation.json"), "w"), indent=1)
    print("rig_cleanup: done", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
