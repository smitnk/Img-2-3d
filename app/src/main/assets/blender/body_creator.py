"""Body Creator: a MakeHuman-based character from a slider spec (no AI, no primitives).

blender -b --factory-startup --python body_creator.py -- --pack /opt/bodypack --spec spec.json --out DIR
spec.json: {"name": "Hero", "macros": {"gender": 1, "age": 0.5, "muscle": 0.7, ...},
            "sliders": {"nose/nose-scale-horiz-decr|incr": 0.3, ...}, "skin": "#C68C6E", "eyes": "#5B3A1E",
            "subdiv": 1, "rig": true}
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "bx"), os.path.join(HERE, "body")]
sys.path.append("/opt/forge/scripts")
import humanmorph  # noqa: E402
import vcbake  # noqa: E402


def to_blender(v):
    """MakeHuman: decimetres, Y up, facing +Z -> Blender: metres, Z up, facing -Y."""
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]]) * 0.1


def hex_rgb(h, default):
    h = (h or default).lstrip("#")
    srgb = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return tuple(np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)) + (1.0,)


def mesh_object(name, verts, faces, uvs=None, uvf=None, groups=None):
    """Quads (a triangle repeats its last index) -> mesh with optional per-corner UVs.
    groups: {bone: (base vertex indices, weights)} in base-mesh numbering, remapped to this mesh."""
    tri = faces[:, 3] == faces[:, 2]
    polys = [list(f[:3]) if t else list(f) for f, t in zip(faces, tri)]
    used = np.unique(np.concatenate([np.array(p) for p in polys]))
    remap = -np.ones(len(verts), np.int64)
    remap[used] = np.arange(len(used))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts[used].tolist(), [], [[int(remap[i]) for i in p] for p in polys])
    if uvs is not None:
        layer = me.uv_layers.new(name="UVMap")
        corner = np.concatenate([uvf[i, :3] if t else uvf[i] for i, t in enumerate(tri)])
        layer.data.foreach_set("uv", uvs[corner].ravel())
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for bone, (idx, w) in (groups or {}).items():
        local = remap[idx]
        keep = local >= 0
        if keep.any():
            vg = ob.vertex_groups.new(name=bone)
            for i, wt in zip(local[keep].tolist(), w[keep].tolist()):
                vg.add([i], wt, "REPLACE")
    return ob


def build_skeleton(skel, hm, ground):
    """MakeHuman skeleton fitted to this body: each joint = mean of its vertex group on the morphed mesh;
    roll from the bone's plane so local X = plane normal (MakeHuman skeleton.get_roll_to)."""
    from mathutils import Vector
    jpos = {j: Vector(to_blender(hm[idx].mean(0, keepdims=True))[0] - np.array([0, 0, ground])) for j, idx in skel["joints"].items()}
    arm = bpy.data.armatures.new("Skeleton")
    rig = bpy.data.objects.new("Skeleton", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for name, b in skel["bones"].items():
        e = arm.edit_bones.new(name)
        e.head, e.tail = jpos[b["head"]], jpos[b["tail"]]
        if (e.tail - e.head).length < 1e-4:
            e.tail = e.head + Vector((0, 0, 0.01))
        eb[name] = e
    for name, b in skel["bones"].items():
        if b.get("parent"):
            eb[name].parent = eb[b["parent"]]
        planes = b.get("rotation_plane")
        planes = [planes] if isinstance(planes, str) else (planes or [])
        normal = Vector()
        for pl in planes:
            j = skel["planes"].get(pl)
            if j:
                p1, p2, p3 = (jpos[x] for x in j)
                n = (p3 - p2).normalized().cross((p2 - p1).normalized())
                if n.length > 1e-6:
                    normal += n.normalized()
        if normal.length > 1e-6:
            y = (eb[name].tail - eb[name].head).normalized()
            eb[name].align_roll(normal.normalized().cross(y))  # Z = X x Y with X = plane normal
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def material(name, color, roughness=0.5, sss=0.0, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = roughness
    if sss and "Subsurface Weight" in b.inputs:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        b.inputs["Subsurface Scale"].default_value = 0.01
    if image:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(image)
        tex.image.pack()
        m.node_tree.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    return m


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/opt/bodypack")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    pack = humanmorph.Pack(a.pack)
    z = pack.npz
    hm = pack.morph(spec.get("macros"), spec.get("sliders"))
    v = to_blender(hm)
    ground = v[np.unique(z["f_body"]), 2].min()
    v[:, 2] -= ground  # feet on the ground

    skel = json.load(open(os.path.join(a.pack, "skeleton.json")))
    wj = json.load(open(os.path.join(a.pack, "weights.json")))
    groups = {bone: (np.array([p[0] for p in vw], np.int64), np.array([p[1] for p in vw], np.float64)) for bone, vw in wj.items()}

    skin = material("Skin", hex_rgb(spec.get("skin"), "#C9A08A"), 0.55, sss=0.15)
    body = mesh_object(spec.get("name", "Body"), v, z["f_body"], z["uvs"], z["uvf_body"], groups)
    body.data.materials.append(skin)
    subdiv = int(spec.get("subdiv", 1))
    if subdiv > 0:  # the evaluated subdivision interpolates the vertex groups along with the shape
        m = body.modifiers.new("subdiv", "SUBSURF")
        m.levels = m.render_levels = subdiv
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(body.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        old = body.data
        body.modifiers.clear()
        body.data = me
        bpy.data.meshes.remove(old)

    mats = {"Mouth": material("Teeth", hex_rgb("#EEE6D8", ""), 0.3), "Tongue": material("Tongue", hex_rgb("#B5525A", ""), 0.4)}
    parts = []
    for part, nm in (("helper-upper-teeth", "TeethUpper"), ("helper-lower-teeth", "TeethLower"), ("helper-tongue", "Tongue")):
        o = mesh_object(nm, v, z[f"f_{part}"], groups=groups)
        o.data.materials.append(mats["Tongue"] if nm == "Tongue" else mats["Mouth"])
        parts.append(o)

    ev = to_blender(pack.fit_proxy(hm, z["eye_refs"], z["eye_w"], z["eye_off"], z["eye_scale"]))
    ev[:, 2] -= ground
    eyes = mesh_object("Eyes", ev, z["eye_faces"], z["eye_uvs"], z["eye_uvf"])
    eye_png = os.path.join(a.pack, "eye.png")
    eyes.data.materials.append(material("Eyes", (1, 1, 1, 1), 0.1, image=eye_png if os.path.isfile(eye_png) else None))
    # rigid per side to the eye bones (x > 0 is the character's left)
    left = [i for i, p in enumerate(eyes.data.vertices) if p.co.x > 0]
    right = [i for i, p in enumerate(eyes.data.vertices) if p.co.x <= 0]
    eyes.vertex_groups.new(name="eye.L").add(left, 1.0, "REPLACE")
    eyes.vertex_groups.new(name="eye.R").add(right, 1.0, "REPLACE")
    parts.append(eyes)

    # One skinned mesh (body + mouth + eyes, one material each): what engines expect from a character.
    for o in bpy.context.view_layer.objects:
        o.select_set(o is body or o in parts)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()

    rig_info = None
    if spec.get("rig", True):
        rig = build_skeleton(skel, hm, ground)
        body.parent = rig
        mod = body.modifiers.new("Armature", "ARMATURE")
        mod.object = rig
        empty = [vg.name for vg in body.vertex_groups if vg.name not in rig.data.bones]
        rig_info = {"bones": len(rig.data.bones), "skinning": "MakeHuman weights (CC0)", "unmatched_groups": empty[:5]}
        # Deformation self-check: raise the left upper arm 60 deg; the left hand must move, the right must not.
        dg = bpy.context.evaluated_depsgraph_get()
        def wrist(side):
            return rig.matrix_world @ rig.pose.bones[f"wrist.{side}"].head
        before = {s_: wrist(s_).copy() for s_ in "LR"}
        pb = rig.pose.bones["upperarm01.L"]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 1.05)
        bpy.context.view_layer.update()
        ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        moved = {s_: (wrist(s_) - before[s_]).length for s_ in "LR"}
        pb.rotation_euler = (0, 0, 0)
        bpy.context.view_layer.update()
        rig_info["selftest"] = {"left_wrist_moved_m": round(moved["L"], 3), "right_wrist_moved_m": round(moved["R"], 3),
                                "skinned_verts": len(ev.data.vertices)}
        assert moved["L"] > 0.1 and moved["R"] < 1e-4, rig_info["selftest"]

    slug = "".join(c if c.isalnum() else "_" for c in spec.get("name", "body")).strip("_").lower() or "body"
    glb = vcbake.export_glb(os.path.join(a.out, f"{slug}.glb"))
    tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
    info = {"generator": "body_creator", "name": spec.get("name", "Body"), "triangles": tris,
            "vertices": len(body.data.vertices),
            "height_m": round(float(v[:, 2].max()), 3), "rig": rig_info, "files": {"glb": glb}, "spec": spec}
    json.dump(info, open(os.path.join(a.out, "generation.json"), "w"), indent=2)
    print("BODY", json.dumps({k: info[k] for k in ("triangles", "height_m", "rig")}))


if __name__ == "__main__":
    main()
