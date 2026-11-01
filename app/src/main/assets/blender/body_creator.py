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


def mesh_object(name, verts, faces, uvs=None, uvf=None):
    """Quads (a triangle repeats its last index) -> mesh with optional per-corner UVs."""
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
    return ob


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

    skin = material("Skin", hex_rgb(spec.get("skin"), "#C9A08A"), 0.55, sss=0.15)
    body = mesh_object(spec.get("name", "Body"), v, z["f_body"], z["uvs"], z["uvf_body"])
    body.data.materials.append(skin)
    subdiv = int(spec.get("subdiv", 1))
    if subdiv > 0:
        m = body.modifiers.new("subdiv", "SUBSURF")
        m.levels = m.render_levels = subdiv
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(body.evaluated_get(dg))
        old = body.data
        body.modifiers.clear()
        body.data = me
        bpy.data.meshes.remove(old)

    mouth = material("Mouth", hex_rgb("#E8DCC8", ""), 0.3)
    parts = []
    for part, nm in (("helper-upper-teeth", "TeethUpper"), ("helper-lower-teeth", "TeethLower"), ("helper-tongue", "Tongue")):
        o = mesh_object(nm, v, z[f"f_{part}"])
        o.data.materials.append(mouth if "Teeth" in nm else material("Tongue", hex_rgb("#B5525A", ""), 0.4))
        parts.append(o)

    ev = to_blender(pack.fit_proxy(hm, z["eye_refs"], z["eye_w"], z["eye_off"], z["eye_scale"]))
    ev[:, 2] -= ground
    eyes = mesh_object("Eyes", ev, z["eye_faces"], z["eye_uvs"], z["eye_uvf"])
    eye_png = os.path.join(a.pack, "eye.png")
    eyes.data.materials.append(material("Eyes", (1, 1, 1, 1), 0.1, image=eye_png if os.path.isfile(eye_png) else None))
    parts.append(eyes)

    rig_info = None
    if spec.get("rig", True):
        from refine_mesh import rigify
        rig, rig_info = rigify(body, rigid={o: None for o in parts})
        if rig is None:
            print("body rig failed:", rig_info)

    slug = "".join(c if c.isalnum() else "_" for c in spec.get("name", "body")).strip("_").lower() or "body"
    glb = vcbake.export_glb(os.path.join(a.out, f"{slug}.glb"), deform_only=bool(rig_info and rig_info.get("route")))
    tris = sum(len(p.vertices) - 2 for o in [body] + parts for p in o.data.polygons)
    info = {"generator": "body_creator", "name": spec.get("name", "Body"), "triangles": tris,
            "vertices": sum(len(o.data.vertices) for o in [body] + parts),
            "height_m": round(float(v[:, 2].max()), 3), "rig": rig_info, "files": {"glb": glb}, "spec": spec}
    json.dump(info, open(os.path.join(a.out, "generation.json"), "w"), indent=2)
    print("BODY", json.dumps({k: info[k] for k in ("triangles", "height_m", "rig")}))


if __name__ == "__main__":
    main()
