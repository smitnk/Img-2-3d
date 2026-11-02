"""Toon Creator: reshape any rigged cartoon/chibi character (Model Forge presets, Kenney / Quaternius CC0 models,
imported GLBs) with proportion sliders, toon head shapes and recolouring, keeping its rig working.

blender -b --factory-startup --python toon_edit.py -- --base model.glb --spec spec.json --out DIR
spec.json: {"name", "head": 1.0, "head_shape": {"round": 0..1, "oval", "square", "buddy"}, "arms", "legs", "limb_w",
            "hands", "feet", "torso_w", "belly", "neck_len", "colors": {"<material>": "#RRGGBB"}}
Bones are sorted into body regions by name (Mixamo, Rigify, Model Forge, Kenney, Quaternius conventions all match);
each vertex's region weights come from its skin weights, so the edits blend exactly like the character bends.
"""
import argparse
import json
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "bx")]
import vcbake  # noqa: E402

REGIONS = ("head", "neck", "hand", "arm", "foot", "leg", "torso")
# Bone names are split into words ("mixamorig:LeftForeArm" -> left fore arm, "upper_arm.L" -> upper arm l) and
# matched as words: substring matching sent "forearm" to the head ("ear").
WORDS = [  # first region whose word list hits wins; hand before arm, foot before leg
    ("head", {"head", "skull", "face", "jaw", "eye", "eyes", "ear", "ears", "hair", "brow", "eyebrow", "mouth", "nose",
              "teeth", "tongue", "cheek", "lip", "lips", "eyelid", "chin", "headtop", "headtopend"}),
    ("neck", {"neck"}),
    ("hand", {"hand", "wrist", "finger", "thumb", "index", "middle", "ring", "pinky", "palm", "fist", "hands"}),
    ("arm", {"arm", "forearm", "upperarm", "lowerarm", "shoulder", "clavicle", "elbow", "collar", "bicep", "arms"}),
    ("foot", {"foot", "toe", "toes", "ankle", "heel", "feet", "toebase"}),
    ("leg", {"leg", "thigh", "knee", "shin", "calf", "upleg", "upperleg", "lowerleg", "legs"}),
]


def words_of(name):
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", name.split(":")[-1])  # camelCase -> words, drop "mixamorig:"
    w = [t for t in re.split(r"[^a-z]+", n.lower()) if t]
    return set(w) | {"".join(w)} | {a + b for a, b in zip(w, w[1:])}  # "fore arm" also as "forearm"


SHAPES = ("round", "oval", "square", "buddy")


def region_of(name):
    w = words_of(name)
    for r, words in WORDS:
        if w & words:
            return r
    return "torso"


def side_of(name, x):
    w = words_of(name)
    if w & {"l", "left"}:
        return "L"
    if w & {"r", "right"}:
        return "R"
    return "L" if x > 1e-4 else "R" if x < -1e-4 else "C"


class Rig:
    """Region pivots measured on the armature in rest pose (world space, Z up, character facing -Y)."""

    def __init__(self, rig):
        self.rig = rig
        mw = rig.matrix_world
        self.bones = {b.name: (region_of(b.name), mw @ b.head_local, mw @ b.tail_local) for b in rig.data.bones}
        self.side = {n: side_of(n, h.x) for n, (_, h, _) in self.bones.items()}

        def first(region, side=None, key=lambda p: p.z, top=True):
            pts = [h for n, (r, h, _) in self.bones.items() if r == region and (side is None or self.side[n] == side)]
            if not pts:
                return None
            return max(pts, key=key) if top else min(pts, key=key)

        self.head = first("head", top=False)  # lowest head bone joint = where the head sits on the neck
        self.neck = first("neck", top=False) or self.head
        self.shoulder = {s: first("arm", s, key=lambda p: -abs(p.x)) for s in "LR"}  # arm joint nearest the spine
        self.wrist = {s: first("hand", s, key=lambda p: -abs(p.x)) for s in "LR"}
        self.hip = {s: first("leg", s) for s in "LR"}
        self.ankle = {s: first("foot", s) for s in "LR"}


def masks(obj, rig):
    """Per-vertex region weights (n, len(REGIONS)) and side (+1 left / -1 right) from the skin weights."""
    n = len(obj.data.vertices)
    m = np.zeros((n, len(REGIONS)))
    side = np.zeros(n)
    gi = {g.index: g.name for g in obj.vertex_groups}
    for v in obj.data.vertices:
        for ge in v.groups:
            name = gi.get(ge.group)
            if name in rig.bones:
                m[v.index, REGIONS.index(rig.bones[name][0])] += ge.weight
                side[v.index] += ge.weight * {"L": 1, "R": -1}.get(rig.side[name], 0)
    total = m.sum(1)
    m[total > 0] /= total[total > 0, None]
    m[total == 0, REGIONS.index("torso")] = 1.0  # unweighted: treat as body
    return m, side


def displacement(p, m, side, rig, k, head_target=None):
    """Same proportion edits as Body Creator's art styles, on arbitrary rigs. p: (n,3) world rest positions."""
    R = {r: m[:, i] for i, r in enumerate(REGIONS)}
    d = np.zeros_like(p)
    up = lambda v: np.array(v)  # noqa: E731
    if rig.head is not None:
        hp = up(rig.head)
        d += (R["head"])[:, None] * (p - hp) * (k["head"] - 1)
        nb = up(rig.neck)
        nl = k["neck_len"] - 1
        d[:, 2] += R["neck"] * (p[:, 2] - nb[2]) * nl + R["head"] * (hp[2] - nb[2]) * nl
        if head_target is not None:
            d += head_target
    for s, sg in (("L", side > 0), ("R", side < 0)):
        if rig.shoulder[s] is not None:
            sh = up(rig.shoulder[s])
            arm = (R["arm"] + R["hand"]) * sg
            d += arm[:, None] * (p - sh) * (k["arms"] - 1)
            if rig.wrist[s] is not None:
                wr = up(rig.wrist[s])
                ax = (wr - sh) / max(np.linalg.norm(wr - sh), 1e-9)
                r = p - sh
                perp = r - (r @ ax)[:, None] * ax
                d += (R["arm"] * sg)[:, None] * perp * (k["limb_w"] - 1)
                d += (R["hand"] * sg)[:, None] * (p - wr) * (k["hands"] - 1)
        if rig.hip[s] is not None:
            hp = up(rig.hip[s])
            leg = (R["leg"] + R["foot"]) * sg
            d += leg[:, None] * (p - hp) * (k["legs"] - 1)
            if rig.ankle[s] is not None:
                an = up(rig.ankle[s])
                ax = (an - hp) / max(np.linalg.norm(an - hp), 1e-9)
                r = p - hp
                perp = r - (r @ ax)[:, None] * ax
                d += (R["leg"] * sg)[:, None] * perp * (k["limb_w"] - 1)
                d += (R["foot"] * sg)[:, None] * (p - an) * (k["feet"] - 1)
    tor = R["torso"]
    if tor.any():
        c = (p * tor[:, None]).sum(0) / max(tor.sum(), 1e-9)
        d[:, 0] += tor * (p[:, 0] - c[0]) * (k["torso_w"] - 1)
        d[:, 1] += tor * (p[:, 1] - c[1]) * (k["torso_w"] - 1)
        front = np.maximum(-(p[:, 1] - c[1]), 0)  # belly: push the front (-Y) out, most at belly height
        z0, z1 = p[tor > 0.5, 2].min() if (tor > 0.5).any() else c[2], c[2]
        bump = np.exp(-((p[:, 2] - (z0 + z1) / 2) / max(z1 - z0, 1e-3)) ** 2)
        d[:, 1] -= tor * bump * front * (k["belly"] - 1)
    return d


def head_shape_target(p, head_w, amounts, centre, r0):
    """Displacement pulling head vertices onto a blend of simple forms around the head centre."""
    rel = p - centre
    dist = np.linalg.norm(rel, axis=1) + 1e-9
    u = rel / dist[:, None]
    d = np.zeros_like(p)
    for kind, amt in amounts.items():
        if amt <= 0 or kind not in SHAPES:
            continue
        if kind == "oval":
            a = np.array([0.88, 0.96, 1.16]) * r0  # x, y (depth), z (up)
            r = 1 / np.sqrt(((u / a) ** 2).sum(1))
        elif kind == "square":
            a = np.array([1.0, 1.0, 1.06]) * r0
            r = 1 / ((np.abs(u / a) ** 4).sum(1)) ** 0.25
        else:
            r = np.full(len(p), r0 * (1.06 if kind == "buddy" else 1.0))
        d += (amt * head_w * (r - dist))[:, None] * u
    return d


def edges_of(me):
    e = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def smooth(d, edges, n, it=12, p=None):
    """Laplacian-smooth the displacement over the mesh. Low-poly/flat-shaded meshes store one vertex per face corner:
    weld coincident vertices first (edges between welded ids) so the pieces move together instead of tearing."""
    if p is not None:
        key = np.round(p / max(np.ptp(p, axis=0).max(), 1e-9) * 1e5).astype(np.int64)
        _, wid, inv_counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        wid = wid.ravel()
        m = wid.max() + 1
        dw = np.zeros((m, 3))
        np.add.at(dw, wid, d)
        dw /= np.bincount(wid, minlength=m)[:, None]
        ew = wid[edges] if len(edges) else edges
        ew = ew[ew[:, 0] != ew[:, 1]] if len(ew) else ew
        return smooth(dw, ew, m, it)[wid]
    if not len(edges):
        return d
    deg = np.bincount(edges.ravel(), minlength=n).astype(float)
    for _ in range(it):
        acc = np.zeros_like(d)
        np.add.at(acc, edges[:, 0], d[edges[:, 1]])
        np.add.at(acc, edges[:, 1], d[edges[:, 0]])
        d = np.where((deg > 0)[:, None], 0.5 * d + 0.5 * acc / np.maximum(deg, 1)[:, None], d)
    return d


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec))
    os.makedirs(a.out, exist_ok=True)
    k = {x: float(spec.get(x, 1.0)) for x in ("head", "neck_len", "arms", "legs", "limb_w", "hands", "feet", "torso_w", "belly")}

    vcbake.import_glb(a.base)
    rigs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not rigs:
        raise SystemExit("This model has no rig. Rig it first (Refine in Blender with Rig, or Body Creator).")
    rig_obj = rigs[0]
    rig = Rig(rig_obj)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and len(o.data.vertices)]
    print("toon: bones", {r: sum(1 for b in rig.bones.values() if b[0] == r) for r in REGIONS})

    # Head-shape frame from the skull: the largest mesh part that is mostly head-weighted.
    data = []
    for o in meshes:
        m, side = masks(o, rig)
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        mw = np.array(o.matrix_world)
        p = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        data.append((o, m, side, p))
    head_i = REGIONS.index("head")
    skull = max(((o, m, s, p) for o, m, s, p in data if (m[:, head_i] > 0.9).mean() > 0.3),
                key=lambda t: (t[1][:, head_i] > 0.9).sum(), default=None)
    amounts = {s: float(v) for s, v in (spec.get("head_shape") or {}).items() if float(v) > 0}
    centre = r0 = None
    if skull is not None and amounts:
        hp = skull[3][skull[1][:, head_i] > 0.9]
        centre = (hp.min(0) + hp.max(0)) / 2
        r0 = float(np.median(np.linalg.norm(hp - centre, axis=1)))

    skull_p = skull_d = None
    if skull is not None and amounts:
        o, m, s, p = skull
        skull_d = head_shape_target(p, m[:, head_i], amounts, centre, r0)
        skull_p = p
    for o, m, side, p in data:
        ht = None
        if skull_d is not None:
            if o is skull[0]:
                ht = skull_d
            else:  # eyes, hair, ears ride on the reshaped skull surface (nearest skull vertex)
                near = np.argmin(((p[:, None, :] - skull_p[None, ::max(1, len(skull_p) // 4000), :]) ** 2).sum(2), axis=1)
                ht = skull_d[::max(1, len(skull_p) // 4000)][near] * m[:, head_i:head_i + 1]
        d = displacement(p, m, side, rig, k, ht)
        d = smooth(d, edges_of(o.data), len(p), p=p)
        q = p + d
        inv = np.array(o.matrix_world.inverted())
        o.data.vertices.foreach_set("co", (q @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32).ravel())
        o.data.update()

    # Move the joints with the same edits (each bone uses its own region), so skinning still lines up.
    bpy.context.view_layer.objects.active = rig_obj
    bpy.ops.object.mode_set(mode="EDIT")
    inv = rig_obj.matrix_world.inverted()
    for eb in rig_obj.data.edit_bones:
        reg, h, t = rig.bones[eb.name]
        onehot = np.zeros((2, len(REGIONS)))
        onehot[:, REGIONS.index(reg)] = 1
        sd = np.full(2, {"L": 1.0, "R": -1.0}.get(rig.side[eb.name], 0.0))
        pts = np.array([list(h), list(t)])
        moved = pts + displacement(pts, onehot, sd, rig, k)
        eb.head, eb.tail = inv @ Vector(moved[0]), inv @ Vector(moved[1])
    bpy.ops.object.mode_set(mode="OBJECT")

    # Ground the feet again (shorter legs lift the body).
    zmin = min(min((o.matrix_world @ v.co).z for v in o.data.vertices) for o in meshes)
    roots = [o for o in bpy.context.scene.objects if o.parent is None]
    for o in roots:
        o.location.z -= zmin
    scale = float(spec.get("scale", 1.0))
    if scale != 1.0:  # overall size about the feet
        for o in roots:
            o.matrix_world = __import__("mathutils").Matrix.Scale(scale, 4) @ o.matrix_world
    bpy.context.view_layer.update()

    colors = spec.get("colors") or {}
    mats = {}
    for o in meshes:
        for slot in o.material_slots:
            mt = slot.material
            if not mt or mt.name in mats:
                continue
            b = mt.node_tree.nodes.get("Principled BSDF") if mt.use_nodes else None
            if b is None:
                continue
            if mt.name in colors:
                h = colors[mt.name].lstrip("#")
                srgb = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
                lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
                b.inputs["Base Color"].default_value = tuple(lin) + (1.0,)
            c = np.array(b.inputs["Base Color"].default_value[:3])
            srgb = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
            textured = any(lk.to_socket == b.inputs["Base Color"] for lk in mt.node_tree.links)
            mats[mt.name] = {"color": "#%02X%02X%02X" % tuple(int(round(x * 255)) for x in np.clip(srgb, 0, 1)), "textured": textured}

    name = spec.get("name") or "Toon"
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "toon"
    glb = vcbake.export_glb(os.path.join(a.out, f"{slug}.glb"))
    tris = sum(len(p.vertices) - 2 for o in meshes for p in o.data.polygons)
    info = {"generator": "toon_edit", "name": name, "triangles": tris, "vertices": sum(len(o.data.vertices) for o in meshes),
            "rig": {"bones": len(rig_obj.data.bones), "skinning": "base model weights"}, "materials": mats,
            "files": {"glb": glb}, "spec": spec}
    json.dump(info, open(os.path.join(a.out, "generation.json"), "w"), indent=2)
    print("TOON", json.dumps({"triangles": tris, "bones": len(rig_obj.data.bones), "materials": len(mats)}))


if __name__ == "__main__":
    main()
