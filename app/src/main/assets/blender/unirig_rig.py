"""Rig with UniRig: AI skeleton + skin weights for a character mesh, on the phone (headless Blender 4.2, ONNX, no torch).

The UniRig pack (/opt/unirig, release unirig-v1) holds the ONNX models as flat files plus unirig_py.zip (onnxruntime + scipy
unpacked wheels in pylibs/, and tools/unirig/phone/*.py + onnx_skeleton.py + onnx_skin.py). The pipeline itself is
unirig_phone.main(); this wrapper unpacks the zip once, runs it, renames the bones to Blender/Rigify names (so Poseme-3D and
the export presets recognise them) and writes generation.json.

usage: blender -b --factory-startup --python unirig_rig.py -- --in in.glb --out DIR [--name N] [--threads 4]
       python unirig_rig.py --selftest        (the bone-naming check; needs no Blender)

Axis convention: the rename runs in Blender's frame (Z up, character faces -Y), where the character's LEFT is +X, as in Rigify
(.L = +X) and as in the exported glTF (Y up, faces +Z, left = +X). The mesh must stand upright; the skeleton is read from topology
plus head positions of UniRig's bone_i bones.
"""
import argparse
import json
import os
import sys
import zipfile

PACK = "/opt/unirig"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Rigged")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:])


def unpack():
    """Unpack the python libs + phone scripts once (the pack ships them as one zip; the app's downloader only handles flat files)."""
    root = os.path.join(PACK, "unirig_py")
    if not os.path.isfile(os.path.join(root, "unirig_phone.py")):
        with zipfile.ZipFile(os.path.join(PACK, "unirig_py.zip")) as z:
            z.extractall(root)
    return root


# ---------------------------------------------------------------- bone naming (pure python)
def blender_names(names, parents, heads):
    """{old: new} for the humanoid core of a UniRig skeleton. names[i], parents[i] (index or None), heads[i] (x, y, z) in Blender's frame.
    Root = pelvis; the most upward chain from it = spine ... neck, head (head = its top end, spine = the bone the arms leave from);
    two sideways chains off the upper spine = arms (upper_arm, forearm, hand = the last three bones of the chain, up to the finger fork);
    two downward chains off the pelvis = legs (thigh, shin, foot = its first three bones). L = +X. Unmatched bones keep their names.
    shortcut: hair / hip / toe bones that continue a chain upward or sit between pelvis and thigh shift the names by one; no manual fix-up."""
    n = len(names)
    kids = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p is not None:
            kids[p].append(i)
    def below(i):
        out, todo = [], [i]
        while todo:
            j = todo.pop()
            out.append(j)
            todo += kids[j]
        return out

    def dz(a, b):
        return heads[b][2] - heads[a][2]

    def side(c, root):  # mean x of the bones below c relative to the root: +1 left, -1 right
        d = sum(heads[j][0] for j in below(c)) / len(below(c)) - heads[root][0]
        return 1 if d > 0 else -1

    def lateral(c, root):
        b = below(c)
        mx = abs(sum(heads[j][0] for j in b) / len(b) - heads[root][0])
        mz = abs(sum(heads[j][2] for j in b) / len(b) - heads[root][2])
        return mx > 0.5 * mz

    roots = [i for i in range(n) if parents[i] is None]
    if len(roots) != 1:
        return {}
    pelvis = roots[0]
    new = {pelvis: "pelvis"}
    ups = [c for c in kids[pelvis] if dz(pelvis, c) > 0]
    if not ups:
        return {names[pelvis]: "pelvis"}
    spine_chain = [pelvis]
    cur = pelvis
    arms = {}  # side -> (junction, arm root)
    while True:
        up = [c for c in kids[cur] if dz(cur, c) > 0 and not (cur != pelvis and lateral(c, cur))]
        if cur == pelvis:
            up = [max(ups, key=lambda c: dz(pelvis, c))]
        if not up:
            break
        nxt = max(up, key=lambda c: dz(cur, c))
        if cur != pelvis:
            for c in kids[cur]:
                if c != nxt and lateral(c, cur) and len(below(c)) >= 2:
                    s = side(c, cur)
                    if s not in arms or len(below(c)) > len(below(arms[s][1])):
                        arms[s] = (cur, c)
        spine_chain.append(nxt)
        cur = nxt
    if len(spine_chain) >= 4:
        new[spine_chain[-1]], new[spine_chain[-2]] = "head", "neck"
        j = arms[1][0] if 1 in arms else (arms[-1][0] if -1 in arms else None)
        new[j if j in spine_chain[1:-2] else spine_chain[1]] = "spine"

    def chain(c, stop_at_fork):
        out = [c]
        while kids[out[-1]]:
            k = kids[out[-1]]
            if stop_at_fork and len(k) > 1:
                break
            out.append(max(k, key=lambda x: len(below(x))))
        return out

    for s, (_, c) in arms.items():
        ch = chain(c, True)
        if len(ch) >= 3:
            sfx = ".L" if s == 1 else ".R"
            for i, nm in zip(ch[-3:], ("upper_arm", "forearm", "hand")):
                new[i] = nm + sfx
    legs = {}
    for c in kids[pelvis]:
        if c not in ups and dz(pelvis, c) <= 0:
            s = side(c, pelvis)
            if s not in legs or len(below(c)) > len(below(legs[s])):
                legs[s] = c
    for s, c in legs.items():
        ch = chain(c, False)
        if len(ch) >= 3:
            sfx = ".L" if s == 1 else ".R"
            for i, nm in zip(ch[:3], ("thigh", "shin", "foot")):
                new[i] = nm + sfx
    return {names[i]: v for i, v in new.items()}


def selftest():
    """Synthetic 28-bone humanoid (Blender frame: Z up, left = +X): pelvis, 3 spine/neck/head, 2 x (clavicle, 3 arm, 2 fingers),
    2 x (thigh, shin, foot, toe), 3 ponytail bones hanging down off the head."""
    B = []

    def add(parent, pos):
        B.append((f"bone_{len(B)}", parent, pos))
        return len(B) - 1

    pel = add(None, (0, 0, 1.0))
    sp = add(pel, (0, 0, 1.2))
    ch = add(sp, (0, 0, 1.4))
    ne = add(ch, (0, 0, 1.55))
    hd = add(ne, (0, 0, 1.65))
    for k in (0, 1, 2):
        add(hd if k == 0 else len(B) - 1, (0, 0.1 * (k + 1), 1.6 - 0.1 * k))  # ponytail: down and back
    for sx in (1, -1):
        cl = add(ch, (0.05 * sx, 0, 1.45))
        ua = add(cl, (0.2 * sx, 0, 1.45))
        fa = add(ua, (0.5 * sx, 0, 1.45))
        ha = add(fa, (0.75 * sx, 0, 1.45))
        add(ha, (0.85 * sx, 0.02, 1.45))
        add(ha, (0.85 * sx, -0.02, 1.45))
        th = add(pel, (0.1 * sx, 0, 0.95))
        sh = add(th, (0.1 * sx, 0, 0.5))
        ft = add(sh, (0.1 * sx, 0, 0.08))
        add(ft, (0.1 * sx, -0.15, 0.02))
    assert len(B) == 28, len(B)
    m = blender_names([b[0] for b in B], [b[1] for b in B], [b[2] for b in B])
    want = {pel: "pelvis", sp: "spine", ne: "neck", hd: "head"}
    for sx, sfx in ((1, ".L"), (-1, ".R")):
        base = 8 + (0 if sx == 1 else 10)
        want.update({base + 1: "upper_arm" + sfx, base + 2: "forearm" + sfx, base + 3: "hand" + sfx,
                     base + 6: "thigh" + sfx, base + 7: "shin" + sfx, base + 8: "foot" + sfx})
    # note: ch (index 2) is the chest where the arms leave, so it is "spine"
    want.pop(sp)
    want[ch] = "spine"
    got = {i: m.get(B[i][0]) for i in want}
    assert got == want, (got, want)
    assert len(m) == len(want), sorted(set(m) - {B[i][0] for i in want})  # clavicles, fingers, toes, ponytail keep their names
    assert not blender_names(["a", "b"], [None, 0], [(0, 0, 0), (0, 0, 1)]).get("b")  # too small for a humanoid: nothing renamed
    print("selftest ok:", sorted(m.values()))


# ---------------------------------------------------------------- Blender side
def main():
    a = args()
    root = unpack()
    sys.path[:0] = [os.path.join(root, "pylibs"), root]
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "rigged.glb")
    sys.argv = [sys.argv[0], "--", "--in", a.inp, "--out", out, "--models", PACK, "--pylibs", os.path.join(root, "pylibs"),
                "--threads", str(a.threads)]
    import unirig_phone
    unirig_phone.main()  # leaves the rigged scene open in bpy
    import bpy
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bones = list(arm.data.bones)
    idx = {b.name: i for i, b in enumerate(bones)}
    m = blender_names([b.name for b in bones], [idx[b.parent.name] if b.parent else None for b in bones],
                      [tuple(b.head_local) for b in bones])
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for old, new in m.items():
        arm.data.bones[old].name = new
        for o in meshes:
            g = o.vertex_groups.get(old)  # Blender usually renames the group with the bone; make sure
            if g:
                g.name = new
    print("UniRig: renamed", len(m), "of", len(bones), "bones:", json.dumps(m), flush=True)
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB")
    info = {"name": a.name, "vertices": sum(len(o.data.vertices) for o in meshes),
            "triangles": sum(len(p.vertices) - 2 for o in meshes for p in o.data.polygons),
            "rig": {"bones": len(bones), "skinning": "UniRig", "renamed": len(m)}}
    json.dump(info, open(os.path.join(a.out, "generation.json"), "w"), indent=1)


if "--selftest" in sys.argv:
    selftest()
else:
    main()
