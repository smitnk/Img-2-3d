"""A/B check of new UniRig outputs against the reference run (branch ci-logs-unirig).

usage: python ab_check.py <ref_dir> <new_dir> <name> [<name> ...]
Prints "AB skeleton <n>: max=.. mean=.. parents_equal=..", "AB skin <n>: max=.. mean=..", "AB glb <n>: joints ref/new".
Needs bpy (FBX import). Exit code is always 0; read the lines.
"""
import json
import struct
import sys

import bpy
import numpy as np


def load_fbx(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    heads = np.array([np.array(arm.matrix_world @ b.head_local) for b in bones])
    parents = [b.parent.name if b.parent else None for b in bones]
    mesh = next((o for o in bpy.data.objects if o.type == "MESH"), None)
    w = None
    if mesh is not None:
        col = {g.index: names.index(g.name) for g in mesh.vertex_groups if g.name in names}
        w = np.zeros((len(mesh.data.vertices), len(names)), np.float32)
        for v in mesh.data.vertices:
            for g in v.groups:
                if g.group in col:
                    w[v.index, col[g.group]] = g.weight
    return names, heads, parents, w


def glb_joints(path):
    b = open(path, "rb").read()
    j = json.loads(b[20:20 + struct.unpack("<I", b[12:16])[0]])
    return [len(s["joints"]) for s in j.get("skins", [])]


ref, new = sys.argv[1], sys.argv[2]
for n in sys.argv[3:]:
    try:
        rn, rh, rp, _ = load_fbx(f"{ref}/skel_{n}.fbx")
        nn_, nh, np_, _ = load_fbx(f"{new}/skel_{n}.fbx")
        if rh.shape == nh.shape:
            d = np.abs(rh - nh)
            print(f"AB skeleton {n}: max={d.max():.3e} mean={d.mean():.3e} parents_equal={rp == np_ and rn == nn_} joints={len(rn)}")
        else:
            print(f"AB skeleton {n}: max=nan mean=nan parents_equal=False joints ref={len(rn)} new={len(nn_)}")
    except Exception as e:
        print(f"AB skeleton {n}: ERROR {e}")
    try:
        rn, _, _, rw = load_fbx(f"{ref}/skin_{n}.fbx")
        nn_, _, _, nw = load_fbx(f"{new}/skin_{n}.fbx")
        if rw.shape == nw.shape and rn == nn_:
            d = np.abs(rw - nw)
            print(f"AB skin {n}: max={d.max():.3e} mean={d.mean():.3e} verts={rw.shape[0]} joints={rw.shape[1]}")
        else:
            print(f"AB skin {n}: max=nan mean=nan shape ref={rw.shape} new={nw.shape}")
    except Exception as e:
        print(f"AB skin {n}: ERROR {e}")
    try:
        print(f"AB glb {n}: joints ref={glb_joints(f'{ref}/unirig_{n}.glb')} new={glb_joints(f'{new}/unirig_{n}.glb')}")
    except Exception as e:
        print(f"AB glb {n}: ERROR {e}")
