"""Compare two rigged GLBs (inside Blender):  blender -b --factory-startup --python compare_rigs.py -- --ref ref.glb --new new.glb [--label x]

The skeletons are sampled (different RNG), so nothing is compared index by index: joints by chamfer distance, skin by where each
vertex's weights point (weight-averaged bone head), both relative to the mesh bounding-box diagonal. Prints CMP lines.
"""
import argparse
import json
import sys

import bpy
import numpy as np
from mathutils import kdtree


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    bones = list(arm.data.bones)
    heads = np.array([np.array(arm.matrix_world @ b.head_local) for b in bones])
    parents = [b.parent.name if b.parent else None for b in bones]
    names = [b.name for b in bones]
    me, mw = mesh.data, np.array(mesh.matrix_world)
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    col = {g.index: names.index(g.name) for g in mesh.vertex_groups if g.name in names}
    W = np.zeros((len(V), len(bones)))
    for v in me.vertices:
        for g in v.groups:
            if g.group in col:
                W[v.index, col[g.group]] = g.weight
    tails = np.array([np.array(arm.matrix_world @ b.tail_local) for b in bones])
    return dict(names=names, heads=heads, tails=tails, parents=parents, V=V, W=W, meshes=sum(o.type == "MESH" for o in bpy.data.objects))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    R, N = load(a.ref), load(a.new)
    diag = float(np.linalg.norm(R["V"].max(0) - R["V"].min(0)))
    out = {"label": a.label, "bones_ref": len(R["names"]), "bones_new": len(N["names"]), "vertices_ref": len(R["V"]), "vertices_new": len(N["V"])}
    d = lambda x, y: np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1))
    D = d(R["heads"], N["heads"])
    out["joint_chamfer_over_diag"] = round(float((D.min(1).mean() + D.min(0).mean()) / 2 / diag), 4)
    out["joint_hausdorff_over_diag"] = round(float(max(D.min(1).max(), D.min(0).max()) / diag), 4)
    for k, X in (("ref", R), ("new", N)):
        L = np.linalg.norm(X["tails"] - X["heads"], axis=1)
        out[f"bone_len_min_over_diag_{k}"] = round(float(L.min() / diag), 5)
        out[f"leaf_bones_{k}"] = sum(1 for n in X["names"] if n not in X["parents"])
        s = X["W"].sum(1)
        out[f"weight_sum_min_max_{k}"] = [round(float(s.min()), 3), round(float(s.max()), 3)]
        out[f"max_influences_{k}"] = int((X["W"] > 1e-6).sum(1).max())
        out[f"unweighted_vertices_{k}"] = int((s < 0.5).sum())
    if len(R["V"]) == len(N["V"]):  # same export of the same mesh -> same vertex order; otherwise map by nearest vertex
        idx = np.arange(len(R["V"]))
        out["vertex_order_equal"] = bool(np.abs(R["V"] - N["V"]).max() < 1e-5)
    else:
        kd = kdtree.KDTree(len(N["V"]))
        for i, v in enumerate(N["V"]):
            kd.insert(v, i)
        kd.balance()
        idx = np.array([kd.find(v)[1] for v in R["V"]])
        out["vertex_order_equal"] = False
    if R["names"] == N["names"]:  # same skeleton (e.g. 16384 vs 32768 samples with one skeleton file): compare the weights themselves
        Wn = N["W"][idx]
        out["weight_l1_per_vertex_mean"] = round(float(np.abs(R["W"] - Wn).sum(1).mean()), 4)
        out["weight_abs_diff_mean"] = round(float(np.abs(R["W"] - Wn).mean()), 6)
        out["argmax_bone_agreement_pct"] = round(float((R["W"].argmax(1) == Wn.argmax(1)).mean() * 100), 2)
    cen = lambda X: (X["W"] @ X["heads"]) / np.maximum(X["W"].sum(1, keepdims=True), 1e-9)  # where the vertex's weights point
    e = np.linalg.norm(cen(R) - cen(N)[idx], axis=1) / diag
    out.update(skin_centroid_err_mean=round(float(e.mean()), 4), skin_centroid_err_p95=round(float(np.percentile(e, 95)), 4),
               skin_centroid_err_max=round(float(e.max()), 4))
    # does each vertex's dominant bone sit near the vertex? (a sane skin keeps it within a fraction of the body size)
    for k, X in (("ref", R), ("new", N)):
        dom = np.linalg.norm(X["heads"][X["W"].argmax(1)] - X["V"], axis=1) / diag
        out[f"dominant_bone_dist_over_diag_mean_{k}"] = round(float(dom.mean()), 4)
    print("CMP", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
