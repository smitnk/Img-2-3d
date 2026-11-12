"""The phone runner's numpy ports against the real UniRig / torch code (x86 CI job; also runs locally with what is installed).

  python parity_x86.py --unirig <UniRig checkout> [--glb-dir DIR with cm_game20k.glb ...] [--skin-art DIR] [--skel-art DIR]
Prints PARITY <check>: ok|FAIL (hard: exit 1) and INFO lines (soft: mesh/voxel checks that need trimesh / bpy / pyrender / the artifacts).
Sections: serialization vs torch (random clouds + the real captured clouds), tokenizer vs UniRig's (grammar masks at every step + detokenize),
sampler / affine vs UniRig's with the same seed, merge+normals vs trimesh, voxelization vs pyrender, voxel_skin / reskin vs UniRig's.
"""
import argparse
import ast
import glob
import importlib
import os
import sys
import types
import typing

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import prep_np as P  # noqa: E402
import serialization_np as Z  # noqa: E402
import tokenizer_np as T  # noqa: E402

hard, soft = [], []


def check(name, ok, detail="", is_hard=True):
    print(f"PARITY {name}: {'ok' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        (hard if is_hard else soft).append(name)


def ensure_box():
    """UniRig's order.py needs python-box; if it is not installed, a minimal stand-in (attribute access + wrapped sub-dicts)."""
    try:
        import box  # noqa: F401
    except ImportError:
        class Box(dict):
            def __getattr__(self, k):
                return self.get(k)

            def get(self, k, d=None):
                v = dict.get(self, k, d)
                return Box(v) if isinstance(v, dict) else v
        m = types.ModuleType("box")
        m.Box = Box
        sys.modules["box"] = m


def section(fn, *a):
    try:
        fn(*a)
    except Exception as e:  # a missing dependency / artifact must not hide the other sections
        import traceback
        traceback.print_exc()
        print(f"INFO {fn.__name__}: skipped or crashed: {e!r}", flush=True)
        soft.append(fn.__name__)


# ---------------------------------------------------------------- serialization vs torch
def serialization(unirig, skin_art):
    import torch
    d = f"{unirig}/src/model/pointcept/models/utils/serialization"
    pkg = types.ModuleType("ptser")
    pkg.__path__ = [d]
    sys.modules["ptser"] = pkg
    dflt = importlib.import_module("ptser.default")  # the very files UniRig runs, minus the pointcept package __init__ chain (spconv ...)

    def torch_ser(pts):
        c = torch.from_numpy(pts)
        g = torch.div(c - c.min(0)[0], 0.005, rounding_mode="trunc").int()  # Point.serialization
        depth = int(g.max()).bit_length()
        code = torch.stack([dflt.encode(g, torch.zeros(len(g), dtype=torch.long), depth, order=o) for o in Z.ORDERS])
        order = torch.argsort(code)
        inv = torch.zeros_like(order).scatter_(dim=1, index=order, src=torch.arange(0, code.shape[1]).repeat(code.shape[0], 1))
        return g.numpy(), code.numpy(), order.numpy(), inv.numpy()

    rng = np.random.default_rng(0)
    clouds = [("uniform 32768", rng.random((32768, 3), dtype=np.float32) * 2 - 1), ("small 1000", rng.random((1000, 3), dtype=np.float32) * .3),
              ("duplicates 8192", (np.round(rng.random((8192, 3)) * 60) / 60).astype(np.float32))]
    for f in sorted(glob.glob(f"{skin_art}/inputs_*.npz")) if skin_art else []:
        inp = np.load(f)
        clouds.append((f"captured {os.path.basename(f)}", inp["vertices"][0] if inp["vertices"].ndim == 3 else inp["vertices"]))
        s = Z.serialize(np.asarray(clouds[-1][1], np.float32))
        perm = inp["perm"]
        check(f"serialization order/inverse == the real run's ({os.path.basename(f)})",
              (s["order"][perm] == inp["order"]).all() and (s["inverse"][perm] == inp["inverse"]).all())
    for name, pts in clouds:
        g, code, order, inv = torch_ser(pts)
        s = Z.serialize(pts)
        check(f"serialization {name}: grid", (s["grid"] == g).all())
        check(f"serialization {name}: codes (bit exact)", (s["code"] == code).all(), f"depth {int(g.max()).bit_length()}")
        check(f"serialization {name}: order/inverse", (s["order"] == order).all() and (s["inverse"] == inv).all(),
              f"order differs in {int((s['order'] != order).sum())} slots (ties of equal codes)")


# ---------------------------------------------------------------- tokenizer vs UniRig
def tokenizer(unirig, skel_art):
    import yaml
    from box import Box
    sys.path.insert(0, unirig)
    os.chdir(unirig)  # the tokenizer config points at ./configs/skeleton/*.yaml
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig
    ref = get_tokenizer(config=TokenizerConfig.parse(config=Box(yaml.safe_load(open("configs/tokenizer/tokenizer_parts_articulationxl_256.yaml")))))
    mine, rng = T.Tokenizer(), np.random.default_rng(5)
    check("tokenizer constants", (ref.bos, ref.eos, ref.pad, ref.vocab_size, ref.cls_name_to_token("articulationxl")) ==
          (T.BOS, T.EOS, T.PAD, 267, T.CLS["articulationxl"]))
    streams = []
    for _ in range(30):
        ids = [T.BOS, T.CLS["articulationxl"]]
        want = int(rng.integers(3, 60)) * 3
        while True:
            a, b = sorted(ref.next_posible_token(ids=np.array(ids))), sorted(mine.next_posible_token(np.array(ids)))
            if a != b:
                check("tokenizer grammar mask", False, f"after {ids}")
                break
            if len(ids) > want and T.EOS in a:
                ids.append(T.EOS)
                break
            ids.append(int(rng.choice(a)))
        streams.append(np.array(ids))
    check("tokenizer grammar masks at every step of 30 random streams", "tokenizer grammar mask" not in hard)
    for f in glob.glob(f"{skel_art}/inputs_*.npz") if skel_art else []:
        streams.append(np.load(f)["ref_ids"])
    ok = 0
    for ids in streams:
        r, m = ref.detokenize(ids=ids), mine.detokenize(ids)
        ok += (np.allclose(r.bones, m.bones, atol=0) and np.allclose(r.tails, m.tails, atol=1e-6) and r.parents == m.parents
               and list(r.names) == list(m.names))
    check(f"tokenizer detokenize bones/tails/parents/names on {len(streams)} streams (30 random + captured real ones)", ok == len(streams), f"{ok} equal")


# ---------------------------------------------------------------- mesh, sampler, affine
def glb_meshes(glb_dir):
    import unirig_phone as U
    out = {}
    for f in sorted(glob.glob(f"{glb_dir}/*.glb")):
        _, V0, F0 = U.load_meshes(f)
        out[os.path.basename(f)[:-4]] = (V0, F0)
    return out


def mesh_vs_trimesh(meshes):
    import trimesh
    for name, (V0, F0) in meshes.items():
        m = trimesh.Trimesh(vertices=V0, faces=F0)  # what extract.py's save_raw_data starts with
        V, F = P.merge_vertices(V0, F0)
        vn, fn = P.mesh_normals(V, F)
        check(f"mesh {name}: merged vertices/faces count", (len(V), len(F)) == (len(m.vertices), len(m.faces)), f"{len(V)}/{len(F)} vs {len(m.vertices)}/{len(m.faces)}",
              False)
        if len(V) != len(m.vertices):
            continue
        key = lambda v: np.lexsort(np.round(v.astype(np.float64) * 1e6).T[::-1])
        a, b = key(V), key(np.asarray(m.vertices))
        dv = np.abs(V[a] - np.asarray(m.vertices, np.float32)[b]).max()
        dn = np.abs(vn[a] - np.asarray(m.vertex_normals, np.float32)[b])
        fa, fb = key(V[F].mean(1)), key(np.asarray(m.vertices)[m.faces].mean(1))
        dfn = np.abs(fn[fa] - np.asarray(m.face_normals, np.float32)[fb])
        check(f"mesh {name}: vertices / vertex normals / face normals", dv < 1e-6 and dn.max() < 1e-3 and dfn.max() < 1e-3,
              f"max diff {dv:.1e} / {dn.max():.1e} (mean {dn.mean():.1e}) / {dfn.max():.1e}", False)


def sampler_affine(unirig, meshes):
    from types import SimpleNamespace
    sys.path.insert(0, unirig)
    from src.data.augment import AugmentAffine, AugmentAffineConfig
    from src.data.sampler import SamplerConfig, get_sampler
    rng = np.random.default_rng(2)
    V = (rng.random((3000, 3)) * [2, 5, 1] + 3).astype(np.float32)
    F = rng.integers(0, 3000, (6000, 3))
    VN, FN = rng.standard_normal((3000, 3)).astype(np.float32), rng.standard_normal((6000, 3)).astype(np.float32)
    J = (V[:5] + 0.1).astype(np.float32)
    vg = rng.random((3000, 5)).astype(np.float32)
    for joints in (None, J):
        asset = SimpleNamespace(vertices=V.copy(), vertex_normals=VN, faces=F, face_normals=FN, joints=None if joints is None else joints.copy(),
                                tails=None if joints is None else joints.copy(), matrix_local=None, pose_matrix=None, vertex_groups={"g": vg})
        np.random.seed(7)
        AugmentAffine(AugmentAffineConfig(normalize_into=[-1.0, 1.0], random_scale_p=0.0, random_scale=[1.0, 1.0], random_shift_p=0.0,
                                          random_shift=[0.0, 0.0])).transform(asset=asset)
        r = get_sampler(SamplerConfig(method="mix", num_samples=2000, vertex_samples=300, kwargs={})).sample(asset=asset)
        rs = np.random.RandomState(7)
        v2, j2, t2, _ = P.affine(rs, V, None if joints is None else joints.copy(), None if joints is None else joints.copy())
        pts, nrm, g = P.sample_mix(rs, v2, VN, F, FN, 2000, 300, vg)
        tag = "mesh only" if joints is None else "mesh + joints"
        check(f"affine ({tag}) == AugmentAffine", (v2 == asset.vertices).all() and (joints is None or ((j2 == asset.joints).all() and (t2 == asset.tails).all())))
        check(f"sample_mix ({tag}) == SamplerMix with np.random.seed(7)",
              (pts == r.vertices).all() and (nrm == r.normals).all() and (g == r.vertex_groups["g"]).all())


# ---------------------------------------------------------------- voxelization, voxel_skin, reskin vs UniRig's functions
def funcs(path, names):
    """The named top-level functions of a UniRig source file, without importing the file (vertex_group.py forces EGL on import)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial import cKDTree
    ns = dict(np=np, ndarray=np.ndarray, csr_matrix=csr_matrix, shortest_path=shortest_path, cKDTree=cKDTree, **{k: v for k, v in vars(typing).items() if not k.startswith("__")})
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns


def voxel_stuff(unirig, meshes):
    import unirig_phone as U
    vg = funcs(f"{unirig}/src/data/vertex_group.py", {"voxelization", "voxel_skin"})
    rk = funcs(f"{unirig}/src/system/skin.py", {"reskin"})
    names = [n for n in meshes if "cm_game20k" in n] or list(meshes)[:1]
    key = lambda c, g: set(map(tuple, np.round((c + 1 - 1 / g) * g / 2).astype(np.int32).tolist()))
    for name in names:
        V0, F0 = meshes[name]
        V, F = P.merge_vertices(V0, F0)
        lo, hi = V.min(0), V.max(0)
        nv = ((V - (lo + hi) / 2) / (np.max(hi - lo) / 2)).astype(np.float32)
        for g in (64, 196):
            ref = vg["voxelization"](vertices=nv, faces=F, grid=g, backend="pyrender")
            mine = P.voxelization(g, U.bvh_cast_factory(nv, F))
            a, b = key(ref, g), key(mine, g)
            iou = len(a & b) / max(len(a | b), 1)
            check(f"voxelization {name} grid {g} vs pyrender", iou > 0.98, f"IoU {iou:.4f}, pyrender {len(a)} voxels, ray cast {len(b)}", False)
            if iou <= 0.98:  # diagnostics: is it a flipped axis?
                for ax in range(3):
                    fl = mine.copy()
                    fl[:, ax] *= -1
                    print(f"INFO   IoU with axis {ax} mirrored: {len(a & key(fl, g)) / max(len(a | key(fl, g)), 1):.4f}", flush=True)
        # voxel_skin / reskin: UniRig's own functions on identical inputs (small grid)
        g, rng = 48, np.random.default_rng(4)
        joints = nv[rng.choice(len(nv), 6, replace=False)]
        gc = P.voxelization(g, U.bvh_cast_factory(nv, F))
        if len(gc):
            a = vg["voxel_skin"](grid=g, grid_coords=gc, joints=joints, vertices=nv, faces=F, grid_query=7, vertex_query=1)
            b = P.voxel_skin(g, gc, joints, nv, F, grid_query=7, vertex_query=1)
            check(f"voxel_skin {name} == UniRig's", np.allclose(a, b, atol=1e-9), f"max diff {np.abs(a - b).max():.1e}")
        parents = [None, 0, 1, 1, 0, 4]
        samp = rng.choice(len(nv), 4000, replace=False)
        sk = rng.random((4000, 6)).astype(np.float32)
        sk /= sk.sum(1, keepdims=True)
        a = rk["reskin"](sampled_vertices=nv[samp], vertices=nv, parents=parents, faces=F, sampled_skin=sk, sample_method="median", alpha=2.0,
                         threshold=0.03)
        b = P.reskin(nv[samp], nv, parents, F, sk)
        check(f"reskin {name} == UniRig's", np.allclose(a, b, atol=1e-6), f"max diff {np.abs(a - b).max():.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unirig", required=True)
    ap.add_argument("--glb-dir")
    ap.add_argument("--skin-art")
    ap.add_argument("--skel-art")
    a = ap.parse_args()
    a.unirig, a.glb_dir, a.skin_art, a.skel_art = (os.path.abspath(p) if p else p for p in (a.unirig, a.glb_dir, a.skin_art, a.skel_art))
    ensure_box()
    section(serialization, a.unirig, a.skin_art)
    section(tokenizer, a.unirig, a.skel_art)
    meshes = {}
    if a.glb_dir:
        try:
            meshes = glb_meshes(a.glb_dir)
        except Exception as e:
            print(f"INFO glb loading failed (no bpy?): {e!r}")
    section(sampler_affine, a.unirig, meshes)
    if meshes:
        section(mesh_vs_trimesh, meshes)
        section(voxel_stuff, a.unirig, meshes)
    print(f"PARITY SUMMARY hard failures: {hard or 'none'}; soft: {soft or 'none'}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
