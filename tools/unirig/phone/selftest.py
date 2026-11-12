"""Stand-alone checks of the phone runner's numpy ports (numpy + scipy only; no torch, UniRig or Blender needed).

  python selftest.py            -> prints SELFTEST lines, exit 1 on a failure
The comparisons against the real torch / UniRig code are in parity_x86.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_np as P  # noqa: E402
import serialization_np as Z  # noqa: E402
import tokenizer_np as T  # noqa: E402

bad = []


def check(name, ok, detail=""):
    print(f"SELFTEST {name}: {'ok' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        bad.append(name)


def skilling(x, p):
    """Pure-python Hilbert index (J. Skilling, Programming the Hilbert curve, 2004), the algorithm serialization/hilbert.py vectorizes."""
    x, n, M = [int(v) for v in x], 3, 1 << (p - 1)
    Q = M
    while Q > 1:
        for i in range(n):
            if x[i] & Q:
                x[0] ^= Q - 1
            else:
                t = (x[0] ^ x[i]) & (Q - 1)
                x[0] ^= t
                x[i] ^= t
        Q >>= 1
    for i in range(1, n):
        x[i] ^= x[i - 1]
    t, Q = 0, M
    while Q > 1:
        if x[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    x = [v ^ t for v in x]
    h = 0
    for b in range(p - 1, -1, -1):
        for i in range(n):
            h = (h << 1) | ((x[i] >> b) & 1)
    return h


def morton(x, y, z, p):
    return sum((((x >> i) & 1) << (3 * i + 2)) | (((y >> i) & 1) << (3 * i + 1)) | (((z >> i) & 1) << (3 * i)) for i in range(p))


def serialization():
    rng = np.random.default_rng(0)
    for p in (1, 2, 5, 9, 12, 16):
        g = rng.integers(0, 1 << p, (200, 3))
        ok = (Z.hilbert_encode(g, p) == [skilling(r, p) for r in g]).all() and (Z.z_encode(g, p) == [morton(*r, p) for r in g]).all()
        check(f"serialization codes depth {p} vs pure python", bool(ok))
    pts = (np.round(rng.random((4096, 3)) * 100) / 100).astype(np.float32)  # many duplicate voxels
    s = Z.serialize(pts)
    inv_ok = all((s["order"][k][s["inverse"][k]] == np.arange(len(pts))).all() for k in range(4))
    sorted_ok = all((np.diff(s["code"][k][s["order"][k]]) >= 0).all() for k in range(4))
    check("serialization order/inverse", inv_ok and sorted_ok, f"depth {int(s['grid'].max()).bit_length()}")


def stream(rng, tok, max_tokens=300):
    """A random token stream the grammar accepts (bos, articulationxl class, joints / branches, eos)."""
    ids = [T.BOS, T.CLS["articulationxl"]]
    while True:
        ok = tok.next_posible_token(np.array(ids))
        if len(ids) > max_tokens and T.EOS in ok:
            return ids + [T.EOS]
        t = int(rng.choice(ok))
        ids.append(t)
        if t == T.EOS:
            return ids


def tokenizer():
    rng, tok = np.random.default_rng(1), T.Tokenizer()
    ok = True
    for _ in range(5):
        r = tok.detokenize(np.array(stream(rng, tok)))
        J = len(r.parents)
        ok &= r.parents[0] is None and all(p is not None and p < i for i, p in enumerate(r.parents) if i) and r.tails.shape == (J, 3)
        ok &= r.joints.shape == (J, 3) and r.names == [f"bone_{i}" for i in range(J)] and np.isfinite(r.tails).all()
    check("tokenizer grammar + detokenize on random streams", bool(ok))
    # a hand-made skeleton: root, child, two branches off the child -> tails extrude / point at the single child
    ids = [T.BOS, T.CLS["articulationxl"], 128, 128, 128, 128, 160, 128, T.BRANCH, 128, 160, 128, 100, 200, 128, T.EOS]
    r = tok.detokenize(np.array(ids))
    check("tokenizer hand-made skeleton", r.parents == [None, 0, 1] and np.allclose(r.tails[0], r.joints[1]), f"parents {r.parents}")


def icosphere(c, r, sub=2):
    t = (1 + 5 ** .5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1],
                  [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v = list(v / np.linalg.norm(v, axis=1, keepdims=True))
    for _ in range(sub):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = (v[a] + v[b]) / 2
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c2 in f:
            ab, bc, ca = mid(a, b), mid(b, c2), mid(c2, a)
            nf += [[a, ab, ca], [b, bc, ab], [c2, ca, bc], [ab, bc, ca]]
        f = np.array(nf)
    return (np.array(v) * r + c).astype(np.float32), f


def brute_cast(V, F):
    tri = V[F].astype(np.float64)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]

    def cast(o, d):  # Moller-Trumbore against every triangle, one ray at a time (small meshes only)
        out = np.full(len(o), np.inf)
        for n in range(len(o)):
            p = np.cross(d[n], e2)
            det = (e1 * p).sum(1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
            s = o[n] - tri[:, 0]
            u = (s * p).sum(1) * inv
            q = np.cross(s, e1)
            w, t = (d[n] * q).sum(1) * inv, (e2 * q).sum(1) * inv
            hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0)
            if hit.any():
                out[n] = t[hit].min()
        return out
    return cast


def voxels():
    g, r = 24, 0.5
    V, F = icosphere(np.zeros(3), r)
    vox = P.voxelization(g, brute_cast(V, F))
    ideal = 4 / 3 * np.pi * r ** 3 / (2 / g) ** 3
    check("voxelization of a centred sphere", len(vox) > 0 and abs(len(vox) / ideal - 1) < 0.15 and np.abs(vox.mean(0)).max() < 0.05,
          f"{len(vox)} voxels, ideal {ideal:.0f}, centroid {vox.mean(0).round(3).tolist() if len(vox) else None}")
    # the end-to-end skin on the same mesh: weights are a distribution per vertex, closer joint = more weight
    nv = V / r
    joints = np.array([[0, 0, -.5], [0, 0, .5]], np.float32)
    sk = P.voxel_skin(g, vox / r * 1.0, joints, nv, F)
    check("voxel_skin rows sum to 1, nearer joint wins", np.allclose(sk.sum(1), 1) and (sk[nv[:, 2] > .8, 1] > sk[nv[:, 2] > .8, 0]).all(),
          f"shape {sk.shape}")
    parents = [None, 0]
    rs = P.reskin(nv[::3], nv, parents, F, sk[::3].astype(np.float32))
    check("reskin rows sum to 1", np.allclose(rs.sum(1), 1) and rs.shape == sk.shape)


def mesh():
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], np.float32)
    V = np.concatenate([cube, cube + 0])  # duplicated vertices (glTF seams) must merge
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    V, F = P.merge_vertices(np.concatenate([cube, cube]), np.concatenate([F, F + 8]) % 16)
    vn, fn = P.mesh_normals(V, F)
    check("merge_vertices + normals", len(V) == 8 and np.allclose(np.linalg.norm(vn, axis=1), 1, atol=1e-6) and np.allclose(np.abs(vn), 3 ** -.5, atol=1e-5),
          f"{len(V)} vertices")
    rs = np.random.RandomState(0)
    v2, _, _, m = P.affine(rs, V * np.array([2., 4., 1.], np.float32) + 5)
    check("affine into [-1,1]", np.allclose(v2.min(0), [-.5, -1, -.25], atol=1e-6) and np.allclose(v2.max(0), [.5, 1, .25], atol=1e-6))
    a, n, _ = P.sample_mix(np.random.RandomState(3), v2, vn, F, fn, 200, 5)
    b, _, _ = P.sample_mix(np.random.RandomState(3), v2, vn, F, fn, 200, 5)
    check("sample_mix deterministic per seed", (a == b).all() and len(a) == 200 and np.abs(a).max() <= 1)


if __name__ == "__main__":
    for f in (serialization, tokenizer, voxels, mesh):
        f()
    sys.exit(1 if bad else 0)
