"""ONNX Runtime runner for the UniRig SKIN model (export: export_skin.py).

numpy-only core (no torch at import time), so it also runs on the ARM job / a phone:
  build_nbr   neighbour table of a submanifold sparse conv (replaces spconv, see export_skin.GatherConv)
  fps_np      numpy farthest point sampling (== torch_cluster.fps, random_start=False)
  prepare     everything the graph takes as input; the 4 z/hilbert serialization orders still come from
              UniRig's torch code (serialize(); needs the UniRig checkout on sys.path)
  skin_mask   parents-accumulated voxel skin
  run         onnxruntime session -> per-sampled-point weights (N, J); voxel_skin postprocess/reskin stay UniRig's

CLI:
  selftest                                          numpy-only unit test of build_nbr + gather conv vs a dense conv
  check --onnx F --inputs inputs.npz --ref ref.npz  ORT vs the torch outputs saved by `export_skin.py capture`
  ort-run <run.py args>                             (inside the UniRig checkout) UniRig skin pipeline with the model
                                                    replaced by ORT; env SKIN_ONNX, SKIN_INPUTS
"""
import os
import sys
import time

import numpy as np

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")
GRID = 0.005
INPUT_NAMES = ["vertices", "normals", "joints", "voxel_skin", "skin_mask", "gsel", "nbr3", "nbr5", "order", "inverse"]
OUT_NAMES = ["skin", "dbg_global", "dbg_bone", "dbg_ptv3"]


def kperms(K):
    """4 candidate maps natural kernel offset index -> weight index (cross-correlation / convolution x axis order);
    export_skin picks the one that matches spconv on real data."""
    a = np.arange(K ** 3).reshape(K, K, K)
    f = a[::-1, ::-1, ::-1]
    return [a.reshape(-1), f.reshape(-1), a.transpose(2, 1, 0).reshape(-1), f.transpose(2, 1, 0).reshape(-1)]


def build_nbr(coords, K, tie=0):
    """(N, K^3) int64: row of the voxel at coords[i] + offset_k, or N (-> a zero row) if empty.
    Offset k = (a, b, c) over range(K), shifted by -K//2, k = (a*K + b)*K + c.
    Duplicate voxels: tie=0 -> the lowest row index owns the voxel, tie=1 -> the highest (spconv's hash table keeps one)."""
    c = np.asarray(coords).astype(np.int64)
    r = K // 2
    c = c - c.min(0) + r
    S = int(c.max()) + r + 1
    key = (c[:, 0] * S + c[:, 1]) * S + c[:, 2]
    N = len(key)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    nbr = np.empty((N, K ** 3), np.int64)
    k = 0
    for a in range(K):
        for b in range(K):
            for cc in range(K):
                q = key + ((a - r) * S + (b - r)) * S + (cc - r)
                pos = np.searchsorted(sk, q, side="left" if tie == 0 else "right") - tie
                pc = np.clip(pos, 0, N - 1)
                ok = (pos >= 0) & (pos < N) & (sk[pc] == q)
                nbr[:, k] = np.where(ok, order[pc], N)
                k += 1
    return nbr


def gather_conv_np(f, nbr, w, bias=None):
    """f (N, I), nbr (N, K3), w (K3, I, O) -> (N, O)."""
    fp = np.concatenate([f, np.zeros((1, f.shape[1]), f.dtype)])
    out = np.zeros((len(f), w.shape[2]), np.float64)
    for k in range(nbr.shape[1]):
        out += fp[nbr[:, k]].astype(np.float64) @ w[k]
    return (out + (0 if bias is None else bias)).astype(np.float32)


def fps_np(p, n):
    """Farthest point sampling from index 0 (torch_cluster.fps with random_start=False)."""
    p = p.astype(np.float32)
    d = np.full(len(p), np.inf, np.float32)
    sel = [0]
    for _ in range(n - 1):
        diff = p - p[sel[-1]]
        d = np.minimum(d, (diff * diff).sum(1))
        sel.append(int(np.argmax(d)))
    return np.array(sel, np.int64)


def global_sel(vertices, token_num=1024):
    """Rows of `vertices` used as the Michelangelo query points: 4*token_num random (seed 0) -> FPS keeps a quarter."""
    ind = np.random.default_rng(seed=0).choice(len(vertices), token_num * 4, replace=token_num * 4 > len(vertices))
    return ind[fps_np(vertices[ind], token_num)].astype(np.int64)


def skin_mask(voxel_skin, parents):
    m = voxel_skin.copy()
    for i, p in enumerate(parents):
        if p >= 0:
            m[:, p] += m[:, i]
    return m


def serialize(vertices, perm=None):
    """UniRig's own serialization (needs torch + the checkout). Returns grid_coord (N,3), order/inverse/code (4,N).
    perm: the random order shuffle the real run drew (export_skin saves it); None = unshuffled."""
    import torch
    from src.model.pointcept.models.utils.structure import Point
    p = Point({"coord": torch.from_numpy(vertices), "offset": torch.tensor([len(vertices)]), "grid_size": GRID})
    p.serialization(order=ORDERS, shuffle_orders=False)
    out = {"grid": p.grid_coord.numpy(), "order": p.serialized_order.numpy(), "inverse": p.serialized_inverse.numpy(),
           "code": p.serialized_code.numpy()}
    if perm is not None:
        for k in ("order", "inverse", "code"):
            out[k] = out[k][perm]
    return out


def prepare(vertices, normals, joints, voxel_skin, parents, perm=None, tie=0, serial=None):
    """All graph inputs for ONE mesh. vertices/normals (N,3), joints (J,3), voxel_skin (N,J), parents (J,) float32/int."""
    N = len(vertices)
    assert N % 1024 == 0, "patch attention without padding needs N % 1024 == 0"
    s = serial or serialize(vertices, perm)
    return {
        "vertices": vertices[None].astype(np.float32), "normals": normals[None].astype(np.float32),
        "joints": joints[None].astype(np.float32), "voxel_skin": voxel_skin.astype(np.float32),
        "skin_mask": skin_mask(voxel_skin, parents).astype(np.float32), "gsel": global_sel(vertices),
        "nbr3": build_nbr(s["grid"], 3, tie), "nbr5": build_nbr(s["grid"], 5, tie),
        "order": s["order"].astype(np.int64), "inverse": s["inverse"].astype(np.int64),
    }


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def run(sess, inp, outs=("skin",)):
    return sess.run(list(outs), {k: inp[k] for k in INPUT_NAMES})


def _hwm_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return -1.0


def diff_line(label, name, a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return f"ONNXDIFF {label} {name}: max={d.max():.3e} mean={d.mean():.3e} refmax={np.abs(b).max():.3e} shape={tuple(a.shape)}"


def check(onnx_path, inputs_path, ref_path, label, repeat=1):
    inp = dict(np.load(inputs_path))
    ref = dict(np.load(ref_path))
    t0 = time.time()
    sess = session(onnx_path)
    print(f"ORT session ready {time.time() - t0:.1f}s hwm={_hwm_gb():.2f}GB", flush=True)
    for i in range(repeat):
        t0 = time.time()
        res = run(sess, inp, OUT_NAMES)
        print(f"ORT run {i} {time.time() - t0:.1f}s hwm={_hwm_gb():.2f}GB", flush=True)
    for name, r in zip(OUT_NAMES, res):
        print(diff_line(label, name, r, ref[name]), flush=True)
    print(f"ONNXPEAK {label} hwm_gb={_hwm_gb():.2f}", flush=True)


def _run_py(run_args):
    import runpy
    sys.path.insert(0, os.getcwd())
    sys.argv = ["run.py"] + run_args
    runpy.run_path("run.py", run_name="__main__")


def ort_run(run_args):
    """UniRig skin pipeline (data extraction, voxel_skin, reskin, FBX export all stock) with UniRigSkin.predict_step
    answered by onnxruntime. SKIN_ONNX = model, SKIN_INPUTS = inputs npz of the same mesh from `export_skin.py capture`
    (only its shuffle permutation and voxel tie policy are used; everything else is recomputed by prepare() and compared)."""
    sys.path.insert(0, os.getcwd())
    import torch
    from src.model import unirig_skin as us
    sess = session(os.environ["SKIN_ONNX"])
    meta = dict(np.load(os.environ["SKIN_INPUTS"]))

    def predict_step(self, batch):
        t0 = time.time()
        g = lambda k: batch[k][0].detach().cpu().numpy()
        J = int(batch["num_bones"][0])
        inp = prepare(g("vertices").astype(np.float32), g("normals").astype(np.float32), g("joints")[:J].astype(np.float32),
                      g("voxel_skin")[:, :J].astype(np.float32), g("parents")[:J], perm=meta["perm"], tie=int(meta["tie"]))
        for k in INPUT_NAMES:
            ok = inp[k].shape == meta[k].shape and np.allclose(inp[k], meta[k], atol=1e-6)
            print(f"PREP {k}: {'equal to capture' if ok else 'DIFFERS from capture'}", flush=True)
        t1 = time.time()
        skin = run(sess, inp)[0]
        print(f"ORT skin {skin.shape} prepare={t1 - t0:.1f}s ort={time.time() - t1:.1f}s hwm={_hwm_gb():.2f}GB", flush=True)
        return [torch.from_numpy(skin)]

    us.UniRigSkin.predict_step = predict_step
    _run_py(run_args)


def selftest():
    """numpy only: build_nbr + gather conv == a dense 3D correlation on a small random voxel set with duplicates."""
    rng = np.random.default_rng(1)
    for K in (3, 5):
        r = K // 2
        coords = rng.integers(0, 6, size=(120, 3))  # 216 cells, 120 points -> plenty of duplicates and neighbours
        N, I, O = len(coords), 4, 5
        f = rng.standard_normal((N, I)).astype(np.float32)
        W = rng.standard_normal((K, K, K, I, O))
        bias = rng.standard_normal(O)
        for tie in (0, 1):
            # dense grid holding the owner row's feature of every voxel (first / last point wins)
            grid = np.zeros((6 + 2 * r,) * 3 + (I,))
            seen = {}
            for i in (reversed(range(N)) if tie == 0 else range(N)):  # later write wins
                seen[tuple(coords[i])] = i
            for c, i in seen.items():
                grid[c[0] + r, c[1] + r, c[2] + r] = f[i]
            want = np.zeros((N, O))
            for i, (x, y, z) in enumerate(coords):
                for a in range(K):
                    for b in range(K):
                        for c in range(K):
                            want[i] += grid[x + a, y + b, z + c] @ W[a, b, c]
            want += bias
            got = gather_conv_np(f, build_nbr(coords, K, tie), W.reshape(K ** 3, I, O), bias)
            err = np.abs(got - want).max()
            print(f"selftest K={K} tie={tie}: max abs err {err:.2e}")
            assert err < 1e-4, err
    assert sorted(kperms(3)[1]) == list(range(27)) and kperms(3)[2][1] == 9
    assert np.allclose(skin_mask(np.eye(3, dtype=np.float32), [-1, 0, 1])[:, 0], [1, 1, 0])  # sequential like UniRig: child before its own child
    p = np.random.default_rng(2).standard_normal((50, 3))
    s = fps_np(p, 10)
    assert s[0] == 0 and len(set(s)) == 10
    print("selftest OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "check":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--onnx", required=True)
        ap.add_argument("--inputs", required=True)
        ap.add_argument("--ref", required=True)
        ap.add_argument("--label", default="ort")
        ap.add_argument("--repeat", type=int, default=1)
        a = ap.parse_args()
        check(a.onnx, a.inputs, a.ref, a.label, a.repeat)
    elif cmd == "ort-run":
        ort_run(sys.argv[2:])
    else:
        sys.exit(__doc__)
