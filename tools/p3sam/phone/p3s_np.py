"""P3-SAM (Tencent Hunyuan3D-Part) on the phone: everything around the two ONNX graphs in numpy / scipy / onnxruntime only.

  p3sam_enc.onnx   Sonata encoder (+ MLP 1232->512). Inputs are numpy tables (dynamic in every voxel count), see build_tables().
  p3sam_heads.onnx prompt heads. feats (N,512) points (N,3) prompts (K,3) -> mask logits (N,K,3), predicted IoU (K,3).

No torch / spconv / torch_scatter / trimesh. Deterministic: stratified surface sampling from a seed, mean-of-voxel features instead of
the random point per voxel, identity order permutation instead of Sonata's random "shuffle_orders", farthest-point prompts.
"""
import os
import time

import numpy as np

GRID = 0.005  # Sonata GridSample size on the normalised cloud
PATCH = 1024  # serialized-attention patch size (all stages)
DEPTHS = (3, 3, 3, 12, 3)
N_ORDERS = 4  # z, z-trans, hilbert, hilbert-trans; block i of a stage uses order i % 4
COLOR = 1.0 / 127.5 * 1.0 - 1.0  # NormalizeColor of an all-ones colour


def used_orders(s):
    return sorted({i % N_ORDERS for i in range(DEPTHS[s])})


def input_names():
    """Encoder input names, in graph order."""
    names = ["feat0"]
    names += [f"nbr{s}" for s in range(5)]
    for s in range(1, 5):
        names += [f"pg{s}", f"pinv{s}"]
    for s in range(5):
        for oi in used_orders(s):
            names += [f"ord{s}_{oi}", f"inv{s}_{oi}"]
    return names


# ---------------------------------------------------------------------------------------------------- runtime plumbing
def tame_malloc(mmap_threshold=4 << 20):
    """Fixed glibc mmap threshold + one arena: big blocks go back to the kernel on free (peak RSS on the phone). No-op without glibc."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        return all([libc.mallopt(-3, mmap_threshold), libc.mallopt(-1, mmap_threshold), libc.mallopt(-2, 0), libc.mallopt(-8, 1)])
    except Exception:
        return False


def trim_malloc():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def hwm_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def session(path, threads=0):
    """Low-memory CPU session: no arena, no memory pattern (dynamic shapes), external weights memory-mapped, no MatMul prepacking copy."""
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.log_severity_level = 3
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    o.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    o.add_session_config_entry("session.disable_prepacking", "1")
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------------------------------- points
def normalize_pc(pc):
    mx, mn = pc.max(0), pc.min(0)
    c, s = (mx + mn) / 2, np.abs((mx - mn) / 2).max()
    return (pc - c) / (s + 1e-10)


def face_normals_areas(v, f):
    t = v[f]
    c = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(c, axis=1)
    return c / np.maximum(a, 1e-30)[:, None], a * 0.5


def sample_surface(v, f, n, seed):
    """n area-proportional surface points: stratified along the cumulative area (low variance between seeds), random barycentric
    position inside the face. -> points (n,3), normals (n,3), face index (n,)"""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    nrm, area = face_normals_areas(v, f)
    rng = np.random.default_rng(seed)
    cum = np.cumsum(area)
    pos = (np.arange(n) + rng.random(n)) / n * cum[-1]
    fi = np.minimum(np.searchsorted(cum, pos, side="right"), len(f) - 1)
    u = rng.random((n, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    t = v[f[fi]]
    p = t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])
    return p, nrm[fi], fi


def fps(p, n, start=None):
    """Farthest point sampling; deterministic start = the point farthest from the centroid unless given."""
    p = np.asarray(p, np.float32)
    n = min(n, len(p))
    if start is None:
        start = int(np.argmax(((p - p.mean(0)) ** 2).sum(1)))
    d = np.full(len(p), np.inf, np.float32)
    sel = np.empty(n, np.int64)
    sel[0] = start
    for i in range(1, n):
        diff = p - p[sel[i - 1]]
        d = np.minimum(d, (diff * diff).sum(1))
        sel[i] = int(np.argmax(d))
    return sel


# ---------------------------------------------------------------------------------------------------- voxels + tables
def voxelize(pts_norm, normals, grid=GRID):
    """Sonata's CenterShift(apply_z) + GridSample + NormalizeColor, deterministic: the voxel's feature is the MEAN of its points.
    -> grid_coord (n0,3) int64 (sorted by x,y,z), feat0 (n0,9) float32 = [coord, colour, normal], inverse (N,) point -> voxel"""
    pts = np.asarray(pts_norm, np.float64)
    mn, mx = pts.min(0), pts.max(0)
    coord = pts - np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]])
    g = np.floor(coord / grid).astype(np.int64)
    g -= g.min(0)
    S = int(g.max()) + 1
    key = (g[:, 0] * S + g[:, 1]) * S + g[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    c = np.stack([np.bincount(inv, coord[:, k]) for k in range(3)], 1) / cnt[:, None]
    nm = np.stack([np.bincount(inv, normals[:, k]) for k in range(3)], 1)
    nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
    gc = np.stack([uk // (S * S), (uk // S) % S, uk % S], 1).astype(np.int64)
    feat0 = np.concatenate([c, np.full((len(uk), 3), COLOR), nm], 1).astype(np.float32)
    return gc, feat0, inv


def natural_offsets(K=3):
    return np.stack(np.meshgrid(*[np.arange(K)] * 3, indexing="ij"), -1).reshape(-1, 3) - K // 2


def build_nbr(coords, K=3):
    """(N, K^3) int64: row of the voxel at coords[i] + offset k (column (a*3+b)*3+c, offset (a-1,b-1,c-1)), N = empty (zero row). Centre = itself."""
    c = np.asarray(coords).astype(np.int64)
    offs = natural_offsets(K)
    r = int(np.abs(offs).max())
    c = c - c.min(0) + r
    S = int(c.max()) + r + 1
    key = (c[:, 0] * S + c[:, 1]) * S + c[:, 2]
    N = len(key)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    nbr = np.empty((N, len(offs)), np.int64)
    for k, (dx, dy, dz) in enumerate(offs):
        q = key + (dx * S + dy) * S + dz
        pos = np.searchsorted(sk, q, side="left")
        pc = np.clip(pos, 0, N - 1)
        ok = (pos < N) & (sk[pc] == q)
        nbr[:, k] = np.where(ok, order[pc], N)
        if not (dx or dy or dz):
            nbr[:, k] = np.arange(N)
    return nbr


def pool_group_table(cluster, n_out, cols=8):
    """(n_out, cols): child rows per cluster (stride 2 in 3-D: at most 8), short clusters padded with their own last member (max is idempotent)."""
    order = np.argsort(cluster, kind="stable")
    counts = np.bincount(cluster, minlength=n_out)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    assert counts.max() <= cols
    G = np.empty((n_out, cols), np.int64)
    for j in range(G.shape[1]):
        G[:, j] = order[np.minimum(starts + j, starts + counts - 1)]
    return G


def z_code(g, depth):
    g = g.astype(np.int64)
    key = np.zeros(len(g), np.int64)
    for i in range(depth):
        key |= ((g[:, 0] >> i) & 1) << (3 * i + 2)
        key |= ((g[:, 1] >> i) & 1) << (3 * i + 1)
        key |= ((g[:, 2] >> i) & 1) << (3 * i)
    return key


def hilbert_code(locs, nbits):
    """Skilling's Hilbert index of integer 3-D locations, the same bit handling as Pointcept's serialization/hilbert.py (numpy)."""
    n = len(locs)
    sh = np.arange(nbits - 1, -1, -1)
    gray = ((locs.astype(np.int64)[:, :, None] >> sh) & 1).astype(np.uint8)  # (n, 3, nbits) MSB first
    for bit in range(nbits):
        for dim in range(3):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] ^= mask[:, None]
            to_flip = (1 - mask)[:, None] & (gray[:, 0, bit + 1:] ^ gray[:, dim, bit + 1:])
            gray[:, dim, bit + 1:] ^= to_flip
            gray[:, 0, bit + 1:] ^= to_flip
    flat = gray.swapaxes(1, 2).reshape(n, nbits * 3)
    b = np.bitwise_xor.accumulate(flat, axis=1).astype(np.int64)
    return (b << np.arange(nbits * 3 - 1, -1, -1)).sum(1)


def serialize(grid):
    """z, z-trans, hilbert, hilbert-trans orders of one stage: [(order, inverse)] with order = argsort(code), inverse = its inverse."""
    depth = int(int(grid.max()) + 1).bit_length()
    tr = grid[:, [1, 0, 2]]
    codes = [z_code(grid, depth), z_code(tr, depth), hilbert_code(grid, depth), hilbert_code(tr, depth)]
    out = []
    for c in codes:
        o = np.argsort(c, kind="stable")
        i = np.empty_like(o)
        i[o] = np.arange(len(o))
        out.append((o, i))
    return out


def patch_tables(n, K):
    """Sonata's get_padding_and_inverse for one cloud: index of the sorted point at each padded slot."""
    npad = -(-n // K) * K if n > K else n
    pad = np.arange(npad)
    if npad != n:
        r = n % K
        pad[npad - K + r:npad] = pad[npad - 2 * K + r:npad - K]
    return pad


def build_tables(grid0):
    """Every structure table of the encoder for a voxel grid -> dict name -> int64 array, and the per-stage voxel counts."""
    T, ns = {}, []
    grid = grid0
    for s in range(5):
        if s > 0:
            gp = grid // 2
            S = int(gp.max()) + 1
            key = (gp[:, 0] * S + gp[:, 1]) * S + gp[:, 2]
            uk, cl = np.unique(key, return_inverse=True)
            cl = cl.reshape(-1)
            grid = np.stack([uk // (S * S), (uk // S) % S, uk % S], 1).astype(np.int64)
            T[f"pg{s}"] = pool_group_table(cl, len(uk))
            T[f"pinv{s}"] = cl.astype(np.int64)
        n = len(grid)
        ns.append(n)
        T[f"nbr{s}"] = build_nbr(grid)
        K = min(n, PATCH)
        pad = patch_tables(n, K)
        ser = serialize(grid)
        for oi in used_orders(s):
            o, i = ser[oi]
            T[f"ord{s}_{oi}"] = np.ascontiguousarray(o[pad].reshape(-1, K))
            T[f"inv{s}_{oi}"] = i.astype(np.int64)
    return T, ns


# ---------------------------------------------------------------------------------------------------- model
class P3Sam:
    """Loads one graph at a time (peak RSS): features() then masks()."""

    def __init__(self, enc_path, heads_path, threads=0, log=print):
        self.enc_path, self.heads_path, self.threads, self.log = enc_path, heads_path, threads, log

    def features(self, pts_norm, normals):
        """(N,512) float32 features of the points."""
        t = time.time()
        grid, feat0, inv = voxelize(pts_norm, normals)
        T, ns = build_tables(grid)
        T["feat0"] = feat0
        self.log(f"P3S voxels per stage {ns}; tables {(time.time() - t):.1f}s; rss {rss_mb():.0f} MB")
        t = time.time()
        s = session(self.enc_path, self.threads)
        self.log(f"P3S encoder loaded {(time.time() - t):.1f}s; rss {rss_mb():.0f} MB")
        t = time.time()
        f0 = s.run(None, {i.name: T[i.name] for i in s.get_inputs()})[0]
        self.log(f"P3S encoder run {(time.time() - t):.1f}s; peak rss {hwm_mb():.0f} MB")
        del s, T
        trim_malloc()
        return f0[inv].astype(np.float32)

    def masks(self, feats, pts_norm, prompts, kchunk=4):
        """-> logits (N,K,3) float16, iou (K,3) float32"""
        t = time.time()
        s = session(self.heads_path, self.threads)
        L, I = [], []
        for i in range(0, len(prompts), kchunk):
            l, io = s.run(None, {"feats": feats, "points": pts_norm.astype(np.float32), "prompts": prompts[i:i + kchunk].astype(np.float32)})
            L.append(l.astype(np.float16))
            I.append(io)
        self.log(f"P3S heads {len(prompts)} prompts {(time.time() - t):.1f}s; peak rss {hwm_mb():.0f} MB")
        del s
        trim_malloc()
        return np.concatenate(L, 1), np.concatenate(I)
