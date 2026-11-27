"""P3-SAM post-processing in numpy / scipy only (runs in Blender's Python on the phone).

per-prompt masks (N points x K prompts) -> part labels on the sample points -> part labels on the mesh faces -> clean parts.

Stable part count comes from four things on top of the reference recipe (see tune_post.py for the sweep that picked the defaults):
  * masks of one NMS cluster vote (a point belongs to the instance when most members say so) instead of the head mask alone,
  * clusters with too little support or too low predicted IoU are dropped,
  * connected pieces smaller than min_frac of the surface area merge into the neighbour they share the longest border with,
  * optional target: merge the smallest part into its longest-border neighbour until the count is reached (or, when the
    model found fewer, relax the thresholds first).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DEFAULTS = dict(nms_iou=0.75, min_support=2, min_iou_pred=0.0, vote=0.5, merge_bbox=0.0, min_frac=0.01, smooth=2)


def unpack_masks(packed, n):
    return np.unpackbits(packed, axis=0)[:n].astype(bool)


# ---------------------------------------------------------------------------------------------------- masks -> point labels
def instances(masks, iou, pts, p):
    """masks (N,K) bool, iou (K,) -> list of boolean instance masks (N,), biggest first."""
    N, K = masks.shape
    keep = np.nonzero((iou >= p["min_iou_pred"]) & (masks.sum(0) > 0))[0]
    if len(keep) == 0:
        return []
    order = keep[np.argsort(-iou[keep], kind="stable")]
    M = masks[:, order].T.astype(np.float32)
    area = M.sum(1)
    inter = M @ M.T
    IoU = inter / np.maximum(area[:, None] + area[None] - inter, 1)
    heads, members = [], []
    for i in range(len(order)):
        for h, mem in zip(heads, members):
            if IoU[i, h] > p["nms_iou"]:
                mem.append(i)
                break
        else:
            heads.append(i)
            members.append([i])
    out = []
    for h, mem in zip(heads, members):
        if len(mem) < p["min_support"]:
            continue
        if p["vote"] > 0 and len(mem) > 2:
            m = M[mem].mean(0) >= p["vote"]
            if m.sum() == 0:
                m = M[h] > 0
        else:
            m = M[h] > 0
        out.append(m)
    if not out:  # nothing had enough support: the best mask alone
        out = [M[0] > 0]
    if p["merge_bbox"] > 0:  # reference recipe: instances whose bounding boxes overlap more than this are one
        out = merge_bbox(out, pts, p["merge_bbox"])
    out.sort(key=lambda m: -m.sum())
    return out


def bbox_iou(a, b):
    a0, a1, b0, b1 = a.min(0), a.max(0), b.min(0), b.max(0)
    lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
    if (lo >= hi).any():
        return 0.0
    inter = np.prod(hi - lo)
    u = np.prod(a1 - a0) + np.prod(b1 - b0) - inter
    return float(inter / u) if u > 0 else 0.0


def merge_bbox(ms, pts, thr):
    ms = sorted(ms, key=lambda m: -m.sum())
    used = [False] * len(ms)
    out = []
    for i, m in enumerate(ms):
        if used[i]:
            continue
        out.append(m)
        for j in range(i + 1, len(ms)):
            if not used[j] and bbox_iou(pts[m], pts[ms[j]]) > thr:
                used[j] = True
    return out


def point_labels(inst, n):
    """Bigger instances first, smaller ones overwrite -> labels (n,) with -1 = no instance."""
    lab = -np.ones(n, np.int64)
    for k, m in enumerate(inst):
        lab[m] = k
    ids = np.unique(lab[lab >= 0])
    remap = -np.ones(len(inst) + 1, np.int64)
    remap[ids] = np.arange(len(ids))
    out = np.where(lab >= 0, remap[np.maximum(lab, 0)], -1)
    return out


# ---------------------------------------------------------------------------------------------------- faces
class FaceGraph:
    """Face adjacency of a triangle mesh: A (sparse, symmetric), shared-edge lengths, face areas."""

    def __init__(self, v, f):
        v = np.asarray(v, np.float64)
        f = np.asarray(f, np.int64)
        self.v, self.f = v, f
        t = v[f]
        self.area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        self.centers = t.mean(1)
        nF = len(f)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        fid = np.tile(np.arange(nF), 3)
        key = np.minimum(e[:, 0], e[:, 1]) * len(v) + np.maximum(e[:, 0], e[:, 1])
        o = np.argsort(key, kind="stable")
        ks, fs = key[o], fid[o]
        same = ks[1:] == ks[:-1]
        a, b = fs[:-1][same], fs[1:][same]
        e0, e1 = e[o][:-1][same][:, 0], e[o][:-1][same][:, 1]
        ln = np.linalg.norm(v[e0] - v[e1], axis=1)
        mid = 0.5 * (v[e0] + v[e1])
        ok = a != b
        a, b, ln, mid = a[ok], b[ok], ln[ok], mid[ok]
        self.pairs, self.plen, self.pmid = np.stack([a, b], 1), ln, mid
        self.A = sp.coo_matrix((np.ones(2 * len(a), np.float32), (np.r_[a, b], np.r_[b, a])), shape=(nF, nF)).tocsr()


def vote(A, lab, n_lab):
    ok = lab >= 0
    oh = sp.coo_matrix((np.ones(ok.sum(), np.float32), (np.nonzero(ok)[0], lab[ok])), shape=(len(lab), n_lab)).tocsr()
    V = (A @ oh).toarray()
    return V.argmax(1), V.max(1), V


def faces_from_points(G, pts, lab, smooth=2):
    n_lab = int(lab.max()) + 1
    _, nn = cKDTree(pts).query(G.centers)
    fl = lab[nn].copy()
    for _ in range(200):
        un = fl < 0
        if not un.any():
            break
        best, cnt, _ = vote(G.A, fl, n_lab)
        fill = un & (cnt > 0)
        if not fill.any():
            break
        fl[fill] = best[fill]
    if (fl < 0).any():
        good = lab >= 0
        _, nn2 = cKDTree(pts[good]).query(G.centers[fl < 0])
        fl[fl < 0] = lab[good][nn2]
    for _ in range(smooth):
        best, cnt, V = vote(G.A, fl, n_lab)
        own = V[np.arange(len(fl)), fl]
        flip = (best != fl) & (cnt > own) & (cnt >= 2)
        fl[flip] = best[flip]
    return fl


def split_components(G, fl):
    """Every connected piece of a label becomes its own label."""
    same = fl[G.pairs[:, 0]] == fl[G.pairs[:, 1]]
    a, b = G.pairs[same, 0], G.pairs[same, 1]
    n = len(fl)
    g = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    nc, cc = sp.csgraph.connected_components(g, directed=False)
    return cc.astype(np.int64)


def region_graph(G, fl):
    """areas (R,), border length matrix (R,R) sparse symmetric between different labels."""
    R = int(fl.max()) + 1
    area = np.bincount(fl, weights=G.area, minlength=R)
    la, lb = fl[G.pairs[:, 0]], fl[G.pairs[:, 1]]
    d = la != lb
    B = sp.coo_matrix((np.r_[G.plen[d], G.plen[d]], (np.r_[la[d], lb[d]], np.r_[lb[d], la[d]])), shape=(R, R)).tocsr()
    return area, B


def renumber(fl):
    ids, inv = np.unique(fl, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def merge_small(G, fl, min_frac=0.01, target=None, min_parts=1):
    """Merge the smallest part into the neighbour it shares the longest border with, while it is below min_frac of the area
    (or while there are more than target parts). Isolated parts with no neighbour stay."""
    fl = renumber(fl)
    total = G.area.sum()
    while True:
        area, B = region_graph(G, fl)
        R = len(area)
        if R <= min_parts:
            break
        deg = np.asarray((B > 0).sum(1)).ravel()
        order = np.argsort(area)
        pick = None
        for r in order:
            if deg[r] == 0:
                continue
            if (target is not None and R > target) or area[r] < min_frac * total:
                pick = r
                break
            break
        if pick is None:
            break
        row = B.getrow(pick)
        nb = row.indices[np.argmax(row.data)]
        fl[fl == pick] = nb
        fl = renumber(fl)
    return fl


# ---------------------------------------------------------------------------------------------------- driver
def segment(G, pts, masks, iou, params=None, target=None, log=print):
    """-> face labels (nF,) 0..P-1, info. target = wanted number of parts (None = auto)."""
    p = {**DEFAULTS, **(params or {})}
    N = len(pts)
    levels = [p]
    if target is not None:  # relax thresholds step by step while the model finds fewer parts than wanted
        levels += [{**p, "nms_iou": 0.85, "min_support": 1, "min_frac": 0.004, "vote": 0.0}, {**p, "nms_iou": 0.92, "min_support": 1, "min_frac": 0.0, "vote": 0.0, "min_iou_pred": 0.0}]
    best = None
    for li, q in enumerate(levels):
        inst = instances(masks, iou, pts, q)
        lab = point_labels(inst, N)
        fl = faces_from_points(G, pts, lab, q["smooth"])
        fl = split_components(G, fl)
        fl = merge_small(G, fl, q["min_frac"], target=target)
        best = fl
        info = dict(level=li, instances=len(inst), parts=int(fl.max()) + 1)
        if target is None or info["parts"] >= target:
            break
    return renumber(best), info


def agreement(la, lb, w=None):
    """Permutation-invariant agreement of two labelings (area-weighted): the best one-to-one label matching."""
    w = np.ones(len(la)) if w is None else w
    a, b = np.unique(la, return_inverse=True)[1].reshape(-1), np.unique(lb, return_inverse=True)[1].reshape(-1)
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), w)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / w.sum())
