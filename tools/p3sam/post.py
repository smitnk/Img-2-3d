"""P3-SAM post-processing (numpy/scipy only): per-prompt masks -> part labels on points -> part labels on mesh faces.

Same steps as Hunyuan3D-Part/P3-SAM/demo/auto_mask.py (sort by predicted IoU, mask NMS at IoU 0.9, drop clusters with too few prompts,
merge clusters whose point bounding boxes overlap > 0.5, patch missed masks, smaller masks win overlaps, project to faces, fill
unlabelled faces from neighbours, split by connectivity and merge tiny pieces), vectorised, and simplified to what the phone needs.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def best_masks(logits, iou):
    """logits (N,K,3) from the heads graph, iou (K,3) -> boolean masks (N,K) of the head with the highest predicted IoU, iou (K,)."""
    j = iou.argmax(1)
    m = np.take_along_axis(logits, j[None, :, None], axis=2)[:, :, 0] > 0
    return m, iou[np.arange(len(j)), j]


def bbox_iou(p1, p2):
    a0, a1, b0, b1 = p1.min(0), p1.max(0), p2.min(0), p2.max(0)
    lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
    if (lo >= hi).any():
        return 0.0
    inter = np.prod(hi - lo)
    u = np.prod(a1 - a0) + np.prod(b1 - b0) - inter
    return float(inter / u) if u > 0 else 0.0


def masks_to_point_labels(pts, masks, iou, nms_iou=0.9, min_members=None, merge_bbox=0.5, patch=0.7):
    """pts (N,3) normalised, masks (N,K) bool, iou (K,) -> labels (N,) in 0..P-1 (-1 = no mask), info."""
    N, K = masks.shape
    if min_members is None:
        min_members = max(1, int(round(3 * K / 400)))  # 3 of 400 prompts in the reference setup
    order = np.argsort(-iou, kind="stable")
    M = masks[:, order].T.astype(np.float32)  # (K, N) sorted by IoU
    area = M.sum(1)
    inter = M @ M.T
    IoU = inter / np.maximum(area[:, None] + area[None] - inter, 1)
    heads, members = [], {}
    for i in range(K):
        for h in heads:
            if IoU[i, h] > nms_iou:
                members[h].append(i)
                break
        else:
            heads.append(i)
            members[i] = [i]
    kept = [h for h in heads if len(members[h]) >= min_members and area[h] > 0]
    if not kept:
        kept = [h for h in heads if area[h] > 0]
    cluster2, used = [], [False] * len(kept)
    for a, ha in enumerate(kept):
        if used[a]:
            continue
        cluster2.append(ha)
        for b in range(a + 1, len(kept)):
            hb = kept[b]
            if not used[b] and M[ha].any() and M[hb].any() and bbox_iou(pts[M[ha] > 0], pts[M[hb] > 0]) > merge_bbox:
                used[b] = True
    no_mask = np.ones(N, bool)
    for h in cluster2:
        no_mask[M[h] > 0] = False
    for i in range(K):
        if i in cluster2 or area[i] == 0:
            continue
        if (M[i] > 0)[no_mask].sum() / area[i] > patch:
            cluster2.append(i)
            no_mask[M[i] > 0] = False
    cluster2.sort(key=lambda i: -area[i])
    lab = -np.ones(N, np.int64)
    for rank, i in enumerate(cluster2):  # biggest first, smaller masks overwrite
        lab[M[i] > 0] = rank
    ids = [u for u in np.unique(lab) if u >= 0]
    remap = {u: n for n, u in enumerate(ids)}
    lab = np.array([remap.get(int(u), -1) for u in lab], np.int64)
    info = dict(prompts=K, nms_clusters=len(heads), kept_clusters=len(kept), min_members=min_members, merged=len(cluster2), unlabelled_points=int((lab < 0).sum()))
    return lab, info


def face_adjacency_matrix(mesh):
    a = mesh.face_adjacency
    n = len(mesh.faces)
    A = sp.coo_matrix((np.ones(len(a) * 2, np.float32), (np.r_[a[:, 0], a[:, 1]], np.r_[a[:, 1], a[:, 0]])), shape=(n, n)).tocsr()
    return A


def neighbour_vote(A, lab, n_lab, mask_faces=None):
    """For each face the label with the most neighbours (unlabelled -1 ignored) and its vote count."""
    ok = lab >= 0
    oh = sp.coo_matrix((np.ones(ok.sum(), np.float32), (np.nonzero(ok)[0], lab[ok])), shape=(len(lab), n_lab)).tocsr()
    V = (A @ oh).toarray()
    return V.argmax(1), V.max(1), V


def faces_from_points(mesh, pts, lab, A=None, smooth=2, min_piece=0.004):
    """Project point labels to faces (nearest sample), fill gaps, majority-smooth, then merge tiny disconnected pieces."""
    A = face_adjacency_matrix(mesh) if A is None else A
    n_lab = int(lab.max()) + 1
    cent = mesh.triangles_center
    _, nn = cKDTree(pts).query(cent)
    fl = lab[nn].copy()
    for _ in range(100):  # fill unlabelled faces from their neighbours
        un = fl < 0
        if not un.any():
            break
        best, cnt, _ = neighbour_vote(A, fl, n_lab)
        fill = un & (cnt > 0)
        if not fill.any():
            break
        fl[fill] = best[fill]
    if (fl < 0).any():  # isolated shells with no labelled neighbours: nearest labelled sample
        _, nn2 = cKDTree(pts[lab >= 0]).query(cent[fl < 0])
        fl[fl < 0] = lab[lab >= 0][nn2]
    for _ in range(smooth):  # a face flips to a label that holds the strict majority of its (<=3) neighbours
        best, cnt, V = neighbour_vote(A, fl, n_lab)
        own = V[np.arange(len(fl)), fl]
        flip = (best != fl) & (cnt > own) & (cnt >= 2)
        fl[flip] = best[flip]
    area = mesh.area_faces
    tot = area.sum()
    for _ in range(10):  # tiny connected pieces of a label join the neighbouring label they touch most
        changed = False
        for l in np.unique(fl):
            idx = np.nonzero(fl == l)[0]
            sub = A[idx][:, idx]
            nc, cc = sp.csgraph.connected_components(sub, directed=False)
            if nc <= 1:
                continue
            sizes = np.bincount(cc, weights=area[idx], minlength=nc)
            big = sizes.argmax()
            for c in range(nc):
                if c == big or sizes[c] >= min_piece * tot:
                    continue
                fs = idx[cc == c]
                _, _, V = neighbour_vote(A, np.where(fl == l, -1, fl), n_lab)
                v = V[fs].sum(0)
                if v.max() > 0:
                    fl[fs] = v.argmax()
                    changed = True
        if not changed:
            break
    ids = np.unique(fl)
    remap = -np.ones(ids.max() + 1, np.int64)
    remap[ids] = np.arange(len(ids))
    return remap[fl]


def agreement(la, lb, weights=None):
    """Permutation-invariant agreement of two labelings: fraction (by weight) covered by the best one-to-one label matching."""
    w = np.ones(len(la)) if weights is None else weights
    a, b = np.unique(la, return_inverse=True)[1], np.unique(lb, return_inverse=True)[1]
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), w)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / w.sum())
