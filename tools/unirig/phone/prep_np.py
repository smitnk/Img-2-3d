"""numpy + scipy ports of the UniRig data steps the phone runner needs (no torch, trimesh, fast_simplification, pyrender, open3d).

Verbatim copies (MIT, VAST-AI-Research/UniRig @6793c66) unless noted:
  merge_vertices / mesh_normals  trimesh.Trimesh(process=True) + vertex_normals / face_normals (re-implemented, float64 maths)
  affine / sample_mix            src/data/augment.py AugmentAffine, src/data/sampler.py SamplerMix; they draw from a RandomState in
                                 the same order as the np.random globals do in UniRig, so RandomState(s) == np.random.seed(s)
  connect_tails                  src/data/tail.py (connect_tail_to_unique_son)
  voxelization                   vertex_group.voxelization(backend='pyrender') with the pyrender depth images replaced by ray casts
                                 (same camera poses / pixel centres / index maths), see `voxelization`
  voxel_skin / reskin            src/data/vertex_group.py voxel_skin, src/system/skin.py reskin (scipy)
"""
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- mesh (what extract.py's save_raw_data gets from trimesh)
def merge_vertices(V, F, digits=8):
    """trimesh merge_vertices: vertices equal after rounding to 1e-8 become one (glTF import splits them at UV seams / hard edges)."""
    key = np.round(np.asarray(V, np.float64) * 10.0 ** digits).astype(np.int64)
    o = np.lexsort(key.T[::-1])  # stable: the first vertex of a group is its lowest original index
    ks = key[o]
    new = np.concatenate([[True], (ks[1:] != ks[:-1]).any(1)])
    inv = np.empty(len(V), np.int64)
    inv[o] = np.cumsum(new) - 1
    return np.asarray(V)[o[new]], inv[F]


def mesh_normals(V, F):
    """(vertex_normals, face_normals) float32, as trimesh: unit face normals (0 for degenerate faces), vertex normals = sum of
    face normals weighted by the corner angle, unitised."""
    V = np.asarray(V, np.float64)
    t = V[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    fn = np.where(ln > 1e-12, n / np.maximum(ln, 1e-300), 0.0)
    vn = np.zeros((len(V), 3))
    for k in range(3):
        a, b = t[:, (k + 1) % 3] - t[:, k], t[:, (k + 2) % 3] - t[:, k]
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-300)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-300)
        w = np.arccos(np.clip((a * b).sum(1), -1, 1))[:, None] * fn
        for c in range(3):
            vn[:, c] += np.bincount(F[:, k], weights=w[:, c], minlength=len(V))
    l = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.where(l > 1e-12, vn / np.maximum(l, 1e-300), 0.0).astype(np.float32), fn.astype(np.float32)


# ---------------------------------------------------------------- augment / sampler
def _t(v):
    m = np.eye(4, dtype=np.float32)
    m[0:3, 3] = v
    return m


def _s(r):
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = r
    m[3, 3] = 1.
    return m


def affine(rs, vertices, joints=None, tails=None, into=(-1.0, 1.0)):
    """AugmentAffine(normalize_into=into, random_*_p=0).transform: bounds of vertices (+ joints) -> centred cube. Returns
    (vertices, joints, tails, 4x4 matrix). The two rs.rand() are the random_scale_p / random_shift_p draws UniRig makes with p == 0."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    if joints is not None:
        lo, hi = np.minimum(lo, joints.min(axis=0)), np.maximum(hi, joints.max(axis=0))
    m = _t(-(hi + lo) / 2) @ np.eye(4, dtype=np.float32)
    m = _s(1. / np.max((hi - lo) / (into[1] - into[0]))) @ m
    b = (into[0] + into[1]) / 2
    m = _t(np.array([b, b, b], dtype=np.float32)) @ m
    rs.rand()
    rs.rand()
    ap = lambda v: None if v is None else np.matmul(v, m[:3, :3].transpose()) + m[:3, 3]
    return ap(vertices), ap(joints), ap(tails), m


def sample_surface(rs, num_samples, vertices, faces):
    """sampler.sample_surface(return_weight=True): area^2-weighted faces (sic), random barycentric points."""
    o0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    o1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    w = np.cross(o0, o1, axis=-1)
    w = (w * w).sum(axis=1)
    cum = np.cumsum(w, axis=0)
    face_index = np.searchsorted(cum, rs.rand(num_samples) * cum[-1])
    origins = vertices[faces[:, 0]]
    vec = vertices[faces[:, 1:]]
    vec -= np.tile(origins, (1, 2)).reshape((-1, 2, 3))
    origins, vec = origins[face_index], vec[face_index]
    rl = rs.rand(len(vec), 2, 1)
    rl[rl.sum(axis=1).reshape(-1) > 1.0] -= 1.0
    rl = np.abs(rl)
    return (vec * rl).sum(axis=1) + origins, face_index, rl


def sample_mix(rs, vertices, vertex_normals, faces, face_normals, num_samples, vertex_samples, vgroup=None):
    """SamplerMix.sample: vertex_samples random vertices + (num_samples - vertex_samples) surface points. vgroup (N, J): per-vertex
    values sampled the same way (barycentric on the surface part). Returns (points, normals, vgroup samples or None)."""
    perm = rs.permutation(len(vertices))
    vs = min(vertex_samples, len(vertices))
    num_samples -= vs
    perm = perm[:vs]
    rs.permutation(num_samples)  # drawn and unused in UniRig, kept so the stream stays aligned
    pts, fi, rl = sample_surface(rs, num_samples, vertices, faces)
    out_v = np.concatenate([vertices[perm], pts], axis=0)
    out_n = np.concatenate([vertex_normals[perm], face_normals[fi]], axis=0)
    out_g = None
    if vgroup is not None:
        g = vgroup[faces[fi, 1:]]
        g -= vgroup[faces[fi, 0]][:, np.newaxis, :]
        out_g = np.concatenate([vgroup[perm], (g * rl).sum(axis=1) + vgroup[faces[fi, 0]]], axis=0)
    return out_v, out_n, out_g


def connect_tails(joints, tails, parents):
    """TailConfig(connect_tail_to_unique_son=True)."""
    children = defaultdict(list)
    for i, p in enumerate(parents):
        if p is not None:
            children[p].append(i)
    tails = tails.copy()
    for i in range(len(joints)):
        if len(children[i]) == 1:
            tails[i] = joints[children[i][0]]
    return tails


# ---------------------------------------------------------------- voxel skin
# pyrender camera poses (OpenGL convention: columns are right / up / back, the camera looks along -back, image row 0 is the top)
_E = 2.0
POSES = {
    "+z": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, _E], [0, 0, 0, 1]],
    "-z": [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -_E], [0, 0, 0, 1]],
    "+y": [[1, 0, 0, 0], [0, 0, -1, -_E], [0, 1, 0, 0], [0, 0, 0, 1]],
    "-y": [[1, 0, 0, 0], [0, 0, 1, _E], [0, -1, 0, 0], [0, 0, 0, 1]],
    "+x": [[0, 0, -1, -_E], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    "-x": [[0, 0, 1, _E], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
}


def depth_images(grid, cast, znear=0.05, zfar=4.0):
    """What `pyrender.OffscreenRenderer(grid, grid).render(DEPTH_ONLY)` + UniRig's `znear + zfar - znear * zfar / d` give for the six
    orthographic cameras (xmag = ymag = 1): the distance along the view direction of the nearest surface through every pixel
    centre (double sided), zfar where nothing is hit. cast(origins (n, 3), dirs (n, 3)) -> hit distance (inf = miss)."""
    c = (np.arange(grid) + 0.5) * 2 / grid - 1  # camera-space x of the pixel columns
    r = 1 - (np.arange(grid) + 0.5) * 2 / grid  # camera-space y of the pixel rows
    out = {}
    for name, pose in POSES.items():
        P = np.array(pose, np.float64)
        o = P[:3, 3] + c[None, :, None] * P[:3, 0] + r[:, None, None] * P[:3, 1]  # (row, col, 3), on the camera plane
        d = np.broadcast_to(-P[:3, 2], o.shape)
        t = cast(o.reshape(-1, 3), np.ascontiguousarray(d).reshape(-1, 3)).reshape(grid, grid)
        out[name] = np.where(np.isfinite(t) & (t >= znear) & (t <= zfar), t, zfar).astype(np.float32)
    return out


def voxelization(grid, cast, eye_dis=2.0):
    """vertex_group.voxelization(backend='pyrender') after the rendering: voxel centres seen 'inside' by at least two of the three
    axes (between the nearest surfaces of the two opposite cameras). Same index conventions as UniRig; (M, 3) float32 centres."""
    depths = depth_images(grid, cast)
    i, j, k = np.indices((grid, grid, grid), dtype=np.int32)
    gi = np.stack((i.ravel(), j.ravel(), k.ravel()), axis=1)
    del i, j, k
    gc = np.stack((gi[:, 0], gi[:, 1], grid - 1 - gi[:, 2]), axis=1).astype(np.float32) * 2 / grid - 1.0 + 1.0 / grid
    mz = -gc[:, 2] + depths["+z"][grid - 1 - gi[:, 1], gi[:, 0]] <= eye_dis
    mz &= gc[:, 2] + depths["-z"][grid - 1 - gi[:, 1], grid - 1 - gi[:, 0]] <= eye_dis
    mx = -gc[:, 0] + depths["+x"][grid - 1 - gi[:, 1], gi[:, 2]] <= eye_dis
    mx &= gc[:, 0] + depths["-x"][grid - 1 - gi[:, 1], gi[:, 2]] <= eye_dis
    my = -gc[:, 1] + depths["+y"][gi[:, 2], gi[:, 0]] <= eye_dis
    my &= gc[:, 1] + depths["-y"][grid - 1 - gi[:, 2], gi[:, 0]] <= eye_dis
    return gc[(mx & my) | (mx & mz) | (my & mz)]


def voxel_skin_weights(vertices, faces, joints, make_cast, grid=196, alpha=0.5, link_dis=0.00001, grid_query=7, vertex_query=1,
                       grid_weight=3.0):
    """VertexGroupVoxelSkin.get_vertex_group (mode 'square'): normalise by the vertex bounds, voxelize, geodesic voxel skin. (N, J).
    make_cast(vertices, faces) -> cast(origins, dirs) -> hit distances, see depth_images."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center, scale = (lo + hi) / 2, np.max(hi - lo) / 2
    nv, nj = (vertices - center) / scale, (joints - center) / scale
    skin = voxel_skin(grid, voxelization(grid, make_cast(nv, faces)), nj, nv, faces, alpha, link_dis, grid_query, vertex_query, grid_weight)
    return np.nan_to_num(skin, nan=0., posinf=0., neginf=0.)


def voxel_skin(grid, grid_coords, joints, vertices, faces, alpha=0.5, link_dis=0.00001, grid_query=27, vertex_query=27, grid_weight=3.0):
    J, M, N = joints.shape[0], grid_coords.shape[0], vertices.shape[0]
    grid_tree, vertex_tree, joint_tree = cKDTree(grid_coords), cKDTree(vertices), cKDTree(joints)
    combined_vertices = np.concatenate([vertices, grid_coords], axis=0)
    dist, idx = grid_tree.query(grid_coords, grid_query)  # link adjacent grids
    dist, idx = dist[:, 1:], idx[:, 1:]
    mask = (0 < dist) & (dist < 2 / grid * 1.001)
    source_grid2grid = np.repeat(np.arange(M), grid_query - 1)[mask.ravel()] + N
    to_grid2grid = idx[mask] + N
    weight_grid2grid = dist[mask] * grid_weight
    dist, idx = vertex_tree.query(vertices, 4)  # link very close vertices
    dist, idx = dist[:, 1:], idx[:, 1:]
    mask = (0 < dist) & (dist < link_dis * 1.001)
    source_close = np.repeat(np.arange(N), 3)[mask.ravel()]
    to_close = idx[mask]
    weight_close = dist[mask]
    dist, idx = vertex_tree.query(grid_coords, vertex_query)  # link grids to mesh vertices
    mask = (0 < dist) & (dist < 2 / grid * 1.001)
    source_grid2vertex = np.repeat(np.arange(M), vertex_query)[mask.ravel()] + N
    to_grid2vertex = idx[mask]
    weight_grid2vertex = dist[mask]
    _, joint_indices = cKDTree(combined_vertices).query(joints)  # joints -> nearest node
    src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]], axis=0)
    dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]], axis=0)
    graph = csr_matrix(
        (np.concatenate([weight_close, np.sqrt(((vertices[src] - vertices[dst]) ** 2).sum(axis=-1)), weight_grid2grid, weight_grid2vertex]),
         (np.concatenate([source_close, src, source_grid2grid, source_grid2vertex], axis=0),
          np.concatenate([to_close, dst, to_grid2grid, to_grid2vertex], axis=0))),
        shape=(N + M, N + M))
    dis = shortest_path(graph, method='D', directed=False, indices=joint_indices)[:, :N]  # (J, N)
    unreachable = np.isinf(dis).all(axis=0)
    k = min(J, 3)
    d3, i3 = joint_tree.query(vertices[unreachable], k)
    dis[i3, np.repeat(np.where(unreachable)[0], k).reshape(-1, k)] = d3
    finite = dis[np.isfinite(dis)]
    max_dis = np.max(finite)
    dis = np.maximum(np.nan_to_num(dis, nan=max_dis, posinf=max_dis, neginf=max_dis), 1e-6)
    skin = (1. / ((1 - alpha) * dis + alpha * dis ** 2)) ** 2
    return (skin / skin.sum(axis=0)).transpose()


def reskin(sampled_vertices, vertices, parents, faces, sampled_skin, nearest_samples=7, iter_steps=1, threshold=0.03, alpha=2.0):
    """system/skin.py reskin(sample_method='median', alpha=2.0, threshold=0.03): sampled-point weights -> every mesh vertex."""
    N, J = vertices.shape[0], sampled_skin.shape[1]
    _, nearest = cKDTree(sampled_vertices).query(vertices, k=nearest_samples, p=2)
    skin = np.median(sampled_skin[nearest], axis=1)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
    for _ in range(iter_steps):
        sum_skin = skin.copy()
        for i in reversed(range(J)):
            if parents[i] is not None:
                sum_skin[:, parents[i]] += sum_skin[:, i]
        mask = sum_skin[edges[:, 1]] < sum_skin[edges[:, 0]]  # only transfer from hotter to cooler
        neighbor_skin = np.zeros_like(sum_skin)
        neighbor_co = np.zeros((N, J), dtype=np.float32)
        dis = np.sqrt(((vertices[edges[:, 1]] - vertices[edges[:, 0]]) ** 2).sum(axis=1, keepdims=True))
        co = np.exp(-dis * alpha)
        neighbor_skin[edges[:, 1]] += sum_skin[edges[:, 0]] * co * mask
        neighbor_co[edges[:, 1]] += co * mask
        sum_skin = (sum_skin + neighbor_skin) / (1. + neighbor_co)
        for i in range(J):
            if parents[i] is not None:
                sum_skin[:, parents[i]] -= sum_skin[:, i]
        skin = sum_skin / sum_skin.sum(axis=-1, keepdims=True)  # (UniRig does this inside the loop above; same result for J > 1)
    mask = (skin >= threshold).any(axis=-1, keepdims=True)
    skin[(skin < threshold) & mask] = 0.
    return skin / skin.sum(axis=-1, keepdims=True)
