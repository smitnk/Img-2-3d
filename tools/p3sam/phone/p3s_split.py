"""Cut a mesh into closed parts from face labels, optional pegs + holes (manifold3d), GLB / ZIP writers. numpy + scipy + manifold3d only.

  1. every label -> its faces, boundary loops closed with a centroid fan -> closed solid, validated / normalised by manifold3d
  2. at the biggest contacts between two parts a cylinder peg is unioned to the smaller part and a slightly bigger cylinder is
     subtracted from the larger one (axis = normal of the shared border, from small to large)
Without manifold3d the capped parts are written as they are (no pegs).
"""
import json
import os
import struct
import zipfile

import numpy as np
import scipy.sparse as sp

try:
    import manifold3d as m3
except Exception:  # pragma: no cover
    m3 = None


def clean(v, f, rel=1e-7):
    """Weld coincident vertices, drop degenerate and duplicate faces."""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    diag = np.linalg.norm(v.max(0) - v.min(0)) + 1e-30
    q = np.round(v / (diag * rel)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    v2 = v[first]
    f2 = inv[f]
    ok = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    f2 = f2[ok]
    _, u = np.unique(np.sort(f2, axis=1), axis=0, return_index=True)
    f2 = f2[np.sort(u)]
    used, inv2 = np.unique(f2, return_inverse=True)
    return v2[used], inv2.reshape(-1, 3)


def cap_part(v, f, face_mask):
    """Closed (vertices, faces, loops closed) for the faces in face_mask; open boundary components are closed by a fan around their centroid."""
    ff = f[face_mask]
    used, inv = np.unique(ff, return_inverse=True)
    vv = v[used]
    ff = inv.reshape(-1, 3)
    V = len(vv)
    e = np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]])
    key = e[:, 0] * V + e[:, 1]
    rev = e[:, 1] * V + e[:, 0]
    bnd = e[~np.isin(key, rev)]
    if len(bnd) == 0:
        return vv, ff, 0
    g = sp.coo_matrix((np.ones(len(bnd)), (bnd[:, 0], bnd[:, 1])), shape=(V, V))
    _, comp = sp.csgraph.connected_components(g, directed=False)
    vs, fs, nc, nv = [vv], [ff], 0, V
    for c in np.unique(comp[bnd[:, 0]]):
        be = bnd[comp[bnd[:, 0]] == c]
        cv = np.unique(be)
        vs.append(vv[cv].mean(0)[None])
        fs.append(np.stack([be[:, 1], be[:, 0], np.full(len(be), nv)], 1))
        nv += 1
        nc += 1
    return np.concatenate(vs), np.concatenate(fs), nc


def to_manifold(v, f):
    """(Manifold or None, status string)."""
    if m3 is None:
        return None, "no manifold3d"

    def make(v_, f_):
        m = m3.Manifold(m3.Mesh(vert_properties=np.asarray(v_, np.float32), tri_verts=np.asarray(f_, np.uint32)))
        return m, str(m.status())
    try:
        m, st = make(v, f)
        if "NoError" in st and not m.is_empty():
            if m.volume() < 0:
                m2, st2 = make(v, np.asarray(f)[:, ::-1])
                if "NoError" in st2 and m2.volume() > 0:
                    return m2, "ok_flipped"
            return m, "ok"
        v2, f2 = clean(v, f)
        m, st2 = make(v2, f2)
        if "NoError" in st2 and not m.is_empty():
            if m.volume() < 0:
                m, st2 = make(v2, f2[:, ::-1])
            return m, "ok_after_cleanup"
        return None, st
    except Exception as e:  # pragma: no cover
        return None, repr(e)[:80]


def rot_z_to(n):
    z = np.array([0, 0, 1.0])
    n = n / np.linalg.norm(n)
    v = np.cross(z, n)
    c = float(z @ n)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def cylinder(start, axis, length, radius, segments=40):
    cyl = m3.Manifold.cylinder(length, radius, radius, segments)
    mat = np.concatenate([rot_z_to(axis), np.asarray(start, float)[:, None]], 1)
    return cyl.transform(mat)


def contacts(G, fl):
    """{(a,b): midpoints of the mesh edges between label a and label b faces}."""
    la, lb = fl[G.pairs[:, 0]], fl[G.pairs[:, 1]]
    sel = la != lb
    out = {}
    for a, b, p in zip(np.minimum(la[sel], lb[sel]), np.maximum(la[sel], lb[sel]), G.pmid[sel]):
        out.setdefault((int(a), int(b)), []).append(p)
    return {k: np.array(x) for k, x in out.items()}


def split_parts(G, fl, pegs=True, peg_size=0.4, max_pegs=24, min_contact_edges=8, log=print):
    """-> dict(parts={label: dict(v, f, ok, status, ...)}, pegs=[...]) with the final (closed, pegged) geometry in parts[l]['v'], ['f']."""
    v, f = G.v, G.f
    labels = np.unique(fl)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    parts, peg_info = {}, []
    for l in labels:
        pv, pf, ncap = cap_part(v, f, fl == l)
        man, st = to_manifold(pv, pf)
        parts[int(l)] = dict(v=pv, f=pf, man=man, status=st, caps=int(ncap), centroid=pv.mean(0))
        log(f"PART {l}: faces={len(pf)} caps={ncap} manifold={st}" + (f" volume={man.volume():.5f}" if man is not None else ""))
    if pegs and m3 is not None:
        for (a, b), p in sorted(contacts(G, fl).items(), key=lambda kv: -len(kv[1]))[:max_pegs]:
            if len(p) < min_contact_edges or parts[a]["man"] is None or parts[b]["man"] is None:
                continue
            va, vb = parts[a]["man"].volume(), parts[b]["man"].volume()
            small, big = (a, b) if va <= vb else (b, a)
            c = p.mean(0)
            w, ev = np.linalg.eigh(np.cov((p - c).T))
            n = ev[:, 0]
            if n @ (parts[big]["centroid"] - parts[small]["centroid"]) < 0:
                n = -n
            R = float(np.sqrt(max(w[1] + w[2], 0)))
            r = min(peg_size * R, 0.05 * diag)
            if r < 0.004 * diag:
                peg_info.append(dict(a=small, b=big, skipped="contact too small"))
                continue
            tol = 0.06 * r
            peg = cylinder(c - n * 1.6 * r, n, 3.2 * r, r)
            hole = cylinder(c - n * 0.6 * r, n, 2.4 * r + tol, r + tol)
            ms, mb = parts[small]["man"] + peg, parts[big]["man"] - hole
            ok = (not ms.is_empty()) and (not mb.is_empty()) and "NoError" in str(ms.status()) and "NoError" in str(mb.status())
            peg_info.append(dict(male=small, hole=big, radius=r, ok=bool(ok)))
            log(f"PEG male={small} hole={big} r={r:.4f} ok={ok}")
            if ok:
                parts[small]["man"], parts[big]["man"] = ms, mb
    for l, p in parts.items():
        if p["man"] is not None:
            mm = p["man"].to_mesh()
            p["v"], p["f"] = np.asarray(mm.vert_properties, np.float64)[:, :3], np.asarray(mm.tri_verts, np.int64)
        p["ok"] = p["man"] is not None
        p["man"] = None
    return dict(parts=parts, pegs=peg_info)


# ---------------------------------------------------------------------------------------------------- GLB / ZIP
def glb_bytes(v, f, colors=None, name="mesh"):
    """Minimal binary glTF 2.0: one triangle mesh, optional COLOR_0 (n,3 float), flat grey material."""
    v = np.ascontiguousarray(v, np.float32)
    idx = np.ascontiguousarray(f, np.uint32).reshape(-1)
    nrm = np.zeros((len(v), 3), np.float32)
    t = v[f.reshape(-1, 3)]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    for k in range(3):
        np.add.at(nrm, f.reshape(-1, 3)[:, k], fn)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    chunks, views, acc = [], [], []

    def add(arr, target, comps, ctype, count, extra=None):
        raw = arr.tobytes()
        off = sum(len(c) for c in chunks)
        chunks.append(raw + b"\0" * (-len(raw) % 4))
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw), "target": target})
        a = {"bufferView": len(views) - 1, "componentType": ctype, "count": count, "type": comps}
        a.update(extra or {})
        acc.append(a)
        return len(acc) - 1
    attrs = {"POSITION": add(v, 34962, "VEC3", 5126, len(v), {"min": v.min(0).tolist(), "max": v.max(0).tolist()}),
             "NORMAL": add(nrm, 34962, "VEC3", 5126, len(v))}
    if colors is not None:
        attrs["COLOR_0"] = add(np.ascontiguousarray(colors, np.float32), 34962, "VEC3", 5126, len(v))
    ind = add(idx, 34963, "SCALAR", 5125, len(idx))
    binb = b"".join(chunks)
    j = {"asset": {"version": "2.0", "generator": "p3sam"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
         "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": ind, "material": 0}]}],
         "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 0.8}, "doubleSided": True}],
         "buffers": [{"byteLength": len(binb)}], "bufferViews": views, "accessors": acc}
    jb = json.dumps(j, separators=(",", ":")).encode()
    jb += b" " * (-len(jb) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binb)
    return struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(jb), 0x4E4F534A) + jb + struct.pack("<II", len(binb), 0x004E4942) + binb


def palette(n):
    import colorsys
    return np.array([colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.65, 0.95) for i in range(n)])


def colored_parts_glb(v, f, fl, name="parts"):
    """One colour per label; vertices are split per (vertex, label) so the colours stay sharp at the borders."""
    labels = np.unique(fl)
    pal = palette(len(labels))
    corner = np.repeat(fl, 3)
    key = f.reshape(-1).astype(np.int64) * (len(labels) + 1) + corner
    uk, inv = np.unique(key, return_inverse=True)
    vi = uk // (len(labels) + 1)
    lab = uk % (len(labels) + 1)
    return glb_bytes(v[vi], inv.reshape(-1, 3), pal[np.searchsorted(labels, lab)], name)


def write_zip(path, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
