"""Cut a mesh into closed parts from face labels (manifold3d), add one peg + matching hole between adjacent parts, export GLBs + preview.

  1. every label -> its own sub-mesh (faces of that label), boundary loops closed with a centroid fan -> closed solid
  2. manifold3d validates / normalises each solid (Manifold(...).status()), gives volume
  3. for the biggest contacts between two parts: a cylinder peg is unioned to the smaller part and a slightly bigger cylinder
     is subtracted from the larger part (peg axis = principal normal of the shared boundary, pointing from small to large)
"""
import json
import os
import zipfile

import numpy as np
import scipy.sparse as sp
import trimesh

import manifold3d as m3


def cap_part(mesh, face_mask):
    """Closed (vertices, faces) for the faces in face_mask; open boundary components are closed by a fan around their centroid."""
    f = mesh.faces[face_mask]
    used, inv = np.unique(f, return_inverse=True)
    v = mesh.vertices[used]
    f = inv.reshape(-1, 3)
    V = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = e[:, 0] * V + e[:, 1]
    rev = e[:, 1] * V + e[:, 0]
    bnd = e[~np.isin(key, rev)]  # directed edges without an opposite twin
    if len(bnd) == 0:
        return v, f, 0
    g = sp.coo_matrix((np.ones(len(bnd)), (bnd[:, 0], bnd[:, 1])), shape=(V, V))
    _, comp = sp.csgraph.connected_components(g, directed=False)
    vs, fs, nc = [v], [f], 0
    nv = V
    for c in np.unique(comp[bnd[:, 0]]):
        be = bnd[comp[bnd[:, 0]] == c]
        cv = np.unique(be)
        centre = v[cv].mean(0)
        vs.append(centre[None])
        fs.append(np.stack([be[:, 1], be[:, 0], np.full(len(be), nv)], 1))  # opposite orientation of the boundary edge
        nv += 1
        nc += 1
    return np.concatenate(vs), np.concatenate(fs), nc


def to_manifold(v, f):
    """(Manifold or None, status string). Tries the capped mesh as is, then after trimesh clean-up."""
    def make(v_, f_):
        m = m3.Manifold(m3.Mesh(vert_properties=np.asarray(v_, np.float32), tri_verts=np.asarray(f_, np.uint32)))
        return m, str(m.status())
    m, st = make(v, f)
    if "NoError" in st and not m.is_empty():
        return m, "ok"
    t = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_winding(t)
    trimesh.repair.fix_inversion(t)
    m, st2 = make(t.vertices, t.faces)
    if "NoError" in st2 and not m.is_empty():
        return m, "ok_after_cleanup"
    return None, st


def rot_z_to(n):
    """3x3 rotation taking +z to the unit vector n."""
    z = np.array([0, 0, 1.0])
    n = n / np.linalg.norm(n)
    v = np.cross(z, n)
    c = float(z @ n)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def cylinder(start, axis, length, radius, segments=48):
    """Cylinder from `start` along unit `axis`."""
    cyl = m3.Manifold.cylinder(length, radius, radius, segments)
    R = rot_z_to(axis)
    mat = np.concatenate([R, np.asarray(start, float)[:, None]], 1)
    return cyl.transform(mat)


def contacts(mesh, fl):
    """{(a,b): midpoints of the mesh edges between label a and label b faces}."""
    adj = mesh.face_adjacency
    la, lb = fl[adj[:, 0]], fl[adj[:, 1]]
    sel = la != lb
    ed = mesh.vertices[mesh.face_adjacency_edges[sel]].mean(1)
    out = {}
    for (a, b), p in zip(zip(np.minimum(la[sel], lb[sel]), np.maximum(la[sel], lb[sel])), ed):
        out.setdefault((int(a), int(b)), []).append(p)
    return {k: np.array(v) for k, v in out.items()}


def split_parts(mesh, fl, outdir, pegs=True, max_pegs=24, min_contact_edges=8, log=print):
    os.makedirs(outdir, exist_ok=True)
    labels = np.unique(fl)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    parts, info = {}, {}
    for l in labels:
        v, f, ncap = cap_part(mesh, fl == l)
        man, st = to_manifold(v, f)
        info[int(l)] = dict(faces_in=int(len(f)), caps=int(ncap), status=st, manifold=man is not None)
        parts[int(l)] = dict(v=v, f=f, man=man, centroid=v.mean(0))
        if man is not None:
            info[int(l)]["volume"] = float(man.volume())
            info[int(l)]["genus"] = int(man.genus())
        log(f"PART {l}: faces={len(f)} caps={ncap} manifold={st} " + (f"volume={man.volume():.5f}" if man is not None else ""))
    peg_info = []
    if pegs:
        cons = contacts(mesh, fl)
        for (a, b), p in sorted(cons.items(), key=lambda kv: -len(kv[1]))[:max_pegs]:
            if len(p) < min_contact_edges or parts[a]["man"] is None or parts[b]["man"] is None:
                continue
            va, vb = parts[a]["man"].volume(), parts[b]["man"].volume()
            small, big = (a, b) if va <= vb else (b, a)
            c = p.mean(0)
            w, ev = np.linalg.eigh(np.cov((p - c).T))
            n = ev[:, 0]
            if n @ (parts[big]["centroid"] - parts[small]["centroid"]) < 0:
                n = -n
            R = float(np.sqrt(max(w[1] + w[2], 0)))  # radius of a circular contact loop = sqrt(var1 + var2)
            r = min(0.4 * R, 0.05 * diag)
            if r < 0.004 * diag:
                peg_info.append(dict(a=small, b=big, skipped="contact too small", contact_radius=R))
                continue
            tol = 0.06 * r
            peg = cylinder(c - n * 1.6 * r, n, 3.2 * r, r)
            hole = cylinder(c - n * 0.6 * r, n, 2.4 * r + tol, r + tol)
            ms, mb = parts[small]["man"] + peg, parts[big]["man"] - hole
            ok = (not ms.is_empty()) and (not mb.is_empty()) and "NoError" in str(ms.status()) and "NoError" in str(mb.status())
            peg_info.append(dict(male=small, hole=big, radius=r, contact_radius=R, contact_edges=int(len(p)), ok=bool(ok),
                                 volume_added=float(ms.volume() - parts[small]["man"].volume()), volume_removed=float(parts[big]["man"].volume() - mb.volume())))
            log(f"PEG male={small} hole={big} r={r:.4f} ok={ok}")
            if ok:
                parts[small]["man"], parts[big]["man"] = ms, mb
    files = []
    for l, p in parts.items():
        if p["man"] is not None:
            mm = p["man"].to_mesh()
            vv, ff = np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts)
        else:
            vv, ff = p["v"], p["f"]
        t = trimesh.Trimesh(vv, ff, process=False)
        path = f"{outdir}/part_{l:02d}.glb"
        t.export(path)
        files.append(path)
        info[l].update(out_faces=int(len(ff)), out_vertices=int(len(vv)), watertight=bool(t.is_watertight), bytes=os.path.getsize(path))
    return dict(parts=info, pegs=peg_info, files=files)


def zip_files(files, extra, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + extra:
            z.write(f, os.path.basename(f))


def _view(v, yaw, pitch):
    cy, sy, cp, sp_ = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp_], [0, sp_, cp]])
    return v @ Ry.T @ Rx.T  # x right, y up, z towards viewer


def render(ax, verts, faces, colors, yaw, pitch=0.15, title=""):
    from matplotlib.collections import PolyCollection
    v = _view(verts - verts.mean(0), yaw, pitch)
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.45 + 0.55 * np.abs(n[:, 2])
    order = np.argsort(tri[:, :, 2].mean(1))
    col = np.clip(colors[order] * shade[order, None], 0, 1)
    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=col, edgecolors="none", linewidths=0))
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def palette(n):
    import colorsys
    return np.array([colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.65, 0.95) for i in range(n)])


def preview(mesh, fl, part_files, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = np.unique(fl)
    pal = palette(len(labels))
    fc = pal[np.searchsorted(labels, fl)]
    fig, axs = plt.subplots(1, 4, figsize=(16, 5), dpi=110)
    for ax, (yaw, name) in zip(axs[:3], [(0.0, "front"), (np.pi / 2, "side"), (np.pi / 4, "3/4")]):
        render(ax, np.asarray(mesh.vertices), np.asarray(mesh.faces), fc, yaw, title=f"{name} - {len(labels)} parts {title}")
    # exploded: each exported part pushed away from the centre
    allv, allf, allc = [], [], []
    base = 0
    cen = mesh.vertices.mean(0)
    for i, f in enumerate(part_files):
        t = trimesh.load(f, force="mesh")
        d = t.vertices.mean(0) - cen
        allv.append(t.vertices + 0.35 * d)
        allf.append(t.faces + base)
        allc.append(np.tile(pal[i % len(pal)], (len(t.faces), 1)))
        base += len(t.vertices)
    render(axs[3], np.concatenate(allv), np.concatenate(allf), np.concatenate(allc), np.pi / 4, title="exploded (pegs on)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
