"""Build the Body Creator asset pack from MakeHuman 1.x data (CC0 assets).

python build_pack.py <makehuman/data> <out_dir>
Writes:
  body.npz     base mesh (hm08, decimetres, Y-up), helper geometry (teeth, tongue), fitted-eye proxy,
               and every morph target as sparse (index, float16 delta) arrays
  targets.json [{"key": "head-head-fat-decr", "vars": ["female", ...]}, ...] in npz order
  modifiers.json  MakeHuman's slider groups (modeling_modifiers.json)
A target's weight = product of its macro variable values x its group factor (MakeHuman lib/targets.py rules).
"""
import json
import os
import shutil
import sys

import numpy as np

MACRO = {"male", "female", "baby", "child", "young", "old", "caucasian", "asian", "african",
         "maxmuscle", "averagemuscle", "minmuscle", "minweight", "averageweight", "maxweight",
         "minheight", "averageheight", "maxheight", "mincup", "averagecup", "maxcup",
         "minfirmness", "averagefirmness", "maxfirmness", "uncommonproportions", "regularproportions",
         "idealproportions"}
SKIP_DIRS = {"expression", "genitals"}  # old expression units (we pose with the rig); not shipped


def read_obj(path):
    v, vt, faces, groups, g = [], [], [], [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            v.append([float(x) for x in p[1:4]])
        elif p[0] == "vt":
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == "g":
            g = p[1] if len(p) > 1 else None
        elif p[0] == "f":
            idx = [[int(t) if t else 0 for t in c.split("/")] + [0] for c in p[1:]]
            vi = [c[0] - 1 for c in idx]
            ti = [(c[1] - 1) if len(c) > 2 and c[1] else -1 for c in idx]
            if len(vi) == 3:
                vi.append(vi[2]); ti.append(ti[2])  # degenerate quad = triangle
            faces.append((vi[:4], ti[:4], g))
    return np.array(v, np.float32), np.array(vt, np.float32), faces


def group_faces(faces, name):
    sel = [(a, b) for a, b, g in faces if g == name]
    return np.array([a for a, _ in sel], np.int32).reshape(-1, 4), np.array([b for _, b in sel], np.int32).reshape(-1, 4)


def read_mhclo(path):
    """MakeHuman proxy: each proxy vertex = sum(w_i * base[ref_i]) + offset * scale."""
    refs, wts, offs, scales = [], [], [], {}
    in_verts = False
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] in ("x_scale", "y_scale", "z_scale"):
            scales[p[0][0]] = (int(p[1]), int(p[2]), float(p[3]))
        elif p[0] == "verts":
            in_verts = True
        elif in_verts and len(p) == 9:
            refs.append([int(x) for x in p[:3]]); wts.append([float(x) for x in p[3:6]]); offs.append([float(x) for x in p[6:9]])
        elif in_verts and len(p) == 1 and p[0].isdigit():
            refs.append([int(p[0])] * 3); wts.append([1, 0, 0]); offs.append([0, 0, 0])
        elif in_verts and p[0] in ("delete_verts", "material", "obj_file"):
            in_verts = False
    sc = np.array([[*scales[k][:2], scales[k][2]] for k in "xyz"], np.float64)
    return np.array(refs, np.int32), np.array(wts, np.float32), np.array(offs, np.float32), sc


def main(data, out):
    os.makedirs(out, exist_ok=True)
    v, vt, faces = read_obj(os.path.join(data, "3dobjs", "base.obj"))
    arrs = {"verts": v, "uvs": vt}
    for part in ("body", "helper-upper-teeth", "helper-lower-teeth", "helper-tongue"):
        f, t = group_faces(faces, part)
        assert len(f), part
        arrs[f"f_{part}"], arrs[f"uvf_{part}"] = f, t

    ev, evt, efaces = read_obj(os.path.join(data, "eyes", "high-poly", "high-poly.obj"))
    refs, wts, offs, sc = read_mhclo(os.path.join(data, "eyes", "high-poly", "high-poly.mhclo"))
    assert len(refs) == len(ev), (len(refs), len(ev))
    arrs.update(eye_uvs=evt, eye_faces=np.array([a for a, _, _ in efaces], np.int32),
                eye_uvf=np.array([b for _, b, _ in efaces], np.int32), eye_refs=refs, eye_w=wts, eye_off=offs, eye_scale=sc)

    root = os.path.join(data, "targets")
    index, idx_all, d_all, off = [], [], [], [0]
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fn in sorted(files):
            if not fn.endswith(".target"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            toks = [t for t in rel.replace(os.sep, "/").replace("/", "-").replace("_", "-").replace(".", "-").split("-")
                    if t and t != "target"]
            key = "-".join(t for t in toks if t not in MACRO)
            vars_ = [t for t in toks if t in MACRO]
            rows = [l.split() for l in open(os.path.join(dirpath, fn), encoding="utf-8") if l.strip() and not l.startswith("#")]
            if not rows:
                continue
            a = np.array(rows, np.float64)
            idx_all.append(a[:, 0].astype(np.int32)); d_all.append(a[:, 1:4].astype(np.float16))
            off.append(off[-1] + len(a))
            index.append({"key": key, "vars": vars_})
    arrs.update(t_idx=np.concatenate(idx_all), t_delta=np.concatenate(d_all), t_off=np.array(off, np.int64))
    np.savez_compressed(os.path.join(out, "body.npz"), **arrs)
    json.dump(index, open(os.path.join(out, "targets.json"), "w"))
    shutil.copy(os.path.join(data, "modifiers", "modeling_modifiers.json"), os.path.join(out, "modifiers.json"))
    shutil.copy(os.path.join(data, "eyes", "materials", "brown_eye.png"), os.path.join(out, "eye.png"))
    with open(os.path.join(out, "LICENSE-ASSETS.txt"), "w") as f:
        f.write("Base mesh, morph targets and eyes: MakeHuman (makehumancommunity.org), released CC0 1.0.\n")
    mb = os.path.getsize(os.path.join(out, "body.npz")) / 1e6
    print(f"pack: {len(v)} verts, {len(arrs['f_body'])} body quads, {len(index)} targets, "
          f"{off[-1]} deltas, body.npz {mb:.1f} MB")
    # self-check: every target index is inside the base mesh
    assert arrs["t_idx"].max() < len(v) and np.isfinite(arrs["t_delta"].astype(np.float32)).all()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
