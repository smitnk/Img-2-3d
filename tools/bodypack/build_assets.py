"""Convert MakeHuman system assets (CC0) into Body Creator pack files.

python build_assets.py <unzipped makehuman_system_assets> <pack dir>
Adds to the pack:
  assets.npz    per proxy "<kind>.<name>.<field>": faces/uvf (quads, a triangle repeats its last index), uvs, refs/w/off
                (MakeHuman proxy fitting), scale (3x3 rows: v1, v2, denominator; NaN row = no scale), delete (body
                vertices a garment hides)
  assets.json   catalog {kind: [{name, label, tex, color, alpha, thumb}]} for the app and body_creator.py
  tex_*.png/jpg textures (max 1024 px), thumb_*.png pictures for the app's cards
Kinds: hair, eyebrows, eyelashes, clothes, skins, eyes.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

KINDS = ("hair", "eyebrows", "eyelashes", "clothes")
TEX = 1024


def read_mhmat(path):
    m = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if len(p) >= 2 and not p[0].startswith(("#", "//")):
            m[p[0]] = p[1:]
    return m


def read_obj(path):
    vt, faces, nv = [], [], 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            nv += 1
        elif p[0] == "vt":
            vt.append([float(x) for x in p[1:3]])
        elif p[0] == "f":
            c = [x.split("/") for x in p[1:]]
            vi = [int(x[0]) - 1 for x in c]
            ti = [int(x[1]) - 1 if len(x) > 1 and x[1] else -1 for x in c]
            for k in range(1, len(vi) - 1, 2):  # fan an n-gon into quads/triangles
                q = [vi[0], vi[k], vi[k + 1], vi[k + 2] if k + 2 < len(vi) else vi[k + 1]]
                t = [ti[0], ti[k], ti[k + 1], ti[k + 2] if k + 2 < len(ti) else ti[k + 1]]
                faces.append((q, t))
    return nv, np.array(vt, np.float32).reshape(-1, 2), faces


def read_mhclo(path):
    refs, w, off, scale, delete = [], [], [], np.full((3, 3), np.nan), []
    mode, obj, mat, name, zdepth = None, None, None, None, 50
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        k = p[0]
        if k in ("x_scale", "y_scale", "z_scale"):
            scale["xyz".index(k[0])] = [float(p[1]), float(p[2]), float(p[3])]
        elif k == "verts":
            mode = "verts"
        elif k == "delete_verts":
            mode = "delete"
        elif k == "obj_file":
            obj = p[1]
        elif k == "material":
            mat = p[1]
        elif k == "name":
            name = " ".join(p[1:])
        elif k == "z_depth":
            zdepth = int(p[1])
        elif mode == "verts" and len(p) >= 9 and p[0].lstrip("-").isdigit():
            refs.append([int(x) for x in p[:3]]); w.append([float(x) for x in p[3:6]]); off.append([float(x) for x in p[6:9]])
        elif mode == "verts" and len(p) == 1 and p[0].isdigit():
            refs.append([int(p[0])] * 3); w.append([1.0, 0.0, 0.0]); off.append([0.0, 0.0, 0.0])
        elif mode == "delete" and p[0].isdigit():
            i = 0
            while i < len(p):  # "a - b" is a range
                if i + 2 < len(p) and p[i + 1] == "-":
                    delete += range(int(p[i]), int(p[i + 2]) + 1); i += 3
                else:
                    delete.append(int(p[i])); i += 1
        elif mode in ("verts", "delete") and not p[0].lstrip("-").replace(".", "").isdigit():
            mode = None
    return dict(refs=np.array(refs, np.int32), w=np.array(w, np.float32), off=np.array(off, np.float32),
                scale=scale.astype(np.float64), delete=np.array(sorted(set(delete)), np.int32)), obj, mat, name, zdepth


def texture(src, dst_base, alpha):
    im = Image.open(src)
    im.thumbnail((TEX, TEX), Image.LANCZOS)
    if alpha:
        im.convert("RGBA").save(dst_base + ".png", optimize=True)
        return os.path.basename(dst_base) + ".png"
    im.convert("RGB").save(dst_base + ".jpg", quality=88)
    return os.path.basename(dst_base) + ".jpg"


def thumb(src, dst):
    if os.path.isfile(src):
        im = Image.open(src).convert("RGBA")
        im.thumbnail((160, 160), Image.LANCZOS)
        im.save(dst, optimize=True)
        return os.path.basename(dst)
    return None


def material(folder, mhmat, kind, name, out):
    m = read_mhmat(os.path.join(folder, mhmat)) if mhmat and os.path.isfile(os.path.join(folder, mhmat)) else {}
    color = [float(x) for x in m.get("diffuseColor", ["0.8", "0.8", "0.8"])[:3]]
    alpha = m.get("transparent", ["False"])[0].lower() == "true" or kind in ("hair", "eyebrows", "eyelashes")
    tex = None
    if "diffuseTexture" in m:
        src = os.path.join(folder, m["diffuseTexture"][0])
        if not os.path.isfile(src):  # paths are sometimes relative to the data root
            src = os.path.join(folder, os.path.basename(m["diffuseTexture"][0]))
        if os.path.isfile(src):
            tex = texture(src, os.path.join(out, f"tex_{kind}_{name}"), alpha)
    return color, alpha, tex


TOON = {"brown": (92, 52, 24), "blue": (40, 110, 210), "green": (40, 150, 80), "amber": (210, 140, 30),
        "violet": (130, 70, 200), "black": (25, 22, 28)}


def toon_eyes(out):
    """Cartoon eye textures: a big flat iris, big pupil and a white highlight, drawn where the MakeHuman eye texture
    (eye.png, from build_pack) has its irises, so they map onto the same eyeball UVs."""
    from PIL import ImageDraw
    base = Image.open(os.path.join(out, "eye.png")).convert("RGB")
    w, h = base.size
    a = np.asarray(base, np.float32)
    # iris pixels: clearly darker than the sclera; two blobs (one per eye island), split by the image diagonal
    dark = a.mean(2) < 0.55 * np.median(a.mean(2))
    ys, xs = np.nonzero(dark)
    upper = xs > ys  # two eyes along the anti-diagonal: one above-right of the main diagonal, one below-left
    blobs = [(xs[m], ys[m]) for m in (upper, ~upper) if m.sum() > 50]
    entries = []
    for name, rgb in TOON.items():
        im = Image.new("RGB", (w, h), (246, 244, 240))
        d = ImageDraw.Draw(im)
        for bx, by in blobs:
            cx, cy = np.median(bx), np.median(by)
            r = 1.75 * np.sqrt(len(bx) / np.pi)  # 1.75x the real iris
            dark_rgb = tuple(int(c * 0.55) for c in rgb)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dark_rgb)
            d.ellipse([cx - 0.82 * r, cy - 0.82 * r, cx + 0.82 * r, cy + 0.82 * r], fill=rgb)
            d.ellipse([cx - 0.45 * r, cy - 0.45 * r, cx + 0.45 * r, cy + 0.45 * r], fill=(12, 10, 14))
            hx, hy, hr = cx - 0.38 * r, cy - 0.38 * r, 0.22 * r
            d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 255, 255))
        tex = f"tex_eyes_toon_{name}.png"
        im.resize((512, 512), Image.LANCZOS).save(os.path.join(out, tex), optimize=True)
        th = f"thumb_eyes_toon_{name}.png"
        im.crop((0, 0, w // 2, h // 2)).resize((160, 160)).save(os.path.join(out, th))
        entries.append(dict(name=f"toon_{name}", label=f"toon {name}", tex=tex, color=[1, 1, 1], alpha=False, thumb=th))
    print("toon eyes from", len(blobs), "iris blobs")
    assert len(blobs) == 2, "expected two irises in eye.png"
    return entries


def main(src, out):
    arrays, catalog = {}, {}
    for kind in KINDS:
        root = os.path.join(src, kind)
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            folder = os.path.join(root, name)
            clo = [f for f in os.listdir(folder) if f.endswith(".mhclo")]
            if not clo:
                continue
            proxy, obj, mat, label, zdepth = read_mhclo(os.path.join(folder, clo[0]))
            nv, uvs, faces = read_obj(os.path.join(folder, obj or name + ".obj"))
            if nv != len(proxy["refs"]):
                print(f"skip {kind}/{name}: {nv} obj verts vs {len(proxy['refs'])} refs")
                continue
            key = f"{kind}.{name}."
            arrays[key + "faces"] = np.array([q for q, _ in faces], np.int32)
            arrays[key + "uvf"] = np.array([t for _, t in faces], np.int32)
            arrays[key + "uvs"] = uvs
            for f in ("refs", "w", "off", "scale", "delete"):
                arrays[key + f] = proxy[f]
            color, alpha, tex = material(folder, mat, kind, name, out)
            catalog.setdefault(kind, []).append(dict(
                name=name, label=(label or name).replace("_", " "), tex=tex, color=color, alpha=alpha, z=zdepth,
                thumb=thumb(os.path.join(folder, name + ".thumb"), os.path.join(out, f"thumb_{kind}_{name}.png"))))
    for kind, sub in (("skins", "skins"), ("eyes", "eyes/materials")):
        root = os.path.join(src, sub)
        for f in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            path = os.path.join(root, f, f + ".mhmat") if kind == "skins" else os.path.join(root, f)
            if not path.endswith(".mhmat") or not os.path.isfile(path):
                continue
            name = os.path.basename(path)[:-6]
            folder = os.path.dirname(path)
            color, _, tex = material(folder, os.path.basename(path), kind, name, out)
            if tex:
                catalog.setdefault(kind, []).append(dict(name=name, label=name.replace("_", " "), tex=tex, color=color, alpha=False,
                                                         thumb=thumb(os.path.join(folder, name + ".thumb"), os.path.join(out, f"thumb_{kind}_{name}.png"))))
    catalog.setdefault("eyes", []).extend(toon_eyes(out))
    np.savez_compressed(os.path.join(out, "assets.npz"), **arrays)
    json.dump(catalog, open(os.path.join(out, "assets.json"), "w"), indent=1)
    print({k: len(v) for k, v in catalog.items()}, f"assets.npz {os.path.getsize(os.path.join(out, 'assets.npz')) / 1e6:.1f} MB")
    for k in KINDS:  # self-check: every proxy references base-mesh vertices only (hm08 has 19158)
        for a in catalog.get(k, []):
            assert arrays[f"{k}.{a['name']}.refs"].max() < 19158, a["name"]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
