"""Does the int8 SF3D encoder change the final model? Original SF3D vs ONNX fp32 vs ONNX int8, end to end.

The ONNX encoders replace only SF3D.get_scene_codes; everything after it (marching tetrahedra, UV unwrap,
texture bake, material estimator) is the original SF3D code, so differences come from the encoder alone.

Outputs in --out: compare.json (metrics), sheet.png (renders), <image>_<variant>.glb
"""
import argparse, gc, json, os, sys, time, types

import numpy as np
import PIL.Image  # noqa: F401  (sf3d.models.utils uses PIL.Image without importing it)
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

VARIANTS = [("fp32", "sf3d_encoder.onnx"), ("int8", "sf3d_encoder_int8.onnx")]


def surface_samples(mesh, n=30000, seed=0):
    """Points on the surface with their albedo (from the baked texture)."""
    import trimesh
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fi], pts)
    uv = (mesh.visual.uv[mesh.faces[fi]] * bary[:, :, None]).sum(1)
    h, w = tex.shape[:2]
    px = np.clip((uv[:, 0] * (w - 1)).round().astype(int), 0, w - 1)
    py = np.clip(((1 - uv[:, 1]) * (h - 1)).round().astype(int), 0, h - 1)
    return pts, tex[py, px]


def compare(ref, other):
    p0, c0 = surface_samples(ref)
    p1, c1 = surface_samples(other)
    diag = float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
    d01, i01 = cKDTree(p1).query(p0)
    d10, _ = cKDTree(p0).query(p1)
    return {
        "faces": int(len(other.faces)),
        "chamfer_pct_of_diag": float((d01.mean() + d10.mean()) / 2 / diag * 100),
        "hausdorff95_pct_of_diag": float(max(np.percentile(d01, 95), np.percentile(d10, 95)) / diag * 100),
        "albedo_mean_abs_diff_0_255": float(np.abs(c0 - c1[i01]).mean()),
        "albedo_p95_abs_diff_0_255": float(np.percentile(np.abs(c0 - c1[i01]).mean(1), 95)),
    }


def render(mesh, yaw_deg, size=300):
    """Tiny textured z-buffer rasterizer (orthographic), enough to eyeball differences."""
    v = mesh.vertices - (mesh.bounds[0] + mesh.bounds[1]) / 2
    a = np.radians(yaw_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    v = v @ rot.T
    s = size * 0.9 / max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), 1e-6)
    sx = v[:, 0] * s + size / 2
    sy = size / 2 - v[:, 1] * s
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32)
    th, tw = tex.shape[:2]
    uv = mesh.visual.uv
    img = np.full((size, size, 3), 40.0)
    zb = np.full((size, size), -np.inf)
    tri = v[mesh.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    shade = 0.35 + 0.65 * np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
    for f, (a_, b_, c_) in enumerate(mesh.faces):
        xs, ys = sx[[a_, b_, c_]], sy[[a_, b_, c_]]
        x0, x1 = max(int(xs.min()), 0), min(int(xs.max()) + 1, size)
        y0, y1 = max(int(ys.min()), 0), min(int(ys.max()) + 1, size)
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * v[a_, 2] + w1 * v[b_, 2] + w2 * v[c_, 2]
        sub = zb[y0:y1, x0:x1]
        win = inside & (z > sub)
        if not win.any():
            continue
        u = w0 * uv[a_, 0] + w1 * uv[b_, 0] + w2 * uv[c_, 0]
        vv = w0 * uv[a_, 1] + w1 * uv[b_, 1] + w2 * uv[c_, 1]
        col = tex[np.clip(((1 - vv) * (th - 1)).astype(int), 0, th - 1), np.clip((u * (tw - 1)).astype(int), 0, tw - 1)]
        sub[win] = z[win]
        img[y0:y1, x0:x1][win] = col[win] * shade[f]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf3d", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.sf3d)
    import onnxruntime as ort
    import rembg
    from sf3d.system import SF3D
    from sf3d.utils import remove_background, resize_foreground

    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(os.path.join(a.models, "model_info.json")))
    stub = types.SimpleNamespace(cfg=types.SimpleNamespace(cond_image_size=info["cond_image_size"],
                                 background_color=info["background_color"]), device="cpu")
    session = rembg.new_session()
    names, imgs, rgbs = [], [], []
    for p in a.images:
        img = resize_foreground(remove_background(Image.open(p).convert("RGBA"), session), 0.85)
        _, rgb = SF3D.prepare_image(stub, img)  # identical to SF3D.run_image's preprocessing
        names.append(os.path.splitext(os.path.basename(p))[0]); imgs.append(img); rgbs.append(rgb[None].numpy())
    del session

    # Stage 1: scene codes from each ONNX encoder, one session at a time (only one ~GB model in RAM).
    codes, enc_seconds = {}, {}
    for v, f in VARIANTS:
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        s = ort.InferenceSession(os.path.join(a.models, f), so, providers=["CPUExecutionProvider"])
        t = time.time()
        codes[v] = [s.run(None, {"rgb": r})[0] for r in rgbs]
        enc_seconds[v] = (time.time() - t) / len(rgbs)
        del s; gc.collect()
        print(f"{v} encoder {enc_seconds[v]:.1f}s/image", flush=True)

    # Stage 2: the original SF3D for everything; for ONNX variants only get_scene_codes is replaced.
    torch.set_grad_enabled(False)
    model = SF3D.from_pretrained("stabilityai/stable-fast-3d", config_name="config.yaml", weight_name="model.safetensors").eval()
    results, tiles = {}, []
    for i, (name, img) in enumerate(zip(names, imgs)):
        meshes = {}
        torch.manual_seed(0)
        meshes["original"], _ = model.run_image(img, bake_resolution=1024, remesh="none")
        for v, _ in VARIANTS:
            model.get_scene_codes = lambda batch, c=codes[v][i]: (torch.from_numpy(c), None)
            torch.manual_seed(0)
            meshes[v], _ = model.run_image(img, bake_resolution=1024, remesh="none")
            del model.get_scene_codes  # back to the class method
        results[name] = {v: compare(meshes["original"], meshes[v]) for v, _ in VARIANTS}
        results[name]["original_faces"] = int(len(meshes["original"].faces))
        print(name, json.dumps(results[name]), flush=True)
        row = []
        for k in ("original", "fp32", "int8"):
            meshes[k].export(os.path.join(a.out, f"{name}_{k}.glb"))
            row += [render(meshes[k], 0), render(meshes[k], 90)]
        tiles.append((name, row))

    w = 300
    sheet = Image.new("RGB", (w * 6, (w + 20) * len(tiles) + 24), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for c, label in enumerate(["original 0°", "original 90°", "onnx fp32 0°", "onnx fp32 90°", "onnx int8 0°", "onnx int8 90°"]):
        d.text((c * w + 6, 6), label, fill=(255, 255, 255))
    for r, (name, row) in enumerate(tiles):
        y = 24 + r * (w + 20)
        d.text((6, y), name, fill=(255, 200, 120))
        for c, im in enumerate(row):
            sheet.paste(im, (c * w, y + 16))
    sheet.save(os.path.join(a.out, "sheet.png"))
    summary = {"encoder_seconds_per_image_ci_cpu": enc_seconds, "per_image": results}
    for v, _ in VARIANTS:
        for k in ("chamfer_pct_of_diag", "albedo_mean_abs_diff_0_255"):
            summary.setdefault("mean", {}).setdefault(v, {})[k] = float(np.mean([results[n][v][k] for n in names]))
    json.dump(summary, open(os.path.join(a.out, "compare.json"), "w"), indent=2)
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
