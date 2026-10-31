"""Export the original Stable Fast 3D (stabilityai/stable-fast-3d, Stability-AI/stable-fast-3d code) to ONNX.

Outputs in --out:
  sf3d_encoder.onnx(+.data)  rgb [1,S,S,3] (composited on background), mask [1,S,S,1] -> scene_codes [1,3,C,H,W]
                             (camera conditioning = the fixed default camera of SF3D.run_image, baked in)
  sf3d_field.onnx            scene_code [3,C,H,W], positions [N,3] -> every decoder head (activations applied)
  sf3d_estimator.onnx        rgb*mask [1,1,S,S,3] -> image-estimator material outputs (deterministic eval)
  sf3d_tets.bin              160^3 tet grid: float32 vertices then int32 indices (shapes in model_info.json)
  model_info.json            config values the phone pipeline needs + checksums + verification numbers
  reference/                 the original SF3D run_image GLB + inputs, for end-to-end comparison
"""
import argparse, hashlib, json, os, sys

import numpy as np
import PIL.Image  # noqa: F401  (sf3d.models.utils uses PIL.Image without importing it)
import torch


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage(name):
    """GitHub annotation: survives even if the runner is OOM-killed before the log upload."""
    import resource
    print(f"::notice::{name} (peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.sf3d)
    import sf3d.models.network as net
    net.trunc_exp = torch.exp  # autograd.Function is not traceable; forward is identical
    from sf3d.system import SF3D
    from sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w, remove_background, resize_foreground
    import onnx, onnxruntime as ort, rembg
    from PIL import Image

    torch.set_grad_enabled(False)
    os.makedirs(os.path.join(a.out, "reference"), exist_ok=True)
    model = SF3D.from_pretrained("stabilityai/stable-fast-3d", config_name="config.yaml", weight_name="model.safetensors").eval()
    cfg = model.cfg
    S = cfg.cond_image_size
    print("cond_image_size", S, "isosurface", cfg.isosurface_resolution, cfg.isosurface_threshold, "radius", cfg.radius, flush=True)

    # --- reference: the original pipeline exactly as run.py (bake 1024, no remesh) ---
    img = remove_background(Image.open(a.sample).convert("RGBA"), rembg.new_session())
    img = resize_foreground(img, 0.85)
    img.save(os.path.join(a.out, "reference", "input_rgba.png"))
    torch.manual_seed(0)
    mesh, glob_dict = model.run_image(img, bake_resolution=1024, remesh="none")
    mesh.export(os.path.join(a.out, "reference", "sf3d_reference.glb"), include_normals=True)
    ref_counts = (len(mesh.vertices), len(mesh.faces))
    stage(f"reference done {ref_counts}")
    print("reference mesh", *ref_counts, flush=True)

    mask, rgb = model.prepare_image(img)
    rgb, mask = rgb[None], mask[None]  # [1,S,S,3], [1,S,S,1]
    c2w = default_cond_c2w(cfg.default_distance)
    intr, intr_n = create_intrinsic_from_fov_deg(cfg.default_fovy_deg, S, S)

    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("c2w", c2w.view(1, 1, 4, 4))
            self.register_buffer("intr", intr.view(1, 1, 3, 3))
            self.register_buffer("intr_n", intr_n.view(1, 1, 3, 3))

        def forward(self, rgb_cond, mask_cond):
            batch = {"rgb_cond": rgb_cond[:, None], "mask_cond": mask_cond[:, None], "c2w_cond": self.c2w,
                     "intrinsic_cond": self.intr, "intrinsic_normed_cond": self.intr_n}
            return self.m.get_scene_codes(batch)[0]

    heads = [h.name for h in model.decoder.cfg.heads]

    class Field(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, scene_code, positions):
            out = self.m.decoder(self.m.query_triplane(positions, scene_code)[0])
            return tuple(out[h] for h in heads)

    enc, field = Encoder(model).eval(), Field(model).eval()
    # DINOv2's interpolate_pos_encoding feeds tensor scale factors to bicubic interpolate, which the tracer
    # rejects. Input size is fixed, so record the eager result once and replay it: identical values.
    emb = model.image_tokenizer.model.embeddings
    orig_pe, cache = emb.interpolate_pos_encoding, {}
    emb.interpolate_pos_encoding = lambda e, h, w: cache.setdefault("pe", orig_pe(e, h, w).detach())
    scene_ref = enc(rgb, mask)
    emb.interpolate_pos_encoding = lambda e, h, w: cache["pe"]
    assert torch.allclose(enc(rgb, mask), scene_ref, atol=1e-5), "frozen position encoding changed the output"
    pts = (torch.rand(4096, 3) * 2 - 1) * cfg.radius

    enc_path = os.path.join(a.out, "sf3d_encoder.onnx")
    torch.onnx.export(enc, (rgb, mask), enc_path, input_names=["rgb", "mask"], output_names=["scene_codes"],
                      opset_version=17, dynamo=False)
    stage("encoder exported")
    field_path = os.path.join(a.out, "sf3d_field.onnx")
    torch.onnx.export(field, (scene_ref[0], pts), field_path, input_names=["scene_code", "positions"], output_names=heads,
                      dynamic_axes={"positions": {0: "n"}, **{h: {0: "n"} for h in heads}}, opset_version=17, dynamo=False)

    est_names, est_path = [], None
    if model.image_estimator is not None:
        probe = model.image_estimator((rgb * mask)[:, None])
        est_names = sorted(k for k, v in probe.items() if torch.is_tensor(v))

        class Estimator(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                o = self.m(x)
                return tuple(o[k] for k in est_names)

        est_path = os.path.join(a.out, "sf3d_estimator.onnx")
        torch.onnx.export(Estimator(model.image_estimator).eval(), ((rgb * mask)[:, None],), est_path,
                          input_names=["image"], output_names=est_names, opset_version=17, dynamo=False)
        print("estimator outputs", est_names, "eval", model.image_estimator.cfg.distribution_eval, flush=True)

    # Keep only small reference values, then free PyTorch before loading the ~1.7 GB ONNX encoder:
    # holding both exceeds the 16 GB CI runner (job was OOM-killed, exit 143).
    f_ref = [t.numpy() for t in field(scene_ref[0], pts)]
    scene_np, rgb_np, mask_np, pts_np = scene_ref.numpy(), rgb.numpy(), mask.numpy(), pts.numpy()
    tets = model.isosurface_helper
    tv = tets.grid_vertices.numpy().astype(np.float32)
    ti = tets.indices.numpy().astype(np.int32)
    points_range, bbox = list(tets.points_range), model.bbox.tolist()
    radius, iso_res, iso_thr, bg = cfg.radius, cfg.isosurface_resolution, cfg.isosurface_threshold, list(cfg.background_color)
    del enc, field, model, tets, mesh
    import gc; gc.collect()
    stage("pytorch freed")

    # >2 GB models are exported as one file per tensor; merge into a single data file (loaded once, then freed).
    m = onnx.load(enc_path, load_external_data=True)
    stray = [f for f in os.listdir(a.out) if f not in ("sf3d_field.onnx", "sf3d_estimator.onnx", "reference", "sf3d_encoder.onnx")]
    onnx.save_model(m, enc_path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="sf3d_encoder.onnx.data", size_threshold=1024)
    params = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    del m; gc.collect()
    for f in stray:
        if f != "sf3d_encoder.onnx.data":
            os.remove(os.path.join(a.out, f))
    enc_bytes = os.path.getsize(os.path.join(a.out, "sf3d_encoder.onnx.data"))
    stage(f"encoder merged: {params/1e6:.0f}M params, {enc_bytes/1e9:.2f} GB")

    # Weight-only int8 candidate, so the app can pick it if fp32 does not fit in phone RAM. Accuracy measured below.
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = os.path.join(a.out, "sf3d_encoder_int8.onnx")
    quantize_dynamic(enc_path, q_path, weight_type=QuantType.QInt8, use_external_data_format=False)
    stage(f"int8 encoder {os.path.getsize(q_path)/1e9:.2f} GB")

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    s_enc = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    # get_scene_codes never reads mask_cond, so ONNX prunes that input; feed only what the graph declares.
    feed = {k: v for k, v in {"rgb": rgb_np, "mask": mask_np}.items() if k in {i.name for i in s_enc.get_inputs()}}
    print("encoder inputs", list(feed), flush=True)
    scene_onnx = s_enc.run(None, feed)[0]
    enc_rel = float(np.abs(scene_onnx - scene_np).max() / np.abs(scene_np).max())
    enc_inputs = [i.name for i in s_enc.get_inputs()]
    del s_enc, scene_onnx; gc.collect()
    s_q = ort.InferenceSession(q_path, so, providers=["CPUExecutionProvider"])
    scene_q = s_q.run(None, feed)[0]
    int8_rel = float(np.abs(scene_q - scene_np).max() / np.abs(scene_np).max())
    int8_mean = float(np.abs(scene_q - scene_np).mean() / np.abs(scene_np).mean())
    del s_q, scene_q; gc.collect()
    stage(f"int8 encoder rel err max {int8_rel:.3e} mean {int8_mean:.3e}")
    s_f = ort.InferenceSession(field_path, so, providers=["CPUExecutionProvider"])
    f_onnx = s_f.run(None, {"scene_code": scene_np[0], "positions": pts_np})
    field_rel = max(float(np.abs(o - r).max() / (np.abs(r).max() + 1e-6)) for o, r in zip(f_onnx, f_ref))
    stage(f"verified enc {enc_rel:.2e} field {field_rel:.2e}")
    print(f"encoder rel err {enc_rel:.2e}, field rel err {field_rel:.2e}", flush=True)
    assert enc_rel < 1e-3 and field_rel < 1e-3, "ONNX export does not match PyTorch"

    with open(os.path.join(a.out, "sf3d_tets.bin"), "wb") as f:
        f.write(tv.tobytes()); f.write(ti.tobytes())
    np.save(os.path.join(a.out, "reference", "scene_codes.npy"), scene_np)
    rgb_np.tofile(os.path.join(a.out, "reference", "rgb.f32"))
    mask_np.tofile(os.path.join(a.out, "reference", "mask.f32"))

    files = ["sf3d_encoder.onnx", "sf3d_encoder.onnx.data", "sf3d_encoder_int8.onnx", "sf3d_field.onnx", "sf3d_tets.bin"] + (["sf3d_estimator.onnx"] if est_path else [])
    info = {
        "version": 1, "encoder_inputs": enc_inputs, "cond_image_size": S, "radius": radius, "isosurface_resolution": iso_res,
        "isosurface_threshold": iso_thr, "background_color": bg, "points_range": points_range, "bbox": bbox, "decoder_heads": heads, "estimator_outputs": est_names,
        "tets": {"vertices": list(tv.shape), "indices": list(ti.shape)}, "scene_shape": list(scene_np.shape[1:]),
        "files": {f: {"bytes": os.path.getsize(os.path.join(a.out, f)), "sha256": sha256(os.path.join(a.out, f))} for f in files},
        "encoder_params": params, "verification": {"encoder_rel_err": enc_rel, "int8_rel_err_max": int8_rel, "int8_rel_err_mean": int8_mean, "field_rel_err": field_rel,
                         "reference_vertices": ref_counts[0], "reference_faces": ref_counts[1]},
    }
    json.dump(info, open(os.path.join(a.out, "model_info.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
