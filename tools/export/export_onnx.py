"""Export the original TripoSR (stabilityai/TripoSR weights, VAST-AI-Research/TripoSR code) to ONNX.

Produces, in --out:
  triposr_encoder.onnx (+ .data)  image [1,512,512,3] in 0..1 (gray-composited) -> scene_codes [1,3,40,64,64]
  triposr_field.onnx              scene_code [3,40,64,64], positions [N,3] in (-r,r) -> density_act [N,1], color [N,3]
  u2net.onnx                      rembg's u2net, copied unchanged
  model_info.json                 radius, threshold, image size, checksums
  reference/                      PyTorch outputs used to verify the Android/JVM pipeline

The graphs are traced straight from the original modules (TSR.forward minus the PIL
image processor, and TriplaneNeRFRenderer.query_triplane), so weights and math are unchanged.
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch
from einops import rearrange


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Encoder(torch.nn.Module):
    """TSR.forward from rgb_cond onward (tsr/system.py)."""

    def __init__(self, tsr):
        super().__init__()
        self.tsr = tsr

    def forward(self, image):  # [1,H,W,3]
        rgb_cond = image[:, None]
        t = self.tsr
        tokens_img = t.image_tokenizer(rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1))
        tokens_img = rearrange(tokens_img, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
        tokens = t.tokenizer(1)
        tokens = t.backbone(tokens, encoder_hidden_states=tokens_img)
        return t.post_processor(t.tokenizer.detokenize(tokens))


class Field(torch.nn.Module):
    def __init__(self, tsr):
        super().__init__()
        self.tsr = tsr

    def forward(self, scene_code, positions):
        out = self.tsr.renderer.query_triplane(self.tsr.decoder, positions, scene_code)
        return out["density_act"], out["color"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triposr", required=True, help="path to cloned TripoSR repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--u2net", required=True, help="path to rembg u2net.onnx")
    ap.add_argument("--sample", required=True, help="sample image for reference outputs")
    args = ap.parse_args()
    sys.path.insert(0, args.triposr)
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    import onnx, onnxruntime as ort, rembg
    from PIL import Image

    os.makedirs(os.path.join(args.out, "reference"), exist_ok=True)
    torch.set_grad_enabled(False)
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt").eval()
    size = model.cfg.cond_image_size
    radius = float(model.renderer.cfg.radius)

    # --- reference preprocessing exactly as run.py ---
    session = rembg.new_session("u2net")
    img = remove_background(Image.open(args.sample), session)
    img = resize_foreground(img, 0.85)
    a = np.array(img).astype(np.float32) / 255.0
    a = a[:, :, :3] * a[:, :, 3:4] + (1 - a[:, :, 3:4]) * 0.5
    img = Image.fromarray((a * 255.0).astype(np.uint8))
    img.save(os.path.join(args.out, "reference", "preprocessed.png"))
    rgb_cond = model.image_processor(img, size)  # [1,512,512,3]

    t0 = time.time()
    scene_ref = model([img], device="cpu")
    print(f"torch encoder: {time.time() - t0:.1f}s", flush=True)

    enc = Encoder(model).eval()
    assert torch.allclose(enc(rgb_cond), scene_ref, atol=1e-5), "encoder wrapper diverges from TSR.forward"

    # ViT's interpolate_pos_encoding passes tensor scale factors to bicubic interpolate, which the
    # tracer rejects. The input size is fixed (512), so freeze its eager result: identical values.
    emb = model.image_tokenizer.model.embeddings
    n_patches = (size // emb.patch_embeddings.patch_size[0]) ** 2
    frozen_pe = emb.interpolate_pos_encoding(torch.zeros(1, n_patches + 1, emb.position_embeddings.shape[-1]), size, size).detach()
    emb.interpolate_pos_encoding = lambda embeddings, height, width: frozen_pe
    assert torch.allclose(enc(rgb_cond), scene_ref, atol=1e-5), "frozen position encoding changed the output"

    enc_path = os.path.join(args.out, "triposr_encoder.onnx")
    torch.onnx.export(enc, (rgb_cond,), enc_path, input_names=["image"], output_names=["scene_codes"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    m = onnx.load(enc_path, load_external_data=True)
    for f in os.listdir(args.out):
        if f.startswith("triposr_encoder") and f != "triposr_encoder.onnx":
            os.remove(os.path.join(args.out, f))
    onnx.save_model(m, enc_path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="triposr_encoder.onnx.data", size_threshold=1024)
    del m

    field = Field(model).eval()
    pts = (torch.rand(4096, 3) * 2 - 1) * radius
    field_path = os.path.join(args.out, "triposr_field.onnx")
    torch.onnx.export(field, (scene_ref[0], pts), field_path, input_names=["scene_code", "positions"],
                      output_names=["density_act", "color"], dynamic_axes={"positions": {0: "n"},
                      "density_act": {0: "n"}, "color": {0: "n"}}, opset_version=17, dynamo=False)

    # --- verify ONNX == torch ---
    so = ort.SessionOptions()
    s_enc = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    scene_onnx = s_enc.run(None, {"image": rgb_cond.numpy()})[0]
    enc_err = float(np.abs(scene_onnx - scene_ref.numpy()).max())
    # fp32 reassociation across ~30 transformer layers gives ~1e-2 abs on large activations; gate relatively,
    # and the real equivalence gate is the mesh chamfer check in CI.
    enc_rel = enc_err / float(np.abs(scene_ref.numpy()).max())
    s_f = ort.InferenceSession(field_path, so, providers=["CPUExecutionProvider"])
    d_ref, c_ref = field(scene_ref[0], pts)
    d_o, c_o = s_f.run(None, {"scene_code": scene_ref[0].numpy(), "positions": pts.numpy()})
    rel = float(np.abs(d_o - d_ref.numpy()).max() / (np.abs(d_ref.numpy()).max() + 1e-6))
    print(f"encoder max abs err {enc_err:.2e} (rel {enc_rel:.2e}), field density rel err {rel:.2e}", flush=True)
    assert enc_rel < 1e-3 and rel < 1e-3, "ONNX export does not match PyTorch"

    # --- reference mesh from the original extract_mesh (torchmcubes) ---
    res = 128
    t0 = time.time()
    mesh = model.extract_mesh(scene_ref, True, resolution=res)[0]
    print(f"torch mesh @ {res}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, {time.time()-t0:.1f}s")
    mesh.export(os.path.join(args.out, "reference", f"mesh_{res}.obj"))
    np.save(os.path.join(args.out, "reference", "scene_codes.npy"), scene_ref.numpy())
    rgb_cond.numpy().astype(np.float32).tofile(os.path.join(args.out, "reference", "image_512.f32"))

    import shutil
    shutil.copy(args.u2net, os.path.join(args.out, "u2net.onnx"))
    files = ["triposr_encoder.onnx", "triposr_encoder.onnx.data", "triposr_field.onnx", "u2net.onnx"]
    info = {
        "version": 1, "image_size": size, "radius": radius, "threshold": 25.0,
        "foreground_ratio": 0.85, "scene_shape": list(scene_ref.shape[1:]),
        "files": {f: {"bytes": os.path.getsize(os.path.join(args.out, f)), "sha256": sha256(os.path.join(args.out, f))} for f in files},
        "verification": {"encoder_max_abs_err": enc_err, "encoder_rel_err": enc_rel, "field_density_rel_err": rel,
                         "reference_mesh_res": res, "reference_vertices": len(mesh.vertices),
                         "reference_faces": len(mesh.faces),
                         "reference_bbox": [mesh.bounds[0].tolist(), mesh.bounds[1].tolist()]},
    }
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info["verification"], indent=2))


if __name__ == "__main__":
    main()
