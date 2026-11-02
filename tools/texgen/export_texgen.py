"""Export the on-device texture painter (LCM Dreamshaper v7, SD1.5 + latent consistency, 4 steps) to ONNX.

python export_texgen.py --out texgen --sample sample.png
Writes:
  unet_q.onnx       int8 (dynamic) UNet: sample[1,4,64,64], timestep[1] int64, encoder_hidden_states[1,77,768],
                    timestep_cond[1,256] -> noise prediction
  vae_encoder.onnx  image[1,3,512,512] in -1..1 -> latent mean[1,4,64,64] (unscaled)
  vae_decoder.onnx  latent[1,4,64,64] (unscaled) -> image[1,3,512,512] in -1..1
  prompts.bin       float32: per view a [77,768] text embedding, then the [256] guidance embedding (w = 8)
  texgen_info.json  views order, settings, verification numbers
  reference/        init.f32, noise.f32, out.f32 (int8 pipeline) for the Java pipeline's CI check
The scheduler (LCMScheduler, scaled-linear betas, 50 original steps) is re-implemented in numpy here and in Java
(core TextureDiffusion); this script checks numpy+ONNX against the same loop on the PyTorch modules.
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

VIEWS = ["front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"]
PROMPT = ("{v} view of a 3d game character, full body, consistent colors, clean detailed hand-painted texture, "
          "flat even lighting, no shadows, plain gray background")
GUIDANCE = 8.0
STEPS, STRENGTH = 4, 0.5
SCALE = 0.18215


def alphas_cumprod():
    b = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
    return np.cumprod(1 - b)


def lcm_timesteps(steps, strength, original=50):
    k = 1000 // original
    origin = np.arange(1, int(original * strength) + 1) * k - 1
    skip = len(origin) // steps
    return origin[::-1][::skip][:steps]


def lcm_loop(unet, x0_latent, noise_list, emb, tcond, steps=STEPS, strength=STRENGTH):
    """x0_latent scaled latent; noise_list: [initial noise] + one noise per intermediate step."""
    ac = alphas_cumprod()
    ts = lcm_timesteps(steps, strength)
    a = ac[ts[0]]
    x = np.sqrt(a) * x0_latent + np.sqrt(1 - a) * noise_list[0]
    for i, t in enumerate(ts):
        eps = unet(x.astype(np.float32), np.array([t], np.int64), emb, tcond)
        a = ac[t]
        scaled = t * 10.0
        c_skip = 0.25 / (scaled ** 2 + 0.25)
        c_out = scaled / np.sqrt(scaled ** 2 + 0.25)
        x0 = (x - np.sqrt(1 - a) * eps) / np.sqrt(a)
        den = c_out * x0 + c_skip * x
        if i + 1 < len(ts):
            ap = ac[ts[i + 1]]
            x = np.sqrt(ap) * den + np.sqrt(1 - ap) * noise_list[i + 1]
        else:
            x = den
    return x


def psnr(a, b):
    return float(10 * np.log10(4.0 / max(np.mean((a - b) ** 2), 1e-12)))  # range -1..1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sample", required=True)
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "reference"), exist_ok=True)
    tmp = os.path.join(a.out, "_fp32")
    os.makedirs(tmp, exist_ok=True)
    from diffusers import LatentConsistencyModelPipeline
    pipe = LatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float32)
    unet, vae = pipe.unet.eval().requires_grad_(False), pipe.vae.eval().requires_grad_(False)

    # text + guidance embeddings for every view prompt
    embs = []
    with torch.no_grad():
        for v in VIEWS:
            ids = pipe.tokenizer(PROMPT.format(v=v.replace("_", " ")), padding="max_length", max_length=77,
                                 truncation=True, return_tensors="pt").input_ids
            embs.append(pipe.text_encoder(ids)[0][0].numpy().astype(np.float32))
        tcond = pipe.get_guidance_scale_embedding(torch.tensor([GUIDANCE - 1.0]), embedding_dim=256)[0].numpy()
    np.concatenate([e.ravel() for e in embs] + [tcond.astype(np.float32).ravel()]).tofile(os.path.join(a.out, "prompts.bin"))

    class Wrap(torch.nn.Module):  # weights must be registered submodules, or tracing bakes them as constants
        def __init__(self, m, fn):
            super().__init__()
            self.m, self.fn = m, fn

        def forward(self, *args):
            return self.fn(self.m, *args)

    U = lambda: Wrap(unet, lambda m, s, t, e, c: m(s, t, e, timestep_cond=c).sample)
    E = lambda: Wrap(vae, lambda m, x: m.encode(x).latent_dist.mean)
    D = lambda: Wrap(vae, lambda m, z: m.decode(z).sample)

    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(U(), (torch.randn(1, 4, 64, 64), torch.tensor([499]), torch.randn(1, 77, 768), torch.randn(1, 256)),
                          os.path.join(tmp, "unet.onnx"), opset_version=17, do_constant_folding=True,
                          input_names=["sample", "timestep", "encoder_hidden_states", "timestep_cond"], output_names=["out"])
        torch.onnx.export(E(), (torch.randn(1, 3, 512, 512),), os.path.join(a.out, "vae_encoder.onnx"), opset_version=17,
                          input_names=["image"], output_names=["latent"])
        torch.onnx.export(D(), (torch.randn(1, 4, 64, 64),), os.path.join(a.out, "vae_decoder.onnx"), opset_version=17,
                          input_names=["latent"], output_names=["image"])
    m = onnx.load(os.path.join(tmp, "unet.onnx"))  # >2 GB: gather the exporter's many weight files into one
    onnx.save_model(m, os.path.join(tmp, "unet_fp32.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="unet_fp32.onnx.data")
    del m
    print(f"exported in {time.time() - t0:.0f}s")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    variants = {"unet_q.onnx": ["MatMul", "Gemm", "Conv"], "unet_qmm.onnx": ["MatMul", "Gemm"]}
    for name, ops in variants.items():
        quantize_dynamic(os.path.join(tmp, "unet_fp32.onnx"), os.path.join(tmp, name), weight_type=QuantType.QUInt8,
                         op_types_to_quantize=ops, use_external_data_format=name != "unet_q.onnx")
        print(name, "MB", round(sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp) if f.startswith(name)) / 1e6))

    # verification: same loop with torch UNet, fp32 ONNX, int8 ONNX on one init image
    img = Image.open(a.sample).convert("RGBA").resize((512, 512))
    bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
    rgb = np.asarray(Image.alpha_composite(bg, img).convert("RGB"), np.float32) / 127.5 - 1
    init = rgb.transpose(2, 0, 1)[None].copy()
    rng = np.random.default_rng(7)
    noise = [rng.standard_normal((1, 4, 64, 64)).astype(np.float32) for _ in range(STEPS)]
    so = ort.SessionOptions()
    enc = ort.InferenceSession(os.path.join(a.out, "vae_encoder.onnx"), so)
    dec = ort.InferenceSession(os.path.join(a.out, "vae_decoder.onnx"), so)
    lat = enc.run(None, {"image": init})[0] * SCALE
    emb, tc = embs[1][None], tcond[None].astype(np.float32)

    def torch_unet(x, t, e, c):
        with torch.no_grad():
            return unet(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(e), timestep_cond=torch.from_numpy(c)).sample.numpy()

    def onnx_unet(path):
        s = ort.InferenceSession(path, so)
        return lambda x, t, e, c: s.run(None, {"sample": x, "timestep": t, "encoder_hidden_states": e, "timestep_cond": c})[0]

    outs, info = {}, {"views": VIEWS, "prompt": PROMPT, "guidance": GUIDANCE, "steps": STEPS, "strength": STRENGTH,
                      "scale": SCALE, "check": {}}
    for key, fn in [("torch", torch_unet), ("fp32", onnx_unet(os.path.join(tmp, "unet_fp32.onnx"))),
                    ("q", onnx_unet(os.path.join(tmp, "unet_q.onnx"))), ("qmm", onnx_unet(os.path.join(tmp, "unet_qmm.onnx")))]:
        t0 = time.time()
        z = lcm_loop(fn, lat, noise, emb, tc)
        outs[key] = np.clip(dec.run(None, {"latent": (z / SCALE).astype(np.float32)})[0], -1, 1)
        info["check"][key] = {"seconds": round(time.time() - t0, 1)}
        Image.fromarray(((outs[key][0].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)).save(os.path.join(a.out, "reference", f"out_{key}.png"))
    for key in ("fp32", "q", "qmm"):
        info["check"][key]["psnr_vs_torch"] = round(psnr(outs[key], outs["torch"]), 2)
    print(json.dumps(info["check"]))
    assert info["check"]["fp32"]["psnr_vs_torch"] > 35, "ONNX export differs from PyTorch"
    best = "q" if info["check"]["q"]["psnr_vs_torch"] > 22 else "qmm"
    info["unet"] = "unet_q.onnx" if best == "q" else "unet_qmm.onnx"
    for f in os.listdir(tmp):
        if f.startswith(info["unet"]):
            os.replace(os.path.join(tmp, f), os.path.join(a.out, f))
    init.astype(np.float32).tofile(os.path.join(a.out, "reference", "init.f32"))
    np.concatenate([n.ravel() for n in noise]).tofile(os.path.join(a.out, "reference", "noise.f32"))
    outs[best].astype(np.float32).tofile(os.path.join(a.out, "reference", "out.f32"))
    json.dump(info, open(os.path.join(a.out, "texgen_info.json"), "w"), indent=1)
    print("chosen", info["unet"])


if __name__ == "__main__":
    main()
