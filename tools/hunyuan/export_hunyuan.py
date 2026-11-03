"""Export Hunyuan3D-2mini Turbo (Tencent Hunyuan community licence) to ONNX for the phone, one part per process.

python export_hunyuan.py part {cond|dit|vae|geo} OUT     export one part (fp32), then int8 with ../skintokens/quant_llm.py
python export_hunyuan.py check OUT IMAGE                 torch pipeline vs the int8 ONNX parts on one image
Parts (the app runs them one at a time):
  cond.onnx  image [1,3,512,512] in -1..1 (recentred, white bg)  -> cond [1,T,1024]   (DINOv2 inside; uncond = zeros)
  dit.onnx   x [B,L,C], t [B] (sigma), cond [B,T,1024]           -> velocity [B,L,C]
  vae.onnx   latents [1,L,C] (already / scale_factor)             -> kv latents [1,L,W]
  geo.onnx   queries [1,M,3], kv latents [1,L,W]                  -> logits [1,M]
Sampler (Java): sigmas from hunyuan_info.json; x += (sigma_next - sigma) * v; t passed as sigma (= timestep / 1000).
Mesh (Java): hierarchical octree decode with geo.onnx + marching cubes at level 0, bounds ±1.01.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

SUB = "hunyuan3d-dit-v2-mini-turbo"
STEPS = 5
QUANT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "skintokens", "quant_llm.py")


def load():
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    p = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder=SUB, use_safetensors=True,
                                                         device="cpu", dtype=torch.float32)
    for m in (p.model, p.vae, p.conditioner):
        m.float().eval().requires_grad_(False)
    p.dtype = torch.float32
    return p


def rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


class Wrap(torch.nn.Module):  # weights must be registered submodules, or tracing bakes them as constants
    def __init__(self, m, fn):
        super().__init__()
        self.m, self.fn = m, fn

    def forward(self, *a):
        return self.fn(self.m, *a)


def guidance_free(p):
    return bool(getattr(p.model, "guidance_embed", False))


def part(name, out):
    os.makedirs(out, exist_ok=True)
    p = load()
    L, C = p.vae.latent_shape
    with torch.no_grad():
        cond = p.conditioner(image=torch.zeros(1, 3, 512, 512))["main"]
    T, D = cond.shape[1:]
    print(f"part {name}: latents {L}x{C}, cond {T}x{D}, guidance_embed {guidance_free(p)}, rss {rss_mb()} MB", flush=True)
    keep = {"cond": p.conditioner, "dit": p.model, "vae": p.vae, "geo": p.vae}[name]
    for m in (p.conditioner, p.model, p.vae):  # free the parts this process doesn't export
        if m is not keep:
            for prm in m.parameters():
                prm.data = torch.empty(0)
    ge = guidance_free(p)
    if name == "cond":
        mod, args = Wrap(p.conditioner, lambda m, im: m(image=im)["main"]), (torch.zeros(1, 3, 512, 512),)
        names, outs, dyn = ["image"], ["cond"], None
    elif name == "dit":
        g = torch.full((1,), 5.0)
        fn = (lambda m, x, t, c: m(x, t, {"main": c}, guidance=g.expand(x.shape[0]))) if ge else \
            (lambda m, x, t, c: m(x, t, {"main": c}, guidance=None))
        mod, args = Wrap(p.model, fn), (torch.randn(2, L, C), torch.full((2,), 0.5), cond.repeat(2, 1, 1))
        names, outs, dyn = ["x", "t", "cond"], ["v"], {"x": {0: "b"}, "t": {0: "b"}, "cond": {0: "b"}, "v": {0: "b"}}
    elif name == "vae":
        mod, args = Wrap(p.vae, lambda m, z: m(z)), (torch.randn(1, L, C),)
        names, outs, dyn = ["latents"], ["kv"], None
    else:
        with torch.no_grad():
            kv = p.vae(torch.randn(1, L, C))
        mod, args = Wrap(p.vae.geo_decoder, lambda m, q, k: m(queries=q, latents=k).squeeze(-1)), (torch.rand(1, 4096, 3) * 2 - 1, kv)
        names, outs, dyn = ["queries", "kv"], ["logits"], {"queries": {1: "m"}, "logits": {1: "m"}}
    t0 = time.time()
    with torch.no_grad():
        ref = mod(*args)
        torch.onnx.export(mod, args, f"{out}/{name}_fp32.onnx", opset_version=17, input_names=names, output_names=outs,
                          dynamic_axes=dyn)
    print(f"exported {name} in {time.time() - t0:.0f}s, rss {rss_mb()} MB", flush=True)
    import onnxruntime as ort
    s = ort.InferenceSession(f"{out}/{name}_fp32.onnx")
    feed = {n: a.numpy() for n, a in zip(names, args)}
    d32 = float(np.abs(s.run(None, feed)[0] - ref.numpy()).max())
    del s
    subprocess.run([sys.executable, QUANT, f"{out}/{name}_fp32.onnx", f"{out}/{name}.onnx"], check=True)
    s = ort.InferenceSession(f"{out}/{name}.onnx")
    r = ref.numpy()
    q = s.run(None, feed)[0]
    rel = float(np.abs(q - r).max() / (np.abs(r).max() + 1e-9))
    info = {"part": name, "fp32_maxdiff": d32, "int8_rel_maxdiff": rel, "L": L, "C": C, "T": int(T), "D": int(D),
            "guidance_embed": ge, "scale_factor": float(p.vae.scale_factor)}
    if name == "dit":
        p.scheduler.set_timesteps(STEPS, sigmas=np.linspace(0, 1, STEPS))
        info["sigmas"] = [float(x) for x in p.scheduler.sigmas]
        info["guidance_scale"] = 5.0
    json.dump(info, open(f"{out}/{name}_info.json", "w"))
    print("PART", json.dumps(info), flush=True)


def check(out, image):
    """Torch pipeline vs ONNX int8 parts (numpy sampler), same noise; both latents decoded by torch to meshes."""
    from PIL import Image
    import onnxruntime as ort
    p = load()
    info = {**json.load(open(f"{out}/dit_info.json")), **{k: v for k, v in json.load(open(f"{out}/cond_info.json")).items()
                                                        if k not in ("part", "fp32_maxdiff", "int8_rel_maxdiff")}}
    from hy3dgen.rembg import BackgroundRemover
    im = Image.open(image).convert("RGBA")
    if im.getextrema()[3][0] == 255:
        im = BackgroundRemover()(im.convert("RGB"))
    img = p.prepare_image(im)["image"]
    np.save(f"{out}/ref_image.npy", img.numpy())
    L, C = info["L"], info["C"]
    noise = torch.from_numpy(np.random.default_rng(7).standard_normal((1, L, C)).astype(np.float32))
    np.save(f"{out}/ref_noise.npy", noise.numpy())
    t0 = time.time()
    p.prepare_latents = lambda *a, **k: noise.clone()  # same starting noise as the ONNX run
    with torch.no_grad():
        mesh_t = p(image=im, num_inference_steps=STEPS, octree_resolution=256, num_chunks=20000, output_type="trimesh")[0]
    print(f"torch pipeline {time.time() - t0:.0f}s", flush=True)
    # sampler with ONNX parts (what the app does)
    sess = lambda n: ort.InferenceSession(f"{out}/{n}.onnx")
    cond = sess("cond").run(None, {"image": img.numpy()})[0]
    ge = info["guidance_embed"]
    c_in = cond if ge else np.concatenate([cond, np.zeros_like(cond)])
    dit = sess("dit")
    x = noise.numpy().copy()
    sig = info["sigmas"]
    t1 = time.time()
    for i in range(STEPS):
        xin = x if ge else np.concatenate([x, x])
        v = dit.run(None, {"x": xin, "t": np.full((xin.shape[0],), sig[i], np.float32), "cond": c_in})[0]
        if not ge:
            v = v[1:] + info["guidance_scale"] * (v[:1] - v[1:])
        x = x + (sig[i + 1] - sig[i]) * v
    print(f"onnx int8 sampling {time.time() - t1:.0f}s", flush=True)
    np.save(f"{out}/ref_latents_onnx.npy", x)
    with torch.no_grad():
        mesh_o = p._export(torch.from_numpy(x), "trimesh", 1.01, 0.0, 20000, 256, "mc")[0]
    mesh_o.export(f"{out}/check_onnx_int8.glb")
    mesh_t.export(f"{out}/check_torch.glb")
    print("CHECK", json.dumps({"faces_onnx": int(len(mesh_o.faces)), "faces_torch": int(len(mesh_t.faces)),
                               "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "part":
        part(sys.argv[2], sys.argv[3])
    else:
        check(sys.argv[2], sys.argv[3])
