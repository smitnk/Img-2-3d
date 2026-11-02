"""CPU probe of Hunyuan3D-2mini (Tencent, Hunyuan community licence) before any phone port.
python probe.py out_dir image1.png [image2.png ...]
Per image and config: wall time per stage (encoder, DiT steps, volume decode + marching cubes), peak RSS,
mesh stats, a GLB and a 4-view preview PNG.
"""
import json, os, resource, sys, time

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(os.cpu_count())
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def preview(mesh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v, f = mesh.vertices, mesh.faces
    if len(f) > 40000:
        f = f[np.random.default_rng(0).choice(len(f), 40000, replace=False)]
    fig = plt.figure(figsize=(16, 4))
    for i, az in enumerate((0, 90, 180, 270)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.plot_trisurf(v[:, 0], v[:, 2], v[:, 1], triangles=f, color="#ccc", linewidth=0, shade=True)
        ax.view_init(elev=10, azim=az - 90)
        ax.set_box_aspect(np.ptp(v[:, [0, 2, 1]], axis=0))
        ax.axis("off")
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def main(out, images):
    os.makedirs(out, exist_ok=True)
    rem = BackgroundRemover()
    configs = [("turbo", "hunyuan3d-dit-v2-mini-turbo", 5, 256)]
    report = {}
    pipes = {}
    for name, sub, steps, octree in configs:
        t0 = time.time()
        if sub not in pipes:
            pipes[sub] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mini", subfolder=sub, use_safetensors=True, device="cpu", dtype=torch.float32)
            pipes[sub].enable_flashvdm(mc_algo="mc")  # dense decoding took ~60 min per mesh on CPU
        pipe = pipes[sub]
        for m in (pipe.model, pipe.vae, pipe.conditioner):  # some weights load as fp16; CPU needs fp32
            m.float()
        pipe.dtype = torch.float32
        vd = type(pipe.vae.volume_decoder).__name__
        print("volume decoder:", vd, flush=True)
        assert vd.startswith("FlashVDM"), vd
        n_dit = sum(p.numel() for p in pipe.model.parameters()) / 1e6
        n_vae = sum(p.numel() for p in pipe.vae.parameters()) / 1e6
        n_enc = sum(p.numel() for p in pipe.conditioner.parameters()) / 1e6
        print(f"{name}: load {time.time() - t0:.0f}s, params dit {n_dit:.0f}M vae {n_vae:.0f}M encoder {n_enc:.0f}M, rss {rss_mb()} MB", flush=True)
        for img in images:
            key = f"{os.path.splitext(os.path.basename(img))[0]}_{name}"
            im = Image.open(img).convert("RGBA")
            if im.getextrema()[3][0] == 255:
                im = rem(im.convert("RGB"))
            t0 = time.time()
            mesh = pipe(image=im, num_inference_steps=steps, octree_resolution=octree, num_chunks=20000,
                        generator=torch.manual_seed(12345), output_type="trimesh")[0]
            dt = time.time() - t0
            mesh.export(f"{out}/{key}.glb")
            preview(mesh, f"{out}/{key}.png")
            report[key] = {"seconds": round(dt, 1), "faces": int(len(mesh.faces)), "rss_mb": rss_mb(),
                           "watertight": bool(mesh.is_watertight), "bodies": int(len(mesh.split(only_watertight=False)))}
            print(key, json.dumps(report[key]), flush=True)
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
