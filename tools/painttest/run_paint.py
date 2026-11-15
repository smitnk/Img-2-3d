#!/usr/bin/env python
"""CPU test of Tencent Hunyuan3D-2 Paint (turbo) texture model: peak RSS / time per variant.

python run_paint.py --variant NAME --out OUT [--mode batch|seq] [--dtype fp32|bf16] [--quant none|int8w|int8dyn]
                    [--free_dual 0|1] [--steps 10] [--refs img ...] [--mesh glb] [--stage full|probe]
Everything non-neural (render of normal/position maps, back-projection bake, inpaint) goes through Hunyuan's own
MeshRender with a numba CPU rasterizer in place of the CUDA custom_rasterizer.
"""
import argparse, gc, json, math, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cr_shim"))
sys.path.insert(0, HERE)
HY = os.environ.get("HY_SRC", "/tmp/hy")
sys.path.insert(0, HY)

import numpy as np
import torch
import trimesh
from PIL import Image
from einops import rearrange

import paintlib as PL

torch.set_num_threads(os.cpu_count())

ELEVS = [0, 0, 0, 0, 90, -90]
AZIMS = [0, 90, 180, 270, 0, 180]
VWEIGHTS = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
VNAMES = ["front", "left", "back", "right", "top", "bottom"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", "/tmp/hfm/hunyuan3d-paint-v2-0-turbo"))
    p.add_argument("--mesh", default="/tmp/char.glb")
    p.add_argument("--refs", nargs="+", default=[os.path.join(HY, "assets/example_images/1316.png")])
    p.add_argument("--mode", default="batch", choices=["batch", "seq"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--quant", default="none", choices=["none", "int8w"])
    p.add_argument("--free_dual", type=int, default=0)   # 1: ref stream computed once, dual UNet freed (loaded first)
    p.add_argument("--vae_batch", type=int, default=0)   # 1: encode/decode all views in one VAE batch
    p.add_argument("--vae_tile", type=int, default=0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--sweeps0", type=int, default=2)     # seq mode: sweeps over the views at the first step
    p.add_argument("--faces", type=int, default=40000)
    p.add_argument("--render_size", type=int, default=1024)
    p.add_argument("--tex_size", type=int, default=1024)
    p.add_argument("--rot_y180", type=int, default=1)
    p.add_argument("--vae_decode_probe", type=int, default=0)  # after results are written: batch-decode all views
    return p.parse_args()


# ------------------------------------------------------------------ mesh
def prep_mesh(a):
    sc = trimesh.load(a.mesh)
    m = sc.dump(concatenate=True) if isinstance(sc, trimesh.Scene) else sc
    m = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces), process=False)
    nv0, nf0 = len(m.vertices), len(m.faces)
    if a.rot_y180:
        m.vertices = m.vertices * np.array([-1, 1, -1])
    if nf0 > a.faces:
        try:
            m = m.simplify_quadric_decimation(face_count=a.faces)
        except TypeError:
            m = m.simplify_quadric_decimation(a.faces)
    from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
    m = mesh_uv_wrap(m)
    print(f"mesh {nv0}v/{nf0}f -> {len(m.vertices)}v/{len(m.faces)}f", flush=True)
    return m, (nv0, nf0)


def camera_info():
    return [(((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el] +
            {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el] for az, el in zip(AZIMS, ELEVS)]


def ref_image(path):
    im = Image.open(path)
    if im.mode == "P":
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        al = np.array(im)[:, :, 3]
        ys, xs = np.nonzero(al > 0)
        im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        w, h = im.size
        bw, bh = int(w * 0.2), int(h * 0.2)
        sq = max(w + 2 * bw, h + 2 * bh)
        new = Image.new("RGBA", (sq, sq), (255, 255, 255, 0))
        new.paste(im, ((sq - w - 2 * bw) // 2 + bw, (sq - h - 2 * bh) // 2 + bh))
        bg = Image.new("RGB", new.size, (255, 255, 255))
        bg.paste(new, mask=new.getchannel("A"))
        im = bg
    return im.convert("RGB").resize((512, 512), Image.BICUBIC)


def to_t(imgs, dtype):  # list of PIL -> [1,N,3,H,W] in 0..1
    arr = []
    for im in imgs:
        x = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        arr.append(torch.from_numpy(x).permute(2, 0, 1))
    return torch.stack(arr)[None].to(dtype)


# ------------------------------------------------------------------ model
def build_unet(model_dir):
    from diffusers import UNet2DConditionModel
    from hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2p5DConditionModel, Basic2p5DTransformerBlock
    Basic2p5DTransformerBlock.forward = PL.patched_forward
    cfg = json.load(open(os.path.join(model_dir, "unet", "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with torch.device("meta"):
        unet = UNet2p5DConditionModel(UNet2DConditionModel(**cfg))
        ted = 4 * cfg["block_out_channels"][0]
        if unet.unet.class_embedding.embedding_dim != ted:  # only differs for tiny test configs
            unet.unet.class_embedding = torch.nn.Embedding(unet.unet.class_embedding.num_embeddings, ted)
    return unet


def vae_encode(vae, x, gen, each):
    """x [1,N,3,H,W] in 0..1 -> latents [1,N,4,h,w]"""
    x = (x[0] - 0.5) * 2.0
    outs = []
    chunks = [x] if not each else [x[i:i + 1] for i in range(len(x))]
    for c in chunks:
        post = vae.encode(c.to(vae.dtype)).latent_dist
        outs.append(post.sample(generator=gen) * vae.config.scaling_factor)
    return torch.cat(outs)[None]


def vae_decode(vae, lat, each):
    lat = lat / vae.config.scaling_factor
    chunks = [lat] if not each else [lat[i:i + 1] for i in range(len(lat))]
    imgs = []
    for c in chunks:
        img = vae.decode(c.to(vae.dtype), return_dict=False)[0].float()
        imgs.append(((img / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return [Image.fromarray(x) for b in imgs for x in b]


def textured_render(render, tex, elev, azim, res=512, bg=0.92):
    _, pos_clip = render.get_pos_from_mvp(elev, azim, None, None)
    rast, _ = render.raster_rasterize(pos_clip, render.pos_idx, resolution=[res, res])
    uv, _ = render.raster_interpolate(render.vtx_uv[None], rast, render.uv_idx)
    mask = (rast[0, ..., -1] > 0)
    H, W = tex.shape[:2]
    r = (uv[0][..., 1] * (H - 1)).round().long().clamp(0, H - 1)
    c = (uv[0][..., 0] * (W - 1)).round().long().clamp(0, W - 1)
    col = torch.from_numpy(tex).float()[r, c]
    col = torch.where(mask[..., None], col, torch.full_like(col, bg))
    return Image.fromarray((col.numpy() * 255).clip(0, 255).astype(np.uint8))


def consistency(textures, cos_maps, wts):
    """Mean weighted std (0..255) of the colour that different views put on the same texel (lower = more consistent)."""
    T = torch.stack(textures)          # V,H,W,3 in 0..1
    C = torch.stack(cos_maps)          # V,H,W,1
    ok = (C > 0.3).float()
    w = ok * (C ** 4) * torch.tensor(wts).view(-1, 1, 1, 1)
    cnt = ok.sum(0)[..., 0]
    sel = cnt >= 2
    ws = w.sum(0).clamp(min=1e-8)
    mean = (w * T).sum(0) / ws
    var = (w * (T - mean) ** 2).sum(0) / ws
    std = var.mean(-1).sqrt()[sel]
    return float(std.mean() * 255) if sel.any() else None, int(sel.sum())


# ------------------------------------------------------------------ main
def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    vdir = os.path.join(a.out, "views")
    os.makedirs(vdir, exist_ok=True)
    mon = PL.Mon()
    res = {"variant": a.variant, "config": vars(a), "ok": False, "notes": []}
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    T0 = time.time()

    def dump():
        res["peak_rss_gb"] = round(mon.peak("hwm_gb"), 3)
        res["peak_footprint_gb"] = round(mon.peak("peak_footprint_gb"), 3)
        res["peak_sampled_rss_gb"] = round(mon.peak("sampled_peak_rss_gb"), 3)
        res["time_s"] = round(time.time() - T0, 1)
        res["stages"] = mon.stages
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    try:
        # ---- mesh + conditioning renders
        mon.begin("prep_mesh_render")
        mesh, (nv0, nf0) = prep_mesh(a)
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        render = MeshRender(default_resolution=a.render_size, texture_size=a.tex_size, device="cpu")
        # CPU-only bug in upstream: get_mesh() edits vtx_pos/vtx_uv in place (.cpu().numpy() shares memory on CPU)
        def _get_mesh(self):
            vp = self.vtx_pos.clone().numpy(); pi = self.pos_idx.clone().numpy()
            uv = self.vtx_uv.clone().numpy(); ui = self.uv_idx.clone().numpy()
            vp[:, [1, 2]] = vp[:, [2, 1]]; vp[:, [0, 1]] = -vp[:, [0, 1]]; uv[:, 1] = 1.0 - uv[:, 1]
            return vp, pi, uv, ui
        MeshRender.get_mesh = _get_mesh
        render.load_mesh(mesh)
        normals = [render.render_normal(e, z, use_abs_coor=True, return_type="pl") for e, z in zip(ELEVS, AZIMS)]
        poss = [render.render_position(e, z, return_type="pl") for e, z in zip(ELEVS, AZIMS)]
        normals = [n.resize((512, 512)) for n in normals]
        poss = [p.resize((512, 512)) for p in poss]
        cam = camera_info()
        refs = [ref_image(p) for p in a.refs]
        for i, (n, p) in enumerate(zip(normals, poss)):
            n.save(os.path.join(a.out, f"cond_normal_{i}_{VNAMES[i]}.png"))
            p.save(os.path.join(a.out, f"cond_position_{i}_{VNAMES[i]}.png"))
        for i, r in enumerate(refs):
            r.save(os.path.join(a.out, f"ref_{i}.png"))
        res["mesh"] = {"verts_in": nv0, "faces_in": nf0, "faces_used": int(len(mesh.faces))}

        gen = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        nviews = 6

        # ---- VAE + conditioning latents
        mon.begin("load_vae_encode")
        from diffusers import AutoencoderKL, LCMScheduler
        vae = AutoencoderKL.from_pretrained(os.path.join(a.model_dir, "vae"), torch_dtype=dtype).eval()
        if a.vae_tile:
            vae.enable_tiling()
            vae.tile_sample_min_size, vae.tile_latent_min_size, vae.tile_overlap_factor = 256, 32, 0.25
        with torch.no_grad():
            ref_lat = vae_encode(vae, to_t(refs, dtype), gen, not a.vae_batch)
            nrm_lat = vae_encode(vae, to_t(normals, dtype), gen, not a.vae_batch)
            pos_lat = vae_encode(vae, to_t(poss, dtype), gen, not a.vae_batch)
        print("latents", ref_lat.shape, nrm_lat.shape, pos_lat.shape, flush=True)

        # ---- UNet
        unet = build_unet(a.model_dir)
        st_path = os.path.join(a.model_dir, "unet", "diffusion_pytorch_model.safetensors")
        n_gen = sum(p.numel() for n, p in unet.named_parameters() if n.startswith("unet."))
        n_dual = sum(p.numel() for n, p in unet.named_parameters() if n.startswith("unet_dual."))
        res["params"] = {"gen_unet": n_gen, "dual_ref_unet": n_dual, "total": n_gen + n_dual,
                         "vae": sum(p.numel() for p in vae.parameters())}
        print("params", res["params"], flush=True)
        qmode = a.quant
        ctx_ref = torch.tensor([[0]])
        cam_t = torch.tensor([cam])
        ref_dict = None
        enc_h = None

        def finish_load(prefix):
            return PL.load_weights(unet, st_path, dtype, qmode, prefix=prefix)

        if a.free_dual:
            mon.begin("load_dual_ref_features")
            finish_load("unet_dual.")
            res["resident_bytes_dual"] = PL.tensor_bytes(unet, "unet_dual.")
            PL.load_weights(unet, st_path, dtype, "none", prefix="unet.learned_text_clip_ref")
            with torch.no_grad():
                ref_dict = {}
                N_ref = ref_lat.shape[1]
                eh_ref = unet.unet.learned_text_clip_ref.to(dtype).unsqueeze(1).repeat(1, N_ref, 1, 1)
                eh_ref = rearrange(eh_ref, "b n l c -> (b n) l c")
                unet.unet_dual(rearrange(ref_lat, "b n c h w -> (b n) c h w"), 0, encoder_hidden_states=eh_ref,
                               class_labels=ctx_ref.reshape(-1),
                               return_dict=False,
                               cross_attention_kwargs={"mode": "w", "num_in_batch": N_ref, "condition_embed_dict": ref_dict})
            res["ref_feature_bytes"] = sum(t.numel() * t.element_size() for t in ref_dict.values())
            del unet._modules["unet_dual"]
            gc.collect()
            mon.begin("load_gen_unet")
            finish_load("unet.")
        else:
            mon.begin("load_unet_all")
            finish_load(None)
        miss = [n for n, p in list(unet.named_parameters()) + list(unet.named_buffers()) if p is not None and p.device.type == "meta"]
        assert not miss, f"meta tensors left: {miss[:5]}"
        unet.eval()
        res["unexpected_ckpt_keys"] = {"count": len(PL.UNEXPECTED), "examples": PL.UNEXPECTED[:12]}
        print("unexpected keys", res["unexpected_ckpt_keys"], flush=True)
        res["resident_weight_gb"] = round(PL.tensor_bytes(unet) / PL.GB, 3)
        print("resident weights GB", res["resident_weight_gb"], flush=True)

        # ---- schedule
        sched_cfg = json.load(open(os.path.join(a.model_dir, "scheduler", "scheduler_config.json")))
        sched = LCMScheduler.from_config(sched_cfg, timestep_spacing="trailing")
        bsz = max(1, 30 // a.steps)
        ddim_ts = (np.arange(1, 31) * (1000 // 30)).round().astype(np.int64) - 1
        idx = list(range(29, 0, -bsz))[: a.steps]
        ts = [int(ddim_ts[i]) for i in idx]
        sched.set_timesteps(timesteps=ts, device="cpu")
        res["timesteps"] = ts
        prompt = unet.unet.learned_text_clip_gen.to(dtype)  # [1,77,1024]

        # ---- denoise
        mon.begin("denoise")
        lat = torch.randn(nviews, 4, 64, 64, generator=gen)
        step_times, view_times = [], []

        def unet_call(x, ni, nl, pl, cm, rd_kwargs):
            return unet(x, t, encoder_hidden_states=prompt, normal_imgs=nl, position_imgs=pl, camera_info_gen=cm,
                        camera_info_ref=ctx_ref, return_dict=False, **rd_kwargs)[0]

        rd_kwargs = {"condition_embed_dict": ref_dict} if ref_dict is not None else {"ref_latents": ref_lat}
        PL.SEQ = PL.SeqCtx(nviews)
        with torch.no_grad():
            for si, t in enumerate(sched.timesteps):
                t0 = time.time()
                xin = lat[None].to(dtype)  # 1,N,4,h,w
                if a.mode == "batch":
                    PL.SEQ.active = False
                    eps = unet_call(xin, None, nrm_lat, pos_lat, cam_t, rd_kwargs).float()
                else:
                    PL.SEQ.active = True
                    outs = [None] * nviews
                    sweeps = a.sweeps0 if si == 0 else 1
                    for sw in range(sweeps):
                        for v in range(nviews):
                            PL.SEQ.cur = v
                            tv = time.time()
                            o = unet_call(xin[:, v:v + 1], None, nrm_lat[:, v:v + 1], pos_lat[:, v:v + 1],
                                          cam_t[:, v:v + 1], rd_kwargs)
                            outs[v] = o.float()
                            view_times.append(round(time.time() - tv, 2))
                    eps = torch.cat(outs, 0)
                lat = sched.step(eps, t, lat, generator=gen, return_dict=False)[0]
                step_times.append(round(time.time() - t0, 1))
                print(f"[step {si + 1}/{len(sched.timesteps)}] t={int(t)} {step_times[-1]}s "
                      f"rss={PL._status()['VmRSS']:.2f}GB", flush=True)
                dump()
        res["step_times_s"] = step_times
        res["view_times_s"] = view_times
        res["steps"] = len(step_times)
        res["seq_ctx_gb"] = round(PL.SEQ.nbytes() / PL.GB, 3)
        torch.save(lat, os.path.join(a.out, "latents.pt"))
        del unet
        gc.collect()

        # ---- decode
        mon.begin("vae_decode")
        with torch.no_grad():
            views = vae_decode(vae, lat.to(dtype), not a.vae_batch)
        for i, v in enumerate(views):
            v.save(os.path.join(vdir, f"view_{i}_{VNAMES[i]}.png"))
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in views])
        res["view_pixel_std"] = round(float(arr.std()), 2)
        res["nan"] = bool(np.isnan(arr).any())

        # ---- bake
        mon.begin("bake_inpaint")
        views_r = [v.resize((a.render_size, a.render_size)) for v in views]
        texs, coss, bcos = [], [], []
        for v, e, z, w in zip(views_r, ELEVS, AZIMS, VWEIGHTS):
            t_, c_, _ = render.back_project(v, e, z)
            texs.append(t_)
            coss.append(c_)
            bcos.append(w * (c_ ** 4))
        tex, trust = render.fast_bake_texture(texs, bcos)
        cstd, ncov = consistency(texs, coss, VWEIGHTS)
        res["consistency_std_0_255"], res["consistency_texels"] = (round(cstd, 2) if cstd else None), ncov
        mask = (trust.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        tex = render.uv_inpaint(tex, mask)
        tex = torch.tensor(tex / 255).float()
        render.set_texture(tex)
        Image.fromarray((tex.numpy() * 255).astype(np.uint8)).save(os.path.join(a.out, "texture.png"))
        tx = render.get_texture()
        rs = [textured_render(render, tx, 0, z) for z in (0, 90, 180, 270)]
        sheet = Image.new("RGB", (4 * 512, 512))
        for i, r in enumerate(rs):
            sheet.paste(r, (i * 512, 0))
        sheet.save(os.path.join(a.out, "render_4views.png"))
        frames = [textured_render(render, tx, 10, z, res=320) for z in range(0, 360, 30)]
        frames[0].save(os.path.join(a.out, "turntable.gif"), save_all=True, append_images=frames[1:], duration=120, loop=0)
        res["ok"] = (not res["nan"]) and res["view_pixel_std"] > 5
        mon.end()
        dump()

        if a.vae_decode_probe:
            mon.begin("probe_vae_decode_batch6")
            dump()
            with torch.no_grad():
                _ = vae_decode(vae, lat.to(dtype), False)
            mon.end()
            dump()
    except Exception as e:
        res["error"] = "".join(traceback.format_exception(e))[-3000:]
        print(res["error"], flush=True)
        mon.end()
        dump()
        raise
    finally:
        mon.end()
        mon.stop = True


if __name__ == "__main__":
    main()
