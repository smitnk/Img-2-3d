#!/usr/bin/env python
"""Hunyuan3D-2 Paint (turbo) -> ONNX graphs for the phone (see paint_phone.py).

  python export_onnx.py --model_dir DIR --out OUT [--tiny 1] [--parity 1]

Graphs written to OUT (weights external, one <graph>.onnx.data each, memory-mapped by onnxruntime):
  ref_unet.onnx   reference (dual) UNet, ONE reference image latent [1,4,64,64] -> 16 per-layer features [1,L,C]
  gen_unet.onnx   generation UNet for ONE view: latent, normal latent, position latent, timestep, camera id,
                  16 ref features [1,Lr,C] (Lr = n_refs*L, dynamic) + 16 cached multiview features of the other views
                  [1,Lo,C] (Lo dynamic, may be 0) -> noise/v prediction [1,4,64,64] + 16 own features (the cache entries)
  vae_enc.onnx    320x320 px tile -> moments [1,8,40,40];  vae_dec.onnx 24x24 latent tile -> [1,3,192,192]
  (features are exchanged as float16; the graphs cast to fp32 inside)
All Linear/Conv weights of the two UNets are int8 per-output-channel weight-only (Cast+Mul in the graph, like C2 of the
torch test; ConstantFolding must stay off in onnxruntime or the fp32 weights get materialised). Attention is written as
query-chunked MatMul/Softmax so no score matrix above ~64 MB exists (onnxruntime has no flash attention on CPU).
There is no image encoder in the turbo paint model (learned text embeddings), nothing to export for it.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("HY_SRC", "/tmp/hy"))
import paintlib as PL  # noqa: E402  (QLinear / QConv2d / load_weights / quant_int8)

MAX_REF, MAX_VIEW = 4, 6
VAE_ENC_TILE, VAE_DEC_TILE = 320, 24
BUDGET = 16_000_000  # floats of one attention-score chunk (64 MB)


# ------------------------------------------------------------------ chunked attention + block forward
class ChunkProc:
    def __init__(self, kmult):
        self.kmult = kmult

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *a, **k):
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, kk, v = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        B, Lq, C = q.shape
        H = attn.heads
        D = C // H
        q = (q * attn.scale).view(B, Lq, H, D).transpose(1, 2)
        kt = kk.view(B, -1, H, D).permute(0, 2, 3, 1)  # B,H,D,K
        v = v.view(B, -1, H, D).transpose(1, 2)
        qc = max(16, min(Lq, BUDGET // (B * H * Lq * self.kmult)))
        qc = 1 << (int(qc).bit_length() - 1)  # power of two
        outs = []
        for i in range(0, Lq, qc):
            outs.append(torch.matmul(torch.softmax(torch.matmul(q[:, :, i:i + qc], kt), -1), v))
        o = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
        return attn.to_out[0](o.transpose(1, 2).reshape(B, Lq, C))


class Ctx:
    mode, ref, mv, out = "gen", {}, {}, {}


CTX = Ctx()


def block_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                  timestep=None, cross_attention_kwargs=None, class_labels=None, added_cond_kwargs=None):
    ln = self.layer_name
    norm = self.norm1(hidden_states)
    h = self.attn1(norm) + hidden_states
    CTX.out[ln] = norm
    if CTX.mode == "gen":
        h = self.attn_refview(norm, encoder_hidden_states=CTX.ref[ln]) + h
        h = self.attn_multiview(norm, encoder_hidden_states=torch.cat([norm, CTX.mv[ln]], 1)) + h
    h = self.attn2(self.norm2(h), encoder_hidden_states=encoder_hidden_states) + h
    return self.ff(self.norm3(h)) + h


def prepare(unet, which):
    """Chunked attention processors + block forward. which: modules to set up (UNet2DConditionModel)."""
    from hy3dgen.texgen.hunyuanpaint.unet.modules import Basic2p5DTransformerBlock
    Basic2p5DTransformerBlock.forward = block_forward
    names = []
    for m in which.modules():
        if isinstance(m, Basic2p5DTransformerBlock):
            names.append(m.layer_name)
            m.attn1.set_processor(ChunkProc(1))
            m.attn2.set_processor(ChunkProc(1))
            if getattr(m, "use_ra", False) and hasattr(m, "attn_refview"):
                m.attn_refview.set_processor(ChunkProc(MAX_REF))
            if getattr(m, "use_ma", False) and hasattr(m, "attn_multiview"):
                m.attn_multiview.set_processor(ChunkProc(MAX_VIEW))
    return names


def san(n):
    return n.replace(".", "_")


class RefWrap(nn.Module):
    def __init__(self, unet_dual, prompt, names):
        super().__init__()
        self.u, self.names = unet_dual, names
        self.register_buffer("prompt", prompt)

    def forward(self, lat):
        CTX.mode, CTX.out = "ref", {}
        self.u(lat, torch.zeros(1, dtype=torch.long), encoder_hidden_states=self.prompt, return_dict=False)
        return tuple(CTX.out[n].half() for n in self.names)


class GenWrap(nn.Module):
    def __init__(self, unet, prompt, names, max_ref):
        super().__init__()
        self.u, self.names = unet, names
        self.register_buffer("prompt", prompt)

    def forward(self, lat, nrm, pos, t, cam, *feats):
        n = len(self.names)
        CTX.mode, CTX.out = "gen", {}
        CTX.ref = {k: f.float() for k, f in zip(self.names, feats[:n])}  # features travel as fp16
        CTX.mv = {k: f.float() for k, f in zip(self.names, feats[n:])}
        x = torch.cat([lat, nrm, pos], 1)
        eps = self.u(x, t, encoder_hidden_states=self.prompt, class_labels=cam + 5, return_dict=False)[0]
        return (eps,) + tuple(CTX.out[nm].half() for nm in self.names)


# ------------------------------------------------------------------ helpers
def quantize_in_place(model):
    """Random-weight test models: same int8 layers as PL.load_weights(quant='int8w') makes."""
    for name, mod in list(model.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, (nn.Linear, nn.Conv2d)):
                q = PL.QLinear(ch) if isinstance(ch, nn.Linear) else PL.QConv2d(ch)
                qw, s = PL.quant_int8(ch.weight.data)
                q._buffers["qweight"], q._buffers["scale"] = qw, s
                setattr(mod, cn, q)


def externalize(path):
    import onnx
    m = onnx.load(path)
    for old in (path + ".data",):
        if os.path.exists(old):
            os.remove(old)
    onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(path) + ".data", size_threshold=1024)
    print("externalized", os.path.basename(path), round(os.path.getsize(path + ".data") / 2 ** 20), "MB", flush=True)


def export(mod, args, path, in_names, out_names, dyn):
    t0 = time.time()
    torch.onnx.export(mod, args, path, opset_version=17, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, do_constant_folding=False, dynamo=False)
    externalize(path)
    print("exported", os.path.basename(path), round(time.time() - t0), "s", flush=True)


def ort_session(path, threads=2):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.log_severity_level = 3
    so.add_session_config_entry("optimization.disable_specified_optimizers", "ConstantFolding")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9)), float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/tmp/hfm/hunyuan3d-paint-v2-0-turbo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", type=int, default=0, help="random small test model instead of the checkpoint")
    ap.add_argument("--parity", type=int, default=1)
    ap.add_argument("--only", default="ref,gen,vae")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    only = a.only.split(",")
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2p5DConditionModel
    meta = {"max_ref": MAX_REF, "max_view": MAX_VIEW, "vae_enc_tile": VAE_ENC_TILE, "vae_dec_tile": VAE_DEC_TILE}
    if os.path.exists(os.path.join(a.out, "meta.json")):
        meta.update(json.load(open(os.path.join(a.out, "meta.json"))))

    if a.tiny:
        cfg = dict(act_fn="silu", attention_head_dim=[1, 2, 4, 4], block_out_channels=[32, 64, 128, 128], cross_attention_dim=64,
                   down_block_types=["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
                   up_block_types=["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3, in_channels=4, out_channels=4, layers_per_block=2,
                   norm_num_groups=8, sample_size=64, use_linear_projection=True, flip_sin_to_cos=True, freq_shift=0)
    else:
        cfg = {k: v for k, v in json.load(open(os.path.join(a.model_dir, "unet", "config.json"))).items() if not k.startswith("_")}
    st_path = os.path.join(a.model_dir, "unet", "diffusion_pytorch_model.safetensors")

    # build the 2.5D UNet on the meta device exactly like run_paint.build_unet
    with torch.device("meta" if not a.tiny else "cpu"):
        unet = UNet2p5DConditionModel(UNet2DConditionModel(**cfg))
        ted = 4 * cfg["block_out_channels"][0]
        if unet.unet.class_embedding.embedding_dim != ted:
            unet.unet.class_embedding = nn.Embedding(unet.unet.class_embedding.num_embeddings, ted)

    def load(prefix):
        if a.tiny:
            return
        PL.load_weights(unet, st_path, torch.float32, "int8w", prefix=prefix)

    if a.tiny:
        quantize_in_place(unet)
        cd = cfg["cross_attention_dim"]
        unet.unet.learned_text_clip_gen.data = torch.randn(1, 77, cd) * 0.1
        unet.unet.learned_text_clip_ref.data = torch.randn(1, 77, cd) * 0.1
        # tiny random weights: make the zero-initialised-looking outputs non-trivial
    # ---------------- reference UNet
    names = prepare(unet.unet_dual, unet.unet_dual)
    prepare(unet.unet, unet.unet)
    meta["layers"] = names
    if "ref" in only:
        load("unet_dual.")
        if not a.tiny:
            PL.load_weights(unet, st_path, torch.float32, "none", prefix="unet.learned_text_clip_ref")
        prompt_ref = unet.unet.learned_text_clip_ref.detach().float().clone()
        rw = RefWrap(unet.unet_dual, prompt_ref, names).eval()
        lat = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            feats_t = rw(lat)
        meta["ref_shapes"] = {n: list(f.shape[1:]) for n, f in zip(names, feats_t)}
        p = os.path.join(a.out, "ref_unet.onnx")
        with torch.no_grad():
            export(rw, (lat,), p, ["lat"], [f"f_{san(n)}" for n in names], None)
        if a.parity:
            s = ort_session(p)
            o = s.run(None, {"lat": lat.numpy()})
            print("PARITY ref_unet (maxrel, l2rel):", [round(rel(x, y.numpy())[1], 5) for x, y in zip(o, feats_t)][:16], flush=True)
            meta["parity_ref_l2"] = max(rel(x, y.numpy())[1] for x, y in zip(o, feats_t))
            del s
        del rw
        unet._modules["unet_dual"] = None
        import gc
        gc.collect()
        torch.save({"prompt_ref": prompt_ref}, os.path.join(a.out, "_tmp_prompt_ref.pt"))

    # ---------------- generation UNet
    if "gen" in only:
        load("unet.")
        prompt_gen = unet.unet.learned_text_clip_gen.detach().float().clone()
        shapes = meta.get("ref_shapes")
        if shapes is None:  # derive from the architecture (token count per layer)
            raise SystemExit("run with ref first (needs shapes)")
        gw = GenWrap(unet.unet, prompt_gen, names, MAX_REF).eval()
        nref, nov = 2, 3
        lat = torch.randn(1, 4, 64, 64)
        nrm, pos = torch.randn(1, 4, 64, 64), torch.randn(1, 4, 64, 64)
        t = torch.tensor([593], dtype=torch.long)
        cam = torch.tensor([2], dtype=torch.long)
        rf = [torch.randn(1, nref * shapes[n][0], shapes[n][1]).half() for n in names]
        mv = [torch.randn(1, nov * shapes[n][0], shapes[n][1]).half() for n in names]
        in_names = ["lat", "nrm", "pos", "t", "cam"] + [f"ref_{san(n)}" for n in names] + [f"mv_{san(n)}" for n in names]
        out_names = ["eps"] + [f"own_{san(n)}" for n in names]
        dyn = {f"ref_{san(n)}": {1: "lr"} for n in names}
        dyn.update({f"mv_{san(n)}": {1: "lo"} for n in names})
        p = os.path.join(a.out, "gen_unet.onnx")
        args = (lat, nrm, pos, t, cam, *rf, *mv)
        with torch.no_grad():
            ref_out = gw(*args)
            export(gw, args, p, in_names, out_names, dyn)
        if a.parity:
            s = ort_session(p)
            feed = {k: v.numpy() for k, v in zip(in_names, args)}
            o = s.run(None, feed)
            r = rel(o[0], ref_out[0].numpy())
            print("PARITY gen_unet eps (maxrel, l2rel):", r, flush=True)
            meta["parity_gen_l2"] = r[1]
            # different n_refs / zero cached views through the dynamic axes
            args2 = list(args)
            for i, n in enumerate(names):
                args2[5 + i] = torch.randn(1, 1 * shapes[n][0], shapes[n][1]).half()
                args2[5 + len(names) + i] = torch.zeros(1, 0, shapes[n][1]).half()
            with torch.no_grad():
                ref2 = gw(*args2)
            o2 = s.run(None, {k: v.numpy() for k, v in zip(in_names, args2)})
            r2 = rel(o2[0], ref2[0].numpy())
            print("PARITY gen_unet eps 1 ref / 0 cached views:", r2, flush=True)
            meta["parity_gen_dyn_l2"] = r2[1]
            del s
        del gw

    # ---------------- VAE
    if "vae" in only:
        if a.tiny or not os.path.isdir(os.path.join(a.model_dir, "vae")):
            vcfg = dict(act_fn="silu", block_out_channels=[128, 256, 512, 512], down_block_types=["DownEncoderBlock2D"] * 4,
                        up_block_types=["UpDecoderBlock2D"] * 4, in_channels=3, latent_channels=4, layers_per_block=2,
                        norm_num_groups=32, out_channels=3, sample_size=768)
            vae = AutoencoderKL(**vcfg)
        else:
            vae = AutoencoderKL.from_pretrained(os.path.join(a.model_dir, "vae"), torch_dtype=torch.float32)
        vae.eval()
        meta["vae_scaling"] = float(vae.config.scaling_factor)

        class Enc(nn.Module):
            def __init__(s):
                super().__init__()
                s.e, s.q = vae.encoder, vae.quant_conv

            def forward(s, x):
                return s.q(s.e(x))

        class Dec(nn.Module):
            def __init__(s):
                super().__init__()
                s.d, s.q = vae.decoder, vae.post_quant_conv

            def forward(s, z):
                return s.d(s.q(z))
        x = torch.randn(1, 3, VAE_ENC_TILE, VAE_ENC_TILE).clamp(-1, 1)
        z = torch.randn(1, 4, VAE_DEC_TILE, VAE_DEC_TILE)
        for nm, m, inp, iname, oname in (("vae_enc", Enc().eval(), x, "x", "moments"), ("vae_dec", Dec().eval(), z, "z", "img")):
            p = os.path.join(a.out, nm + ".onnx")
            with torch.no_grad():
                ref_o = m(inp)
                export(m, (inp,), p, [iname], [oname], None)
            if a.parity:
                o = ort_session(p).run(None, {iname: inp.numpy()})[0]
                r = rel(o, ref_o.numpy())
                print(f"PARITY {nm}:", r, flush=True)
                meta[f"parity_{nm}_l2"] = r[1]
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    tmp = os.path.join(a.out, "_tmp_prompt_ref.pt")
    if os.path.exists(tmp):
        os.remove(tmp)
    print("files:", {f: os.path.getsize(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))}, flush=True)


if __name__ == "__main__":
    main()
