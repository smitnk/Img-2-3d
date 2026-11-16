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
BUDGET = 8_000_000  # floats of one attention-score chunk (32 MB)


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
        # fp16 cache inputs are tied to this block's activations (+0) so onnxruntime does not convert / project the features of
        # all 16 layers up front (they would all be alive at once)
        dep = (norm.reshape(-1)[0] * 0).half()
        h = self.attn_refview(norm, encoder_hidden_states=(CTX.ref[ln] + dep).float()) + h
        h = self.attn_multiview(norm, encoder_hidden_states=torch.cat([norm, (CTX.mv[ln] + dep).float()], 1)) + h
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
        CTX.ref = dict(zip(self.names, feats[:n]))  # features travel as fp16 (cast to fp32 per block)
        CTX.mv = dict(zip(self.names, feats[n:]))
        x = torch.cat([lat, nrm, pos], 1)
        eps = self.u(x, t, encoder_hidden_states=self.prompt, class_labels=cam + 5, return_dict=False)[0]
        return (eps,) + tuple(CTX.out[nm].half() for nm in self.names)


class GenSeg(nn.Module):
    """A run of UNet units (down i / mid / up i) of the generation UNet: the whole model is cut into a few graphs whose weights
    are mapped one at a time, so little of the 0.9 GB of int8 weights is resident at once (the runner goes segment by segment
    over all 6 views; the cross-view dependency is per layer only, so this order gives the same result as view by view)."""

    def __init__(self, unet2d, prompt, units, names, first, last, n_in_skips, ref=False):
        super().__init__()
        self.u, self.units, self.names, self.first, self.last, self.nin, self.ref = unet2d, units, names, first, last, n_in_skips, ref
        self.register_buffer("prompt", prompt)

    def forward(self, *a):
        u, eh = self.u, self.prompt
        a = list(a)
        if self.first and self.ref:  # reference stream: timestep 0, no camera embedding, plain 4-channel latent
            lat = a[0]
            a = a[1:]
            emb = u.time_embedding(u.time_proj(torch.zeros(1, dtype=torch.long)).to(lat.dtype))
            sample = u.conv_in(lat)
            skips = [sample]
        elif self.first:
            lat, nrm, pos, t, cam = a[:5]
            a = a[5:]
            emb = u.time_embedding(u.time_proj(t).to(lat.dtype)) + u.class_embedding(cam + 5).to(lat.dtype)
            sample = u.conv_in(torch.cat([lat, nrm, pos], 1))
            skips = [sample]
        else:
            sample, emb = a[0], a[1]
            skips = [x.float() for x in a[2:2 + self.nin]]
            a = a[2 + self.nin:]
        n = len(self.names)
        CTX.mode, CTX.out = ("ref" if self.ref else "gen"), {}
        if not self.ref:
            CTX.ref, CTX.mv = dict(zip(self.names, a[:n])), dict(zip(self.names, a[n:2 * n]))
        for kind, i in self.units:
            if kind == "down":
                blk = u.down_blocks[i]
                if getattr(blk, "has_cross_attention", False):
                    sample, res = blk(hidden_states=sample, temb=emb, encoder_hidden_states=eh)
                else:
                    sample, res = blk(hidden_states=sample, temb=emb)
                skips += list(res)
            elif kind == "mid":
                sample = u.mid_block(sample, emb, encoder_hidden_states=eh)
            else:
                blk = u.up_blocks[i]
                k = len(blk.resnets)
                res, skips = tuple(skips[-k:]), skips[:-k]
                if getattr(blk, "has_cross_attention", False):
                    sample = blk(hidden_states=sample, temb=emb, res_hidden_states_tuple=res, encoder_hidden_states=eh, upsample_size=None)
                else:
                    sample = blk(hidden_states=sample, temb=emb, res_hidden_states_tuple=res, upsample_size=None)
        owns = tuple(CTX.out[nm].half() for nm in self.names)
        if self.last and self.ref:
            return owns
        if self.last:
            return (u.conv_out(u.conv_act(u.conv_norm_out(sample))),) + owns
        return (sample, emb) + tuple(x.half() for x in skips) + owns


def unit_layers(unit, names):
    kind, i = unit
    pre = {"down": f"down_{i}_", "mid": "mid_", "up": f"up_{i}_"}[kind]
    return [n for n in names if n.startswith(pre)]


def unit_params(u, unit):
    kind, i = unit
    mod = {"down": lambda: u.down_blocks[i], "mid": lambda: u.mid_block, "up": lambda: u.up_blocks[i]}[kind]()
    return sum(p.numel() for p in mod.parameters()) + sum(b.numel() for n, b in mod.named_buffers() if n.endswith("qw"))


def plan_segments(u, nseg):
    units = [("down", i) for i in range(len(u.down_blocks))] + [("mid", 0)] + [("up", i) for i in range(len(u.up_blocks))]
    w = [unit_params(u, x) for x in units]
    tot, segs, cur, acc = sum(w), [], [], 0
    for x, wi in zip(units, w):
        if cur and acc + wi / 2 > tot / nseg * (len(segs) + 1) and len(segs) < nseg - 1:
            segs.append(cur)
            cur = []
        cur.append(x)
        acc += wi
    segs.append(cur)
    return segs, dict(zip(units, w))


# ------------------------------------------------------------------ int8 layers whose dequantisation cannot be hoisted
# onnxruntime schedules every node that does not depend on the activations first, i.e. Cast(int8)->Mul(scale)->Transpose of ALL
# weights (4 GB fp32 for the gen UNet) before the first MatMul. Making the int8 weight tensor a Gather whose index depends on the
# layer's input (+ 0) ties each dequantisation to its own layer, so only one dequantised weight exists at a time.
class GLinear(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.register_buffer("qw", q.qweight.t().contiguous())  # [in, out] int8 (MatMul layout, no Transpose node)
        self.register_buffer("scale", q.scale.reshape(1, -1).float())
        self.register_buffer("ar", torch.arange(q.in_features))
        self.bias = q.bias

    def forward(self, x):
        dep = (x.reshape(-1)[0] * 0).long()
        y = torch.matmul(x, self.qw.index_select(0, self.ar + dep).float() * self.scale)
        return y if self.bias is None else y + self.bias


class GConv2d(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = q.stride, q.padding, q.dilation, q.groups
        self.register_buffer("qw", q.qweight)
        self.register_buffer("scale", q.scale.float())
        self.register_buffer("ar", torch.arange(q.qweight.shape[0]))
        self.bias = q.bias

    def forward(self, x):
        dep = (x.reshape(-1)[0] * 0).long()
        w = self.qw.index_select(0, self.ar + dep).float() * self.scale
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def gate_quant(model):
    for _, mod in list(model.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, PL.QLinear):
                setattr(mod, cn, GLinear(ch))
            elif isinstance(ch, PL.QConv2d):
                setattr(mod, cn, GConv2d(ch))


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
    ap.add_argument("--segments", type=int, default=4, help="number of gen_seg*.onnx graphs")
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
        gate_quant(unet)

    if a.tiny:
        quantize_in_place(unet)
        cd = cfg["cross_attention_dim"]
        unet.unet.learned_text_clip_gen.data = torch.randn(1, 77, cd) * 0.1
        unet.unet.learned_text_clip_ref.data = torch.randn(1, 77, cd) * 0.1
        # tiny random weights: make the zero-initialised-looking outputs non-trivial
    # ---------------- reference UNet
    names = prepare(unet.unet_dual, unet.unet_dual)
    prepare(unet.unet, unet.unet)
    gate_quant(unet)  # no-op until weights are loaded for the real model (see load())
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
        segs, w = plan_segments(unet.unet_dual, a.segments)
        print("ref segments:", [[f"{k}{i}" for k, i in sg] for sg in segs], "params(M):", [round(sum(w[x] for x in sg) / 1e6) for sg in segs], flush=True)
        meta["ref_segments"], nskip, files = [], 0, []
        state = None
        for k, sg in enumerate(segs):
            first, last = k == 0, k == len(segs) - 1
            lnames = [n for x in sg for n in unit_layers(x, names)]
            nin = nskip
            for kind, i in sg:
                if kind == "down":
                    blk = unet.unet_dual.down_blocks[i]
                    nskip += len(blk.resnets) + (1 if blk.downsamplers else 0) + (1 if i == 0 else 0)
                elif kind == "up":
                    nskip -= len(unet.unet_dual.up_blocks[i].resnets)
            gs = GenSeg(unet.unet_dual, prompt_ref, sg, lnames, first, last, nin, ref=True).eval()
            args = (lat,) if first else tuple(state)
            in_names = ["lat"] if first else ["i_h", "i_emb"] + [f"i_sk{j}" for j in range(nin)]
            out_state = [] if last else ["h", "emb"] + [f"sk{j}" for j in range(nskip)]
            out_names = out_state + [f"own_{san(n)}" for n in lnames]
            p = os.path.join(a.out, f"ref_seg{k}.onnx")
            with torch.no_grad():
                outs = gs(*args)
                export(gs, args, p, in_names, out_names, None)
            state = outs[:len(out_state)]
            meta["ref_segments"].append({"file": f"ref_seg{k}.onnx", "layers": lnames, "first": first, "last": last,
                                         "in_state": in_names, "out_state": out_state})
            files.append(p)
            del gs
        if a.parity:
            st, got = None, {}
            for p, sm in zip(files, meta["ref_segments"]):
                s = ort_session(p)
                o = s.run(None, {"lat": lat.numpy()} if sm["first"] else dict(zip(sm["in_state"], st)))
                del s
                st = o[:len(sm["out_state"])]
                got.update(dict(zip(sm["layers"], o[len(sm["out_state"]):])))
            errs = [round(rel(got[n], f.numpy())[1], 5) for n, f in zip(names, feats_t)]
            print("PARITY ref (segments chained, per layer l2rel):", errs, flush=True)
            meta["parity_ref_l2"] = max(errs)
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
        gw = GenWrap(unet.unet, prompt_gen, names, MAX_REF).eval()  # monolithic torch reference for the parity check only
        segs, w = plan_segments(unet.unet, a.segments)
        print("segments:", [[f"{k}{i}" for k, i in sg] for sg in segs], "params(M):",
              [round(sum(w[x] for x in sg) / 1e6) for sg in segs], flush=True)
        nref, nov = 2, 3
        lat = torch.randn(1, 4, 64, 64)
        nrm, pos = torch.randn(1, 4, 64, 64), torch.randn(1, 4, 64, 64)
        t = torch.tensor([593], dtype=torch.long)
        cam = torch.tensor([2], dtype=torch.long)

        def rand_feats(nm, nr, no):
            return ([torch.randn(1, nr * shapes[n][0], shapes[n][1]).half() if nr else torch.zeros(1, 0, shapes[n][1]).half() for n in nm],
                    [torch.randn(1, no * shapes[n][0], shapes[n][1]).half() if no else torch.zeros(1, 0, shapes[n][1]).half() for n in nm])
        rf, mv = rand_feats(names, nref, nov)
        full_args = (lat, nrm, pos, t, cam, *rf, *mv)
        with torch.no_grad():
            ref_out = gw(*full_args)
        meta["segments"] = []
        nskip = 0
        gsegs = []
        for k, sg in enumerate(segs):
            first, last = k == 0, k == len(segs) - 1
            lnames = [n for x in sg for n in unit_layers(x, names)]
            nin = nskip
            for kind, i in sg:  # skip-connection bookkeeping (diffusers: conv_in output, every resnet(+attn) and downsampler output)
                if kind == "down":
                    blk = unet.unet.down_blocks[i]
                    nskip += len(blk.resnets) + (1 if blk.downsamplers else 0) + (1 if i == 0 else 0)
                elif kind == "up":
                    nskip -= len(unet.unet.up_blocks[i].resnets)
            gs = GenSeg(unet.unet, prompt_gen, sg, lnames, first, last, nin).eval()
            gsegs.append(gs)
            meta["segments"].append({"file": f"gen_seg{k}.onnx", "layers": lnames, "first": first, "last": last, "n_in_skips": nin,
                                     "n_out_skips": nskip})
        # dry run in torch (monolith-equivalent chain) to get example state tensors, then export every segment
        state, p_files = None, []
        for k, gs in enumerate(gsegs):
            sm = meta["segments"][k]
            lnames = sm["layers"]
            sr, sv = [rf[names.index(n)] for n in lnames], [mv[names.index(n)] for n in lnames]
            if gs.first:
                args = (lat, nrm, pos, t, cam, *sr, *sv)
                in_names = ["lat", "nrm", "pos", "t", "cam"]
            else:
                args = (*state, *sr, *sv)
                in_names = ["i_h", "i_emb"] + [f"i_sk{j}" for j in range(sm["n_in_skips"])]
            in_names += [f"ref_{san(n)}" for n in lnames] + [f"mv_{san(n)}" for n in lnames]
            out_state = ["eps"] if gs.last else ["h", "emb"] + [f"sk{j}" for j in range(sm["n_out_skips"])]
            out_names = out_state + [f"own_{san(n)}" for n in lnames]
            dyn = {f"ref_{san(n)}": {1: "lr"} for n in lnames}
            dyn.update({f"mv_{san(n)}": {1: "lo"} for n in lnames})
            p = os.path.join(a.out, sm["file"])
            with torch.no_grad():
                outs = gs(*args)
                export(gs, args, p, in_names, out_names, dyn)
            sm["in_state"] = in_names[:len(in_names) - 2 * len(lnames)]
            sm["out_state"] = out_state
            state = outs[:len(out_state)]
            p_files.append((p, in_names, args, outs, out_names))
            del gs
        gsegs = None
        meta["n_segments"] = len(segs)
        if a.parity:
            def chain(feeds_rf, feeds_mv):
                st, eps = None, None
                for (p, in_names, args, outs, out_names), sm in zip(p_files, meta["segments"]):
                    s = ort_session(p)
                    ln = sm["layers"]
                    feed = {}
                    if sm["first"]:
                        feed.update({k: v.numpy() for k, v in zip(["lat", "nrm", "pos", "t", "cam"], (lat, nrm, pos, t, cam))})
                    else:
                        feed.update(dict(zip(sm["in_state"], st)))
                    feed.update({f"ref_{san(n)}": feeds_rf[names.index(n)].numpy() for n in ln})
                    feed.update({f"mv_{san(n)}": feeds_mv[names.index(n)].numpy() for n in ln})
                    o = s.run(None, feed)
                    del s
                    st = o[:len(sm["out_state"])]
                    if sm["last"]:
                        eps = o[0]
                return eps
            r = rel(chain(rf, mv), ref_out[0].numpy())
            print("PARITY gen_unet (segments chained) eps (maxrel, l2rel):", r, flush=True)
            meta["parity_gen_l2"] = r[1]
            rf2, mv2 = rand_feats(names, 1, 0)
            with torch.no_grad():
                ref2 = gw(lat, nrm, pos, t, cam, *rf2, *mv2)
            r2 = rel(chain(rf2, mv2), ref2[0].numpy())
            print("PARITY gen_unet 1 ref / 0 cached views:", r2, flush=True)
            meta["parity_gen_dyn_l2"] = r2[1]
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
