"""P3-SAM (Tencent Hunyuan3D-Part, part segmentation) -> CPU torch + ONNX port helpers.

P3-SAM = Sonata (a Point Transformer V3 encoder, 5 stages, spconv submanifold conv + serialized attention + grid pooling)
         + MLP 1232->512 + prompt heads (per point x per prompt MLPs, max-pooled "global" feature, IoU head).

Ported like tools/unirig (see export_skin.py / onnx_skin.py):
  * spconv SubMConv3d      -> gather + matmul over a neighbour table built in numpy (onnx_skin.build_nbr, verified against spconv)
  * flash-attn             -> plain (chunked) softmax attention over the serialized patches
  * torch_scatter segment_csr(max) in GridPooling -> padded group table + max over columns
  * serialization orders / padding / pooling clusters -> recorded from the stock torch run and fed as int64 inputs
  * prompt heads: the first Linear of each head acts on concat(feature, xyz, prompt, ...), so it is split into per-point and
    per-prompt parts (exactly the same maths, 3x fewer FLOPs, no (N, K, 512) concat tensors)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unirig"))
import onnx_skin as O  # noqa: E402
import export_skin as E  # noqa: E402

CONV_BUDGET = 1 << 24


CONV = {"mode": "ref", "cache": {}}


def nbr_for(x):
    """(neighbour table (N, 27), voxel coords) of a SparseConvTensor; natural offset order (a*3+b)*3+c with offset a-1,b-1,c-1, centre = row itself."""
    key = (x.indices.data_ptr(), int(x.indices.shape[0]), int(x.indices.sum()))
    t = CONV["cache"].get(key)
    if t is None:
        coords = x.indices[:, 1:].cpu().numpy().astype(np.int64)
        t = (O.build_nbr(coords, 3), coords)
        CONV["cache"][key] = t
    return t


# ----------------------------------------------------------------------------------------------- stock model (torch, CPU)
def build_p3(src, enable_cpu_patches=True, shuffle=True):
    """Stock P3-SAM modules on CPU. src = snapshot of the HF space tencent/Hunyuan3D-Part (P3-SAM/, XPart/)."""
    sys.path.insert(0, f"{src}/XPart/partgen")
    import spconv.pytorch as spconv

    class SubMConv3dCPU(spconv.SubMConv3d):
        """CONV['mode'] == 'ref': exact submanifold conv by gathers over a numpy neighbour table (the maths of spconv's CUDA kernels);
        'spconv': the pip spconv CPU build (its CPU kernels returned wrong / run-to-run different neighbours on 15k voxels, see capture.py)."""

        def forward(self, input, add_input=None):
            if CONV["mode"] == "spconv":  # CPU build asserts on the fused bias of an eval conv, see unirig/patch_cpu.py
                return self._conv_forward(True, input, self.weight, self.bias, add_input, name=self.name,
                                          sparse_unique_name=self._sparse_unique_name, act_type=self.act_type,
                                          act_alpha=self.act_alpha, act_beta=self.act_beta)
            nbr = torch.from_numpy(nbr_for(input)[0])
            if not hasattr(self, "_wn"):
                self._wn = E.native_w(self).detach().clone()
            f = input.features
            fp = torch.cat([f, f.new_zeros(1, f.shape[1])])
            I = f.shape[1]
            g = max(1, min(self._wn.shape[0], CONV_BUDGET // (len(f) * I)))
            out = None
            for a in range(0, self._wn.shape[0], g):
                b = min(a + g, self._wn.shape[0])
                x = torch.index_select(fp, 0, nbr[:, a:b].reshape(-1)).reshape(len(f), (b - a) * I)
                y = x @ self._wn[a:b].reshape((b - a) * I, -1)
                out = y if out is None else out + y
            if self.bias is not None:
                out = out + self.bias
            return input.replace_feature(out)

    spconv.SubMConv3d = SubMConv3dCPU
    from models import sonata
    from models.sonata import model as sm

    ckpt = sonata.load("sonata", repo_id="facebook/sonata", download_root=os.environ.get("SONATA_DIR", "sonata_ckpt"), ckpt_only=True)
    cfg = dict(ckpt["config"])
    print("SONATA CONFIG", cfg, flush=True)
    cfg["enable_flash"] = False
    cfg["upcast_attention"] = False
    cfg["upcast_softmax"] = False
    cfg["shuffle_orders"] = shuffle

    class P3(nn.Module):
        def __init__(self):
            super().__init__()
            self.sonata = sm.PointTransformerV3(**cfg)
            self.mlp = nn.Sequential(nn.Linear(1232, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512))

            def head(i, o, h):
                return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))
            self.seg_mlp_1, self.seg_mlp_2, self.seg_mlp_3 = [head(512 + 6, 1, 512) for _ in range(3)]
            self.seg_s2_mlp_g = head(512 + 9, 256, 256)
            self.seg_s2_mlp_1, self.seg_s2_mlp_2, self.seg_s2_mlp_3 = [head(512 + 9 + 256, 1, 256) for _ in range(3)]
            self.iou_mlp = head(512 + 9 + 256, 256, 256)
            self.iou_mlp_out = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 3))

    p3 = P3()
    return p3, sonata, sm, cfg


def load_weights(p3, path):
    from safetensors.torch import load_file
    sd = load_file(path)
    sd = {(k[4:] if k.startswith("dit.") else k): v for k, v in sd.items()}
    res = p3.load_state_dict(sd, strict=False)
    print(f"WEIGHTS loaded {len(sd)} tensors; missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    for k in res.missing_keys[:20]:
        print("  missing", k)
    for k in res.unexpected_keys[:20]:
        print("  unexpected", k)
    return len(res.missing_keys), len(res.unexpected_keys)


def chunked_attn(q, k, v, scale, C, budget_patches_heads=48):
    """q,k,v (P, H, K, Ch) -> (P*K, C); same as the stock non-flash path, a few patches at a time (bounds the (P,H,K,K) scores)."""
    H = q.shape[1]
    step = max(1, budget_patches_heads // H)
    outs = []
    for s in range(0, q.shape[0], step):
        a = torch.softmax((q[s:s + step] * scale) @ k[s:s + step].transpose(-2, -1), -1)
        outs.append((a @ v[s:s + step]).transpose(1, 2).reshape(-1, C))
    return torch.cat(outs)


# ----------------------------------------------------------------------------------------------- stock run + recording
class Rec:
    """Everything the static graph needs, recorded from one stock run."""

    def __init__(self):
        self.stage = 0
        self.nbr, self.coords, self.ordinv, self.cluster, self.n, self.K = {}, {}, {}, {}, {}, {}


def install_recording(p3, sm, rec):
    def fwd(self, point):
        self.patch_size = min(sm.offset2bincount(point.offset).min().tolist(), self.patch_size_max)
        H, K, C = self.num_heads, self.patch_size, self.channels
        pad, unpad, _ = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        rec.ordinv[(rec.stage, self.order_index)] = (order.cpu().numpy().astype(np.int64), inverse.cpu().numpy().astype(np.int64))
        rec.K[rec.stage] = K
        rec.n[rec.stage] = point.feat.shape[0]
        qkv = self.qkv(point.feat)[order]
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        feat = chunked_attn(q, k, v, self.scale, C)[inverse]
        point.feat = self.proj_drop(self.proj(feat))
        return point

    sm.SerializedAttention.forward = fwd
    hooks = []

    def conv_pre(mod, args):
        x = args[0]
        s = rec.stage
        if s not in rec.nbr:
            rec.nbr[s], rec.coords[s] = nbr_for(x)

    def pool_post(mod, args, out):
        rec.stage += 1
        rec.cluster[rec.stage] = out.pooling_inverse.cpu().numpy().astype(np.int64)

    for m in p3.sonata.modules():
        n = m.__class__.__name__
        if n == "SubMConv3dCPU":
            hooks.append(m.register_forward_pre_hook(conv_pre))
        elif n == "GridPooling":
            hooks.append(m.register_forward_hook(pool_post))
    return hooks


@torch.no_grad()
def stock_features(p3, transform, points, normals):
    """get_feat of auto_mask.py on CPU -> (N, 512), plus the (n0, 9) voxel feature the graph starts from."""
    data = transform({"coord": points.copy(), "normal": normals.copy(), "color": np.ones_like(points), "batch": np.zeros(len(points), dtype=np.int64)})
    point = p3.sonata(data)
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    feat = p3.mlp(point.feat)
    return feat[point.inverse], data


@torch.no_grad()
def stock_heads(p3, feats, points, prompts, bs=4):
    """P3SAM.forward of auto_mask.py, prompts in chunks of bs. feats (N,512) points (N,3) prompts (K,3) -> logits (N,K,3), iou (K,3)."""
    feats, points, prompts = [torch.as_tensor(a, dtype=torch.float32) for a in (feats, points, prompts)]
    N = feats.shape[0]
    L, I = [], []
    for i in range(0, prompts.shape[0], bs):
        pr = prompts[i:i + bs]
        K = pr.shape[0]
        fs = torch.cat([feats[:, None].expand(N, K, 512), points[:, None].expand(N, K, 3), pr[None].expand(N, K, 3)], -1)
        pm = torch.cat([p3.seg_mlp_1(fs), p3.seg_mlp_2(fs), p3.seg_mlp_3(fs)], -1)
        fs2 = torch.cat([fs, pm], -1)
        g = p3.seg_s2_mlp_g(fs2).max(0).values[None].expand(N, K, 256)
        fs3 = torch.cat([g, fs2], -1)
        pm2 = torch.cat([p3.seg_s2_mlp_1(fs3), p3.seg_s2_mlp_2(fs3), p3.seg_s2_mlp_3(fs3)], -1)
        iou = torch.sigmoid(p3.iou_mlp_out(p3.iou_mlp(torch.cat([g, fs, pm2], -1)).max(0).values))
        L.append(pm2)
        I.append(iou)
    return torch.cat(L, 1), torch.cat(I, 0)


# ----------------------------------------------------------------------------------------------- static (ONNX-able) graphs
def pool_group_table(cluster, n_out):
    """(n_out, maxc) rows of the child stage per cluster, short clusters padded with their own last member (max is idempotent)."""
    order = np.argsort(cluster, kind="stable")
    counts = np.bincount(cluster, minlength=n_out)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    G = np.empty((n_out, int(counts.max())), np.int64)
    for j in range(G.shape[1]):
        G[:, j] = order[np.minimum(starts + j, starts + counts - 1)]
    return G


class Conv27(nn.Module):
    """spconv SubMConv3d(k=3) as gathers + matmuls; works for any n (the chunk size is only a memory budget)."""

    def __init__(self, conv, n_ref):
        super().__init__()
        self.register_buffer("w", E.native_w(conv).clone())
        self.register_buffer("b", conv.bias.detach().clone())
        self.K3, self.I, self.O = self.w.shape
        self.g = max(1, min(self.K3, CONV_BUDGET // (n_ref * self.I)))

    def forward(self, f, nbr):
        fp = torch.cat([f, f.new_zeros(1, self.I)])
        out = None
        for a in range(0, self.K3, self.g):
            b = min(a + self.g, self.K3)
            x = torch.index_select(fp, 0, nbr[:, a:b].reshape(-1)).reshape(-1, (b - a) * self.I)
            y = x @ self.w[a:b].reshape((b - a) * self.I, self.O)
            out = y if out is None else out + y
        return out + self.b


def enc_input_spec(rec):
    """Names of the encoder graph inputs in order + dynamic axes."""
    names, dyn = ["feat0"], {"feat0": {0: "n0"}}
    for s in sorted(rec.nbr):
        names.append(f"nbr{s}")
        dyn[f"nbr{s}"] = {0: f"n{s}"}
    for s in sorted(rec.cluster):
        names += [f"pg{s}", f"pinv{s}"]
        dyn[f"pg{s}"] = {0: f"n{s}"}
        dyn[f"pinv{s}"] = {0: f"n{s - 1}"}
    for (s, oi) in sorted(rec.ordinv):
        names += [f"ord{s}_{oi}", f"inv{s}_{oi}"]
        dyn[f"ord{s}_{oi}"] = {0: f"np{s}"}
        dyn[f"inv{s}_{oi}"] = {0: f"n{s}"}
    return names, dyn


def enc_inputs(rec, feat0):
    d = {"feat0": np.ascontiguousarray(feat0, np.float32)}
    for s in sorted(rec.nbr):
        d[f"nbr{s}"] = rec.nbr[s].astype(np.int64)
    for s in sorted(rec.cluster):
        d[f"pg{s}"] = pool_group_table(rec.cluster[s], rec.n[s])
        d[f"pinv{s}"] = rec.cluster[s]
    for (s, oi), (o, i) in sorted(rec.ordinv.items()):
        d[f"ord{s}_{oi}"], d[f"inv{s}_{oi}"] = o, i
    return d


class EncStatic(nn.Module):
    """Sonata encoder (embedding + 5 stages + multi-scale concat) + the 1232->512 MLP, for a recorded structure. -> (n0, 512)."""

    def __init__(self, p3, rec):
        super().__init__()
        son = p3.sonata
        self.names, _ = enc_input_spec(rec)
        self.K = dict(rec.K)
        self.n_stages = len(son.enc._modules)
        self.stem = son.embedding.stem
        self.stages = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.plan = []  # per stage: list of (block_index_in_self.blocks)
        self.blocks = nn.ModuleList()
        for s, stage in enumerate(son.enc._modules.values()):
            ids = []
            for mod in stage._modules.values():
                if mod.__class__.__name__ == "GridPooling":
                    self.pool.append(mod)
                else:
                    self.blocks.append(mod)
                    self.convs.append(Conv27(mod.cpe[0], rec.n[s]))
                    ids.append(len(self.blocks) - 1)
            self.plan.append(ids)
        self.mlp = p3.mlp

    def attn(self, a, h, order, inverse, K):
        H, C = a.num_heads, a.channels
        qkv = a.qkv(h)[order]
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
        return a.proj(chunked_attn(q, k, v, a.scale, C, budget_patches_heads=48)[inverse])

    def forward(self, *ins):
        t = dict(zip(self.names, ins))
        f = self.stem[2](self.stem[1](self.stem[0](t["feat0"])))
        feats = []
        for s, ids in enumerate(self.plan):
            if s > 0:
                p = self.pool[s - 1]
                x = p.proj(f)
                G = t[f"pg{s}"]
                f = x[G[:, 0]]
                for j in range(1, G.shape[1]):
                    f = torch.maximum(f, x[G[:, j]])
                f = p.act[0](p.norm[0](f))
            for i in ids:
                blk = self.blocks[i]
                oi = blk.attn.order_index
                f = f + blk.cpe[2](blk.cpe[1](self.convs[i](f, t[f"nbr{s}"])))
                h = self.attn(blk.attn, blk.norm1[0](f), t[f"ord{s}_{oi}"], t[f"inv{s}_{oi}"], self.K[s])
                f = f + blk.drop_path[0](blk.ls1[0](h))
                f = f + blk.drop_path[0](blk.ls2[0](blk.mlp[0](blk.norm2[0](f))))
            feats.append(f)
        cur = feats[-1]
        for s in range(self.n_stages - 2, -1, -1):
            cur = torch.cat([feats[s], cur[t[f"pinv{s + 1}"]]], -1)
        return self.mlp(cur)


def lin_sum(lin, sizes, xs):
    """Linear over concat(xs, -1) = sum of the per-part linears (xs broadcast against each other)."""
    out = None
    for w, x in zip(torch.split(lin.weight, sizes, dim=1), xs):
        y = F.linear(x, w)
        out = y if out is None else out + y
    return out + lin.bias


class HeadsStatic(nn.Module):
    """P3SAM prompt heads. feats (N,512) points (N,3) prompts (K,3) -> s2 mask logits (N,K,3), iou (K,3). Dynamic N and K."""

    def __init__(self, p3):
        super().__init__()
        for k in ("seg_mlp_1", "seg_mlp_2", "seg_mlp_3", "seg_s2_mlp_g", "seg_s2_mlp_1", "seg_s2_mlp_2", "seg_s2_mlp_3", "iou_mlp", "iou_mlp_out"):
            setattr(self, k, getattr(p3, k))

    @staticmethod
    def run(seq, first_sizes, xs):
        h = lin_sum(seq[0], first_sizes, xs)
        h = seq[2](seq[1](h))
        return seq[4](seq[3](h))

    def forward(self, feats, points, prompts):
        f, p, q = feats[:, None], points[:, None], prompts[None]
        s1 = [512, 3, 3]
        pm = torch.cat([self.run(m, s1, [f, p, q]) for m in (self.seg_mlp_1, self.seg_mlp_2, self.seg_mlp_3)], -1)  # (N,K,3)
        s2 = [512, 3, 3, 3]
        g = self.run(self.seg_s2_mlp_g, s2, [f, p, q, pm]).max(0).values[None]  # (1,K,256)
        s3 = [256, 512, 3, 3, 3]
        l3 = [self.run(m, s3, [g, f, p, q, pm]) for m in (self.seg_s2_mlp_1, self.seg_s2_mlp_2, self.seg_s2_mlp_3)]
        pm2 = torch.cat(l3, -1)
        h = self.run(self.iou_mlp, s3, [g, f, p, q, pm2]).max(0).values  # (K,256)
        # a second Concat for the graph output: an output that also feeds other nodes breaks onnxconverter_common's fp16 conversion
        return torch.cat(l3, -1), torch.sigmoid(self.iou_mlp_out(h))


def export_onnx(mod, args, in_names, out_names, dyn, path):
    tmp = path + ".tmp.onnx"
    torch.onnx.export(mod, args, tmp, input_names=in_names, output_names=out_names, dynamic_axes=dyn, opset_version=17,
                      do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL, dynamo=False)
    O.externalize(tmp, path)
    for f in (tmp, tmp + ".data"):
        if os.path.exists(f):
            os.remove(f)
    sz = os.path.getsize(path) + (os.path.getsize(path + ".data") if os.path.exists(path + ".data") else 0)
    print(f"EXPORT {path}: {sz / 2 ** 20:.1f} MB", flush=True)
    return sz


def fps(p, n):
    """Farthest point sampling from index 0 (numpy)."""
    return O.fps_np(p, n)
