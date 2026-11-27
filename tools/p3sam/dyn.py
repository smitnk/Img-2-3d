"""Dynamic P3-SAM encoder graph: one ONNX file for any mesh. Every structure table comes in as an input computed by phone/p3s_np.py
(numpy); nothing about the capture mesh is baked in (voxel counts, patch count, patch size, pooling fan-in are all shapes of inputs).

Differences to p3.EncStatic: attention order tables are 2-D (patches, K) so K and the patch count are read off the shape; attention is
unrolled over heads (bounds the score tensor to (patches, K, K) without a data-dependent loop); the pooling group table always has
8 columns; the gather-conv chunk size is a fixed memory budget.
"""
import torch
import torch.nn as nn

import p3 as P
import export_skin as E

CONV_BUDGET = 1 << 23
N_REF = (32000, 24000, 14000, 6000, 2000)


class ConvDyn(nn.Module):
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


class EncDyn(nn.Module):
    def __init__(self, p3, names):
        super().__init__()
        son = p3.sonata
        self.names = names
        self.n_stages = len(son.enc._modules)
        self.stem = son.embedding.stem
        self.pool, self.convs, self.blocks, self.plan = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), []
        for s, stage in enumerate(son.enc._modules.values()):
            ids = []
            for mod in stage._modules.values():
                if mod.__class__.__name__ == "GridPooling":
                    self.pool.append(mod)
                else:
                    self.blocks.append(mod)
                    self.convs.append(ConvDyn(mod.cpe[0], N_REF[s]))
                    ids.append(len(self.blocks) - 1)
            self.plan.append(ids)
        self.mlp = p3.mlp

    @staticmethod
    def attn(a, h, order2, inverse):
        H, C = a.num_heads, a.channels
        K = order2.shape[1]
        qkv = a.qkv(h)[order2.reshape(-1)]
        t = qkv.reshape(-1, K, 3, H, C // H)
        outs = []
        for i in range(H):
            q, k, v = t[:, :, 0, i] * a.scale, t[:, :, 1, i], t[:, :, 2, i]
            outs.append(torch.softmax(q @ k.transpose(1, 2), -1) @ v)
        return a.proj(torch.cat(outs, -1).reshape(-1, C)[inverse])

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
                for j in range(1, 8):
                    f = torch.maximum(f, x[G[:, j]])
                f = p.act[0](p.norm[0](f))
            for i in ids:
                blk = self.blocks[i]
                oi = blk.attn.order_index
                f = f + blk.cpe[2](blk.cpe[1](self.convs[i](f, t[f"nbr{s}"])))
                h = self.attn(blk.attn, blk.norm1[0](f), t[f"ord{s}_{oi}"], t[f"inv{s}_{oi}"])
                f = f + blk.drop_path[0](blk.ls1[0](h))
                f = f + blk.drop_path[0](blk.ls2[0](blk.mlp[0](blk.norm2[0](f))))
            feats.append(f)
        cur = feats[-1]
        for s in range(self.n_stages - 2, -1, -1):
            cur = torch.cat([feats[s], cur[t[f"pinv{s + 1}"]]], -1)
        return self.mlp(cur)


def dyn_axes(names):
    d = {"feat0": {0: "n0"}, "feat": {0: "n0"}}
    for n in names:
        if n.startswith("nbr") or n.startswith("pg"):
            d[n] = {0: "n" + n[-1]}
        elif n.startswith("pinv"):
            d[n] = {0: "n" + str(int(n[-1]) - 1)}
        elif n.startswith("ord"):
            d[n] = {0: "P" + n[3:], 1: "K" + n[3:]}
        elif n.startswith("inv"):
            d[n] = {0: "n" + n[3]}
    return d
