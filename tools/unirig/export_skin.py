"""Export the UniRig SKIN model to ONNX (float32) and prove it on real data.

usage (inside the CPU-patched UniRig checkout, instead of `python run.py <args>`):
    SKIN_NAME=<mesh> SKIN_OUT=<dir> python export_skin.py capture <run.py args>

It runs the stock skin pipeline; on the first UniRigSkin._get_predict call it
  1. hooks every spconv SubMConv3d. For the first conv of each kernel size it PROBES spconv itself (probe_table): a
     copy of the layer with a one-hot weight per kernel index and the input-row id as feature reveals, per output
     row and flat weight index k, which input row spconv sums in. From that: the offset vector of every k, the duplicate
     voxel policy and whether the centre is the row itself. build_nbr must reproduce the probed table exactly
     (SPCONV TABLE ... PASS/FAIL). Then every conv is compared with GatherConv on its real tensor (SPCONV ... PASS/FAIL),
  2. runs the stock forward (reference outputs),
  3. runs SkinONNX (the dense-ops re-implementation) in torch and prints WRAPDIFF lines vs the stock forward,
  4. exports <SKIN_OUT>/skin.onnx (first mesh only, or SKIN_EXPORT=1) and saves inputs_<name>.npz / ref_<name>.npz.
The pipeline then carries on with the torch result, so the torch FBX is produced as usual.

The sparse conv: spconv has no ONNX op, so out[i] = bias + sum_k W_k . feat[nbr[i, k]] with the neighbour table
nbr (N, K^3) built in numpy from the voxel coordinates (onnx_skin.build_nbr), row N = an all-zero row for empty
neighbours. Serialization orders, FPS query points and the parent-accumulated voxel mask are also inputs.
Assumes batch size 1, N = 16384 points (the phone's size; the graphs have N baked in, UniRig's own inference config uses 32768; N % 1024 == 0, so the PTv3 patch padding is the identity), no pooling.
"""
import math
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_skin as O  # noqa: E402

CONV_BUDGET = 1 << 24  # elements of gathered features per matmul (64 MB fp32)


def native_w(conv):
    """(K3, I, O) conv weight in spconv's own flat kernel index order (weight (O, K, K, K, I) -> index (a*K+b)*K+c)."""
    w = conv.weight.detach()
    I, Oc = conv.in_channels, conv.out_channels
    if w.shape[0] == Oc and w.shape[-1] == I:  # spconv 2.x
        return w.reshape(Oc, -1, I).permute(1, 2, 0).contiguous()
    assert w.shape[-2:] == (I, Oc), tuple(w.shape)
    return w.reshape(-1, I, Oc)


def probe_table(mod, x):
    """spconv's own neighbour table for layer `mod` on its real input `x` (SparseConvTensor): tab[i, k] = input row that
    spconv adds to output row i through flat weight index k (N = none). Feature = row id + 1 in channel 0, weight =
    one-hot (k -> out channel k mod O), zero bias, so out[i, k mod O] is exactly the id of the contributing row."""
    import collections
    import copy
    import spconv.pytorch as spconv
    p = copy.deepcopy(mod)
    p._forward_hooks, p._forward_pre_hooks = collections.OrderedDict(), collections.OrderedDict()
    N, I, Oc = x.features.shape[0], mod.in_channels, mod.out_channels
    K3 = native_w(mod).shape[0]
    tab = torch.full((N, K3), N, dtype=torch.long)
    feat = torch.zeros(N, I)
    feat[:, 0] = torch.arange(1, N + 1)  # exact in fp32 up to 2^24
    xin = spconv.SparseConvTensor(feat, x.indices, x.spatial_shape, x.batch_size)
    with torch.no_grad():
        if p.bias is not None:
            p.bias.zero_()
        for base in range(0, K3, Oc):
            m = min(Oc, K3 - base)
            wk = torch.zeros(K3, I, Oc)
            wk[base + torch.arange(m), 0, torch.arange(m)] = 1
            p.weight.copy_(wk.permute(2, 0, 1).reshape(p.weight.shape) if p.weight.shape[0] == Oc and p.weight.shape[-1] == I
                           else wk.reshape(p.weight.shape))
            y = p(xin).features[:, :m]
            tab[:, base:base + m] = torch.where(y > 0.5, y.round().long() - 1, torch.full_like(y, N, dtype=torch.long))
    return tab.numpy()


def derive_offsets(tab, coords):
    """Per flat weight index k the offset vector coords[src] - coords[dst] (must be one vector for every pair) -> ((K3, 3), n_inconsistent)."""
    N, K3 = tab.shape
    offs, bad = np.zeros((K3, 3), np.int64), 0
    for k in range(K3):
        dst = np.nonzero(tab[:, k] < N)[0]
        assert len(dst), f"weight index {k} never used"
        u, c = np.unique(coords[tab[dst, k]] - coords[dst], axis=0, return_counts=True)
        offs[k] = u[c.argmax()]
        bad += int(c.sum() - c.max())
    return offs, bad


class GatherConv(nn.Module):
    """spconv SubMConv3d as gathers + matmuls over a neighbour table whose column k is spconv's flat weight index k.
    One gather (<= CONV_BUDGET elements = 64 MB) + one matmul per chunk of offsets, accumulated in place of an (N, K3, I) tensor."""

    def __init__(self, conv, n):
        super().__init__()
        self.register_buffer("w", native_w(conv))  # (K3, I, O)
        self.register_buffer("b", None if conv.bias is None else conv.bias.detach().clone())
        self.K3, self.I, self.O, self.n = self.w.shape[0], conv.in_channels, conv.out_channels, n
        self.g = max(1, min(self.K3, CONV_BUDGET // (n * self.I)))

    def forward(self, f, nbr):
        fp = torch.cat([f, f.new_zeros(1, self.I)])
        out = None
        for a in range(0, self.K3, self.g):
            b = min(a + self.g, self.K3)
            x = torch.index_select(fp, 0, nbr[:, a:b].reshape(-1)).reshape(self.n, (b - a) * self.I)
            y = x @ self.w[a:b].reshape((b - a) * self.I, self.O)
            out = y if out is None else out + y
        return out if self.b is None else out + self.b


def sdpa(q, k, v, sq=None, chunk=None):
    """softmax(q k^T / sqrt(d)) v on (B, H, S, D); query rows in chunks of `chunk` (sq = static query length)."""
    s = q.shape[-1] ** -0.5
    if chunk is None:
        return torch.softmax((q * s) @ k.transpose(-2, -1), -1) @ v
    return torch.cat([torch.softmax((q[:, :, a:a + chunk] * s) @ k.transpose(-2, -1), -1) @ v
                      for a in range(0, sq, chunk)], dim=2)


def heads(t, h):  # (1, S, h*d) -> (1, h, S, d)
    return t.reshape(1, -1, h, t.shape[-1] // h).transpose(1, 2)


def merge(t):  # (1, h, S, d) -> (1, S, h*d)
    return t.transpose(1, 2).reshape(1, -1, t.shape[1] * t.shape[3])


class SkinONNX(nn.Module):
    """Dense re-implementation of UniRigSkin._get_predict (eval, B=1) sharing the stock modules' weights."""

    def __init__(self, m, n):
        super().__init__()
        assert not m.training
        self.m, self.n = m, n
        enc = m.mesh_encoder
        self.stem = GatherConv(enc.embedding.stem.conv, n)
        self.stages = []  # per stage: list of (kind, idx) over the stage's modules
        self.convs = nn.ModuleList()
        for stage in enc.enc._modules.values():
            plan = []
            for mod in stage._modules.values():
                if isinstance(mod, nn.Linear):
                    plan.append(("lin", mod))
                else:
                    assert mod.pre_norm and not mod.attn.enable_rpe and not mod.attn.enable_flash
                    plan.append(("blk", len(self.convs)))
                    self.convs.append(GatherConv(mod.cpe[0], n))
            self.stages.append(plan)
        self.blocks = [mod for stage in enc.enc._modules.values() for mod in stage._modules.values() if not isinstance(mod, nn.Linear)]

    # -- Michelangelo global encoder -> (1, 1024, feat_dim)
    def global_latents(self, vertices, normals, gsel):
        m = self.m
        e = m.global_encoder.encoder
        fe = e.fourier_embedder
        data = e.input_proj(torch.cat([fe(vertices), normals], -1))  # (1, N, 512)
        x = e.input_proj(torch.cat([fe(vertices[:, gsel]), normals[:, gsel]], -1))  # (1, 1024, 512)
        ca = e.cross_attn
        h = ca.attn.heads
        q = heads(ca.attn.c_q(ca.ln_1(x)), h)
        kv = ca.attn.c_kv(ca.ln_2(data)).reshape(1, -1, h, 2 * (data.shape[-1] // h))
        k, v = kv.chunk(2, dim=-1)
        # 128 query rows x 8 heads x 32768 keys (N at 32768) = 134 MB of scores per chunk
        o = sdpa(q, k.transpose(1, 2), v.transpose(1, 2), sq=gsel.shape[0], chunk=128)
        x = x + ca.attn.c_proj(merge(o))
        x = x + ca.mlp(ca.ln_3(x))
        for blk in e.self_attn.resblocks:
            h = blk.attn.heads
            qkv = blk.attn.c_qkv(blk.ln_1(x))
            qkv = qkv.reshape(1, -1, h, qkv.shape[-1] // h)
            q, k, v = qkv.chunk(3, dim=-1)
            x = x + blk.attn.c_proj(merge(sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))))
            x = x + blk.mlp(blk.ln_2(x))
        if e.ln_post is not None:
            x = e.ln_post(x)
        return m.out_proj(x)

    def mha(self, mha, x, xkv, sq=None, chunk=None):  # the CPU MHA shim: Wq / Wkv / out_proj, cross attention
        h = mha.h
        q = heads(mha.Wq(x), h)
        kv = mha.Wkv(xkv)
        kv = kv.reshape(1, -1, 2, h, kv.shape[-1] // (2 * h)).permute(2, 0, 3, 1, 4)
        return mha.out_proj(merge(sdpa(q, kv[0], kv[1], sq, chunk)))

    def res_cross(self, blk, q, kv, sq=None, chunk=None):
        x = blk.norm1(q + self.mha(blk.attention, q, kv, sq, chunk))
        return blk.norm2(x + blk.ffn(x))

    def ptv3_attn(self, a, h, order, inverse):
        H, C, K = a.num_heads, a.channels, min(self.n, a.patch_size_max)
        assert self.n % K == 0
        qkv = a.qkv(h)[order]
        if a.enable_qknorm:
            qkv = a.qknorm(qkv)
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)  # (P, H, K, Ch)
        step = max(1, CONV_BUDGET // (H * K * K))
        outs = []
        for s in range(0, q.shape[0], step):
            att = torch.softmax((q[s:s + step] * a.scale) @ k[s:s + step].transpose(-2, -1), -1)
            outs.append((att @ v[s:s + step]).transpose(1, 2).reshape(-1, C))
        return a.proj(torch.cat(outs)[inverse])

    def ptv3(self, vertices, normals, nbr3, nbr5, order, inverse):
        enc = self.m.mesh_encoder
        f0 = torch.cat([vertices, normals, torch.zeros_like(vertices)], -1).reshape(self.n, 9)
        emb = enc.embedding
        f = emb.stem.act(emb.stem.norm(self.stem(f0, nbr5))) + emb.res_linear(f0)
        nb = 0
        for plan in self.stages:
            for kind, x in plan:
                if kind == "lin":
                    f = x(f)
                    continue
                blk = self.blocks[nb]
                oi = blk.attn.order_index
                f = f + blk.cpe[2](blk.cpe[1](self.convs[x](f, nbr3)))
                f = f + self.ptv3_attn(blk.attn, blk.norm1[0](f), order[oi], inverse[oi])
                f = f + blk.mlp[0](blk.norm2[0](f))
                nb += 1
        return f

    def global_bone(self, vertices, normals, joints, gsel):
        """Part A: Michelangelo global latents g (1, 1024, F) and the bone encoder + its cross attention (1, J, F)."""
        m = self.m
        g = self.global_latents(vertices, normals, gsel)
        be = m.bone_encoder
        min_coord = vertices.amin(dim=1)  # (1, 3)
        x = be.bone_encoder((joints - min_coord[:, None, :]).squeeze(0)).unsqueeze(0)  # (1, J, feat)
        lat = torch.cat([x, g], 1)
        for a in be.attn:
            x = self.res_cross(a, x, lat)
        return g, x

    def head(self, pt, g, bone, voxel_skin, skin_mask):
        """Part C: mesh <- bone/global cross attention, per-vertex skin weights over vertex chunks."""
        m = self.m
        mesh = m.feat_map(pt).unsqueeze(0)
        lat = torch.cat([bone, g], 1)
        for blk in m.mesh_bone_attn:
            mesh = self.res_cross(blk, mesh, lat, sq=self.n, chunk=2048)
        H, Fd = m.num_heads, m.feat_dim
        bk = m.kmesh(bone[0]).reshape(-1, H, Fd).permute(1, 2, 0)  # (H, F, J)
        outs = []
        for i in range(0, self.n, m.num_train_vertex):
            c = m.qmesh(mesh[0, i:i + m.num_train_vertex])
            q = c.reshape(c.shape[0], H, Fd).permute(1, 0, 2)  # (H, n, F)
            a = torch.softmax(torch.bmm(q, bk) / math.sqrt(Fd), -1).permute(1, 2, 0)  # (n, J, H)
            a = m.attn_skin_norm(a)
            e = m.voxel_skin_norm(m.voxel_skin_embed(voxel_skin[i:i + m.num_train_vertex].unsqueeze(-1)))
            p = m.skinweight_pred(m.downscale(torch.cat([a, e], -1))).squeeze(-1)  # (n, J)
            outs.append(torch.softmax(p, -1))
        s = torch.cat(outs, 0) * torch.pow(skin_mask, m.voxel_mask)
        return s / s.sum(-1, keepdim=True)

    def forward(self, vertices, normals, joints, voxel_skin, skin_mask, gsel, nbr3, nbr5, order, inverse):
        g, bone = self.global_bone(vertices, normals, joints, gsel)
        pt = self.ptv3(vertices, normals, nbr3, nbr5, order, inverse)  # (N, 384)
        return self.head(pt, g, bone, voxel_skin, skin_mask), g, bone, pt


class Part(nn.Module):
    """One of the three sub-graphs (same maths as SkinONNX.forward, run one after the other by onnx_skin.run_parts)."""

    def __init__(self, net, fn):
        super().__init__()
        self.net, self.fn = net, fn

    def forward(self, *a):
        return getattr(self.net, self.fn)(*a)


def export_parts(net, inp, outs, out_dir):
    """skin_a.onnx (global latents + bone encoder), skin_pt.onnx (PTv3), skin_head.onnx (mesh-bone attention + weights), each with its
    weights in <name>.onnx.data. outs = net(*args) = (skin, g, bone, pt). Peak memory on the phone = the biggest part, not the sum."""
    import shutil
    I = lambda *k: tuple(_t(inp[x]) for x in k)
    _, g, bone, pt = outs
    spec = {
        "skin_a": ("global_bone", I("vertices", "normals", "joints", "gsel"), ["vertices", "normals", "joints", "gsel"], ["g", "bone"],
                   {"joints": {1: "J"}, "bone": {1: "J"}}),
        "skin_pt": ("ptv3", I("vertices", "normals", "nbr3", "nbr5", "order", "inverse"),
                    ["vertices", "normals", "nbr3", "nbr5", "order", "inverse"], ["pt"], {}),
        "skin_head": ("head", (pt, g, bone) + I("voxel_skin", "skin_mask"), ["pt", "g", "bone", "voxel_skin", "skin_mask"], ["skin"],
                      {"bone": {1: "J"}, "voxel_skin": {1: "J"}, "skin_mask": {1: "J"}, "skin": {1: "J"}}),
    }
    for name, (fn, args, ins, outn, dyn) in spec.items():
        tmp = f"{out_dir}/_{name}"
        os.makedirs(tmp, exist_ok=True)
        torch.onnx.export(Part(net, fn).eval(), args, f"{tmp}/{name}.onnx", input_names=ins, output_names=outn, dynamic_axes=dyn,
                          opset_version=17, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
        O.externalize(f"{tmp}/{name}.onnx", f"{out_dir}/{name}.onnx")  # also merges the legacy exporter's loose per-tensor files
        shutil.rmtree(tmp)
        print(f"EXPORT part {name}: {os.path.getsize(f'{out_dir}/{name}.onnx.data') / 2 ** 30:.2f} GB weights", flush=True)


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def _line(tag, name, a, b):
    d = (a.double() - b.double()).abs()
    print(f"{tag} {name}: max={d.max():.3e} mean={d.mean():.3e} refmax={b.abs().max():.3e} shape={tuple(a.shape)}", flush=True)
    return float(d.max())


def run_capture(model, batch, orig):
    out_dir, name = os.environ["SKIN_OUT"], os.environ["SKIN_NAME"]
    os.makedirs(out_dir, exist_ok=True)
    enc = model.mesh_encoder
    st = {"rows": [], "tab": {}, "offs": {}, "tie": 0, "sc": True, "table_fail": 0}
    ref = {}
    hooks = []

    def pre_emb(mod, args):
        p = args[0]
        grid = p.grid_coord.cpu().numpy()
        S = int(grid.max()) + 1
        key = (grid[:, 0].astype(np.int64) * S + grid[:, 1]) * S + grid[:, 2]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        st["dup"] = torch.from_numpy(cnt[inv] > 1)
        st["grid"] = grid
        print(f"SPCONV points={len(grid)} unique_voxels={int((cnt > 0).sum())} points_in_shared_voxels={int(st['dup'].sum())}", flush=True)
        st["pt"] = {k: p[k].cpu().numpy() for k in ("serialized_order", "serialized_inverse", "serialized_code")}

    def learn(cname, mod, x, K):
        """First conv of kernel size K: read spconv's neighbour table, derive offsets / tie / centre, check build_nbr == table."""
        import spconv
        coords = x.indices[:, 1:].cpu().numpy().astype(np.int64)
        N = len(coords)
        tab = probe_table(mod, x)
        offs, bad = derive_offsets(tab, coords)
        print(f"SPCONV PROBE {cname} K={K} spconv={getattr(spconv, '__version__', '?')} weight={tuple(mod.weight.shape)} "
              f"indices={tuple(x.indices.shape)} spatial_shape={list(x.spatial_shape)} batch={x.batch_size} "
              f"padding={getattr(mod, 'padding', None)} inconsistent_pairs={bad} missing_pairs={int((tab == N).sum())}", flush=True)
        print("SPCONV PROBE offsets by weight index (k: dx,dy,dz): " + " ".join(f"{k}:{o[0]},{o[1]},{o[2]}" for k, o in enumerate(offs)), flush=True)
        print(f"SPCONV PROBE offsets == natural order: {bool((offs == O.natural_offsets(K)).all())}, "
              f"== reversed: {bool((offs == -O.natural_offsets(K)).all())}, coords == serialize grid: "
              f"{bool(coords.shape == st['grid'].shape and (coords == st['grid']).all())}", flush=True)
        best = None
        for t in (0, 1):
            for sc in (True, False):
                mism = int((O.build_nbr(coords, K, t, offs, sc) != tab).sum())
                print(f"SPCONV PROBE build_nbr tie={t} self_center={sc}: {mism} of {tab.size} entries differ", flush=True)
                if best is None or mism < best[0]:
                    best = (mism, t, sc)
        if not st["offs"]:  # the first (stem) probe decides the policy for all
            st["tie"], st["sc"] = best[1], best[2]
        mism = int((O.build_nbr(coords, K, st["tie"], offs, st["sc"]) != tab).sum())
        ok = bad == 0 and mism == 0
        st["table_fail"] += not ok
        print(f"SPCONV TABLE K={K} tie={st['tie']} self_center={st['sc']} differing_entries={mism} {'PASS' if ok else 'FAIL'}", flush=True)
        if mism:  # which rows disagree: shared voxels (policy) or any (offsets)?
            r = np.nonzero((O.build_nbr(coords, K, st["tie"], offs, st["sc"]) != tab).any(1))[0]
            print(f"SPCONV TABLE K={K} rows differing={len(r)} of which in shared voxels={int(st['dup'][r].sum())} first rows={r[:5].tolist()}", flush=True)
        st["offs"][K], st["coords"] = offs, coords
        st["tab"][K] = torch.from_numpy(O.build_nbr(coords, K, st["tie"], offs, st["sc"]))

    def conv_hook(cname):
        def h(mod, args, out):
            x = args[0]
            f, o = x.features.detach(), out.features.detach()
            K = int(mod.kernel_size[0])
            if K not in st["tab"]:
                learn(cname, mod, x, K)
            y = GatherConv(mod, f.shape[0])(f, st["tab"][K])
            d = (y - o).abs().amax(1)
            dup = st["dup"]
            rel = float(d[~dup].max() if (~dup).any() else 0.0) / max(float(o.abs().max()), 1e-12)
            r = (cname, K, f.shape[1], o.shape[1], float(d.max()), float(o.abs().max()),
                 float(d[dup].max()) if dup.any() else 0.0, float(d[~dup].max()) if (~dup).any() else 0.0, rel,
                 "PASS" if rel < 1e-3 else "FAIL")
            st["rows"].append(r)
            print("SPCONV %s K=%d in=%d out=%d max=%.3e refmax=%.3e dup_rows=%.3e unique_rows=%.3e rel_unique=%.2e %s" % r, flush=True)
        return h

    hooks.append(enc.embedding.register_forward_pre_hook(pre_emb))
    for cname, mod in model.named_modules():
        if mod.__class__.__name__.startswith("SubMConv3d"):
            hooks.append(mod.register_forward_hook(conv_hook(cname)))
    hooks.append(model.out_proj.register_forward_hook(lambda mod, a, o: ref.__setitem__("dbg_global", o.detach().clone())))
    hooks.append(model.bone_encoder.register_forward_hook(lambda mod, a, o: ref.__setitem__("dbg_bone", o.detach().clone())))
    hooks.append(enc.register_forward_hook(lambda mod, a, o: ref.__setitem__("dbg_ptv3", o.feat.detach().clone())))
    res = orig(model, batch)
    for h in hooks:
        h.remove()
    ref["skin"] = res[0][0].detach().clone()
    rows = st["rows"]
    print(f"SPCONV SUMMARY convs={len(rows)} worst max abs diff {max(r[4] for r in rows):.3e} "
          f"(dup rows {max(r[6] for r in rows):.3e}, unique rows {max(r[7] for r in rows):.3e}) "
          f"worst relative {max(r[4] / max(r[5], 1e-12) for r in rows):.3e}", flush=True)
    nfail = sum(r[9] == "FAIL" for r in rows)
    print(f"SPCONV RESULT {'PASS' if rows and not nfail and not st['table_fail'] else 'FAIL'} "
          f"(convs failing={nfail} of {len(rows)}, neighbour tables failing={st['table_fail']})", flush=True)

    with torch.inference_mode(False), torch.no_grad():
        g = lambda k: batch[k][0].detach().cpu().numpy()
        J = int(batch["num_bones"][0])
        assert batch["vertices"].shape[0] == 1
        N = batch["vertices"].shape[1]
        pt = st["pt"]
        ser = O.serialize(g("vertices").astype(np.float32))
        perm = np.array([[bool((ser["code"][j] == pt["serialized_code"][r]).all()) for j in range(4)].index(True) for r in range(4)])
        print(f"PREP shuffle perm={perm.tolist()}", flush=True)
        inp = O.prepare(g("vertices").astype(np.float32), g("normals").astype(np.float32), g("joints")[:J].astype(np.float32),
                        g("voxel_skin")[:, :J].astype(np.float32), g("parents")[:J], perm=perm, tie=st["tie"], offs=st["offs"], sc=st["sc"])
        for k, kk in (("order", "serialized_order"), ("inverse", "serialized_inverse")):
            print(f"PREP {k} equals the real run's: {bool((inp[k] == pt[kk]).all())}", flush=True)
        try:  # does the numpy FPS reproduce torch_cluster?
            from torch_cluster import fps
            v = g("vertices").astype(np.float32)
            ind = np.random.default_rng(seed=0).choice(N, 4096, replace=False)
            idx = fps(torch.from_numpy(v[ind]), torch.zeros(4096, dtype=torch.long), ratio=0.25, random_start=False).numpy()
            print(f"PREP fps numpy == torch_cluster: {bool((ind[idx] == inp['gsel']).all())} "
                  f"({int((ind[idx] == inp['gsel']).sum())}/{len(idx)} rows)", flush=True)
            inp["gsel"] = ind[idx].astype(np.int64)  # use torch_cluster's picks so ORT-vs-torch isolates the model
        except Exception as e:
            print("PREP fps check skipped:", repr(e), flush=True)

        net = SkinONNX(model, N).eval()
        args = tuple(_t(inp[k]) for k in O.INPUT_NAMES)
        outs = net(*args)
        for nm, o in zip(O.OUT_NAMES, outs):
            _line("WRAPDIFF", nm, o, ref[nm])
        np.savez_compressed(f"{out_dir}/inputs_{name}.npz", **inp, perm=perm, tie=st["tie"], sc=st["sc"],
                            offs3=st["offs"][3], offs5=st["offs"][5])
        np.savez_compressed(f"{out_dir}/ref_{name}.npz", **{k: v.cpu().numpy() for k, v in ref.items()})
        path = f"{out_dir}/skin.onnx"
        if not os.path.exists(path) or os.environ.get("SKIN_EXPORT") == "1":
            dyn = {"joints": {1: "J"}, "voxel_skin": {1: "J"}, "skin_mask": {1: "J"}, "skin": {1: "J"}, "dbg_bone": {1: "J"}}
            torch.onnx.export(net, args, path, input_names=O.INPUT_NAMES, output_names=O.OUT_NAMES, dynamic_axes=dyn,
                              opset_version=17, do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
            try:
                consolidate(path)
            except Exception:
                traceback.print_exc()
            try:
                export_parts(net, inp, outs, out_dir)
            except Exception:  # the monolithic skin.onnx stays the fallback
                traceback.print_exc()
            size = sum(os.path.getsize(f"{out_dir}/{f}") for f in os.listdir(out_dir) if not f.endswith(".npz"))
            print(f"EXPORT wrote {path} (+external data), {size / 2 ** 30:.2f} GB", flush=True)
    return res


def consolidate(path):
    """The legacy exporter writes one file per tensor when the model is over 2 GB: merge them into skin.onnx.data."""
    d = os.path.dirname(path)
    loose = [f for f in os.listdir(d) if not f.endswith(".npz") and f != os.path.basename(path)]
    if not loose:
        return
    import onnx
    m = onnx.load(path)
    for f in loose:
        os.remove(os.path.join(d, f))
    onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location="skin.onnx.data", size_threshold=1024)
    print(f"EXPORT merged {len(loose)} loose tensor files into skin.onnx.data", flush=True)


def capture(run_args):
    sys.path.insert(0, os.getcwd())
    from src.model import unirig_skin as us
    orig = us.UniRigSkin._get_predict
    done = []

    def wrapped(self, batch):
        if done or self.training:
            return orig(self, batch)
        done.append(1)
        try:
            return run_capture(self, batch, orig)
        except Exception:
            traceback.print_exc()
            print("CAPTURE FAILED (continuing with the plain torch forward)", flush=True)
            open(os.path.join(os.environ["SKIN_OUT"], f"capture_failed_{os.environ['SKIN_NAME']}"), "w").close()
            return orig(self, batch)

    us.UniRigSkin._get_predict = wrapped
    O._run_py(run_args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture(sys.argv[2:])
    else:
        sys.exit(__doc__)
