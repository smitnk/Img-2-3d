"""Pieces for the Hunyuan3D-2 Paint CPU test: RSS monitor, patched transformer block (batch / one-view-at-a-time
multiview attention), int8 weight-only layers, streaming weight loader."""
import os, sys, threading, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

GB = 1 << 30
UNEXPECTED = []


# ---------------------------------------------------------------- RSS monitor
def _status():
    d = {"VmRSS": 0.0, "VmHWM": 0.0, "VmSwap": 0.0}
    with open("/proc/self/status") as f:
        for l in f:
            k = l.split(":")[0]
            if k in d:
                d[k] = int(l.split(":")[1].split()[0]) / 1048576.0  # kB -> GB
    return d


class Mon:
    """Samples RSS(+swap) every 0.2 s per named stage and reads VmHWM (reset with clear_refs) at stage end."""

    def __init__(self):
        self.cur, self.stages, self.t = None, {}, {}
        self.stop = False
        self.t0 = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            st = _status()
            if self.cur:
                m = self.stages[self.cur]
                m["sampled_peak_rss_gb"] = max(m["sampled_peak_rss_gb"], st["VmRSS"])
                m["peak_footprint_gb"] = max(m["peak_footprint_gb"], st["VmRSS"] + st["VmSwap"])
                m["peak_swap_gb"] = max(m["peak_swap_gb"], st["VmSwap"])
            time.sleep(0.2)

    def begin(self, name):
        self.end()
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")  # reset VmHWM
        except Exception:
            pass
        self.stages[name] = {"sampled_peak_rss_gb": 0.0, "peak_footprint_gb": 0.0, "peak_swap_gb": 0.0}
        self.t[name] = time.time()
        self.cur = name
        print(f"[MON] >> {name}  rss={_status()['VmRSS']:.2f}GB", flush=True)

    def end(self):
        if not self.cur:
            return
        st = _status()
        m = self.stages[self.cur]
        m["hwm_gb"] = round(st["VmHWM"], 3)
        m["rss_end_gb"] = round(st["VmRSS"], 3)
        m["time_s"] = round(time.time() - self.t[self.cur], 1)
        for k in ("sampled_peak_rss_gb", "peak_footprint_gb", "peak_swap_gb"):
            m[k] = round(m[k], 3)
        print(f"[MON] << {self.cur}  hwm={m['hwm_gb']}GB sampled={m['sampled_peak_rss_gb']}GB "
              f"footprint={m['peak_footprint_gb']}GB swap={m['peak_swap_gb']}GB rss_end={m['rss_end_gb']}GB t={m['time_s']}s",
              flush=True)
        self.cur = None

    def peak(self, key="hwm_gb"):
        return max((m.get(key, 0) for m in self.stages.values()), default=0)


# ---------------------------------------------------------------- sequential multiview attention context
class SeqCtx:
    """One-view-at-a-time multiview attention. For every transformer block we keep the (norm1) hidden states of all
    views from their most recent forward (this step if already done, otherwise the previous step) and let the view
    being processed attend to [own fresh tokens + cached tokens of the other views]."""

    def __init__(self, n):
        self.n, self.cache, self.cur, self.active = n, {}, 0, False

    def kv(self, layer, norm):
        lst = self.cache.setdefault(layer, [None] * self.n)
        parts = [norm if j == self.cur else lst[j] for j in range(self.n) if (j == self.cur or lst[j] is not None)]
        return torch.cat(parts, 1)

    def store(self, layer, norm):
        self.cache.setdefault(layer, [None] * self.n)[self.cur] = norm.detach().clone()

    def nbytes(self):
        return sum(t.numel() * t.element_size() for l in self.cache.values() for t in l if t is not None)


SEQ = SeqCtx(1)


def patched_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None, class_labels=None,
                    added_cond_kwargs=None):
    """Same maths as Basic2p5DTransformerBlock.forward (layer_norm / SD2 path) + SEQ one-view mode."""
    assert self.norm_type == "layer_norm", self.norm_type
    ckw = dict(cross_attention_kwargs or {})
    n = ckw.pop("num_in_batch", 1)
    mode = ckw.pop("mode", None) or ""
    mva_scale = ckw.pop("mva_scale", 1.0)
    ref_scale = ckw.pop("ref_scale", 1.0)
    ckw.pop("position_attn_mask", None)
    ckw.pop("position_voxel_indices", None)
    ckw.pop("gligen", None)
    cond = ckw.pop("condition_embed_dict", None)
    norm = self.norm1(hidden_states)
    if self.pos_embed is not None:
        norm = self.pos_embed(norm)
    a = self.attn1(norm, encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                   attention_mask=attention_mask, **ckw)
    hidden_states = a + hidden_states
    if "w" in mode:
        cond[self.layer_name] = rearrange(norm, "(b n) l c -> b (n l) c", n=n)
    if "r" in mode and self.use_ra:
        ce = cond[self.layer_name].unsqueeze(1).repeat(1, n, 1, 1)
        ce = rearrange(ce, "b n l c -> (b n) l c")
        a = self.attn_refview(norm, encoder_hidden_states=ce, attention_mask=None, **ckw)
        hidden_states = ref_scale * a + hidden_states
    if self.use_ma:
        if SEQ.active:
            kv = SEQ.kv(self.layer_name, norm)
            a = self.attn_multiview(norm, encoder_hidden_states=kv, **ckw)
            SEQ.store(self.layer_name, norm)
            hidden_states = mva_scale * a + hidden_states
        elif n > 1:
            mh = rearrange(norm, "(b n) l c -> b (n l) c", n=n)
            a = self.attn_multiview(mh, encoder_hidden_states=mh, **ckw)
            a = rearrange(a, "b (n l) c -> (b n) l c", n=n)
            hidden_states = mva_scale * a + hidden_states
    if self.attn2 is not None:
        n2 = self.norm2(hidden_states)
        a = self.attn2(n2, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask, **ckw)
        hidden_states = a + hidden_states
    n3 = self.norm3(hidden_states)
    return self.ff(n3) + hidden_states


# ---------------------------------------------------------------- int8 weight-only layers (per-output-channel)
def quant_int8(w):
    w = w.float()
    s = w.abs().amax(dim=tuple(range(1, w.ndim)), keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(w / s).clamp(-127, 127).to(torch.int8), s


class QLinear(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.in_features, self.out_features = src.in_features, src.out_features
        self.register_buffer("qweight", None)
        self.register_buffer("scale", None)
        self.bias = src.bias

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


class QConv2d(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = src.stride, src.padding, src.dilation, src.groups
        self.padding_mode = src.padding_mode
        self.register_buffer("qweight", None)
        self.register_buffer("scale", None)
        self.bias = src.bias

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return F.conv2d(x, w, None if self.bias is None else self.bias.to(x.dtype), self.stride, self.padding,
                        self.dilation, self.groups)


def load_weights(model, st_path, dtype, quant="none", prefix=None):
    """Stream tensors from a safetensors file straight into (meta-device) `model`. quant: none | int8w."""
    from safetensors import safe_open
    n = 0
    with safe_open(st_path, "pt") as f:
        for k in f.keys():
            if prefix and not k.startswith(prefix):
                continue
            mp, pn = k.rsplit(".", 1)
            try:
                mod = model.get_submodule(mp)
            except AttributeError:
                UNEXPECTED.append(k)
                continue
            if pn not in mod._parameters and pn not in mod._buffers:
                UNEXPECTED.append(k)
                continue
            t = f.get_tensor(k)
            if quant == "int8w" and pn == "weight" and isinstance(mod, (nn.Linear, nn.Conv2d)):
                q = QLinear(mod) if isinstance(mod, nn.Linear) else QConv2d(mod)
                qw, s = quant_int8(t)
                q._buffers["qweight"], q._buffers["scale"] = qw, s
                pp, cn = (mp.rsplit(".", 1) + [None])[:2] if "." in mp else ("", mp)
                parent = model.get_submodule(pp) if pp else model
                setattr(parent, cn, q)
            elif pn in mod._parameters:
                mod._parameters[pn] = nn.Parameter(t.to(dtype if t.is_floating_point() else t.dtype), requires_grad=False)
            else:
                mod._buffers[pn] = t.to(dtype if t.is_floating_point() else t.dtype)
            n += 1
    return n


def tensor_bytes(model, prefix=None):
    tot = 0
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p is None or p.device.type == "meta" or (prefix and not n.startswith(prefix)):
            continue
        tot += p.numel() * p.element_size()
    return tot
