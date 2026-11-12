"""Patch a UniRig checkout to run inference on CPU only (no CUDA, no flash_attn).

usage: python patch_cpu.py <unirig_dir> [pyrender|open3d]   (voxel backend for skinning)
Every replacement asserts the exact expected number of matches so upstream drift fails loudly.
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "pyrender"
assert backend in ("pyrender", "open3d")


def patch(rel, *pairs):
    p = root / rel
    s = p.read_text()
    for old, new, *n in pairs:
        want = n[0] if n else 1
        assert s.count(old) == want, f"{rel}: expected {want}x {old!r}, found {s.count(old)}"
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)


# Lightning trainer: CPU, fp32 (bf16-mixed + accelerator gpu in the shipped tasks)
for task in ("quick_inference_skeleton_articulationxl_ar_256", "quick_inference_unirig_skin"):
    patch(f"configs/task/{task}.yaml",
          ("precision: bf16-mixed", "precision: 32-true"),
          ("accelerator: gpu", "accelerator: cpu"))

# OPT-350m decoder: flash_attention_2 -> sdpa (was eager). Same softmax(QK^T/sqrt(d))V; sdpa's CPU
# kernel is blocked so it never materialises the 15 beams x 16 heads x 1026^2 fp32 score tensor
# (~1 GB, 2-3 live copies) in the prefill. Differs from eager only by float summation order.
patch("configs/model/unirig_ar_350m_1024_81920_float32.yaml",
      ("_attn_implementation: flash_attention_2", "_attn_implementation: sdpa"))

# ---- lossless RAM reductions ----
# Beam search expands the 1026-token prompt (1024 shape latents + bos + cls) to 15 identical rows, so
# HF's DynamicCache stores the prompt K/V 15 times (24 layers x 2 x 15 x 1026 x 1024 x 4 B = 3 GB).
# SharedPrefixCache stores it once and rebuilds the per-layer (B, H, P+G, D) tensor with expand+cat on
# the fly (transient, one layer at a time). The values fed to attention are bit-identical to the stock
# cache; sharing is only enabled for a layer if all prompt rows are exactly equal (torch.equal), else
# that layer falls back to the stock behaviour.
SHARED_CACHE = '''

from transformers import DynamicCache


class SharedPrefixCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.prefix = {}  # layer -> (k, v) of ONE beam
        self.seen_layers = set()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self.seen_layers:  # prefill of this layer
            self.seen_layers.add(layer_idx)
            k, v = key_states, value_states
            if k.shape[0] > 1 and torch.equal(k, k[:1].expand_as(k)) and torch.equal(v, v[:1].expand_as(v)):
                if layer_idx == 0:
                    self._seen_tokens += k.shape[-2]
                while len(self.key_cache) <= layer_idx:
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.prefix[layer_idx] = (k[:1].clone(), v[:1].clone())
                return k, v
            return super().update(k, v, layer_idx, cache_kwargs)
        gk, gv = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx not in self.prefix:
            return gk, gv
        pk, pv = self.prefix[layer_idx]
        B = gk.shape[0]
        return (torch.cat([pk.expand(B, -1, -1, -1), gk], dim=-2),
                torch.cat([pv.expand(B, -1, -1, -1), gv], dim=-2))

    def get_seq_length(self, layer_idx=0):
        n = super().get_seq_length(layer_idx)
        return n + (self.prefix[layer_idx][0].shape[-2] if layer_idx in self.prefix else 0)
    # reorder_cache (stock) only touches key_cache/value_cache = the generated part, which is what
    # differs between beams; the shared prompt never needs reordering.
'''
patch("src/model/unirig_ar.py",
      ("from .spec import ModelSpec, ModelInput\n", "from .spec import ModelSpec, ModelInput\n" + SHARED_CACHE),
      ("        results = self.transformer.generate(\n            inputs_embeds=cond,",
       "        if kwargs.get('num_beams', 1) > 1:\n            kwargs['past_key_values'] = SharedPrefixCache()\n"
       "        results = self.transformer.generate(\n            inputs_embeds=cond,"))

# The skin checkpoint is 4.4 GB for ~1.5 GB of weights: it still holds Adam optimizer state, and Lightning
# keeps the whole dict in RAM next to the model while restoring. Drop optimizer_states/lr_schedulers (never
# read in predict mode) once, mmap-loading so the untouched optimizer pages never become resident. The
# state_dict is byte-identical, so the outputs are too. No-op for the skeleton ckpt (weights only).
patch("run.py",
      ("    resume_from_checkpoint = download(resume_from_checkpoint)\n",
       "    resume_from_checkpoint = download(resume_from_checkpoint)\n"
       "    try:\n"
       "        _ck = torch.load(resume_from_checkpoint, map_location='cpu', mmap=True, weights_only=False)\n"
       "        print('ckpt keys', list(_ck.keys()))\n"
       "        if _ck.get('optimizer_states'):\n"
       "            _ck['optimizer_states'] = []\n"
       "            _ck['lr_schedulers'] = []\n"
       "            torch.save(_ck, '/tmp/slim_model.ckpt')\n"
       "            resume_from_checkpoint = '/tmp/slim_model.ckpt'\n"
       "            print('slimmed ckpt (optimizer state dropped)')\n"
       "        del _ck\n"
       "    except Exception as e:\n"
       "        print('ckpt slimming skipped:', e)\n"))

# voxel_skin renders depth with pyrender (needs OpenGL); the workflow may switch to open3d
patch("configs/transform/inference_skin_transform.yaml",
      ("backend: pyrender", f"backend: {backend}"))

# Skin model: flash_attn.modules.mha.MHA (always used with use_flash_attn=False, i.e. plain
# cross-attention) -> same-named parameters (Wq, Wkv, out_proj) so the checkpoint still loads.
MHA_SHIM = '''class MHA(nn.Module):  # CPU stand-in for flash_attn.modules.mha.MHA (cross-attn only)
    def __init__(self, embed_dim, num_heads, cross_attn=True):
        super().__init__()
        assert cross_attn
        self.h = num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_kv):
        B, S, E = x.shape
        q = self.Wq(x).view(B, S, self.h, -1).transpose(1, 2)
        k, v = self.Wkv(x_kv).view(B, x_kv.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(o.transpose(1, 2).reshape(B, S, E))
'''
patch("src/model/unirig_skin.py",
      ("from flash_attn.modules.mha import MHA\n", MHA_SHIM),
      ("torch.autocast(device_type='cuda', dtype=torch.float32)",
       "torch.autocast(device_type='cpu', dtype=torch.float32)"))

# PTv3: serialized attention without flash_attn (3 constructors default enable_flash=True;
# the configs do not override it)
# spconv's CPU build asserts on the fused bias of an eval-mode conv ("cpu don't support act and
# bias"); the training=True code path adds the bias separately and is otherwise identical here.
CONV_CPU = '''

class SubMConv3dCPU(spconv.SubMConv3d):
    def forward(self, input, add_input=None):
        return self._conv_forward(
            True, input, self.weight, self.bias, add_input, name=self.name,
            sparse_unique_name=self._sparse_unique_name, act_type=self.act_type,
            act_alpha=self.act_alpha, act_beta=self.act_beta)
'''
patch("src/model/pointcept/models/PTv3Object.py",
      ("        enable_flash=True,\n", "        enable_flash=False,\n", 3),
      ("from .modules import PointModule, PointSequential\n",
       "from .modules import PointModule, PointSequential\n" + CONV_CPU),
      ("spconv.SubMConv3d(\n", "SubMConv3dCPU(\n", 2))

# Michelangelo shape encoder (skeleton model): drop the CUDA-only sdp_kernel context
patch("src/model/michelangelo/models/modules/transformer_blocks.py",
      ("torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)",
       "__import__('contextlib').nullcontext()"))
