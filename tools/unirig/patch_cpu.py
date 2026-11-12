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

# OPT-350m decoder: flash_attention_2 -> eager
patch("configs/model/unirig_ar_350m_1024_81920_float32.yaml",
      ("_attn_implementation: flash_attention_2", "_attn_implementation: eager"))

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
