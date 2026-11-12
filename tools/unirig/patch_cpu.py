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

# ---- RAM instrumentation: "MEM <label> rss=..GB hwm=..GB" lines go to stdout, hence into log_*_skin.txt ----
(root / "src/memlog.py").write_text(r"""import time
from collections import Counter

_T0 = time.time()


def _gb(key):
    try:
        for line in open('/proc/self/status'):
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return -1.0


def mem(label):
    print(f"MEM {label} rss={_gb('VmRSS'):.2f}GB hwm={_gb('VmHWM'):.2f}GB t={time.time() - _T0:.0f}s", flush=True)


def tsizes(label, obj, name='batch'):
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        nb = obj.numel() * obj.element_size() if hasattr(obj, 'numel') else obj.nbytes
        print(f"MEM {label} {name} shape={tuple(obj.shape)} dtype={obj.dtype} bytes={nb}", flush=True)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            tsizes(label, v, f"{name}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:8]):
            tsizes(label, v, f"{name}[{i}]")


def _nb(o):
    if hasattr(o, 'numel'):
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(_nb(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_nb(v) for v in o)
    return 0


def ckpt_report(ck, system=None):
    for k, v in ck.items():
        print(f"MEM ckpt key={k} bytes={_nb(v)} ({_nb(v) / 2**30:.2f}GB)", flush=True)
    sd = ck.get('state_dict') or {}
    n = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
    print(f"MEM ckpt state_dict tensors={len(sd)} params={n} fp32_bytes={4 * n} disk_bytes={_nb(sd)} "
          f"disk_dtypes={dict(Counter(str(v.dtype) for v in sd.values() if hasattr(v, 'dtype')))}", flush=True)
    if system is not None:
        ps, bs = list(system.parameters()), list(system.buffers())
        print(f"MEM model params={sum(p.numel() for p in ps)} param_dtypes={dict(Counter(str(p.dtype) for p in ps))} "
              f"buffers={len(bs)} buffer_bytes={sum(_nb(b) for b in bs)} buffer_dtypes={dict(Counter(str(b.dtype) for b in bs))}", flush=True)
""")
print("wrote src/memlog.py")

# The skin checkpoint is 4.4 GB for ~1.5 GB of weights: it still holds Adam optimizer state. Stock flow =
# Lightning torch.load()s the WHOLE file (no mmap) next to the live model. Here: mmap-load once (untouched
# optimizer pages never become resident), copy only state_dict into the system with the same strict
# load_state_dict Lightning would call, free it, and tell trainer.predict not to reload (ckpt_path=None).
# Same tensors -> same weights -> same outputs. Only when optimizer_states exist (skin); the weights-only
# skeleton ckpt keeps the stock path (the code below just logs for it).
patch("run.py",
      ("import argparse\nimport yaml\n",
       "import argparse\nfrom src.memlog import mem, tsizes, ckpt_report\nmem('run.py top (before torch import)')\nimport yaml\n"),
      ("if __name__ == \"__main__\":\n    torch.set_float32_matmul_precision('high')\n",
       "if __name__ == \"__main__\":\n    mem('after imports')\n    torch.set_float32_matmul_precision('high')\n"),
      # voxel_skin (the big scipy graph) is the transient peak and needs nothing from the model: run the
      # dataset transform BEFORE the model exists so the two never overlap in RSS. Same call, same numpy RNG
      # stream (only torch RNG is used by model init), result is handed to the dataset via PRELOADED.
      ("    # get model\n    model_config = task.components.get('model', None)\n",
       "    if mode == 'predict' and datapath is not None and tokenizer_config is None:\n"
       "        import src.data.dataset as _dsm\n"
       "        try:\n"
       "            _pre = _dsm.UniRigDataset(process_fn=lambda b: b, data=datapath.get_data(), name='pre',\n"
       "                                      tokenizer=None, transform_config=predict_transform_config, data_name=data_name)\n"
       "            for _i in range(len(_pre)):\n"
       "                _dsm.PRELOADED[_i] = _pre[_i]\n"
       "            del _pre\n"
       "            mem('after preload transform (before model)')\n"
       "        except Exception as e:\n"
       "            print('preload skipped:', e)\n"
       "            _dsm.PRELOADED.clear()\n"
       "    # get model\n    model_config = task.components.get('model', None)\n"),
      ("    wandb_config = task.get('wandb', None)\n",
       "    mem('after config/model/system construction')\n    wandb_config = task.get('wandb', None)\n"),
      ("    resume_from_checkpoint = download(resume_from_checkpoint)\n",
       "    resume_from_checkpoint = download(resume_from_checkpoint)\n"
       "    _preloaded = False\n"
       "    try:\n"
       "        mem('before ckpt torch.load')\n"
       "        _ck = torch.load(resume_from_checkpoint, map_location='cpu', mmap=True, weights_only=False)\n"
       "        ckpt_report(_ck, system)\n"
       "        mem('after ckpt torch.load (mmap)')\n"
       "        if _ck.get('optimizer_states') and system is not None:\n"
       "            system.load_state_dict(_ck['state_dict'], strict=True)\n"
       "            _preloaded = True\n"
       "            print('state_dict loaded directly (optimizer state never read)')\n"
       "            mem('after load_state_dict')\n"
       "        del _ck\n"
       "        import gc, ctypes\n"
       "        gc.collect()\n"
       "        ctypes.CDLL('libc.so.6').malloc_trim(0)\n"
       "        mem('after ckpt freed')\n"
       "    except Exception as e:\n"
       "        print('ckpt preload skipped:', e)\n"),
      ("    if mode == 'train':\n        trainer.fit(",
       "    mem('after Trainer creation')\n    if mode == 'train':\n        trainer.fit("),
      ("trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)\n",
       "trainer.predict(system, datamodule=data, ckpt_path=None if _preloaded else resume_from_checkpoint, return_predictions=False)\n"
       "        mem('after trainer.predict')\n"))

# dataset: MEM at setup / transform / first batch (with per-tensor sizes); PRELOADED (see run.py) short-circuits __getitem__
patch("src/data/dataset.py",
      ("from .spec import ConfigSpec\n",
       "from .spec import ConfigSpec\nfrom ..memlog import mem, tsizes\n\nPRELOADED = {}  # idx -> ModelInput transformed before the model was built\n"),
      ("        if self.predict_datapath is not None:\n            self._predict_ds = {}\n",
       "        mem('datamodule setup')\n        if self.predict_datapath is not None:\n            self._predict_ds = {}\n"),
      ("        cls, dir_path = self.data[idx]\n",
       "        if idx in PRELOADED:\n            mem('dataset item taken from preload')\n            return PRELOADED.pop(idx)\n        cls, dir_path = self.data[idx]\n"),
      ("        first_augments, second_augments = transform_asset(\n            asset=asset,\n            transform_config=self.transform_config,\n        )\n",
       "        mem('dataset: before transform_asset')\n"
       "        first_augments, second_augments = transform_asset(\n            asset=asset,\n            transform_config=self.transform_config,\n        )\n"
       "        mem('dataset: after transform_asset')\n"),
      ("        return data.dataloader.default_collate(self.process_fn(batch))\n",
       "        out = data.dataloader.default_collate(self.process_fn(batch))\n"
       "        tsizes('first batch', out)\n        mem('first batch built')\n        return out\n"))

# voxel_skin, same math and same operation order:
#  - int32 grid indices (values < 196, float32 cast of them is exact) instead of 3x 7.5M int64 arrays
#  - np.nonzero(mask)[0] == np.repeat(arange(M), k)[mask.ravel()] (row-major order), without the two temporaries
#  - edge arrays / kd-trees freed once the CSR graph exists
#  - Dijkstra is per source, so running 8 joints at a time and keeping only the [:, :N] mesh-vertex columns
#    gives identical numbers but never holds the (J, N+M) float64 matrix (M = interior voxels, often millions)
patch("src/data/vertex_group.py",
      ("from .spec import ConfigSpec\n", "from .spec import ConfigSpec\nfrom ..memlog import mem\n"),
      ("        grid_coords = voxelization(\n", "        mem('voxel_skin: before voxelization')\n        grid_coords = voxelization(\n"),
      ("        skin = voxel_skin(\n",
       "        mem(f'voxel_skin: voxelization done M={len(grid_coords)} N={len(normalized_vertices)} J={len(normalized_joints)} F={len(asset.faces)}')\n        skin = voxel_skin(\n"),
      ("        skin = np.nan_to_num(skin, nan=0., posinf=0., neginf=0.)\n",
       "        mem('voxel_skin: done')\n        skin = np.nan_to_num(skin, nan=0., posinf=0., neginf=0.)\n"),
      ("        i, j, k = np.indices((grid, grid, grid))\n",
       "        i, j, k = np.indices((grid, grid, grid), dtype=np.int32)\n"),
      ("k.ravel()), axis=1, dtype=np.int64)\n", "k.ravel()), axis=1, dtype=np.int32)\n"),
      ("    source_grid2grid = np.repeat(np.arange(M), grid_query-1)[mask.ravel()] + N\n",
       "    source_grid2grid = np.nonzero(mask)[0] + N\n"),
      ("    # make combined vertices\n", "    mem('voxel_skin: trees built')\n    # make combined vertices\n"),
      ("    # get shortest path (J, N+M)\n    dist_matrix = shortest_path(graph, method='D', directed=False, indices=joint_indices)\n",
       "    del (source_grid2grid, to_grid2grid, weight_grid2grid, source_close, to_close, weight_close, source_grid2vertex,\n"
       "         to_grid2vertex, weight_grid2vertex, source_vertex2vertex, to_vertex2vertex, weight_vertex2vertex,\n"
       "         dist, idx, mask, grid_tree, vertex_tree, combined_tree, combined_vertices)\n"
       "    mem(f'voxel_skin: graph built nnz={graph.nnz}')\n"
       "    # get shortest path (J, N), mesh-vertex columns only\n"
       "    parts = []\n"
       "    for a in range(0, len(joint_indices), 8):\n"
       "        d = shortest_path(graph, method='D', directed=False, indices=joint_indices[a:a + 8])\n"
       "        parts.append(d[:, :N].copy())\n"
       "        del d\n"
       "    dist_matrix = np.concatenate(parts, axis=0)\n"
       "    del parts, graph\n"
       "    mem('voxel_skin: dijkstra done')\n"))

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

# MEM points inside the skin forward (HWM after each stage shows which one raised the peak)
patch("src/model/unirig_skin.py",
      ("from .spec import ModelSpec, ModelInput\n", "from .spec import ModelSpec, ModelInput\nfrom ..memlog import mem\n"),
      ("        global_latents = self.encode_mesh_cond(vertices, normals)\n",
       "        mem(f'fwd start N={N} J={J}')\n        global_latents = self.encode_mesh_cond(vertices, normals)\n        mem('fwd after global (Michelangelo) encoder')\n"),
      ("        if isinstance(self.mesh_encoder, MAP_MESH_ENCODER.ptv3obj):\n            feat = torch.cat(",
       "        mem('fwd after bone encoder')\n        if isinstance(self.mesh_encoder, MAP_MESH_ENCODER.ptv3obj):\n            feat = torch.cat("),
      ("        # (B, J + seq_len, feat_dim)\n        latents = torch.cat([bone_feat, global_latents], dim=1)\n",
       "        mem('fwd after PTv3 mesh encoder')\n        # (B, J + seq_len, feat_dim)\n        latents = torch.cat([bone_feat, global_latents], dim=1)\n"),
      ("        # trans to (B, num_heads, J, feat_dim)\n",
       "        mem('fwd after mesh_bone_attn x16')\n        # trans to (B, num_heads, J, feat_dim)\n"),
      ("        skin_pred_list = torch.cat(skin_pred_list, dim=1)\n",
       "        mem('fwd after skin chunk loop')\n        skin_pred_list = torch.cat(skin_pred_list, dim=1)\n"),
      ("        o = F.scaled_dot_product_attention(q, k, v)\n",
       "        o = F.scaled_dot_product_attention(q, k, v)\n"
       "        if not getattr(MHA, '_logged', False):\n            MHA._logged = True\n            mem(f'MHA first call (q {S} x kv {x_kv.shape[1]})')\n"))

patch("src/system/skin.py",
      ("from ..model.spec import ModelSpec\n", "from ..model.spec import ModelSpec\nfrom ..memlog import mem\n"),
      ("        res = self.model.predict_step(batch)\n",
       "        mem('predict_step: before model')\n        res = self.model.predict_step(batch)\n        mem('predict_step: after model')\n"),
      ("            skin_resampled = reskin(\n", "            mem('writer: before reskin')\n            skin_resampled = reskin(\n"),
      ("            def make_path(save_name: str, suffix: str, trim: bool=False):\n",
       "            mem('writer: after reskin')\n            def make_path(save_name: str, suffix: str, trim: bool=False):\n"),
      ("                    exporter = Exporter()\n                    names = RawData.load(",
       "                    mem('writer: before fbx export')\n                    exporter = Exporter()\n                    names = RawData.load("),
      ("                except Exception as e:\n                    print(str(e))\n",
       "                except Exception as e:\n                    print(str(e))\n                mem('writer: after fbx export')\n"))
