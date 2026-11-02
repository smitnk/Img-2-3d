"""Export SkinTokens / TokenRig (VAST-AI-Research, MIT) to ONNX for the phone and verify against PyTorch.

Run inside a SkinTokens checkout (CPU-patched like .github/workflows/skintokens-probe.yml):
  python export_skintokens.py export <mesh.glb> <out_dir>     export + verify (needs torch)
  python export_skintokens.py onnx <out_dir>                  ONNX-only greedy rig of the saved inputs (RAM probe)

Graphs (all fp32 except the int8 LLM):
  mesh_cond.onnx  pc[1,4*T1,3], feats[1,4*T1,3] -> mesh tokens [1,L,D]      (michelangelo encoder + output_proj)
  vae_cond.onnx   cond[1,N,6] -> cond_latents [1,T2,C]                       (skin VAE condition encoder)
  llm.onnx / llm_q.onnx  embeds[1,1,D], position[1,1], past k/v per layer -> logits[1,1,V], present k/v
  skin_dec.onnx   indices[m,K] int64, cond[1,N,6], cond_latents[1,T2,C] -> skin logits [m,N]
Point sampling is outside the graphs: the original picks 4*T random points then farthest-point-samples T of them;
here the caller orders the cloud so those T come first (graphs were traced with identity sampling), see order().
"""
import json
import os
import resource
import sys
import time

import numpy as np

STEP_PAD = 2048


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def fps(points, k):
    """Farthest point sampling from index 0 (the original's eval mode: random_start=False)."""
    idx = np.zeros(k, np.int64)
    d = np.full(len(points), np.inf)
    for i in range(1, k):
        d = np.minimum(d, ((points - points[idx[i - 1]]) ** 2).sum(1))
        idx[i] = int(np.argmax(d))
    return idx


def order(points, t, seed=0):
    """Permutation of the cloud: [T farthest-point picks of 4T random points][the other 3T][the rest]."""
    n = len(points)
    pick = np.random.default_rng(seed).choice(n, 4 * t, replace=4 * t > n)
    f = fps(points[pick], t)
    first = pick[f]
    rest4 = np.delete(pick, f)
    others = np.setdiff1d(np.arange(n), pick)
    return np.concatenate([first, rest4, others])


def greedy(step, embed_rows, prefix, tokenizer, eos, k_skin, start, max_len=STEP_PAD, penalty=2.0, log=print):
    """HF generate(do_sample=False) with repetition penalty + VocabSwitchingLogitsProcessor, one token per step.
    step(embeds[1,1,D], pos) -> logits[V]; prefix: [L,D] mesh tokens followed by start-token embeddings."""
    for p, e in enumerate(prefix[:-1]):
        step(e[None, None], p)
    logits = step(prefix[-1][None, None], len(prefix) - 1)
    out = []
    t0 = time.time()
    while len(out) < max_len:
        s = logits.astype(np.float64).copy()
        if out:  # RepetitionPenaltyLogitsProcessor over generated ids
            u = np.unique(out)
            s[u] = np.where(s[u] < 0, s[u] * penalty, s[u] / penalty)
        seq = np.array(list(start) + out)
        mask = np.full_like(s, -np.inf)
        if eos_switch(tokenizer) in seq:
            mask[eos_switch(tokenizer):] = 0
            where = int(np.nonzero(seq == eos_switch(tokenizer))[0][0])
            J = tokenizer.bones_in_sequence(ids=seq)
            if len(seq) - where == J * k_skin:
                mask[:] = -np.inf
                mask[eos] = 0
            else:
                mask[eos] = -np.inf
        else:
            mask[tokenizer.next_posible_token(ids=seq)] = 0
        nxt = int(np.argmax(s + mask))
        out.append(nxt)
        if nxt == eos:
            break
        logits = step(embed_rows[nxt][None, None], len(prefix) + len(out) - 1)
    log(f"greedy: {len(out)} tokens, {len(out) / max(time.time() - t0, 1e-9):.1f} tok/s")
    return np.array(list(start) + out)


def eos_switch(tokenizer):
    return tokenizer.eos  # skeleton ends at the tokenizer's eos; skin tokens follow until the model's eos


def ort_llm(path, n_layers, kv_heads, head_dim, threads=4):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    s = ort.InferenceSession(path, o)
    past = [np.zeros((1, kv_heads, 0, head_dim), np.float32) for _ in range(2 * n_layers)]

    def step(e, pos):
        feeds = {"embeds": e.astype(np.float32), "position": np.array([[pos]], np.int64)}
        feeds.update({f"past_{i}": p for i, p in enumerate(past)})
        r = s.run(None, feeds)
        past[:] = r[1:]
        return r[0][0, -1]
    return step


def main_export(mesh, out):
    import torch
    sys.path.insert(0, os.getcwd())
    import demo
    from src.model import tokenrig as TR
    import src.model.michelangelo.models.tsal.sal_perceiver as SP
    import src.model.skin_vae.autoencoders.skin_fsq_cvae_model as SV
    from src.data.dataset import DatasetConfig, RigDatasetModule

    os.makedirs(out, exist_ok=True)
    demo.start_bpy_server()
    time.sleep(10)
    demo.load_model(demo.MODEL_CKPTS[0], None)
    model, tok = demo.model.eval().requires_grad_(False), demo.tokenizer  # traced weights must not require grad
    cfg = DatasetConfig.parse(shuffle=False, batch_size=1, num_workers=1, pin_memory=False, persistent_workers=False,
                              datapath={"data_name": None, "loader": "bpy_server",
                                        "filepaths": {"articulation": [mesh]}}).split_by_cls()
    batch = next(iter(RigDatasetModule(predict_dataset_config=cfg, predict_transform=demo.transform, tokenizer=tok,
                                       process_fn=model._process_fn).predict_dataloader()["articulation"]))
    V, Nn = batch["vertices"][0].float(), batch["normals"][0].float()
    cls = batch["cls"][0] if batch.get("cls") is not None else None
    T1 = model.mesh_encoder.encoder.token_num
    T2, K = model.tokens_skin_cond, model.tokens_per_skin
    lm = model.transformer
    cfgl = lm.config
    L, KVH = cfgl.num_hidden_layers, cfgl.num_key_value_heads
    HD = getattr(cfgl, "head_dim", cfgl.hidden_size // cfgl.num_attention_heads)
    print(f"N={len(V)} T1={T1} T2={T2} K={K} layers={L} kv_heads={KVH} head_dim={HD} vocab={cfgl.vocab_size} "
          f"hidden={cfgl.hidden_size} params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    vn = V.numpy()
    o1, o2 = order(vn, T1), order(vn, T2)
    cond = torch.cat([V, Nn], -1)

    # identity sampling: graphs and the patched reference take pre-ordered clouds
    class IdRng:
        def choice(self, n, k, replace=False):
            return np.arange(k)
    real_rng = np.random.default_rng
    np.random.default_rng = lambda seed=None: IdRng()
    idfps = lambda pos, batch, ratio=0.25, random_start=False: torch.arange(int(pos.shape[0]) // 4)  # ratio is always 1/4
    SP.fps, SV.fps = idfps, idfps
    # aten::rms_norm has no opset-17 ONNX export (and callers bind it directly): register the decomposition
    from torch.onnx import register_custom_op_symbolic, symbolic_helper as sh

    def rms_sym(g, x, shape, weight, eps):
        e = sh._maybe_get_const(eps, "f")
        e = 1.1920929e-07 if e is None or sh._is_none(eps) else float(e)
        ms = g.op("ReduceMean", g.op("Mul", x, x), axes_i=[-1], keepdims_i=1)
        y = g.op("Mul", x, g.op("Reciprocal", g.op("Sqrt", g.op("Add", ms, g.op("Constant", value_t=torch.tensor(e))))))
        return y if sh._is_none(weight) else g.op("Mul", y, weight)

    register_custom_op_symbolic("aten::rms_norm", rms_sym, 17)

    class MeshCond(torch.nn.Module):
        def forward(self, pc, feats):
            return TR.encode_mesh_cond(model.mesh_encoder, model.output_proj, T2, {"vertices": pc, "normals": feats})

    class VaeCond(torch.nn.Module):
        def forward(self, c):
            return model.vae.model._encode(x=None, cond=c, num_tokens=K, cond_tokens=T2, return_z=False)[1]

    class SkinDec(torch.nn.Module):
        def forward(self, indices, c, cl):
            m = indices.shape[0]
            z = model.vae.model.FSQ.indices_to_codes(indices).reshape(m, K, -1)
            return model.vae.decode(z=z, sampled_cond=c.repeat(m, 1, 1), cond_tokens=cl.repeat(m, 1, 1)).reshape(m, -1)

    from transformers import DynamicCache

    class Step(torch.nn.Module):
        def forward(self, embeds, position, *past):
            cache = DynamicCache()
            for i in range(L):
                cache.update(past[2 * i], past[2 * i + 1], i)
            r = lm(inputs_embeds=embeds, position_ids=position, past_key_values=cache, use_cache=True)
            pk = r.past_key_values
            kv = []
            for i in range(L):
                if hasattr(pk, "layers"):
                    kv += [pk.layers[i].keys, pk.layers[i].values]
                else:
                    kv += [pk.key_cache[i], pk.value_cache[i]]
            return (r.logits,) + tuple(kv)

    pc1 = (V[o1[:4 * T1]][None], Nn[o1[:4 * T1]][None])
    c2 = cond[o2][None]
    with torch.no_grad():
        mc_ref = MeshCond()(*pc1)
        cl_ref = VaeCond()(c2)
        t0 = time.time()
        torch.onnx.export(MeshCond(), pc1, f"{out}/mesh_cond.onnx", opset_version=17, input_names=["pc", "feats"],
                          output_names=["tokens"])
        torch.onnx.export(VaeCond(), (c2,), f"{out}/vae_cond.onnx", opset_version=17, input_names=["cond"],
                          output_names=["cond_latents"], dynamic_axes={"cond": {1: "n"}})
        idx = torch.zeros((2, K), dtype=torch.long)
        torch.onnx.export(SkinDec(), (idx, cond[None], cl_ref), f"{out}/skin_dec.onnx", opset_version=17,
                          input_names=["indices", "cond", "cond_latents"], output_names=["skin"],
                          dynamic_axes={"indices": {0: "m"}, "cond": {1: "n"}, "skin": {0: "m", 1: "n"}})
        past = tuple(torch.zeros((1, KVH, 3, HD)) for _ in range(2 * L))
        names = [f"past_{i}" for i in range(2 * L)]
        torch.onnx.export(Step(), (torch.zeros((1, 1, cfgl.hidden_size)), torch.tensor([[3]]), *past), f"{out}/llm_tmp.onnx",
                          opset_version=17, input_names=["embeds", "position"] + names,
                          output_names=["logits"] + [f"present_{i}" for i in range(2 * L)],
                          dynamic_axes={**{n: {2: "p"} for n in names}, **{f"present_{i}": {2: "q"} for i in range(2 * L)}})
        print(f"exported in {time.time() - t0:.0f}s, rss {rss_mb()} MB")
    np.random.default_rng = real_rng

    import onnx
    m = onnx.load(f"{out}/llm_tmp.onnx")
    onnx.save_model(m, f"{out}/llm.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="llm.onnx.data")
    del m
    for f in os.listdir(out):
        if f.startswith("llm_tmp") or (f not in ("llm.onnx", "llm.onnx.data") and "." not in f):
            os.remove(os.path.join(out, f))
    # quantize in a child process so its peak doesn't stack on the live torch model (runner was killed here)
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, "-c", "from onnxruntime.quantization import QuantType as Q, quantize_dynamic as qd; "
                    f"qd('{out}/llm.onnx', '{out}/llm_q.onnx', weight_type=Q.QInt8, op_types_to_quantize=['MatMul','Gemm'])"],
                   check=True)
    print(f"quantized, rss {rss_mb()} MB", flush=True)

    import onnxruntime as ort
    mc = ort.InferenceSession(f"{out}/mesh_cond.onnx").run(None, {"pc": pc1[0].numpy(), "feats": pc1[1].numpy()})[0]
    cl = ort.InferenceSession(f"{out}/vae_cond.onnx").run(None, {"cond": c2.numpy()})[0]
    report = {"N": len(V), "T1": T1, "T2": T2, "K": K, "layers": L, "kv_heads": KVH, "head_dim": HD,
              "vocab": cfgl.vocab_size, "hidden": cfgl.hidden_size, "eos": int(model.eos), "tok_eos": int(tok.eos),
              "tok_vocab": int(tok.vocab_size), "bos": int(tok.bos), "pad": int(tok.pad),
              "mesh_cond_maxdiff": float(np.abs(mc - mc_ref.numpy()).max()),
              "vae_cond_maxdiff": float(np.abs(cl - cl_ref.numpy()).max())}
    print(json.dumps(report))

    embed = lm.get_input_embeddings().weight.detach().numpy().astype(np.float32)
    start = model.make_start_tokens(cls=[cls] if cls is not None else None)[0] if cls is not None else [tok.bos] + tok.make_cls_head(cls=None)
    prefix = np.concatenate([mc[0], embed[start]], 0)

    # PyTorch reference: the same greedy loop on the torch step (identical sampling, so ids must agree)
    def torch_step():
        cache = DynamicCache()

        def step(e, pos):
            with torch.no_grad():
                r = lm(inputs_embeds=torch.from_numpy(e).float(), position_ids=torch.tensor([[pos]]), past_key_values=cache, use_cache=True)
            return r.logits[0, -1].numpy()
        return step

    ids = {}
    for name, step in [("torch", torch_step()), ("onnx", ort_llm(f"{out}/llm.onnx", L, KVH, HD)),
                       ("onnx_q", ort_llm(f"{out}/llm_q.onnx", L, KVH, HD))]:
        t0 = time.time()
        ids[name] = greedy(step, embed, prefix, tok, int(model.eos), K, start)
        report[f"{name}_tokens"] = int(len(ids[name]))
        report[f"{name}_seconds"] = round(time.time() - t0, 1)
        print(name, report[f"{name}_tokens"], "tokens", report[f"{name}_seconds"], "s, rss", rss_mb(), "MB")
    for name in ("onnx", "onnx_q"):
        n = min(len(ids[name]), len(ids["torch"]))
        report[f"{name}_same_prefix"] = int(np.argmin(np.append(ids[name][:n] == ids["torch"][:n], False)))
    # decode skins: torch reference vs skin_dec.onnx on the int8 ids
    dec = ort.InferenceSession(f"{out}/skin_dec.onnx")
    for name in ("torch", "onnx_q"):
        d = TR.decode(cond=cond, cond_latents=torch.from_numpy(cl[0]), inputs_ids=torch.from_numpy(ids[name]),
                      tokenizer=tok, tokens_per_skin=K, vae=model.vae)
        report[f"{name}_joints"] = int(d["detokenize_output"].joints.shape[0])
        if d["skin_pred"] is not None and name == "onnx_q":
            seq = ids[name]
            where = int(np.nonzero(seq == tok.eos)[0][0])
            J = report[f"{name}_joints"]
            skin_tok = seq[where + 1:where + 1 + J * K].reshape(J, K) - tok.vocab_size
            s = np.concatenate([dec.run(None, {"indices": skin_tok[i:i + 8].astype(np.int64), "cond": cond[None].numpy(),
                                               "cond_latents": cl})[0] for i in range(0, J, 8)], 0).T
            report["skin_dec_maxdiff"] = float(np.abs(s - d["skin_pred"].numpy()).max())
    np.savez(f"{out}/reference.npz", points=cond.numpy(), o1=o1, o2=o2, start=np.array(start), ids_q=ids["onnx_q"],
             embed=embed.astype(np.float16))
    embed.astype(np.float16).tofile(f"{out}/embed_f16.bin")
    json.dump({**report, "tokenizer_config": model.tokenizer_config}, open(f"{out}/skintokens_info.json", "w"), indent=1,
              default=str)
    print("REPORT", json.dumps(report))


def main_onnx(out):
    """Phone-like run: only ONNX Runtime + numpy + the python tokenizer, int8 LLM. Peak RSS = RAM estimate."""
    sys.path.insert(0, os.getcwd())
    import onnxruntime as ort
    from src.tokenizer.parse import get_tokenizer
    info = json.load(open(f"{out}/skintokens_info.json"))
    tok = get_tokenizer(**info["tokenizer_config"]) if isinstance(info["tokenizer_config"], dict) else None
    r = np.load(f"{out}/reference.npz")
    pts = r["points"]
    o1, o2 = r["o1"], r["o2"]
    T1 = info["T1"]
    t0 = time.time()
    mc = ort.InferenceSession(f"{out}/mesh_cond.onnx").run(None, {"pc": pts[o1[:4 * T1], :3][None], "feats": pts[o1[:4 * T1], 3:][None]})[0]
    cl = ort.InferenceSession(f"{out}/vae_cond.onnx").run(None, {"cond": pts[o2][None]})[0]
    print(f"encoders {time.time() - t0:.1f}s rss {rss_mb()} MB")
    embed = np.fromfile(f"{out}/embed_f16.bin", np.float16).reshape(info["vocab"], info["hidden"]).astype(np.float32)
    start = r["start"].tolist()
    step = ort_llm(f"{out}/llm_q.onnx", info["layers"], info["kv_heads"], info["head_dim"])
    ids = greedy(step, embed, np.concatenate([mc[0], embed[start]], 0), tok, info["eos"], info["K"], start)
    print(f"llm done rss {rss_mb()} MB, same as export run: {np.array_equal(ids, r['ids_q'])}")
    print(f"ONNX_ONLY total {time.time() - t0:.0f}s peak {rss_mb()} MB")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        main_export(sys.argv[2], sys.argv[3])
    else:
        main_onnx(sys.argv[2])
