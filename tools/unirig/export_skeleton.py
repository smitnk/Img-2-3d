"""Export UniRig's SKELETON stage to ONNX (fp32) and prove it against the PyTorch run it replaces.

Run from the CPU-patched UniRig checkout (tools/unirig/patch_cpu.py), in place of `python run.py ...`:
  SKEL_OUT=<dir> SKEL_NAME=<mesh> python export_skeleton.py --task=... --seed=12345 --input=... --output=res/skel_x.fbx --npz_dir=tmp
It runs run.py unchanged, but wraps UniRigAR.generate (called once). Inside, with the model and the real sampled point cloud at hand:
  1. torch beam search = the reference (its token ids are recorded)
  2. export skel_encoder.onnx + skel_decoder.onnx (first mesh only; the graphs have dynamic point count / batch / past length)
  3. encoder: torch (with torch_cluster FPS) vs ORT (numpy FPS), max abs diff of the 1024 latents
  4. greedy decoding (num_beams=1, do_sample=False): HF generate vs the ORT graph, teacher-forced on torch's tokens ->
     per-step raw-logit diff (first 32 steps and all) and argmax agreement
  5. the ORT beam search (onnx_skeleton.beam_sample) with torch's RNG state restored and its exponential noise drawn from torch ->
     expected to give the same tokens as step 1 (compared; any float near-tie flip is reported as first mismatch)
  generate returns the ORT result, so the pipeline writes res/skel_x.fbx from ONNX; ab_check.py then compares it to the reference fbx.
Files in SKEL_OUT: skel_encoder.onnx, skel_decoder.onnx(+data), embed.npy, info.json, inputs_<name>.npz, report_<name>.json
"""
import gc
import json
import os
import runpy
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_skeleton as R  # noqa: E402

OUT = os.environ.get("SKEL_OUT", "onnx")
NAME = os.environ.get("SKEL_NAME", "mesh")


def chunked_cross_attention(torch, chunk=64):
    """Replacement forward for Michelangelo's QKVMultiheadCrossAttention (non-flash path), same maths, lower peak memory.
    The original builds the full [heads, Q, N] score tensor (8 x 1024 x 65536 fp32 = 2 GB) plus a softmax copy. Softmax is per
    query row, so processing `chunk` query rows at a time gives the same rows (only matmul tiling may differ in the last bit).
    Q is unrolled at trace time (it is the fixed token_num=1024); N stays dynamic."""
    import math

    def forward(self, q, kv):
        bs, n_ctx, _ = q.shape
        n_data, width = kv.shape[1], kv.shape[2]
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        k, v = torch.split(kv.view(bs, n_data, self.heads, -1), attn_ch, dim=-1)
        qh = (q * scale).transpose(1, 2)  # [b,h,t,c]
        kh = (k * scale).permute(0, 2, 3, 1)  # [b,h,c,s]
        vh = v.transpose(1, 2)  # [b,h,s,c]
        outs = [torch.softmax(a @ kh, dim=-1) @ vh for a in torch.split(qh, chunk, dim=2)]  # each [b,h,chunk,c]
        return torch.cat(outs, 2).transpose(1, 2).reshape(bs, n_ctx, -1)

    return forward


def export_encoder(model, V, Nn, torch):
    class Enc(torch.nn.Module):  # CrossAttentionEncoder._forward after the (outside, numpy) sampling of the queries
        def forward(self, pc, feats, spc, sfeats):
            e = model.mesh_encoder.encoder
            assert e.use_full_input and not e.query_method
            data = e.input_proj(torch.cat([e.fourier_embedder(pc), feats], -1))
            q = e.input_proj(torch.cat([e.fourier_embedder(spc), sfeats], -1))
            lat = e.self_attn(e.cross_attn(q, data))
            if e.ln_post is not None:
                lat = e.ln_post(lat)
            return model.output_proj(lat)

    spc, sf = R.sample_queries(V, Nn)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
    # patch only for the export: the torch reference (encode_mesh_cond) must keep the original attention
    import types
    mods = [m for m in model.mesh_encoder.encoder.modules() if type(m).__name__ == "QKVMultiheadCrossAttention"]
    assert mods, "cross attention module not found"
    for m in mods:
        m.forward = types.MethodType(chunked_cross_attention(torch), m)
    try:
        torch.onnx.export(Enc().eval(), (t(V), t(Nn), t(spc), t(sf)), f"{OUT}/skel_encoder.onnx", opset_version=17,
                          input_names=["pc", "feats", "spc", "sfeats"], output_names=["cond"],
                          dynamic_axes={"pc": {1: "n"}, "feats": {1: "n"}, "spc": {1: "q"}, "sfeats": {1: "q"}, "cond": {1: "q"}})
    finally:
        for m in mods:
            del m.forward


def export_decoder(lm, L, Hh, Dh, D, torch):
    from transformers import DynamicCache

    class LazyCache(DynamicCache):
        """Prompt K/V (batch 1) + generated K/V (per beam) are joined one layer at a time, inside that layer's attention, so only
        one layer's expanded [b,H,p+g+1,Dh] K and V exist at once (the eager version built all 24 layers x 15 beams up front: ~3 GB)."""
        def __init__(self, pk, gk):
            super().__init__()
            self.pk, self.gk, self.new = pk, gk, {}

        def update(self, k, v, i, cache_kwargs=None):
            self.new[i] = (k, v)
            b = k.shape[0]
            return (torch.cat([self.pk[2 * i].expand(b, -1, -1, -1), self.gk[2 * i], k], 2),
                    torch.cat([self.pk[2 * i + 1].expand(b, -1, -1, -1), self.gk[2 * i + 1], v], 2))

        def get_seq_length(self, layer_idx=0):
            return self.pk[0].shape[2] + self.gk[0].shape[2]

    class Step(torch.nn.Module):
        def forward(self, embeds, position, *kv):
            cache = LazyCache(kv[:2 * L], kv[2 * L:])
            r = lm(inputs_embeds=embeds, position_ids=position, past_key_values=cache, use_cache=True)
            new = []
            for i in range(L):
                new += list(cache.new[i])
            return (r.logits[:, 0],) + tuple(new)

    n = 2 * L
    pk = [f"pk{i}" for i in range(n)]
    gk = [f"gk{i}" for i in range(n)]
    nk = [f"nk{i}" for i in range(n)]
    args = (torch.randn(2, 1, D), torch.tensor([[7], [7]]), *[torch.randn(1, Hh, 3, Dh) for _ in range(n)],
            *[torch.randn(2, Hh, 2, Dh) for _ in range(n)])
    torch.onnx.export(Step().eval(), args, f"{OUT}/skel_decoder.onnx", opset_version=17,
                      do_constant_folding=False,  # folding copies every weight
                      input_names=["embeds", "position"] + pk + gk, output_names=["logits"] + nk,
                      dynamic_axes={"embeds": {0: "b"}, "position": {0: "b"}, "logits": {0: "b"},
                                    **{x: {2: "p"} for x in pk}, **{x: {0: "b", 2: "g"} for x in gk}, **{x: {0: "b"} for x in nk}})


def experiments(model, ref, ref_ids, rng_state, vertices, normals, cls, kw, torch, U):
    from transformers import LogitsProcessorList
    rep = {"name": NAME}
    os.makedirs(OUT, exist_ok=True)
    tok, lm = model.tokenizer, model.transformer
    model.eval().requires_grad_(False)
    V = vertices.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
    Nn = normals.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
    cfgl = lm.config
    L, Hh, D = cfgl.num_hidden_layers, cfgl.num_attention_heads, cfgl.hidden_size
    start = [int(tok.bos)] + ([int(tok.cls_name_to_token(cls))] if cls is not None else [])
    embed = lm.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    info = dict(layers=L, heads=Hh, head_dim=D // Hh, hidden=D, vocab=int(embed.shape[0]), eos=int(tok.eos), bos=int(tok.bos),
                pad=int(tok.pad), start=start, points=len(V),
                **{k: kw[k] for k in ("max_new_tokens", "repetition_penalty", "num_beams", "temperature", "top_k", "top_p")})
    rep.update(points=len(V), prompt_tokens=model.mesh_encoder.encoder.token_num + len(start))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    with torch.inference_mode(False), torch.no_grad():
        # ---- export (once; later meshes reuse the graphs, which also proves they are not tied to one input) ----
        t0 = time.time()
        if not os.path.exists(f"{OUT}/skel_encoder.onnx"):
            export_encoder(model, V, Nn, torch)
            gc.collect()
            print(f"encoder exported {time.time() - t0:.0f}s", flush=True)
        if not os.path.exists(f"{OUT}/skel_decoder.onnx"):
            export_decoder(lm, L, Hh, D // Hh, D, torch)
            gc.collect()
            print(f"decoder exported {time.time() - t0:.0f}s", flush=True)
        np.save(f"{OUT}/embed.npy", embed)
        json.dump(info, open(f"{OUT}/info.json", "w"), indent=1)
        rep["sizes_bytes"] = {f: os.path.getsize(f"{OUT}/{f}") for f in sorted(os.listdir(OUT)) if f.startswith("skel_")}

        # ---- 3. encoder: torch (real path, torch_cluster FPS) vs ORT (numpy FPS) ----
        from torch_cluster import fps as tc_fps
        ind = np.random.default_rng(0).choice(len(V), 4096, replace=4096 > len(V))
        idx_t = tc_fps(t(V[ind]), torch.zeros(4096, dtype=torch.long), ratio=0.25, random_start=False).numpy()
        rep["fps_idx_equal"] = bool(np.array_equal(idx_t, R.fps_np(V[ind], 1024)))
        cond_t = model.encode_mesh_cond(vertices=t(V), normals=t(Nn)).detach().numpy()[0]
        t0 = time.time()
        inp = dict(vertices=V, normals=Nn)
        cond_o = R.encode(OUT, inp)
        rep["encoder_maxdiff"] = float(np.abs(cond_o - cond_t).max())
        rep["encoder_ort_seconds"] = round(time.time() - t0, 1)
        print("encoder maxdiff", rep["encoder_maxdiff"], "fps equal", rep["fps_idx_equal"], flush=True)

        # ---- 4. greedy: HF generate vs ORT graph (teacher forced on torch's tokens) ----
        t0 = time.time()
        sid = torch.tensor(start)
        emb = torch.cat([t(cond_t)[None], lm.get_input_embeddings()(sid)[None]], 1)
        proc = U.VocabSwitchingLogitsProcessor(tokenizer=tok, start_tokens=sid)
        g = lm.generate(inputs_embeds=emb, bos_token_id=tok.bos, eos_token_id=tok.eos, pad_token_id=tok.pad,
                        logits_processor=LogitsProcessorList([proc]), num_beams=1, do_sample=False,
                        max_new_tokens=kw["max_new_tokens"], repetition_penalty=kw["repetition_penalty"],
                        return_dict_in_generate=True, output_logits=True)
        ids_t = g.sequences[0].numpy().astype(np.int64)
        raw_t = torch.stack([x[0] for x in g.logits]).numpy()
        assert len(ids_t) == len(raw_t), (len(ids_t), len(raw_t))
        rep["greedy_torch_seconds"] = round(time.time() - t0, 1)
        del g, emb
        gc.collect()
        dec = R.Decoder(f"{OUT}/skel_decoder.onnx", L)
        t0 = time.time()
        own, raw_o = R.greedy(dec, embed, R.prompt_of(cond_t, embed, start), tok, start, info, forced=ids_t)
        n = min(len(raw_o), len(raw_t))
        step_diff = np.abs(raw_o[:n] - raw_t[:n]).max(-1)
        rep.update(greedy_tokens=int(len(ids_t)), logits_maxdiff_first32=float(step_diff[:32].max()),
                   logits_maxdiff_all=float(step_diff.max()), logits_diff_per_step_first32=[float(x) for x in step_diff[:32]],
                   greedy_argmax_agree=int((own == ids_t).sum()), greedy_tokens_equal=bool(np.array_equal(own, ids_t)),
                   greedy_first_mismatch=int(np.argmin(np.append(own == ids_t, False))), greedy_ort_seconds=round(time.time() - t0, 1))
        print("greedy", {k: v for k, v in rep.items() if k.startswith("greedy") or k.startswith("logits_maxdiff")}, flush=True)
        np.savez(f"{OUT}/inputs_{NAME}.npz", vertices=V, normals=Nn, start=np.array(start), cond_torch=cond_t, greedy_ids=ids_t,
                 greedy_logits32=raw_t[:32], ref_ids=ref_ids)
        json.dump(rep, open(f"{OUT}/report_{NAME}.json", "w"), indent=1)

        # ---- 5. beam search (15 beams, sampling) through ORT, torch RNG restored -> expected == reference tokens ----
        t0 = time.time()
        torch.set_rng_state(rng_state)
        ids_o = R.beam_sample(dec, embed, R.prompt_of(cond_o, embed, start), tok, start, info, R.noise_torch())
        res = tok.detokenize(ids=ids_o)
        n = min(len(ids_o), len(ref_ids))
        rep.update(beam_ort_seconds=round(time.time() - t0, 1), beam_tokens_ort=int(len(ids_o)), beam_tokens_torch=int(len(ref_ids)),
                   beam_tokens_equal=bool(np.array_equal(ids_o, ref_ids)),
                   beam_first_mismatch=int(np.argmin(np.append(ids_o[:n] == ref_ids[:n], False))),
                   joints_ort=int(len(res.joints)), joints_torch=int(len(ref.joints)))
        if res.joints.shape == ref.joints.shape:
            rep["joints_maxdiff_normalized_vs_torch"] = float(np.abs(res.joints - ref.joints).max())
        np.save(f"{OUT}/beam_ids_ort_{NAME}.npy", ids_o)
        print("beam", {k: v for k, v in rep.items() if k.startswith(("beam", "joints"))}, flush=True)
    return res, rep


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    import torch
    from src.model import unirig_ar as U
    orig = U.UniRigAR.generate
    called = []

    def hooked(self, vertices, normals, cls=None, **kw):
        if called:
            return orig(self, vertices, normals, cls, **kw)
        called.append(1)
        rng_state = torch.get_rng_state().clone()
        rec = {}
        det = self.tokenizer.detokenize

        def spy(*a, **k):
            rec["ids"] = np.array(k.get("ids", a[0] if a else None)).copy()
            return det(*a, **k)

        self.tokenizer.detokenize = spy
        t0 = time.time()
        ref = orig(self, vertices, normals, cls, **kw)  # the reference: torch beam search
        del self.tokenizer.detokenize
        print(f"torch beam reference {time.time() - t0:.0f}s, {len(rec['ids'])} tokens", flush=True)
        try:
            res, rep = experiments(self, ref, rec["ids"], rng_state, vertices, normals, cls, kw, torch, U)
            json.dump(rep, open(f"{OUT}/report_{NAME}.json", "w"), indent=1)
            print("REPORT", json.dumps(rep), flush=True)
            return res
        except Exception:
            traceback.print_exc()
            print("SKEL_ONNX_FAILED", flush=True)
            return ref

    U.UniRigAR.generate = hooked
    sys.argv = ["run.py"] + sys.argv[1:]
    runpy.run_path("run.py", run_name="__main__")


if __name__ == "__main__":
    main()
