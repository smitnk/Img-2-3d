"""UniRig SKELETON stage on ONNX Runtime: numpy + onnxruntime only (torch is optional, see noise_torch).

Graphs (fp32, written by export_skeleton.py):
  skel_encoder.onnx  pc[1,N,3] feats[1,N,3] spc[1,Q,3] sfeats[1,Q,3] -> cond[1,Q,D]   (Michelangelo encoder + output_proj)
  skel_decoder.onnx  embeds[b,1,D] position[b,1] pk{i}[1,H,p,Dh] gk{i}[b,H,g,Dh] (i < 2*layers)
                     -> logits[b,V], nk{i}[b,H,1,Dh]       one OPT token step; K/V of the shared prompt (pk) is stored once
                        and expanded to b beams inside the graph, gk is the per-beam generated part.
Prompt (1024 latents + bos + cls) is fed token by token through the same graph (b=1, p=0).

CLI (the ARM job; every phase is its own process so /usr/bin/time -v gives per-phase peak RSS):
  python onnx_skeleton.py encode|greedy|beam --dir <onnx dir> --name <mesh> --unirig <UniRig checkout>
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def log_rss(tag):
    """Current (VmRSS) and peak (VmHWM) resident memory from /proc, one line per call, to find where memory goes."""
    try:
        v = dict(l.split(":", 1) for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM")))
        print(f"MEM {tag}: rss {int(v['VmRSS'].split()[0]) / 1048576:.2f}GB peak {int(v['VmHWM'].split()[0]) / 1048576:.2f}GB", flush=True)
    except Exception:  # not Linux
        print(f"MEM {tag}: peak {rss_gb():.2f}GB", flush=True)


def session(path, threads=0):
    """ORT session tuned for low peak RSS: no arena (it keeps the largest block ever requested and rounds up to powers of two),
    no memory pattern (dynamic shapes), initializers straight from the system allocator instead of the arena."""
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.enable_cpu_mem_arena = False
    o.enable_mem_pattern = False
    o.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    log_rss(f"session {os.path.basename(path)}")
    return s


# ---------- encoder inputs: what Michelangelo's CrossAttentionEncoder does before its first matmul ----------
def fps_np(pts, k):
    """torch_cluster.fps(ratio, random_start=False) on one cloud: start at 0, strict '>' so the first max wins, float32 sums x,y,z in order."""
    pts = pts.astype(np.float32)
    idx = np.zeros(k, np.int64)
    dist = np.full(len(pts), 1e38, np.float32)
    for i in range(1, k):
        d = pts - pts[idx[i - 1]]
        dist = np.minimum(dist, d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        idx[i] = int(np.argmax(dist))
    return idx


def sample_queries(vertices, normals, token_num=1024):
    """4*token_num random points (numpy default_rng(0), as in eval mode), then FPS down to token_num. Returns (points, normals)."""
    n = 4 * token_num
    ind = np.random.default_rng(0).choice(len(vertices), n, replace=n > len(vertices))
    pre, pre_n = vertices[ind], normals[ind]
    f = fps_np(pre, token_num)
    return pre[f], pre_n[f]


# ---------- decoder ----------
class Decoder:
    def __init__(self, path, layers, threads=0):
        self.s = session(path, threads)
        self.n = 2 * layers

    def run(self, emb, pos, pre, gen):
        f = {"embeds": np.ascontiguousarray(emb, np.float32), "position": np.asarray(pos, np.int64)}
        f.update({f"pk{i}": a for i, a in enumerate(pre)})
        f.update({f"gk{i}": a for i, a in enumerate(gen)})
        r = self.s.run(None, f)
        return r[0], r[1:]


def empty_kv(dec, b, heads, hd):
    return [np.zeros((b, heads, 0, hd), np.float32) for _ in range(dec.n)]


def prefill(dec, prompt, heads, hd):
    """prompt [P,D] -> (K/V of the whole prompt, batch 1, as the shared prefix; logits [V] of the last prompt token)."""
    pre0, gen = empty_kv(dec, 1, heads, hd), empty_kv(dec, 1, heads, hd)
    for t in range(len(prompt)):
        logits, new = dec.run(prompt[t][None, None], [[t]], pre0, gen)
        for i, n in enumerate(new):
            gen[i] = np.concatenate([gen[i], n], 2)
    log_rss("prefill done")
    return gen, logits[0]


def repetition_penalty(s, ids, p):
    """HF RepetitionPenaltyLogitsProcessor on one row: ids = tokens generated so far."""
    if len(ids):
        u = np.unique(ids)
        v = s[u]
        s[u] = np.where(v < 0, v * np.float32(p), v / np.float32(p))
    return s


def vocab_switch(s, tok, start, ids):
    """UniRig VocabSwitchingLogitsProcessor on one row: -inf everywhere but the grammar's next tokens."""
    m = np.full_like(s, -np.inf)
    m[tok.next_posible_token(ids=np.array(list(start) + list(ids), dtype=np.int64))] = 0
    return s + m


def greedy(dec, embed, prompt, tok, start, cfg, forced=None):
    """HF generate(num_beams=1, do_sample=False) with repetition penalty + grammar. forced: token ids to feed instead of
    the own argmax (teacher forcing) so that per-step logits stay comparable after a divergence.
    Returns (own argmax ids, raw logits [T,V])."""
    H, Dh, eos, P = cfg["heads"], cfg["head_dim"], cfg["eos"], len(prompt)
    pre, logits = prefill(dec, prompt, heads=H, hd=Dh)
    gen = empty_kv(dec, 1, H, Dh)
    fed, own, raw = [], [], []
    limit = len(forced) if forced is not None else cfg["max_new_tokens"]
    while len(own) < limit:
        raw.append(logits.copy())
        s = vocab_switch(repetition_penalty(logits.astype(np.float32).copy(), fed, cfg["repetition_penalty"]), tok, start, fed)
        own.append(int(np.argmax(s)))
        nxt = int(forced[len(own) - 1]) if forced is not None else own[-1]
        fed.append(nxt)
        if nxt == eos and forced is None:
            break
        if len(own) == limit:
            break
        logits, new = dec.run(embed[nxt][None, None], [[P + len(fed) - 1]], pre, gen)
        logits = logits[0]
        for i, n in enumerate(new):
            gen[i] = np.concatenate([gen[i], n], 2)
    return np.array(own, np.int64), np.stack(raw)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def noise_numpy(seed=12345):
    r = np.random.default_rng(seed)
    return lambda n: r.standard_exponential(n).astype(np.float32)


def noise_torch():
    """Exponential(1) noise from torch's global RNG: the exact stream torch.multinomial(without replacement) consumes
    (it draws q = exponential_(1) of the probs' shape and takes topk(probs / q)). Set torch's RNG state before calling."""
    import torch
    return lambda n: torch.empty(1, n).exponential_(1).numpy()[0]


def beam_sample(dec, embed, prompt, tok, start, cfg, noise):
    """transformers 4.51.3 _beam_search(do_sample=True), batch 1, as UniRig calls it: repetition penalty -> grammar ->
    temperature -> top-k -> top-p on the log-probs of every beam, + running beam score, multinomial (w/o replacement)
    over all beams*vocab, keep 2*beams candidates, stop on eos/max length, return the best finished sequence."""
    H, Dh, eos, P = cfg["heads"], cfg["head_dim"], cfg["eos"], len(prompt)
    nb, V, max_len = cfg["num_beams"], embed.shape[0], cfg["max_new_tokens"]
    keep = 2 * nb
    pre, l0 = prefill(dec, prompt, heads=H, hd=Dh)
    gen = empty_kv(dec, nb, H, Dh)
    run_seq = np.zeros((nb, max_len), np.int64)
    seqs = np.zeros((nb, max_len), np.int64)
    lens = np.zeros(nb, np.int64)
    run_sc = np.full(nb, -1e9, np.float32)
    run_sc[0] = 0
    fin_sc = np.full(nb, -1e9, np.float32)
    is_fin = np.zeros(nb, bool)
    c = 0  # generated length so far (HF cur_len)
    neg = np.float32(-1e9)
    while True:
        if c == 0:
            logits = np.repeat(l0[None], nb, 0)
        else:
            logits, new = dec.run(embed[run_seq[:, c - 1]][:, None], np.full((nb, 1), P + c - 1), pre, gen)
            for i, n in enumerate(new):  # in place per tensor: one transient copy instead of a second whole cache
                gen[i] = np.concatenate([gen[i], n], 2)
        x = logits.astype(np.float32)
        lp = x - x.max(-1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        for b in range(nb):  # RepetitionPenalty, then the grammar
            lp[b] = vocab_switch(repetition_penalty(lp[b], run_seq[b, :c], cfg["repetition_penalty"]), tok, start, run_seq[b, :c])
        lp = lp / np.float32(cfg["temperature"])
        k = cfg["top_k"]
        kth = np.partition(lp, -k, axis=1)[:, -k][:, None]
        lp = np.where(lp < kth, -np.inf, lp).astype(np.float32)
        order = np.argsort(lp, axis=1, kind="stable")  # top-p over ascending sorted scores, keep >= 2 tokens (eos count + 1)
        c_p = np.cumsum(softmax(np.take_along_axis(lp, order, 1)), 1, dtype=np.float32)
        rem = c_p <= np.float32(1 - cfg["top_p"])
        rem[:, -2:] = False
        mask = np.zeros_like(rem)
        np.put_along_axis(mask, order, rem, 1)
        lp = np.where(mask, -np.inf, lp).astype(np.float32)

        acc = (lp + run_sc[:, None]).reshape(-1)
        probs = softmax(acc).astype(np.float32)
        sel = np.argsort(-(probs / noise(acc.size)), kind="stable")[:keep]  # torch.multinomial(probs, keep) == topk(probs / Exp(1))
        t_lp, t_beam, t_tok = acc[sel], sel // V, sel % V
        t_seq = run_seq[t_beam].copy()
        t_seq[:, c] = t_tok
        hits = (t_tok == eos) | (c + 1 >= max_len)
        # next running beams: best nb non-finished
        r_lp = t_lp + hits.astype(np.float32) * neg
        top = np.argsort(-r_lp, kind="stable")[:nb]
        run_seq, run_sc, parent = t_seq[top], r_lp[top], t_beam[top]
        # finished pool: only the first nb of the sampled order may enter (top_num_beam_mask)
        did = hits & (np.arange(keep) < nb)
        f_lp = t_lp / np.float32(c + 1) + (~did).astype(np.float32) * neg
        m_sc = np.concatenate([fin_sc, f_lp])
        m_top = np.argsort(-m_sc, kind="stable")[:nb]
        seqs = np.concatenate([seqs, t_seq])[m_top]
        lens = np.concatenate([lens, np.full(keep, c + 1)])[m_top]
        is_fin = np.concatenate([is_fin, did])[m_top]
        fin_sc = m_sc[m_top]
        if c > 0:
            for i in range(len(gen)):
                gen[i] = gen[i][parent]
        else:  # first step: all beams share the prompt, the generated part is still empty
            gen = empty_kv(dec, nb, H, Dh)
        c += 1
        # _beam_search_has_unfinished_sequences (early_stopping=False, length_penalty=1)
        worst = np.where(is_fin, fin_sc.min(), -1e9)
        if not (np.any(run_sc[0] / np.float32(c) > worst) and not hits.all()):
            break
        if c == 1 or c % 50 == 0:
            print(f"beam step {c} best {run_sc[0]:.2f}", flush=True)
            log_rss(f"beam step {c}")
    assert lens[0] > 0, "no finished beam"
    return np.array(list(start) + seqs[0, :lens[0]].tolist(), np.int64)


# ---------- glue ----------
def load_tokenizer(unirig_dir):
    """UniRig's own tokenizer (numpy + yaml + python-box only; bpy/torch are not needed for it)."""
    import yaml
    from box import Box
    unirig_dir = os.path.abspath(unirig_dir)
    os.chdir(unirig_dir)  # the tokenizer config points at ./configs/skeleton/*.yaml
    sys.path.insert(0, unirig_dir)
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig
    cfg = Box(yaml.safe_load(open("configs/tokenizer/tokenizer_parts_articulationxl_256.yaml")))
    return get_tokenizer(config=TokenizerConfig.parse(config=cfg))


def load(d, name, unirig, threads=0):
    d = os.path.abspath(d)
    info = json.load(open(f"{d}/info.json"))
    embed = np.load(f"{d}/embed.npy")
    inp = np.load(f"{d}/inputs_{name}.npz")
    tok = load_tokenizer(unirig)
    return d, info, embed, inp, tok, threads


def encode(d, inp, threads=0):
    s = session(f"{d}/skel_encoder.onnx", threads)
    V, N = inp["vertices"].astype(np.float32), inp["normals"].astype(np.float32)
    spc, sf = sample_queries(V, N)
    cond = s.run(None, {"pc": V[None], "feats": N[None], "spc": spc[None], "sfeats": sf[None]})[0][0]
    log_rss("encoder run done")
    del s  # frees the encoder weights/activations before any decoder session is built in this process
    return cond


def prompt_of(cond, embed, start):
    return np.concatenate([cond, embed[list(start)]], 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["encode", "greedy", "beam"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--unirig", required=True)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    log_rss("start")
    d, info, embed, inp, tok, th = load(a.dir, a.name, a.unirig, a.threads)
    log_rss("inputs + tokenizer loaded")
    start = [int(x) for x in inp["start"]]
    out = {"phase": a.phase, "name": a.name}
    t0 = time.time()
    if a.phase == "encode":
        cond = encode(d, inp, th)
        np.save(f"{d}/cond_ort_{a.name}.npy", cond)
        out["encoder_maxdiff_vs_torch_x86"] = float(np.abs(cond - inp["cond_torch"]).max())
    else:
        cond = np.load(f"{d}/cond_ort_{a.name}.npy")
        dec = Decoder(f"{d}/skel_decoder.onnx", info["layers"], th)
        prompt = prompt_of(cond, embed, start)
        if a.phase == "greedy":
            ids, raw = greedy(dec, embed, prompt, tok, start, info)
            ref = inp["greedy_ids"]
            n = min(len(ids), len(ref))
            out.update(tokens=len(ids), torch_tokens=len(ref), equal=bool(np.array_equal(ids, ref)),
                       first_mismatch=int(np.argmin(np.append(ids[:n] == ref[:n], False))),
                       logits_maxdiff_first32=float(np.abs(raw[:32] - inp["greedy_logits32"]).max()))
        else:
            ids = beam_sample(dec, embed, prompt, tok, start, info, noise_numpy(12345))
            res, ref = tok.detokenize(ids=ids), tok.detokenize(ids=inp["ref_ids"])
            out.update(tokens=len(ids), torch_tokens=len(inp["ref_ids"]), joints=len(res.joints), torch_joints=len(ref.joints),
                       note="numpy RNG: a different but valid sample, not expected to equal torch's")
            if len(res.joints) and len(ref.joints):
                dist = lambda x, y: np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1)).min(1)
                out["joint_chamfer_normalized"] = float((dist(res.joints, ref.joints).mean() + dist(ref.joints, res.joints).mean()) / 2)
    log_rss("end")
    out.update(seconds=round(time.time() - t0, 1), self_peak_rss_gb=round(rss_gb(), 2))
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
