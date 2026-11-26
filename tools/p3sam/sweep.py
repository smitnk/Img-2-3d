"""Quantisation recipes for enc.onnx / heads.onnx: which op groups can be int8 (dynamic) or 4-bit weight-only (MatMulNBits) before the
features / masks drift. Writes enc_int8 / heads_int8 (best int8 recipe) and enc_w4 / heads_w4, plus sweep.json.

usage: python sweep.py work
"""
import json
import os
import re
import sys
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unirig"))
import onnx_skin as O  # noqa: E402

GROUPS_ENC = {
    "conv": r"/convs\.",
    "attn": r"/attn/|/blocks\.\d+/attn",
    "blkmlp": r"/blocks\.\d+/mlp",
    "cpe": r"/blocks\.\d+/cpe",
    "pool": r"/pool",
    "stem": r"/stem",
    "final": r"^/mlp/|^/mlp\.",
}
GROUPS_HEADS = {"s1": r"seg_mlp_", "s2g": r"seg_s2_mlp_g", "s2": r"seg_s2_mlp_[123]", "iou": r"iou_mlp"}


def safe(n):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", n)


def mm_nodes(path):
    m = onnx.load(path, load_external_data=False)
    return [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]


def classify(names, groups):
    out = {g: [n for n in names if re.search(p, n)] for g, p in groups.items()}
    seen = {n for v in out.values() for n in v}
    out["_other"] = [n for n in names if n not in seen]
    return out


def q_int8(src, dst, exclude, rr=False):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for f in (dst, dst + ".data"):
        if os.path.exists(f):
            os.remove(f)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"],
                     nodes_to_exclude=list(exclude), reduce_range=rr, use_external_data_format=True)


def q_w4(src, dst, exclude, block=32):
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
    for f in (dst, dst + ".data"):
        if os.path.exists(f):
            os.remove(f)
    q = MatMul4BitsQuantizer(onnx.load(src, load_external_data=True), block_size=block, is_symmetric=True, nodes_to_exclude=list(exclude))
    q.process()
    q.model.save_model_to_file(dst, True)


def size(p):
    return os.path.getsize(p) + (os.path.getsize(p + ".data") if os.path.exists(p + ".data") else 0)


def sess(p):
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count()
    so.enable_cpu_mem_arena = False
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])


def main():
    w = sys.argv[1]
    ins = dict(np.load(f"{w}/enc_inputs.npz"))
    inv = ins.pop("inverse")
    rz = np.load(f"{w}/ref.npz")
    ref_feat = rz["feat"].astype(np.float32)
    pts = np.load(f"{w}/points.npz")["pts_norm"]
    prompts, ref_logits, ref_iou = rz["prompts"][:8], rz["logits"][:, :8].astype(np.float32), rz["iou"][:8]
    res = {"enc": {}, "heads": {}}

    def enc_eval(path):
        s = sess(path)
        t = time.time()
        f = s.run(None, {i.name: ins[i.name] for i in s.get_inputs()})[0][inv]
        dt = time.time() - t
        cos = (f * ref_feat).sum(1) / (np.linalg.norm(f, axis=1) * np.linalg.norm(ref_feat, axis=1) + 1e-9)
        return dict(cos_mean=float(cos.mean()), cos_min=float(cos.min()), run_s=round(dt, 2), mb=round(size(path) / 2 ** 20, 1))

    def heads_eval(path):
        s = sess(path)
        t = time.time()
        L, I = [], []
        for i in range(0, 8, 4):
            l, io = s.run(None, {"feats": ref_feat, "points": pts, "prompts": prompts[i:i + 4]})
            L.append(l)
            I.append(io)
        dt = time.time() - t
        L, I = np.concatenate(L, 1), np.concatenate(I)
        j = ref_iou.argmax(1)
        a = np.take_along_axis(L, j[None, :, None], 2)[:, :, 0] > 0
        b = np.take_along_axis(ref_logits, j[None, :, None], 2)[:, :, 0] > 0
        return dict(mask_agree=float((a == b).mean()), iou_maxdiff=float(np.abs(I - ref_iou).max()), run_s=round(dt, 2), mb=round(size(path) / 2 ** 20, 1))

    def node_type(n):
        return "conv" if "/convs." in n else re.sub(r"_\d+$", "", n)

    def pick_and_write(part, kind, key, bar_fn):
        cand = {k: v for k, v in res[part].items() if k.startswith(kind + ":") and "error" not in v}
        if not cand:
            return
        ok = {k: v for k, v in cand.items() if bar_fn(v)}
        pick = min(ok, key=lambda k: ok[k]["mb"]) if ok else max(cand, key=lambda k: cand[k][key])
        res[part][f"pick_{kind}"] = dict(recipe=pick, within_bar=bool(ok), **cand[pick])
        print(f"SWEEP {part} pick {kind}: {pick} (bar met: {bool(ok)}) {cand[pick]}", flush=True)
        O.externalize(f"{w}/_{part}_{kind}_{safe(pick.split(':', 1)[1])}.onnx", f"{w}/{part}_{kind}.onnx")

    def attempt(part, src, kind, name, ex, ev, rr=False):
        dst = f"{w}/_{part}_{kind}_{safe(name)}.onnx"
        try:
            t = time.time()
            (q_int8(src, dst, ex, rr) if kind == "int8" else q_w4(src, dst, ex))
            r = ev(dst)
            r.update(quant_s=round(time.time() - t, 1), excluded=len(ex), reduce_range=rr)
            res[part][f"{kind}:{name}"] = r
            print(f"SWEEP {part} {kind}:{name} {r}", flush=True)
            return r
        except Exception as e:
            traceback.print_exc()
            res[part][f"{kind}:{name}"] = dict(error=repr(e)[:200])
            print(f"SWEEP {part} {kind}:{name} FAILED {repr(e)[:200]}", flush=True)

    # ---------------- encoder: sensitivity per op type, then greedy exclusion
    src = f"{w}/enc_fp32.onnx"
    names = mm_nodes(src)
    types = {}
    for n in names:
        types.setdefault(node_type(n), []).append(n)
    res["enc"]["types"] = {k: len(v) for k, v in types.items()}
    print(f"SWEEP enc: {len(names)} MatMul/Gemm nodes in {len(types)} types: {res['enc']['types']}", flush=True)
    res["enc"]["fp32"] = enc_eval(src)
    print(f"SWEEP enc fp32 {res['enc']['fp32']}", flush=True)
    base = attempt("enc", src, "int8", "all", [], enc_eval)
    attempt("enc", src, "int8", "all_rr", [], enc_eval, rr=True)
    gain = {}
    for t, ex in types.items():
        r = attempt("enc", src, "int8", f"no[{t}]", ex, enc_eval)
        if r and base:
            gain[t] = r["cos_mean"] - base["cos_mean"]
    order = sorted(gain, key=lambda t: -gain[t])
    print("SWEEP enc types by cos gain when left in fp32:", [(t, round(gain[t], 5)) for t in order[:8]], flush=True)
    cum = []
    for t in order[:4]:
        cum += types[t]
        attempt("enc", src, "int8", f"greedy{len([k for k in res['enc'] if k.startswith('int8:greedy')]) + 1}", list(cum), enc_eval)
    attempt("enc", src, "w4", "all", [], enc_eval)
    pick_and_write("enc", "int8", "cos_mean", lambda v: v["cos_mean"] >= 0.997 and v["cos_min"] >= 0.98)
    pick_and_write("enc", "w4", "cos_mean", lambda v: v["cos_mean"] >= 0.995)

    # ---------------- heads
    src = f"{w}/heads_fp32.onnx"
    res["heads"]["fp32"] = heads_eval(src)
    attempt("heads", src, "int8", "all", [], heads_eval)
    attempt("heads", src, "w4", "all", [], heads_eval)
    pick_and_write("heads", "int8", "mask_agree", lambda v: v["mask_agree"] >= 0.985)
    pick_and_write("heads", "w4", "mask_agree", lambda v: v["mask_agree"] >= 0.985)
    for f in os.listdir(w):
        if f.startswith("_enc_") or f.startswith("_heads_"):
            os.remove(f"{w}/{f}")
    json.dump(res, open(f"{w}/sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
