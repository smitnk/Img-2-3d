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


def mm_nodes(path):
    m = onnx.load(path, load_external_data=False)
    return [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]


def classify(names, groups):
    out = {g: [n for n in names if re.search(p, n)] for g, p in groups.items()}
    seen = {n for v in out.values() for n in v}
    out["_other"] = [n for n in names if n not in seen]
    return out


def q_int8(src, dst, exclude):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for f in (dst, dst + ".data"):
        if os.path.exists(f):
            os.remove(f)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul", "Gemm"],
                     nodes_to_exclude=list(exclude), use_external_data_format=True)


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

    for part, groups, ev in (("enc", GROUPS_ENC, enc_eval), ("heads", GROUPS_HEADS, heads_eval)):
        src = f"{w}/{part}_fp32.onnx"
        names = mm_nodes(src)
        cl = classify(names, groups)
        print(f"SWEEP {part}: {len(names)} MatMul/Gemm nodes; groups " + str({k: len(v) for k, v in cl.items()}) + "; other examples " + str(cl["_other"][:6]), flush=True)
        res[part]["groups"] = {k: len(v) for k, v in cl.items()}
        recipes = {"all": []}
        for g in list(groups) + ["_other"]:
            if cl[g]:
                recipes[f"no_{g}"] = cl[g]
        recipes["only_" + "+".join(g for g in ("blkmlp", "final") if g in groups)] = [n for g, v in cl.items() for n in v if g not in ("blkmlp", "final")] if part == "enc" else []
        res[part]["fp32"] = ev(src)
        print(f"SWEEP {part} fp32 {res[part]['fp32']}", flush=True)
        for kind, fn in (("int8", q_int8), ("w4", q_w4)):
            for name, ex in recipes.items():
                if name.startswith("only_") and not ex:
                    continue
                if kind == "w4" and name not in ("all", "no_attn", "no_conv"):
                    continue
                dst = f"{w}/_{part}_{kind}_{name}.onnx"
                try:
                    t = time.time()
                    fn(src, dst, ex)
                    r = ev(dst)
                    r["quant_s"] = round(time.time() - t, 1)
                    r["excluded"] = len(ex)
                    res[part][f"{kind}:{name}"] = r
                    print(f"SWEEP {part} {kind}:{name} {r}", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    res[part][f"{kind}:{name}"] = dict(error=repr(e)[:200])
                    print(f"SWEEP {part} {kind}:{name} FAILED {repr(e)[:200]}", flush=True)
        # pick: smallest recipe within the accuracy bar, else the most accurate
        for kind in ("int8", "w4"):
            cand = {k: v for k, v in res[part].items() if k.startswith(kind + ":") and "error" not in v}
            if not cand:
                continue
            key = "cos_mean" if part == "enc" else "mask_agree"
            bar = 0.995 if part == "enc" else 0.985
            ok = {k: v for k, v in cand.items() if v[key] >= bar}
            pick = min(ok, key=lambda k: ok[k]["mb"]) if ok else max(cand, key=lambda k: cand[k][key])
            res[part][f"pick_{kind}"] = dict(recipe=pick, within_bar=bool(ok), **cand[pick])
            print(f"SWEEP {part} pick {kind}: {pick} (bar {bar} met: {bool(ok)}) {cand[pick]}", flush=True)
            src_f = f"{w}/_{part}_{kind}_{pick.split(':')[1]}.onnx"
            dst_f = f"{w}/{part}_{kind}.onnx"
            O.externalize(src_f, dst_f)
        for f in os.listdir(w):
            if f.startswith(f"_{part}_"):
                os.remove(f"{w}/{f}")
    json.dump(res, open(f"{w}/sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
