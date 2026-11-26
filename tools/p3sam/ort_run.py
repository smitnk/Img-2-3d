"""Stage 2 (ONNX Runtime CPU, one process per variant so the RSS numbers are clean).

usage: python ort_run.py --work work --variant fp32|int8|fp16 [--kchunk 4] [--threads N] [--arena 0|1]
reads  work/enc_inputs.npz, work/points.npz, work/ref.npz, work/enc_<v>.onnx, work/heads_<v>.onnx
writes work/ort_<v>.npz (masks of the best head per prompt, iou, features fp16) and work/ort_<v>.json (time, peak RSS, sizes)
"""
import argparse
import gc
import json
import os
import threading
import time

import numpy as np
import onnxruntime as ort
import psutil


class Rss:
    def __init__(self):
        self.p, self.peak, self.stop = psutil.Process(), 0, False
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.02)

    def mark(self):
        self.peak = max(self.peak, self.p.memory_info().rss)
        return self.p.memory_info().rss


def session(path, threads, arena):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.enable_cpu_mem_arena = bool(arena)
    so.enable_mem_pattern = bool(arena)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def size(path):
    return os.path.getsize(path) + (os.path.getsize(path + ".data") if os.path.exists(path + ".data") else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--kchunk", type=int, default=4)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--arena", type=int, default=0)
    a = ap.parse_args()
    w, v = a.work, a.variant
    rss = Rss()
    base = rss.mark()
    ins = dict(np.load(f"{w}/enc_inputs.npz"))
    inv = ins.pop("inverse")
    pts = np.load(f"{w}/points.npz")["pts_norm"]
    prompts = np.load(f"{w}/ref.npz")["prompts"]
    res = {"variant": v, "threads": a.threads, "arena": a.arena, "rss_after_inputs_mb": round(rss.mark() / 2 ** 20)}
    out = {}
    try:
        t = time.time()
        s = session(f"{w}/enc_{v}.onnx", a.threads, a.arena)
        res["enc_load_s"] = round(time.time() - t, 2)
        res["rss_enc_loaded_mb"] = round(rss.mark() / 2 ** 20)
        feed = {i.name: ins[i.name] for i in s.get_inputs()}
        t = time.time()
        f0 = s.run(None, feed)[0]
        res["enc_run_s"] = round(time.time() - t, 2)
        res["rss_enc_done_mb"] = round(rss.mark() / 2 ** 20)
        del s, feed
        gc.collect()
        feat = f0[inv].astype(np.float32)
        res["rss_enc_freed_mb"] = round(rss.mark() / 2 ** 20)
        t = time.time()
        s = session(f"{w}/heads_{v}.onnx", a.threads, a.arena)
        res["heads_load_s"] = round(time.time() - t, 2)
        t = time.time()
        L, I = [], []
        for i in range(0, len(prompts), a.kchunk):
            l, io = s.run(None, {"feats": feat, "points": pts, "prompts": prompts[i:i + a.kchunk]})
            L.append(l)
            I.append(io)
            rss.mark()
        res["heads_run_s"] = round(time.time() - t, 2)
        logits, iou = np.concatenate(L, 1), np.concatenate(I)
        out = dict(feat=feat.astype(np.float16), logits=logits.astype(np.float16), iou=iou)
        res["ok"] = True
    except Exception as e:  # keep the numbers we have (e.g. fp16 kernels missing on this CPU)
        import traceback
        traceback.print_exc()
        res["ok"], res["error"] = False, repr(e)[:400]
    rss.stop = True
    res["peak_rss_mb"] = round(rss.peak / 2 ** 20)
    res["total_s"] = round(res.get("enc_load_s", 0) + res.get("enc_run_s", 0) + res.get("heads_load_s", 0) + res.get("heads_run_s", 0), 2)
    for k in ("enc", "heads"):
        p = f"{w}/{k}_{v}.onnx"
        if os.path.exists(p):
            res[f"{k}_bytes"] = size(p)
    np.savez_compressed(f"{w}/ort_{v}.npz", **out)
    json.dump(res, open(f"{w}/ort_{v}.json", "w"), indent=1)
    print("ORT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
