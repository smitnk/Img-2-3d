"""Stage 1 (torch, CPU): stock P3-SAM reference run on a mesh + recording of the static structure + ONNX export.

usage: python capture.py --src <HF space snapshot> --weights p3sam.safetensors --mesh m.glb --work work --points 16384 --prompts 128
writes in --work: mesh.npz, points.npz, enc_inputs.npz, ref.npz, ref_meta.json, enc.onnx(+.data), heads.onnx(+.data)
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3 as P  # noqa: E402


def normalize_pc(pc):
    mx, mn = pc.max(0), pc.min(0)
    c, s = (mx + mn) / 2, np.abs((mx - mn) / 2).max()
    return (pc - c) / (s + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--points", type=int, default=16384)
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kchunk", type=int, default=4)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--skip-heads", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    meta = {"points": a.points, "prompts": a.prompts, "seed": a.seed, "torch": torch.__version__}

    t0 = time.time()
    p3, sonata, sm, cfg = P.build_p3(a.src)
    meta["sonata_config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items() if isinstance(v, (int, float, str, bool, tuple, list, type(None)))}
    miss, unexp = P.load_weights(p3, a.weights)
    p3.eval()
    meta["missing_keys"], meta["unexpected_keys"] = miss, unexp
    nparam = sum(p.numel() for p in p3.parameters())
    nparam_son = sum(p.numel() for p in p3.sonata.parameters())
    meta["params_total"], meta["params_sonata"] = nparam, nparam_son
    print(f"PARAMS total {nparam / 1e6:.1f}M sonata {nparam_son / 1e6:.1f}M fp32 {nparam * 4 / 2 ** 20:.0f} MB", flush=True)
    transform = sonata.transform.default()

    # ---- mesh -> points (same as auto_mask.mesh_sam)
    mesh = trimesh.load(a.mesh, force="mesh")
    mesh.merge_vertices()
    mesh.process(True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    print(f"MESH vertices={len(mesh.vertices)} faces={len(mesh.faces)} watertight={mesh.is_watertight} bodies={len(mesh.split(only_watertight=False))}", flush=True)
    meta["mesh"] = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight)}
    np.savez_compressed(f"{a.work}/mesh.npz", vertices=mesh.vertices.astype(np.float64), faces=mesh.faces.astype(np.int64))
    pts, fidx = trimesh.sample.sample_surface(mesh, a.points, seed=a.seed)
    pts_norm = normalize_pc(pts).astype(np.float32)
    normals = mesh.face_normals[fidx].astype(np.float32)
    np.savez_compressed(f"{a.work}/points.npz", pts=pts, pts_norm=pts_norm, normals=normals, face_idx=fidx)

    # ---- stock features (+ recording)
    rec = P.Rec()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    hooks = P.install_recording(p3, sm, rec)
    t = time.time()
    feat_ref, data = P.stock_features(p3, transform, pts_norm, normals)
    meta["torch_features_s"] = round(time.time() - t, 2)
    for h in hooks:
        h.remove()
    feat_ref = feat_ref.numpy()
    feat0 = data["feat"].numpy()
    meta["stages_n"] = {str(k): int(v) for k, v in rec.n.items()}
    meta["stages_K"] = {str(k): int(v) for k, v in rec.K.items()}
    print(f"STOCK features {feat_ref.shape} in {meta['torch_features_s']}s; voxels per stage {rec.n}; patch K per stage {rec.K}; "
          f"rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
    inv = data["inverse"].numpy()

    # ---- static encoder in torch vs stock
    enc = P.EncStatic(p3, rec).eval()
    ins = P.enc_inputs(rec, feat0)
    names, dyn = P.enc_input_spec(rec)
    args = tuple(torch.from_numpy(ins[n]) for n in names)
    with torch.no_grad():
        t = time.time()
        f0 = enc(*args).numpy()
        meta["static_torch_enc_s"] = round(time.time() - t, 2)
    f_static = f0[inv]
    d = np.abs(f_static - feat_ref)
    cos = (f_static * feat_ref).sum(1) / (np.linalg.norm(f_static, axis=1) * np.linalg.norm(feat_ref, axis=1) + 1e-9)
    meta["static_vs_stock_feat"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_max": float(np.abs(feat_ref).max()), "min_cos": float(cos.min())}
    print(f"STATIC encoder vs stock: max abs {d.max():.3e} mean {d.mean():.3e} refmax {np.abs(feat_ref).max():.3e} min cos {cos.min():.6f} "
          f"{'PASS' if cos.min() > 0.9999 else 'FAIL'}", flush=True)
    np.savez_compressed(f"{a.work}/enc_inputs.npz", inverse=inv, **ins)
    meta["enc_inputs_mb"] = round(sum(v.nbytes for v in ins.values()) / 2 ** 20, 1)

    if a.skip_heads:
        json.dump(meta, open(f"{a.work}/ref_meta.json", "w"), indent=1)
        print("SKIP HEADS", flush=True)
        return
    # ---- prompts + stock heads
    sel = P.fps(pts_norm, a.prompts)
    prompts = pts_norm[sel]
    t = time.time()
    logits_ref, iou_ref = P.stock_heads(p3, feat_ref, pts_norm, prompts, bs=a.kchunk)
    meta["torch_heads_s"] = round(time.time() - t, 2)
    logits_ref, iou_ref = logits_ref.numpy(), iou_ref.numpy()
    heads = P.HeadsStatic(p3).eval()
    with torch.no_grad():
        t = time.time()
        l2, i2 = [], []
        for i in range(0, len(prompts), a.kchunk):
            x, y = heads(torch.from_numpy(feat_ref), torch.from_numpy(pts_norm), torch.from_numpy(prompts[i:i + a.kchunk]))
            l2.append(x)
            i2.append(y)
        meta["static_torch_heads_s"] = round(time.time() - t, 2)
    l2, i2 = torch.cat(l2, 1).numpy(), torch.cat(i2).numpy()
    agree = float(((l2 > 0) == (logits_ref > 0)).mean())
    meta["static_vs_stock_heads"] = {"max_abs_logit": float(np.abs(l2 - logits_ref).max()), "max_abs_iou": float(np.abs(i2 - iou_ref).max()), "mask_agree": agree}
    print(f"STATIC heads vs stock: max logit diff {np.abs(l2 - logits_ref).max():.3e} iou diff {np.abs(i2 - iou_ref).max():.3e} mask agreement {agree * 100:.4f}%", flush=True)
    np.savez_compressed(f"{a.work}/ref.npz", feat=feat_ref.astype(np.float16), logits=logits_ref.astype(np.float16), iou=iou_ref, prompts=prompts, prompt_idx=sel)

    # ---- export
    if not a.no_export:
        sizes = {}
        sizes["enc_fp32"] = P.export_onnx(enc, args, names, ["feat"], {**dyn, "feat": {0: "n0"}}, f"{a.work}/enc.onnx")
        K0 = min(a.kchunk, len(prompts))
        sizes["heads_fp32"] = P.export_onnx(heads, (torch.from_numpy(feat_ref), torch.from_numpy(pts_norm), torch.from_numpy(prompts[:K0])),
                                            ["feats", "points", "prompts"], ["logits", "iou"],
                                            {"feats": {0: "N"}, "points": {0: "N"}, "prompts": {0: "K"}, "logits": {0: "N", 1: "K"}, "iou": {0: "K"}},
                                            f"{a.work}/heads.onnx")
        meta["onnx_bytes"] = sizes
    meta["capture_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    meta["capture_total_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(f"{a.work}/ref_meta.json", "w"), indent=1)
    print("CAPTURE DONE", json.dumps(meta)[:2000], flush=True)


if __name__ == "__main__":
    main()
