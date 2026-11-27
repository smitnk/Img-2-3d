"""Stage 3: compare ONNX variants with the torch reference, post-process to part labels, cut + peg + export, write results.json.

usage: python finish.py --work work --out out [--variant auto]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post as PP  # noqa: E402
import split as SP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--min-agree", type=float, default=0.90)
    a = ap.parse_args()
    w, out = a.work, a.out
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{w}/ref_meta.json"))
    mz = np.load(f"{w}/mesh.npz")
    mesh = trimesh.Trimesh(mz["vertices"], mz["faces"], process=False)
    pz = np.load(f"{w}/points.npz")
    pts, pts_n = pz["pts"], pz["pts_norm"]
    rz = np.load(f"{w}/ref.npz")
    A = PP.face_adjacency_matrix(mesh)
    area = mesh.area_faces

    def label(logits, iou, tag):
        masks, best_iou = PP.best_masks(logits.astype(np.float32), iou)
        pl, info = PP.masks_to_point_labels(pts_n, masks, best_iou)
        fl = PP.faces_from_points(mesh, pts, pl, A)
        print(f"LABELS {tag}: point labels={int(pl.max()) + 1} face parts={len(np.unique(fl))} {info}", flush=True)
        return masks, pl, fl, info

    ref_masks, ref_pl, ref_fl, ref_info = label(rz["logits"], rz["iou"], "torch-ref")
    ref_feat = rz["feat"].astype(np.float32)
    res = {"meta": meta, "reference": {"parts": int(len(np.unique(ref_fl))), **ref_info}, "variants": {}}
    fls = {}
    for v in ("fp32", "int8", "w4", "fp16"):
        jf = f"{w}/ort_{v}.json"
        if not os.path.exists(jf):
            continue
        r = json.load(open(jf))
        if r.get("ok"):
            z = np.load(f"{w}/ort_{v}.npz")
            feat = z["feat"].astype(np.float32)
            cos = (feat * ref_feat).sum(1) / (np.linalg.norm(feat, axis=1) * np.linalg.norm(ref_feat, axis=1) + 1e-9)
            masks, pl, fl, info = label(z["logits"], z["iou"], v)
            # same prompt, same head as the reference: raw mask agreement
            jr = rz["iou"].argmax(1)
            lg = z["logits"].astype(np.float32)
            same_head = np.take_along_axis(lg, jr[None, :, None], 2)[:, :, 0] > 0
            ref_same = np.take_along_axis(rz["logits"].astype(np.float32), jr[None, :, None], 2)[:, :, 0] > 0
            r.update(feat_cos_mean=float(cos.mean()), feat_cos_min=float(cos.min()), mask_agree_pct=float((same_head == ref_same).mean() * 100),
                     iou_pred_maxdiff=float(np.abs(z["iou"] - rz["iou"]).max()), point_label_agree_pct=PP.agreement(ref_pl, pl) * 100,
                     face_label_agree_pct=PP.agreement(ref_fl, fl, area) * 100, parts=int(len(np.unique(fl))), **{f"post_{k}": x for k, x in info.items()})
            fls[v] = fl
        res["variants"][v] = r
    chosen = a.variant
    if chosen == "auto":
        cand = [(r.get("enc_bytes", 0) + r.get("heads_bytes", 0), v) for v, r in res["variants"].items()
                if r.get("ok") and r.get("face_label_agree_pct", 0) >= a.min_agree * 100]
        chosen = min(cand)[1] if cand else ("fp32" if "fp32" in fls else None)
    res["chosen_variant"] = chosen
    fl = fls.get(chosen, ref_fl)
    print(f"CHOSEN variant {chosen}; {len(np.unique(fl))} parts", flush=True)

    t = time.time()
    pdir = f"{out}/parts"
    sp = SP.split_parts(mesh, fl, pdir, pegs=True)
    res["split"] = {k: v for k, v in sp.items() if k != "files"}
    res["split_s"] = round(time.time() - t, 1)
    res["split_all_manifold"] = all(p["manifold"] for p in sp["parts"].values())
    res["split_all_watertight"] = all(p["watertight"] for p in sp["parts"].values())
    res["pegs_ok"] = int(sum(1 for p in sp["pegs"] if p.get("ok")))
    res["pegs_total"] = int(len(sp["pegs"]))
    # also the plain (no pegs) cut and the reference-label cut for comparison
    SP.preview(mesh, fl, sp["files"], f"{out}/preview.png", title=f"({chosen})")
    np.save(f"{out}/face_labels.npy", fl)
    SP.zip_files(sp["files"], [f"{out}/preview.png"], f"{out}/parts.zip")
    res["zip_bytes"] = os.path.getsize(f"{out}/parts.zip")
    json.dump(res, open(f"{out}/results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("RESULTS", json.dumps({k: v for k, v in res.items() if k not in ("meta", "split")}, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
