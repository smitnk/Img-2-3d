"""CI-only (uses PIL): textured 4-view sheet + turntable gif from a paint_phone.py output dir (mesh_uv.npz + texture.png)."""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_np as N

import json
import colour_metric as CM

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else None  # optional: output dir of an older run, measured the same way ("before")


def load(dd):
    z = np.load(os.path.join(dd, "mesh_uv.npz"))
    return z["Vr"], z["F"], z["uv"], np.asarray(Image.open(os.path.join(dd, "texture.png")).convert("RGB"), np.float32) / 255


def shot(elev, azim, res, bg=0.92):
    sx, sy, dz = N.project(Vr, elev, azim, res)
    fid, bary, _ = N.rasterize(sx, sy, dz, F, res, res)
    uv = N.interp(uvc, Fuv, fid, bary, 0.0)
    r = np.clip(np.round(uv[..., 1] * (T - 1)).astype(int), 0, T - 1)
    c = np.clip(np.round(uv[..., 0] * (T - 1)).astype(int), 0, T - 1)
    col = tex[T - 1 - r, c]  # v=1 is the top PNG row
    return np.where((fid >= 0)[..., None], col, bg)


def metrics(dd):
    global Vr, F, uvc, tex, T, Fuv
    Vr, F, uvc, tex = load(dd)
    T, Fuv = tex.shape[0], np.arange(len(F) * 3).reshape(-1, 3)
    out = {}
    for az, rf, nm in ((0, "ref_0.png", "front_"), (180, "ref_1.png", "back_")):
        rp = os.path.join(d, rf)
        if os.path.exists(rp):
            img = shot(0, az, 512)
            mask = np.abs(img - 0.92).max(-1) > 1e-4
            out.update(CM.measure(img, mask, np.asarray(Image.open(rp).convert("RGB"), np.float32) / 255, nm))
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


m_after = metrics(d)
m_before = metrics(base) if base and os.path.exists(os.path.join(base, "texture.png")) else None
Vr, F, uvc, tex = load(d)
T, Fuv = tex.shape[0], np.arange(len(F) * 3).reshape(-1, 3)
sp = os.path.join(d, "stats.json")
st = json.load(open(sp)) if os.path.exists(sp) else {}
st["colour_metrics"], st["colour_metrics_before"] = m_after, m_before
json.dump(st, open(sp, "w"), indent=1)
print(json.dumps({"after": m_after, "before": m_before}, indent=1))
sheet = np.concatenate([shot(0, a, 512) for a in (0, 90, 180, 270)], 1)
Image.fromarray((sheet * 255).astype(np.uint8)).save(os.path.join(d, "render_4views.png"))
fr = [Image.fromarray((shot(10, a, 320) * 255).astype(np.uint8)) for a in range(0, 360, 30)]
fr[0].save(os.path.join(d, "turntable.gif"), save_all=True, append_images=fr[1:], duration=120, loop=0)
print("preview ok")
