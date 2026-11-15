"""CI-only (uses PIL): textured 4-view sheet + turntable gif from a paint_phone.py output dir (mesh_uv.npz + texture.png)."""
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paint_np as N

d = sys.argv[1]
z = np.load(os.path.join(d, "mesh_uv.npz"))
Vr, F, uvc = z["Vr"], z["F"], z["uv"]
tex = np.asarray(Image.open(os.path.join(d, "texture.png")).convert("RGB"), np.float32) / 255  # row 0 = top
T = tex.shape[0]
Fuv = np.arange(len(F) * 3).reshape(-1, 3)


def shot(elev, azim, res, bg=0.92):
    sx, sy, dz = N.project(Vr, elev, azim, res)
    fid, bary, _ = N.rasterize(sx, sy, dz, F, res, res)
    uv = N.interp(uvc, Fuv, fid, bary, 0.0)
    r = np.clip(np.round(uv[..., 1] * (T - 1)).astype(int), 0, T - 1)
    c = np.clip(np.round(uv[..., 0] * (T - 1)).astype(int), 0, T - 1)
    col = tex[T - 1 - r, c]  # v=1 is the top PNG row
    return np.where((fid >= 0)[..., None], col, bg)


sheet = np.concatenate([shot(0, a, 512) for a in (0, 90, 180, 270)], 1)
Image.fromarray((sheet * 255).astype(np.uint8)).save(os.path.join(d, "render_4views.png"))
fr = [Image.fromarray((shot(10, a, 320) * 255).astype(np.uint8)) for a in range(0, 360, 30)]
fr[0].save(os.path.join(d, "turntable.gif"), save_all=True, append_images=fr[1:], duration=120, loop=0)
print("preview ok")
