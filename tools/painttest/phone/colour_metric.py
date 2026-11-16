"""Colour-fidelity metric (numpy + PIL): textured front/back render vs the preprocessed reference photos (ref_0.png / ref_1.png).
Render and reference foregrounds are cropped to their bounding boxes and resampled to a common grid; regions are picked in the
reference by hue (orange: scarf/cuffs/boots, green: suit) and compared by mean colour (CIE Lab delta E) and mean chroma."""
import numpy as np
from PIL import Image


def srgb2lab(rgb):
    c = np.clip(rgb, 0, 1).astype(np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * np.array([0.95047, 1.0, 1.08883])
    Mi = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]))
    lin = np.clip(xyz @ Mi.T, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def hsv(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-9)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0), mx


def ref_fg(ref):
    return np.abs(ref - 1.0).max(-1) > 0.05


def crop_resize(img, mask, n=256):
    ys, xs = np.nonzero(mask)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    im = Image.fromarray((np.clip(img[sl], 0, 1) * 255 + 0.5).astype(np.uint8)).resize((n, n), Image.BILINEAR)
    mk = Image.fromarray((mask[sl] * 255).astype(np.uint8)).resize((n, n), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255, np.asarray(mk) > 127


def erode(m, k):
    m = m.copy()
    for ax in (0, 1):
        a = m.copy()
        for s in range(1, k + 1):
            a &= np.roll(m, s, ax) & np.roll(m, -s, ax)
        m = a
    return m


def region_masks(ref):
    h, s, v = hsv(ref)
    fg = ref_fg(ref)
    return {"orange": fg & (h > 12) & (h < 40) & (s > 0.45) & (v > 0.5),
            "green": fg & (h > 100) & (h < 220) & (s > 0.07) & (v > 0.25)}


def measure(render, render_mask, ref, name=""):
    """render (H,W,3) 0..1 + mask, ref (H,W,3) white bg. -> dict of metrics."""
    rf = ref_fg(ref)
    ra, rm = crop_resize(render, render_mask)
    fa, fm = crop_resize(ref, rf)
    out = {}
    both = erode(rm & fm, 2)
    la, lf = srgb2lab(ra), srgb2lab(fa)
    ch = lambda l: np.hypot(l[..., 1], l[..., 2])  # noqa: E731
    out["chroma_render"], out["chroma_ref"] = float(ch(la)[rm].mean()), float(ch(lf)[fm].mean())
    out["chroma_ratio"] = out["chroma_render"] / out["chroma_ref"]
    out["L_render"], out["L_ref"] = float(la[..., 0][rm].mean()), float(lf[..., 0][fm].mean())
    out["deltaE_fg"] = float(np.linalg.norm(la - lf, axis=-1)[both].mean())
    for k, m in region_masks(fa).items():
        m = erode(m, 3) & rm
        if m.sum() < 30:
            continue
        a, b = la[m].mean(0), lf[m].mean(0)
        out[f"{k}_ref_rgb"] = [round(float(x), 3) for x in fa[m].mean(0)]
        out[f"{k}_render_rgb"] = [round(float(x), 3) for x in ra[m].mean(0)]
        out[f"{k}_deltaE"] = float(np.linalg.norm(a - b))
        out[f"{k}_chroma_ratio"] = float(np.hypot(a[1], a[2]) / max(np.hypot(b[1], b[2]), 1e-6))
    return {name + k: v for k, v in out.items()}
