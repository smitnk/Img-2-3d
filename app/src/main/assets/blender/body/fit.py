"""Fit Body Creator sliders to a reference shape (a TripoSR photo mesh, an imported model) by matching silhouettes.

Both shapes are reduced to a height-normalised profile: at 28 heights, the width and depth of the part of the body
that contains the centre line (so arms beside the torso are ignored when there is a gap). Coordinate search over a
few meaningful sliders minimises the profile difference. numpy only: runs in Blender and in plain-Python tests.
"""
import numpy as np

LEVELS = 28

# (spec section, key, candidate values); styles are 0..1 slider weights, macros 0..1.
PARAMS = [
    ("sliders", "style/style-sd", (0, 0.5, 1)),
    ("sliders", "style/style-chibi", (0, 0.5, 1)),
    ("sliders", "style/style-cartoon", (0, 0.5, 1)),
    ("macros", "weight", (0.1, 0.3, 0.5, 0.7, 0.9)),
    ("macros", "muscle", (0.2, 0.5, 0.8)),
    ("macros", "gender", (0.1, 0.5, 0.9)),
    ("sliders", "style/style-chubby", (0, 0.5, 1)),
    ("sliders", "style/style-lanky", (0, 0.5, 1)),
    ("macros", "age", (0.25, 0.5, 0.75)),
    ("sliders", "toonhead/toonhead-round", (0, 0.5, 1)),
]


def _central_extent(c, centre, bins=64):
    """Extent of the occupied interval that contains `centre` (1D histogram walk)."""
    if len(c) == 0:
        return 0.0
    lo, hi = c.min(), c.max()
    if hi - lo < 1e-9:
        return 0.0
    h, edges = np.histogram(c, bins=bins, range=(lo, hi))
    k = int(np.clip((centre - lo) / (hi - lo) * bins, 0, bins - 1))
    if h[k] == 0:  # centre falls in a gap (e.g. between legs): take the nearest occupied bin
        occ = np.nonzero(h)[0]
        k = occ[np.argmin(np.abs(occ - k))]
    a = b = k
    while a > 0 and h[a - 1] > 0:
        a -= 1
    while b < bins - 1 and h[b + 1] > 0:
        b += 1
    return edges[b + 1] - edges[a]


def profile(pts):
    """pts: (N, 3) Blender axes (Z up). Returns a 2*LEVELS vector of width/height and depth/height."""
    p = pts - np.array([0, 0, pts[:, 2].min()])
    h = max(p[:, 2].max(), 1e-9)
    cx, cy = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
    out = np.zeros(2 * LEVELS)
    for i in range(LEVELS):
        z0, z1 = (i + 0.15) / LEVELS * h, (i + 0.85) / LEVELS * h
        s = p[(p[:, 2] >= z0) & (p[:, 2] < z1)]
        out[i] = _central_extent(s[:, 0], cx) / h
        out[LEVELS + i] = _central_extent(s[:, 1], cy) / h
    return out


def fit(pack, to_blender, body_idx, ref_pts, spec, log=print, passes=2):
    """Returns a copy of spec with fitted macros/sliders. to_blender maps MakeHuman coords to Blender axes."""
    target = profile(ref_pts)
    w = np.ones(2 * LEVELS)
    w[LEVELS:] = 0.6  # TripoSR depth is the least reliable (one view)
    cur = {"macros": dict(spec.get("macros", {})), "sliders": dict(spec.get("sliders", {}))}

    def err(c):
        v = to_blender(pack.morph(c["macros"], c["sliders"]))[body_idx]
        return float((w * (profile(v) - target) ** 2).sum())

    best = err(cur)
    log(f"fit: start error {best:.4f}")
    for _ in range(passes):
        for sec, key, values in PARAMS:
            for val in values:
                trial = {"macros": dict(cur["macros"]), "sliders": dict(cur["sliders"])}
                trial[sec][key] = val
                e = err(trial)
                if e < best - 1e-6:
                    best, cur = e, trial
        log(f"fit: pass error {best:.4f}")
    out = dict(spec)
    out["macros"], out["sliders"] = cur["macros"], {k: v for k, v in cur["sliders"].items() if v}
    out["fit_error"] = round(best, 5)
    return out
