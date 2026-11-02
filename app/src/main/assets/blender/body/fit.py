"""Fit Body Creator sliders to a reference shape (a TripoSR photo mesh, an imported model) by matching silhouettes.

Both shapes are reduced to a height-normalised profile: at 28 heights, the width and depth of the part of the body
that contains the centre line (so arms beside the torso are ignored when there is a gap). Coordinate search over a
few meaningful sliders minimises the profile difference. numpy only: runs in Blender and in plain-Python tests.
"""
import numpy as np

LEVELS = 28

# (spec section, key, low, high): searched continuously (pattern search with shrinking steps).
PARAMS = [
    ("sliders", "style/style-sd", 0, 1),
    ("sliders", "style/style-chibi", 0, 1),
    ("sliders", "style/style-cartoon", 0, 1),
    ("macros", "gender", 0, 1),
    ("macros", "weight", 0, 1),
    ("macros", "muscle", 0, 1),
    ("macros", "age", 0.15, 1),
    ("sliders", "style/style-chubby", 0, 1),
    ("sliders", "style/style-lanky", 0, 1),
    ("macros", "proportions", 0, 1),
    ("macros", "height", 0, 1),
]
DEFAULTS = {"gender": 0.5, "weight": 0.5, "muscle": 0.5, "age": 0.5, "proportions": 0.5, "height": 0.5}


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


def _occupied_at(c, centre, lo, hi, bins=64):
    if len(c) == 0 or hi - lo < 1e-9:
        return False
    h, _ = np.histogram(c, bins=bins, range=(lo, hi))
    k = int(np.clip((centre - lo) / (hi - lo) * bins, 0, bins - 1))
    return h[k] > 0


def smooth_profile(pts, n=40, reach=0.22):
    """Smooth proportions signature: at n heights (fraction of total height), the 90th-percentile half-width and
    half-depth of the body near the centre line (|x| < reach*H drops arms held out). Percentiles move continuously
    when the shape changes, so the fit can follow gradients (histogram measures are flat almost everywhere)."""
    p = pts - np.array([0, 0, pts[:, 2].min()])
    H = max(p[:, 2].max(), 1e-9)
    cx = np.median(p[:, 0])
    cy = np.median(p[:, 1])
    rx, ry = np.abs(p[:, 0] - cx) / H, np.abs(p[:, 1] - cy) / H
    near = rx < reach
    z = p[:, 2] / H
    out = np.zeros(2 * n)
    for i in range(n):
        c = (i + 0.5) / n
        m = near & (np.abs(z - c) < 0.75 / n)
        if m.sum() >= 3:
            out[i] = np.percentile(rx[m], 90)
            out[n + i] = np.percentile(ry[m], 90)
    return out


def features(pts):
    """Landmark-style proportions, all / height: head height (from the narrowest neck section), leg length (crotch:
    first height where the slice contains the centre line), head width, chest width and depth, waist, hips.
    The same code measures the reference and every candidate, so its biases cancel."""
    p = pts - np.array([0, 0, pts[:, 2].min()])
    H = max(p[:, 2].max(), 1e-9)
    cx = (p[:, 0].min() + p[:, 0].max()) / 2
    cy = (p[:, 1].min() + p[:, 1].max()) / 2
    n = 60
    zs = (np.arange(n) + 0.5) / n * H
    w, dpt, occ = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for i, z in enumerate(zs):
        s = p[np.abs(p[:, 2] - z) < 0.5 * H / n]
        w[i] = _central_extent(s[:, 0], cx) / H if len(s) else 0
        dpt[i] = _central_extent(s[:, 1], cy) / H if len(s) else 0
        occ[i] = _occupied_at(s[:, 0], cx, p[:, 0].min(), p[:, 0].max()) if len(s) else False
    upper = np.nonzero((zs > 0.55 * H) & (zs < 0.97 * H) & (w > 0))[0]
    neck = upper[np.argmin(w[upper])] if len(upper) else int(0.85 * n)
    crotch = next((i for i in range(int(0.08 * n), neck) if occ[i] and occ[min(i + 1, n - 1)]), int(0.45 * n))
    torso = np.arange(crotch + max(1, n // 20), max(crotch + 2, neck - n // 10))
    head = np.arange(neck, n)
    return np.array([
        (H - zs[neck]) / H,                          # head height ratio (chibi ~0.35, adult ~0.13)
        zs[crotch] / H,                              # leg length ratio
        w[head].max() if len(head) else 0,           # head width
        w[torso].max() if len(torso) else 0,         # chest/shoulder width
        w[torso].min() if len(torso) else 0,         # waist
        w[min(crotch + 1, n - 1)],                   # hips
        dpt[torso].max() if len(torso) else 0,       # body depth (belly / chest)
    ])


FEATURE_SCALE = np.array([0.02, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03])


def fit(pack, to_blender, body_idx, ref_pts, spec, log=print, passes=2):
    """Returns a copy of spec with fitted macros/sliders. to_blender maps MakeHuman coords to Blender axes."""
    target = profile(ref_pts)
    target_f = features(ref_pts)
    log("fit: reference features " + " ".join(f"{x:.3f}" for x in target_f))
    w = np.ones(2 * LEVELS)
    w[LEVELS:] = 0.6  # TripoSR depth is the least reliable (one view)
    cur = {"macros": dict(spec.get("macros", {})), "sliders": dict(spec.get("sliders", {}))}

    target_s = smooth_profile(ref_pts)

    def resid(x):
        c = {"macros": dict(cur["macros"]), "sliders": dict(cur["sliders"])}
        for (sec, key, _, _), val in zip(PARAMS, x):
            c[sec][key] = float(val)
        v = to_blender(pack.morph(c["macros"], c["sliders"]))[body_idx]
        # small pull toward neutral values: of two shapes that fit equally, prefer the simpler one (e.g. not
        # chubby + lanky cancelling out)
        reg = 1.5 * (x - neutral)
        return np.concatenate([(smooth_profile(v) - target_s) / 0.01, reg])

    neutral = np.array([DEFAULTS.get(k, 0.0) for _, k, _, _ in PARAMS], float)
    lo = np.array([p_[2] for p_ in PARAMS], float)
    hi = np.array([p_[3] for p_ in PARAMS], float)
    best_x, best = None, np.inf
    # Proportions respond almost linearly to the sliders, so Levenberg-Marquardt (finite-difference Jacobian, box
    # bounds by clipping) converges in a few steps. Start once per style family so a chibi is not fitted as a
    # squashed adult (the objective has one basin per family).
    starts = []
    for fam in (None, "style/style-chibi", "style/style-sd", "style/style-cartoon"):
        x0 = np.array([cur[sec].get(key, DEFAULTS.get(key, 0.0)) for sec, key, _, _ in PARAMS], float)
        if fam:
            x0[[k for sec, k, _, _ in PARAMS].index(fam)] = 1.0
        starts.append(x0)
    for x in starts:
        r = resid(x)
        e = float(r @ r)
        lam = 1e-2
        for it in range(8 * passes):
            J = np.empty((len(r), len(x)))
            for j in range(len(x)):
                h = 0.05 if x[j] + 0.05 <= hi[j] else -0.05
                xj = x.copy()
                xj[j] += h
                J[:, j] = (resid(xj) - r) / h
            A = J.T @ J
            g = J.T @ r
            moved = False
            for _ in range(4):
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-6), -g)
                xn = np.clip(x + step, lo, hi)
                rn = resid(xn)
                en = float(rn @ rn)
                if en < e:
                    x, r, e, lam, moved = xn, rn, en, lam * 0.3, True
                    break
                lam *= 10
            if not moved or np.abs(step).max() < 0.01:
                break
        log(f"fit: start {np.round(starts[len(starts) - 1] if False else x, 2).tolist()} error {e:.4f}")
        if e < best:
            best, best_x = e, x
    for (sec, key, _, _), val in zip(PARAMS, best_x):
        cur[sec][key] = round(float(val), 3)
    out = dict(spec)
    out["macros"], out["sliders"] = cur["macros"], {k: v for k, v in cur["sliders"].items() if v > 0.02}
    out["fit_error"] = round(best, 5)
    return out
