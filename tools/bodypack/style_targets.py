"""Stylization morphs for Body Creator, made from the base mesh itself (no artist sculpting needed).

Each style is a sum of smooth region edits on the default human: head / eyes / nose / limbs / torso. Regions come from
MakeHuman's own data, so the edits blend the way an artist would paint them:
  - bone-weight masks (weights.json): head = everything the head and its child bones move, arms, legs, torso
  - joint centres (skeleton.json): pivots for scaling the head about the neck, limbs about hip/shoulder
  - target footprints (body.npz): eyes and nose regions = vertices the eye / nose sliders move
Helper vertices (eyeballs, teeth) move with their region, so fitted eyes, hair and clothes follow the style.

Returned as extra targets "style-style-<name>" with slider ids "style/style-<name>" (0..1).
"""
import numpy as np

STYLES = {
    #            head  eyes  nose  arms  legs  torso_w  belly  shoulders
    "cartoon":  (1.30, 1.45, 0.75, 1.00, 0.94, 1.00, 1.00, 1.00),
    "chibi":    (1.65, 1.55, 0.60, 0.75, 0.70, 1.00, 1.08, 0.95),
    "pixar":    (1.15, 1.30, 0.85, 1.02, 0.96, 0.98, 1.00, 1.00),
    "heroic":   (0.90, 0.95, 1.05, 1.05, 1.10, 1.06, 0.95, 1.18),
    "lanky":    (0.95, 1.00, 1.10, 1.18, 1.18, 0.88, 0.90, 0.95),
    "chubby":   (1.08, 1.10, 1.00, 0.95, 0.90, 1.18, 1.35, 1.05),
}


def _subtree(bones, root):
    out, todo = set(), [root]
    while todo:
        b = todo.pop()
        out.add(b)
        todo += [k for k, v in bones.items() if v.get("parent") == b]
    return out


def _mask(n, weights, bones):
    m = np.zeros(n)
    for b in bones:
        for i, w in weights.get(b, []):
            m[i] += w
    return np.clip(m, 0, 1)


def _footprint(n, arrays, index, prefixes):
    """Per-vertex 0..1: how strongly the given slider groups move each vertex (normalised max |delta|)."""
    m = np.zeros(n)
    for t, meta in enumerate(index):
        if any(meta["key"].startswith(p) for p in prefixes):
            a, b = arrays["t_off"][t], arrays["t_off"][t + 1]
            d = np.linalg.norm(arrays["t_delta"][a:b].astype(np.float32), axis=1)
            np.maximum.at(m, arrays["t_idx"][a:b], d)
    return np.clip(m / max(np.percentile(m[m > 0], 90), 1e-6), 0, 1) if m.any() else m


def make(arrays, index, skel, weights):
    v = arrays["verts"].astype(np.float64)  # decimetres, Y up, faces +Z
    n = len(v)
    joint = lambda j: v[skel["joints"][j]].mean(0)  # noqa: E731
    bones = skel["bones"]
    head = _mask(n, weights, _subtree(bones, "head"))
    arms = _mask(n, weights, _subtree(bones, "upperarm01.L") | _subtree(bones, "upperarm01.R"))
    legs = _mask(n, weights, _subtree(bones, "upperleg01.L") | _subtree(bones, "upperleg01.R"))
    torso = _mask(n, weights, {b for b in bones if b.startswith(("spine", "breast", "pelvis"))})
    eyes = _footprint(n, arrays, index, ("eyes-l-eye", "eyes-r-eye")) * head
    nose = _footprint(n, arrays, index, ("nose-",)) * head
    neck = joint("head____head")
    eye_c = {s: joint(f"eye.{s}____head") for s in "LR"}
    nose_c = (v[nose > 0.5].mean(0) if (nose > 0.5).any() else neck)
    hip = {s: joint(f"upperleg01.{s}____head") for s in "LR"}
    sh = {s: joint(f"upperarm01.{s}____head") for s in "LR"}
    chest = joint("spine03____head") if "spine03____head" in skel["joints"] else v[torso > 0.5].mean(0)
    belly_c = joint("spine05____head") if "spine05____head" in skel["joints"] else chest

    def styled(h, e, no, ar, lg, tw, bl, shd):
        d = np.zeros_like(v)
        d += head[:, None] * (v - neck) * (h - 1)  # head about the neck joint (face features ride along)
        d += nose[:, None] * (v - nose_c) * (no - 1)
        tor = torso * (1 - head) * (1 - arms) * (1 - legs)
        d[:, 0] += tor * (v[:, 0] - chest[0]) * (tw - 1)  # torso width / depth
        d[:, 2] += tor * (v[:, 2] - chest[2]) * (tw - 1)
        bel = tor * np.exp(-((v[:, 1] - belly_c[1]) / 1.6) ** 2)  # belly: push the front forward
        d[:, 2] += bel * np.maximum(v[:, 2] - belly_c[2], 0) * (bl - 1)
        top = tor * np.clip((v[:, 1] - chest[1]) / 1.5 + 0.5, 0, 1)  # shoulders: widen the upper chest
        d[:, 0] += top * (v[:, 0] - chest[0]) * (shd - 1)
        for s in "LR":
            side = (v[:, 0] > 0) if s == "L" else (v[:, 0] <= 0)  # +X is the character's left
            # eyes: smooth radial falloff around the eyeball (sockets, lids and brows stretch together, no creases)
            r = np.linalg.norm(v - eye_c[s], axis=1)
            fall = np.exp(-(r / 0.26) ** 2) * head * side
            d += fall[:, None] * (v - eye_c[s]) * (e - 1)
            d[:, 1] += legs * side * (v[:, 1] - hip[s][1]) * (lg - 1)  # leg length about the hip (feet re-grounded later)
            d += (arms * side)[:, None] * (v - sh[s]) * (ar - 1)  # arm length/size about the shoulder
            d[:, 0] += arms * side * (sh[s][0] - chest[0]) * (shd - 1)  # arms ride the wider shoulders
        return d

    out = []
    for name, p in STYLES.items():
        d = styled(*p)
        keep = np.nonzero(np.linalg.norm(d, axis=1) > 1e-4)[0]
        out.append((name, keep.astype(np.int32), d[keep].astype(np.float16)))
    return out


def add_to(arrays, index, modifiers, skel, weights):
    """Append the style targets to the pack arrays (in place) and a "style" slider group to modifiers."""
    styles = make(arrays, index, skel, weights)
    idx, dl, off = [arrays["t_idx"]], [arrays["t_delta"]], list(arrays["t_off"])
    for name, k, d in styles:
        idx.append(k); dl.append(d); off.append(off[-1] + len(k))
        index.append({"key": f"style-style-{name}", "vars": []})
    arrays["t_idx"], arrays["t_delta"], arrays["t_off"] = np.concatenate(idx), np.concatenate(dl), np.array(off, np.int64)
    modifiers.insert(0, {"group": "style", "modifiers": [{"target": f"style-{n}"} for n in STYLES]})
    return [n for n, _, _ in styles]
