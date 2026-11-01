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

# Proportion multipliers (1 = unchanged). Head-height reference: realistic adult ~7.5 heads tall, cartoon ~5,
# chibi ~3, super-deformed (SD) ~2 to 2.5.
P = ("head", "eyes", "eye_round", "nose", "mouth", "neck_len", "neck_w", "arms", "legs", "limb_w", "hands", "feet",
     "torso_w", "belly", "shoulders", "torso_len")
STYLES = {
    "cartoon": dict(eye_round=1.15, head=1.30, eyes=1.70, nose=0.70, mouth=0.90, neck_len=0.85, neck_w=1.10, legs=0.92, limb_w=1.10,
                    hands=1.15, feet=1.15, belly=1.05),
    "pixar":   dict(eye_round=1.08, head=1.20, eyes=1.38, nose=0.85, neck_len=0.90, neck_w=1.05, legs=0.95, limb_w=1.05, hands=1.10, feet=1.05),
    "chibi":   dict(eye_round=1.25, head=1.90, eyes=2.10, nose=0.55, mouth=0.80, neck_len=0.50, neck_w=1.30, arms=0.72, legs=0.62,
                    limb_w=1.55, hands=1.25, feet=1.35, torso_w=1.08, belly=1.15, shoulders=0.95, torso_len=0.85),
    "sd":      dict(eye_round=1.30, head=2.40, eyes=2.40, nose=0.45, mouth=0.75, neck_len=0.30, neck_w=1.50, arms=0.60, legs=0.45,
                    limb_w=1.95, hands=1.40, feet=1.55, torso_w=1.15, belly=1.20, shoulders=0.90, torso_len=0.75),
    "heroic":  dict(head=0.90, eyes=0.95, nose=1.05, neck_w=1.15, arms=1.05, legs=1.10, limb_w=1.10, hands=1.10,
                    torso_w=1.06, belly=0.95, shoulders=1.20),
    "lanky":   dict(head=0.95, nose=1.10, neck_len=1.20, neck_w=0.90, arms=1.18, legs=1.18, limb_w=0.85, hands=1.05,
                    feet=1.05, torso_w=0.88, belly=0.90, shoulders=0.95, torso_len=1.08),
    "chubby":  dict(head=1.08, eyes=1.10, neck_len=0.85, neck_w=1.25, arms=0.95, legs=0.90, limb_w=1.25, hands=1.05,
                    torso_w=1.18, belly=1.35, shoulders=1.05),
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
    sub = lambda b: _mask(n, weights, _subtree(bones, b))  # noqa: E731
    head = sub("head")
    neck = np.clip(sub("neck01") - head, 0, 1)
    arms = np.clip(sub("upperarm01.L") + sub("upperarm01.R"), 0, 1)
    legs = np.clip(sub("upperleg01.L") + sub("upperleg01.R"), 0, 1)
    hands = np.clip(sub("wrist.L") + sub("wrist.R"), 0, 1)
    feet = np.clip(sub("foot.L") + sub("foot.R"), 0, 1)
    torso = np.clip(_mask(n, weights, {b for b in bones if b.startswith(("spine", "breast", "pelvis", "clavicle", "shoulder"))}), 0, 1)
    tor = torso * (1 - head) * (1 - arms) * (1 - legs) * (1 - neck)
    nose = _footprint(n, arrays, index, ("nose-",)) * head
    mouth = _footprint(n, arrays, index, ("mouth-",)) * head
    head_base = joint("head____head")
    # Grow the head from the chin line, so a bigger head rises above the shoulders instead of swallowing the neck.
    head_pivot = np.array([head_base[0], v[head > 0.9, 1].min(), head_base[2]])
    neck_base = joint("neck01____head")
    eye_c = {s: joint(f"eye.{s}____head") for s in "LR"}
    nose_c = v[nose > 0.5].mean(0) if (nose > 0.5).any() else head_base
    mouth_c = v[mouth > 0.5].mean(0) if (mouth > 0.5).any() else head_base
    hip = {s: joint(f"upperleg01.{s}____head") for s in "LR"}
    ankle = {s: joint(f"foot.{s}____head") for s in "LR"}
    sh = {s: joint(f"upperarm01.{s}____head") for s in "LR"}
    wrist = {s: joint(f"wrist.{s}____head") for s in "LR"}
    chest = joint("spine03____head") if "spine03____head" in skel["joints"] else v[torso > 0.5].mean(0)
    belly_c = joint("spine05____head") if "spine05____head" in skel["joints"] else chest
    hip_y = (hip["L"][1] + hip["R"][1]) / 2
    side = {"L": v[:, 0] > 0, "R": v[:, 0] <= 0}  # +X is the character's left

    def radial(mask, a, b, f):
        """Thicken/thin a limb: scale the distance from its axis a->b."""
        ax = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        r = v - a
        perp = r - (r @ ax)[:, None] * ax
        return mask[:, None] * perp * (f - 1)

    f = arrays["f_body"]
    on_body = np.zeros(n, bool)
    on_body[f.ravel()] = True

    def styled(p):
        k = {x: p.get(x, 1.0) for x in P}
        d = np.zeros_like(v)
        # proportions of the whole figure: torso length moves everything above the hips with it
        tl = k["torso_len"] - 1
        d[:, 1] += tor * (v[:, 1] - hip_y) * tl + (head + neck) * (neck_base[1] - hip_y) * tl + arms * (sh["L"][1] - hip_y) * tl
        # neck: shorter/longer about its base, the head rides on top; thicker/thinner around its axis
        nl = k["neck_len"] - 1
        d[:, 1] += neck * (v[:, 1] - neck_base[1]) * nl + head * (head_base[1] - neck_base[1]) * nl
        d += radial(neck, neck_base, head_base, k["neck_w"])
        # head about its pivot, then face features about their own centres (eyes, nose, mouth)
        d += head[:, None] * (v - head_pivot) * (k["head"] - 1)
        d += nose[:, None] * (v - nose_c) * (k["nose"] - 1)
        d += mouth[:, None] * (v - mouth_c) * (k["mouth"] - 1)
        for s in "LR":
            r = np.linalg.norm(v - eye_c[s], axis=1)
            fall = np.exp(-(r / 0.26) ** 2) * head * side[s]  # sockets, lids and brows stretch with the eyeball
            # cartoon eyes are rounder, not just bigger: open them more vertically than horizontally
            # (the eyeball itself stays a sphere, sized to the opened socket, so it never pokes through the lids)
            sc = np.where(on_body[:, None], np.array([1.0, k["eye_round"], 1.0]), (1 + k["eye_round"]) / 2) * k["eyes"]
            d += fall[:, None] * (v - eye_c[s]) * (sc - 1)
            arm, leg = arms * side[s], legs * side[s]
            d += arm[:, None] * (v - sh[s]) * (k["arms"] - 1)  # arm size about the shoulder
            d += radial(arm * (1 - hands), sh[s], wrist[s], k["limb_w"])
            d += (hands * side[s])[:, None] * (v - wrist[s]) * (k["hands"] - 1)
            d += leg[:, None] * (v - hip[s]) * (k["legs"] - 1)  # leg size about the hip (feet re-grounded at build)
            d += radial(leg * (1 - feet), hip[s], ankle[s], k["limb_w"])
            d += (feet * side[s])[:, None] * (v - ankle[s]) * (k["feet"] - 1)
            d[:, 0] += arm * (sh[s][0] - chest[0]) * (k["shoulders"] - 1)  # arms ride the wider shoulders
        d[:, 0] += tor * (v[:, 0] - chest[0]) * (k["torso_w"] - 1)
        d[:, 2] += tor * (v[:, 2] - chest[2]) * (k["torso_w"] - 1)
        bel = tor * np.exp(-((v[:, 1] - belly_c[1]) / 1.6) ** 2)
        d[:, 2] += bel * np.maximum(v[:, 2] - belly_c[2], 0) * (k["belly"] - 1)
        top = tor * np.clip((v[:, 1] - chest[1]) / 1.5 + 0.5, 0, 1)
        d[:, 0] += top * (v[:, 0] - chest[0]) * (k["shoulders"] - 1)
        return d

    # Blend the seams: different edits meet where masks overlap (shoulder, hip, neck) and would leave folds. Laplacian-
    # smoothing the displacement field over the body surface removes the folds; a linear (scale) field is unchanged by it.
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 3]], f[:, [3, 0]]])
    edges = edges[edges[:, 0] != edges[:, 1]]
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.float64)

    def smooth(d, it=40):
        for _ in range(it):
            acc = np.zeros_like(d)
            np.add.at(acc, edges[:, 0], d[edges[:, 1]])
            np.add.at(acc, edges[:, 1], d[edges[:, 0]])
            avg = acc / np.maximum(deg, 1)[:, None]
            d = np.where(on_body[:, None], 0.5 * d + 0.5 * avg, d)
        return d

    out = []
    for name, p in STYLES.items():
        d = smooth(styled(p))
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
