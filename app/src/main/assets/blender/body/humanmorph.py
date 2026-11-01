"""MakeHuman-compatible morph engine (numpy only; runs in Blender and in plain Python tests).

A body = base mesh + sum(weight_t * delta_t). weight_t = product of the macro variable values in the target's
name (gender/age/race/muscle/weight/height/cup/firmness/proportions, MakeHuman lib/targets.py) x its group factor:
1 for macro groups, else the slider value split into left/right (MakeHuman humanmodifier.UniversalModifier).
"""
import json
import os

import numpy as np

MACRO_GROUPS = {"macrodetails", "macrodetails-universal", "macrodetails-height", "macrodetails-proportions", "breast"}
DEFAULT_MACROS = dict(gender=0.5, age=0.5, muscle=0.5, weight=0.5, height=0.5, proportions=0.5,
                      african=1 / 3, asian=1 / 3, caucasian=1 / 3, cup=0.5, firmness=0.5)


def _tri(v, lo, mid, hi):
    """MakeHuman's min/average/max split of a 0..1 macro value."""
    mx, mn = max(0.0, v * 2 - 1), max(0.0, 1 - v * 2)
    return {hi: mx, lo: mn, mid: 1 - max(mx, mn)}


def macro_values(m):
    m = dict(DEFAULT_MACROS, **(m or {}))
    c = lambda k: min(1.0, max(0.0, float(m[k])))  # noqa: E731
    f = {"male": c("gender"), "female": 1 - c("gender")}
    a = c("age")
    if a < 0.5:
        young = max(0.0, (a - 0.1875) * 3.2)
        f.update(old=0.0, baby=max(0.0, 1 - a * 5.333), young=young, child=max(0.0, min(1.0, 5.333 * a) - young))
    else:
        f.update(child=0.0, baby=0.0, old=max(0.0, a * 2 - 1), young=1 - max(0.0, a * 2 - 1))
    race = np.array([c("african"), c("asian"), c("caucasian")], np.float64)
    race = race / race.sum() if race.sum() > 0 else np.full(3, 1 / 3)
    f.update(african=race[0], asian=race[1], caucasian=race[2])
    f.update(_tri(c("muscle"), "minmuscle", "averagemuscle", "maxmuscle"))
    f.update(_tri(c("weight"), "minweight", "averageweight", "maxweight"))
    f.update(_tri(c("height"), "minheight", "averageheight", "maxheight"))
    f.update(_tri(c("cup"), "mincup", "averagecup", "maxcup"))
    f.update(_tri(c("firmness"), "minfirmness", "averagefirmness", "maxfirmness"))
    f.update(_tri(c("proportions"), "uncommonproportions", "regularproportions", "idealproportions"))
    return f


def slider_factors(sliders, modifiers):
    """sliders: {"head/head-fat-decr|incr": -0.4, "head/head-oval": 0.7} -> {target group key: factor}."""
    known = {}
    for g in modifiers:
        for m in g["modifiers"]:
            if "target" not in m:
                continue
            base = f'{g["group"]}-{m["target"]}'
            if "min" in m:
                known[f'{g["group"]}/{m["target"]}-{m["min"]}|{m["max"]}'] = (f'{base}-{m["min"]}', f'{base}-{m["max"]}')
            else:
                known[f'{g["group"]}/{m["target"]}'] = (None, base)
    out = {}
    for name, val in (sliders or {}).items():
        if name not in known:
            raise KeyError(f"unknown slider {name}")
        left, right = known[name]
        v = max(-1.0 if left else 0.0, min(1.0, float(val)))
        if left:
            out[left] = out.get(left, 0.0) + max(0.0, -v)
        out[right] = out.get(right, 0.0) + max(0.0, v)
    return out


class Pack:
    def __init__(self, folder):
        self.folder = folder
        self.npz = np.load(os.path.join(folder, "body.npz"))
        self.index = json.load(open(os.path.join(folder, "targets.json")))
        self.modifiers = json.load(open(os.path.join(folder, "modifiers.json")))
        self.base = self.npz["verts"].astype(np.float64)
        self.t_idx, self.t_delta, self.t_off = self.npz["t_idx"], self.npz["t_delta"], self.npz["t_off"]

    def weights(self, macros=None, sliders=None, extra=None):
        mv = macro_values(macros)
        sf = slider_factors(sliders, self.modifiers)
        sf.update(extra or {})
        w = np.zeros(len(self.index))
        for i, t in enumerate(self.index):
            g = 1.0 if t["key"] in MACRO_GROUPS else sf.get(t["key"], 0.0)
            if g:
                w[i] = g * np.prod([mv[v] for v in t["vars"]]) if t["vars"] else g
        # toon head shapes follow the head size the art styles give (a chibi head is ~2x: its shape change is too)
        hs = 1.0 + sum(w[i] * (t["head"] - 1) for i, t in enumerate(self.index) if "head" in t)
        for i, t in enumerate(self.index):
            if t.get("head_scaled"):
                w[i] *= hs
        return w

    def morph(self, macros=None, sliders=None, extra=None):
        w = self.weights(macros, sliders, extra)
        v = self.base.copy()
        for i in np.nonzero(np.abs(w) > 1e-6)[0]:
            a, b = self.t_off[i], self.t_off[i + 1]
            np.add.at(v, self.t_idx[a:b], w[i] * self.t_delta[a:b].astype(np.float64))
        return v

    def fit_proxy(self, v, refs, w, off, scale):
        """MakeHuman proxy fitting (shared/proxy.py): barycentric refs + offset scaled by the body's own size."""
        s = np.array([1.0 if np.isnan(d) else abs(v[int(a), k] - v[int(b), k]) / d for k, (a, b, d) in enumerate(scale)])
        return (v[refs] * w[..., None]).sum(1) + off * s


if __name__ == "__main__":  # self-check against the pack: default human is ~1.7 m and in front of its own feet
    import sys
    p = Pack(sys.argv[1])
    for name, mac in (("default", {}), ("tall heavy man", dict(gender=1, height=1, weight=1)),
                      ("child", dict(age=0.1)), ("muscular woman", dict(gender=0, muscle=1))):
        v = p.morph(mac)
        body = np.unique(p.npz["f_body"])
        h = (v[body, 1].max() - v[body, 1].min()) / 10
        print(f"{name:16s} height {h:.2f} m")
        assert 0.6 < h < 2.6
