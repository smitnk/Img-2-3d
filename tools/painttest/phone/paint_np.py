"""numpy-only parts of the phone texture pipeline (no torch, no PIL, no scipy): software rasteriser, Hunyuan3D-2 paint
camera/maps, back-projection bake, UV fill, LCM scheduler, tiled VAE blending, reference preprocessing.
Conventions copied from Hunyuan3D-2 (hy3dgen/texgen/differentiable_renderer): orthographic camera scale 1.2 at distance 1.45,
mesh frame (x,y,z)->(-x,z,-y) after centring / scaling to 1.15, pixel row 0 = bottom of the camera image."""
import math

import numpy as np

ELEVS = [0, 0, 0, 0, 90, -90]
AZIMS = [0, 90, 180, 270, 0, 180]
VWEIGHTS = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
VNAMES = ["front", "left", "back", "right", "top", "bottom"]
ORTHO_HALF = 0.6
CAM_DIST = 1.45


def camera_info():
    return [(((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el]
            for az, el in zip(AZIMS, ELEVS)]


# ------------------------------------------------------------------ mesh frame / camera
def to_render_frame(V, rot_y180=True):
    """V: vertices in glTF (Y-up) coordinates as trimesh would load them -> Hunyuan renderer frame, centred, scale 1.15."""
    V = np.asarray(V, np.float64)
    if rot_y180:
        V = V * np.array([-1, 1, -1])
    V = np.stack([-V[:, 0], V[:, 2], -V[:, 1]], 1)
    c = (V.max(0) + V.min(0)) / 2
    s = np.linalg.norm(V - c, axis=1).max() * 2.0
    return ((V - c) * (1.15 / s)).astype(np.float32), 1.15


def mv_matrix(elev, azim, dist=CAM_DIST):
    elev = -elev
    azim = azim + 90
    er, ar = math.radians(elev), math.radians(azim)
    pos = np.array([dist * math.cos(er) * math.cos(ar), dist * math.cos(er) * math.sin(ar), dist * math.sin(er)])
    look = -pos / np.linalg.norm(pos)
    right = np.cross(look, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, look)
    up /= np.linalg.norm(up)
    R = np.stack([right, up, -look], 1)  # c2w rotation
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ pos
    return w2c.astype(np.float32), look.astype(np.float32)


def project(V, elev, azim, res):
    """-> screen x, y (pixel units, shim convention), depth (smaller = nearer)."""
    m, _ = mv_matrix(elev, azim)
    c = V @ m[:3, :3].T + m[:3, 3]
    sx = (c[:, 0] / ORTHO_HALF * 0.5 + 0.5) * (res - 1) + 0.5
    sy = (c[:, 1] / ORTHO_HALF * 0.5 + 0.5) * (res - 1) + 0.5
    return sx, sy, -c[:, 2]


# ------------------------------------------------------------------ rasteriser
def rasterize(sx, sy, z, F, W, H, chunk=1_000_000):
    """z-buffer over screen-space triangles. Returns face id (H,W) int32 (-1 = none), barycentrics (H,W,3), depth (H,W)."""
    sx, sy, z = (np.asarray(a, np.float64) for a in (sx, sy, z))
    x0, y0, x1, y1, x2, y2 = (sx[F[:, 0]], sy[F[:, 0]], sx[F[:, 1]], sy[F[:, 1]], sx[F[:, 2]], sy[F[:, 2]])
    area = (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0)
    xmn = np.ceil(np.minimum(np.minimum(x0, x1), x2) - 0.5).astype(np.int64)
    xmx = np.floor(np.maximum(np.maximum(x0, x1), x2) - 0.5).astype(np.int64)
    ymn = np.ceil(np.minimum(np.minimum(y0, y1), y2) - 0.5).astype(np.int64)
    ymx = np.floor(np.maximum(np.maximum(y0, y1), y2) - 0.5).astype(np.int64)
    xmn, ymn = np.maximum(xmn, 0), np.maximum(ymn, 0)
    xmx, ymx = np.minimum(xmx, W - 1), np.minimum(ymx, H - 1)
    ok = (area != 0) & (xmx >= xmn) & (ymx >= ymn)
    ids = np.nonzero(ok)[0]
    k = np.maximum(xmx[ids] - xmn[ids] + 1, ymx[ids] - ymn[ids] + 1)
    cls = np.ceil(np.log2(np.maximum(k, 1))).astype(np.int64)
    P, D, FI = [], [], []
    for c in np.unique(cls):
        sel = ids[cls == c]
        kk = 1 << int(c)
        step = max(1, chunk // (kk * kk))
        off = np.arange(kk)
        for s in range(0, len(sel), step):
            f = sel[s:s + step]
            px = xmn[f][:, None, None] + off[None, None, :]
            py = ymn[f][:, None, None] + off[None, :, None]
            px, py = np.broadcast_to(px, (len(f), kk, kk)), np.broadcast_to(py, (len(f), kk, kk))
            qx, qy = px + 0.5, py + 0.5
            inv = (1.0 / area[f])[:, None, None]
            X0, Y0 = x0[f][:, None, None], y0[f][:, None, None]
            beta = ((x2[f][:, None, None] - X0) * (qy - Y0) - (qx - X0) * (y2[f][:, None, None] - Y0)) * inv
            gamma = ((qx - X0) * (y1[f][:, None, None] - Y0) - (x1[f][:, None, None] - X0) * (qy - Y0)) * inv
            alpha = 1.0 - beta - gamma
            m = (alpha >= 0) & (beta >= 0) & (gamma >= 0) & (px <= xmx[f][:, None, None]) & (py <= ymx[f][:, None, None])
            fi, yi, xi = np.nonzero(m)
            if not len(fi):
                continue
            a_, b_, g_ = alpha[fi, yi, xi], beta[fi, yi, xi], gamma[fi, yi, xi]
            ff = f[fi]
            P.append(py[fi, yi, xi] * W + px[fi, yi, xi])
            D.append(a_ * z[F[ff, 0]] + b_ * z[F[ff, 1]] + g_ * z[F[ff, 2]])
            FI.append(ff)
    fid = np.full(H * W, -1, np.int32)
    dep = np.full(H * W, np.inf, np.float32)
    if P:
        P, D, FI = np.concatenate(P), np.concatenate(D), np.concatenate(FI)
        o = np.argsort(-D, kind="stable")  # far first, nearest written last
        fid[P[o]] = FI[o]
        dep[P[o]] = D[o]
    fid, dep = fid.reshape(H, W), dep.reshape(H, W)
    bary = np.zeros((H, W, 3), np.float32)
    yy, xx = np.nonzero(fid >= 0)
    if len(yy):
        f = fid[yy, xx]
        qx, qy = xx + 0.5, yy + 0.5
        inv = 1.0 / area[f]
        beta = ((x2[f] - x0[f]) * (qy - y0[f]) - (qx - x0[f]) * (y2[f] - y0[f])) * inv
        gamma = ((qx - x0[f]) * (y1[f] - y0[f]) - (x1[f] - x0[f]) * (qy - y0[f])) * inv
        bary[yy, xx] = np.stack([1 - beta - gamma, beta, gamma], 1)
    return fid, bary, dep


def interp(attr, F, fid, bary, bg):
    """attr (N,C) per vertex -> (H,W,C), bg where no face."""
    out = np.empty(fid.shape + (attr.shape[1],), np.float32)
    out[:] = bg
    yy, xx = np.nonzero(fid >= 0)
    f = fid[yy, xx]
    out[yy, xx] = (attr[F[f, 0]] * bary[yy, xx, 0:1] + attr[F[f, 1]] * bary[yy, xx, 1:2] + attr[F[f, 2]] * bary[yy, xx, 2:3])
    return out


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    acc = np.zeros_like(V)
    for i in range(3):
        np.add.at(acc, F[:, i], n)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)


def box_down(img, f):
    h, w = img.shape[:2]
    return img.reshape(h // f, f, w // f, f, -1).mean((1, 3))


class Views:
    """Renders the 6 views at `res` px: face ids / barycentrics / depth, normal and position maps (0..1, bg 1)."""

    def __init__(self, V, F, res=1024, log=print):
        self.V, self.F, self.res = V, F, res
        self.vn = vertex_normals(V, F)
        self.pos_attr = 0.5 - V / 1.15
        self.r = []
        for e, a in zip(ELEVS, AZIMS):
            sx, sy, z = project(V, e, a, res)
            self.r.append(rasterize(sx, sy, z, F, res, res))
            log(f"raster view {e},{a}: {(self.r[-1][0] >= 0).mean():.3f} covered")

    def maps(self, i, out=512):
        fid, bary, _ = self.r[i]
        n = interp((self.vn + 1) * 0.5, self.F, fid, bary, 1.0)
        p = interp(self.pos_attr, self.F, fid, bary, 1.0)
        f = self.res // out
        # image rows: raster row 0 is the PNG's top row (same as the CUDA rasteriser's output)
        self.r[i] = (fid, None, self.r[i][2])  # barycentrics are not needed any more (bake gathers per texel)
        return [np.clip(box_down(m, f), 0, 1) for m in (n, p)]

    def free(self, i):
        self.r[i] = None


# ------------------------------------------------------------------ bake
def erode(mask, k):
    """binary erosion with a (2k+1)^2 square."""
    m = mask.copy()
    for ax in (0, 1):
        acc = m.copy()
        for s in range(1, k + 1):
            acc &= np.roll(m, s, ax) & np.roll(m, -s, ax)
        m = acc
    return m


def dilate(mask, k):
    return ~erode(~mask, k)


def uv_raster(Vuv, Fuv, attrs, tex):
    """rasterise UV triangles into tex x tex; attrs list of per-corner-vertex arrays (N,C) -> list of (tex,tex,C), mask."""
    sx, sy = Vuv[:, 0] * tex, Vuv[:, 1] * tex
    fid, bary, _ = rasterize(sx, sy, np.zeros(len(sx)), Fuv, tex, tex)
    return [interp(a, Fuv, fid, bary, 0.0) for a in attrs], fid >= 0


def bake(views, gen_imgs, Pt, Nt, cover, log=print, angle=75, power=4, relax=True):
    """Gather bake. Pt/Nt (tex,tex,3) texel position / normal (render frame), cover bool mask.
    views: Views; gen_imgs: list of (512,512,3) floats 0..1 (png orientation). Returns texture (tex,tex,3), trust mask.
    Pass 1: strict visibility (eroded silhouette, no depth edges, angle < `angle`), weights VWEIGHTS * cos^power.
    Pass 2 (relax): texels no view painted in pass 1 (silhouette / depth-edge / grazing-angle texels, which used to be
    smeared by the UV fill into grey-brown patches) take the colour of the views that still see them (cos > 0.02, no edge exclusion)."""
    T = Pt.shape[0]
    idx = np.nonzero(cover)
    P, N = Pt[idx], Nt[idx]
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    acc = np.zeros((len(P), 3), np.float32)
    wsum = np.zeros(len(P), np.float32)
    acc2 = np.zeros((len(P), 3), np.float32)
    wsum2 = np.zeros(len(P), np.float32)
    cthr = math.cos(math.radians(angle))
    R = views.res
    for i, (e, a) in enumerate(zip(ELEVS, AZIMS)):
        fid, _, dep = views.r[i]
        vis = fid >= 0
        k = int(2 / 512 * R)
        vis_s = erode(vis, k)
        d = np.where(vis, dep, 0).astype(np.float32)
        dn = (d - d[vis].min()) / max(float(d[vis].max() - d[vis].min()), 1e-9) if vis.any() else d
        g = np.maximum(np.abs(np.diff(dn, axis=0, prepend=dn[:1])), np.abs(np.diff(dn, axis=1, prepend=dn[:, :1])))
        edge = dilate(g * vis > 0.02, k)
        good = vis_s & ~edge
        sx, sy, z = project(P, e, a, R)
        _, look = mv_matrix(e, a)
        cosv = N @ look  # the frame change mirrors the mesh, so vertex normals point inward (same as Hunyuan's lookat=(0,0,-1) test)
        px = np.clip(np.round(sx - 0.5).astype(np.int64), 0, R - 1)
        py = np.clip(np.round(sy - 0.5).astype(np.int64), 0, R - 1)
        dz = z - dep[py, px]
        ok = good[py, px] & (cosv > cthr) & (dz < 0.01)
        w = np.where(ok, VWEIGHTS[i] * np.clip(cosv, 0, 1) ** power, 0).astype(np.float32)
        # colour: bilinear from the generated (S x S, same orientation as the raster) view
        img = gen_imgs[i]
        S = img.shape[0]
        u = (sx - 0.5) / (R - 1) * (S - 1)
        v = (sy - 0.5) / (R - 1) * (S - 1)
        u, v = np.clip(u, 0, S - 1.001), np.clip(v, 0, S - 1.001)
        u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        fu, fv = (u - u0)[:, None], (v - v0)[:, None]
        col = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv) + img[v0 + 1, u0] * (1 - fu) * fv
               + img[v0 + 1, u0 + 1] * fu * fv)
        if relax:
            ok2 = vis[py, px] & (cosv > 0.02) & (dz < 0.02)
            w2 = np.where(ok2, np.clip(cosv, 0, 1) ** 2 * (0.2 + VWEIGHTS[i]), 0).astype(np.float32)
            acc2 += col * w2[:, None]
            wsum2 += w2
        painted = wsum > 0
        view_sum, painted_sum = (w > 0).sum(), ((w > 0) & painted).sum()
        log(f"bake view {VNAMES[i]}: {int(view_sum)} texels, {int(painted_sum)} already painted")
        if view_sum and painted_sum / view_sum > 0.99:
            continue
        acc += col * w[:, None]
        wsum += w
    if relax:
        miss = (wsum <= 1e-8) & (wsum2 > 1e-8)
        log(f"bake relaxed pass: {int(miss.sum())} of {int((wsum <= 1e-8).sum())} unpainted texels")
        acc[miss], wsum[miss] = acc2[miss], wsum2[miss]
    tex = np.zeros((T, T, 3), np.float32)
    trust = np.zeros((T, T), bool)
    good = wsum > 1e-8
    tex[idx[0][good], idx[1][good]] = acc[good] / wsum[good, None]
    trust[idx[0][good], idx[1][good]] = True
    return tex, trust


def fill(tex, known, iters=64):
    """grow known colours into unknown texels by repeated 8-neighbour averaging."""
    tex, known = tex.copy(), known.copy()
    for _ in range(iters):
        if known.all():
            break
        s, c = np.zeros_like(tex), np.zeros(known.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                kk = np.roll(np.roll(known, dy, 0), dx, 1)
                s += np.roll(np.roll(tex * known[..., None], dy, 0), dx, 1)
                c += kk
        new = (~known) & (c > 0)
        if not new.any():
            break
        tex[new] = s[new] / c[new, None]
        known = known | new
    return tex, known


# ------------------------------------------------------------------ LCM scheduler (v_prediction), numpy
class LCM:
    def __init__(self, steps=10, train=1000, beta_start=0.00085, beta_end=0.012, zero_snr=True):
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, train, dtype=np.float64) ** 2
        if zero_snr:  # diffusers rescale_zero_terminal_snr
            al = 1.0 - betas
            ac = np.cumprod(al)
            s = np.sqrt(ac)
            s0, sT = s[0], s[-1]
            s = (s - sT) * s0 / (s0 - sT)
            ac = s ** 2
            al = np.concatenate([ac[:1], ac[1:] / ac[:-1]])
            betas = 1 - al
        self.ac = np.cumprod(1.0 - betas)
        ddim = (np.arange(1, 31) * (train // 30)).round().astype(np.int64) - 1
        # Hunyuan3D-2 turbo is consistency-distilled on exactly range(29, 0, -3) of the 30 DDIM timesteps (10 steps). Other step
        # counts are spread evenly over the same 29 .. 2 range (steps = 10 reproduces the original list).
        idx = np.unique(np.round(np.linspace(29, 2, steps)).astype(int))[::-1]
        self.ts = [int(ddim[i]) for i in idx]

    def step(self, out, i, x, rng):
        t = self.ts[i]
        prev = self.ts[i + 1] if i + 1 < len(self.ts) else t
        a_t, a_p = self.ac[t], (self.ac[prev] if prev >= 0 else 1.0)
        b_t, b_p = 1 - a_t, 1 - a_p
        x0 = math.sqrt(a_t) * x - math.sqrt(b_t) * out
        st = 10.0 * t
        c_skip = 0.25 / (st ** 2 + 0.25)
        c_out = st / math.sqrt(st ** 2 + 0.25)
        den = c_out * x0 + c_skip * x
        if i != len(self.ts) - 1:
            return (math.sqrt(a_p) * den + math.sqrt(b_p) * rng.standard_normal(x.shape)).astype(np.float32)
        return den.astype(np.float32)


# ------------------------------------------------------------------ tiling
def _starts(n, tile, stride):
    s = list(range(0, max(n - tile, 0) + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def tiled(arr, fn, tile, stride, scale_out, ch_out, blend):
    """arr (1,C,H,W); fn(tile)->(1,ch_out,tile*scale,tile*scale) (scale_out may be a fraction for the encoder)."""
    _, _, H, W = arr.shape
    oh, ow = int(H * scale_out), int(W * scale_out)
    acc = np.zeros((1, ch_out, oh, ow), np.float32)
    wgt = np.zeros((1, 1, oh, ow), np.float32)
    ot = int(tile * scale_out)
    ramp = np.minimum(1.0, (np.arange(ot) + 1) / blend) * np.minimum(1.0, (ot - np.arange(ot)) / blend)
    for y in _starts(H, tile, stride):
        for x in _starts(W, tile, stride):
            o = fn(np.ascontiguousarray(arr[:, :, y:y + tile, x:x + tile]))
            wy = ramp.copy()
            wx = ramp.copy()
            if y == 0:
                wy[:blend] = 1
            if x == 0:
                wx[:blend] = 1
            if y + tile >= H:
                wy[-blend:] = 1
            if x + tile >= W:
                wx[-blend:] = 1
            w = (wy[:, None] * wx[None, :])[None, None].astype(np.float32)
            yo, xo = int(y * scale_out), int(x * scale_out)
            acc[:, :, yo:yo + ot, xo:xo + ot] += o * w
            wgt[:, :, yo:yo + ot, xo:xo + ot] += w
    return acc / wgt


# ------------------------------------------------------------------ images
def resize_area(img, size):
    """float (H,W,C) -> (size,size,C) area / bilinear resize for the reference (numpy only)."""
    h, w = img.shape[:2]
    ys = (np.arange(size) + 0.5) * h / size - 0.5
    xs = (np.arange(size) + 0.5) * w / size - 0.5
    if h > size:  # pre-average for strong down-scaling
        f = max(1, int(h // size))
        if f > 1:
            hh, ww = (h // f) * f, (w // f) * f
            img = img[:hh, :ww].reshape(hh // f, f, ww // f, f, -1).mean((1, 3))
            h, w = img.shape[:2]
            ys = (np.arange(size) + 0.5) * h / size - 0.5
            xs = (np.arange(size) + 0.5) * w / size - 0.5
    ys, xs = np.clip(ys, 0, h - 1.001), np.clip(xs, 0, w - 1.001)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    return (img[y0][:, x0] * (1 - fy) * (1 - fx) + img[y0][:, x0 + 1] * (1 - fy) * fx + img[y0 + 1][:, x0] * fy * (1 - fx)
            + img[y0 + 1][:, x0 + 1] * fy * fx)


def prep_reference(img, size=512, border=0.2):
    """float RGB(A) (H,W,C) 0..1 -> (size,size,3): white background, subject cropped + centred with 20 % border
    (Hunyuan's rembg + padding). Photos without alpha: background = colour distance from the border median, grown from the border."""
    if img.shape[2] == 4:
        alpha = img[..., 3] > 0.5
        rgb = img[..., :3] * img[..., 3:4] + (1 - img[..., 3:4])
    else:
        rgb = img[..., :3]
        H0, W0 = rgb.shape[:2]
        f = max(1, int(math.ceil(max(H0, W0) / 192)))
        hh, ww = (H0 // f) * f, (W0 // f) * f
        sm = rgb[:hh, :ww].reshape(hh // f, f, ww // f, f, 3).mean((1, 3))
        edge = np.concatenate([sm[:3].reshape(-1, 3), sm[-3:].reshape(-1, 3), sm[:, :3].reshape(-1, 3), sm[:, -3:].reshape(-1, 3)])
        bg = np.median(edge, 0)
        d = np.abs(sm - bg).max(-1)
        thr = max(0.08, float(np.percentile(np.abs(edge - bg).max(-1), 99)) * 1.5)
        fg = d > thr
        core = erode(fg, 3)
        if core.any():  # keep the blob connected to the largest solid core
            seed = core
            for _ in range(300):
                nxt = dilate(seed, 1) & fg
                if (nxt == seed).all():
                    break
                seed = nxt
            fg = seed
        outside = ~fg  # holes = background not reachable from the border
        reach = np.zeros_like(fg)
        reach[0, :] = reach[-1, :] = reach[:, 0] = reach[:, -1] = True
        reach &= outside
        for _ in range(400):
            nxt = dilate(reach, 1) & outside
            if (nxt == reach).all():
                break
            reach = nxt
        al = np.repeat(np.repeat(~reach, f, 0), f, 1)
        alpha = np.zeros((H0, W0), bool)
        alpha[:al.shape[0], :al.shape[1]] = al
        rgb = np.where(alpha[..., None], rgb, 1.0)
    ys, xs = np.nonzero(alpha)
    rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = rgb.shape[:2]
    bw, bh = int(w * border), int(h * border)
    sq = max(w + 2 * bw, h + 2 * bh)
    canvas = np.ones((sq, sq, 3), np.float32)
    oy, ox = (sq - h - 2 * bh) // 2 + bh, (sq - w - 2 * bw) // 2 + bw
    canvas[oy:oy + h, ox:ox + w] = rgb
    return np.clip(resize_area(canvas, size), 0, 1)


# ------------------------------------------------------------------ colour fidelity (Lab)
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_W = np.array([0.95047, 1.0, 1.08883])


def srgb2lab(rgb):
    c = np.clip(rgb, 0, 1).astype(np.float32)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M.T.astype(np.float32)) / _W.astype(np.float32)
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1).astype(np.float32)


def lab2srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _W
    lin = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.maximum(lin, 1e-9) ** (1 / 2.4) - 0.055).astype(np.float32)


def ref_foreground(ref):
    """reference prepared by prep_reference (white background): foreground = anything not (nearly) white."""
    return np.abs(ref - 1.0).max(-1) > 0.05


def _cdf_match(src, dst, strength):
    """per-channel histogram matching of the 1-D samples src to dst, blended by strength (0..1); returns a lookup (x -> y)."""
    q = np.linspace(0, 100, 101)
    s, d = np.percentile(src, q), np.percentile(dst, q)
    return lambda x: x + strength * (np.interp(x, s, d) - x)


def match_colours(gen, masks, refs, ref_masks, mode="chroma", strength=1.0, lstrength=0.5, ref_views=None):
    """Pull the colour statistics of every generated view (foreground only) towards the reference photos.
    The turbo model returns paler, lower-saturation colours than its reference (and grey backgrounds); the photos are the ground truth.
    ref_views = the label (front/left/back/right) of each reference; a view with a photo of its own label is matched to that photo, any
    other view to the pooled references (a single photo: to that photo). Default labels: front, back.
    mode 'chroma': L histogram-matched (lstrength), a/b scaled radially so the mean chroma and the chroma spread follow the reference
    (hue preserved); mode 'cdf': L, a, b histogram-matched independently."""
    rl = [srgb2lab(r)[m] for r, m in zip(refs, ref_masks)]
    pooled = np.concatenate(rl)
    labels = list(ref_views) if ref_views else ["front", "back"][:len(rl)] + [""] * max(0, len(rl) - 2)
    out = []
    for i, (img, m) in enumerate(zip(gen, masks)):
        own = labels.index(VNAMES[i]) if VNAMES[i] in labels[:len(rl)] else -1
        tgt = rl[own] if own >= 0 else (rl[0] if len(rl) == 1 else pooled)
        lab = srgb2lab(img)
        src = lab[m]
        if not len(src):
            out.append(img)
            continue
        if mode == "cdf":
            for c in range(3):
                lab[..., c] = _cdf_match(src[:, c], tgt[:, c], strength if c else lstrength)(lab[..., c])
        else:
            lab[..., 0] = _cdf_match(src[:, 0], tgt[:, 0], lstrength)(lab[..., 0])
            cs, ct = np.hypot(src[:, 1], src[:, 2]), np.hypot(tgt[:, 1], tgt[:, 2])
            f = _cdf_match(cs, ct, strength)  # chroma quantile map
            ch = np.hypot(lab[..., 1], lab[..., 2])
            g = np.where(ch > 1e-3, f(ch) / np.maximum(ch, 1e-3), 1.0)
            lab[..., 1] *= g
            lab[..., 2] *= g
        out.append(np.where(m[..., None], lab2srgb(lab), img))
    return out


def gauss_blur(img, sigma):
    """separable gaussian on (H,W,C) float32."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    pad = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = sum(k[i] * pad[i:i + img.shape[0]] for i in range(2 * r + 1))
    pad = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="edge")
    return sum(k[i] * pad[:, i:i + img.shape[1]] for i in range(2 * r + 1)).astype(np.float32)


def sharpen(img, mask, amount=0.6, sigma=1.2):
    """unsharp mask on Lab lightness only (no colour fringes), foreground pixels only."""
    if amount <= 0:
        return img
    lab = srgb2lab(img)
    L = lab[..., :1]
    lab[..., :1] = np.clip(L + amount * (L - gauss_blur(L, sigma)), 0, 100)
    return np.where(mask[..., None], lab2srgb(lab), img)
