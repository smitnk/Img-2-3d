"""CPU (numba) stand-in for Hunyuan3D-2's CUDA `custom_rasterizer`: same API (rasterize / interpolate) and the same
maths as lib/custom_rasterizer_kernel/rasterizer.cpp (z-buffer on quantised depth, face index token, barycentrics)."""
import numpy as np
import torch
from numba import njit

MAXINT = 2147483647


@njit(cache=True, fastmath=False)
def _raster(V, F, width, height):
    nf = F.shape[0]
    BIG = np.int64(MAXINT) * np.int64(MAXINT) + np.int64(MAXINT - 1)
    zb = np.full((height, width), BIG, dtype=np.int64)
    for f in range(nf):
        a, b, c = F[f, 0], F[f, 1], F[f, 2]
        w0, w1, w2 = V[a, 3], V[b, 3], V[c, 3]
        x0 = (V[a, 0] / w0 * 0.5 + 0.5) * (width - 1) + 0.5
        y0 = (0.5 + 0.5 * V[a, 1] / w0) * (height - 1) + 0.5
        z0 = V[a, 2] / w0 * 0.49999 + 0.5
        x1 = (V[b, 0] / w1 * 0.5 + 0.5) * (width - 1) + 0.5
        y1 = (0.5 + 0.5 * V[b, 1] / w1) * (height - 1) + 0.5
        z1 = V[b, 2] / w1 * 0.49999 + 0.5
        x2 = (V[c, 0] / w2 * 0.5 + 0.5) * (width - 1) + 0.5
        y2 = (0.5 + 0.5 * V[c, 1] / w2) * (height - 1) + 0.5
        z2 = V[c, 2] / w2 * 0.49999 + 0.5
        area = (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0)
        if area == 0:
            continue
        inv = 1.0 / area
        xmin = int(min(x0, min(x1, x2)))
        xmax = max(x0, max(x1, x2))
        ymin = int(min(y0, min(y1, y2)))
        ymax = max(y0, max(y1, y2))
        px = xmin
        while px < xmax + 1:
            if px >= 0 and px < width:
                py = ymin
                while py < ymax + 1:
                    if py >= 0 and py < height:
                        qx = px + 0.5
                        qy = py + 0.5
                        beta = ((x2 - x0) * (qy - y0) - (qx - x0) * (y2 - y0)) * inv
                        gamma = ((qx - x0) * (y1 - y0) - (x1 - x0) * (qy - y0)) * inv
                        alpha = 1.0 - beta - gamma
                        if alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0 and gamma >= 0.0 and gamma <= 1.0:
                            depth = alpha * z0 + beta * z1 + gamma * z2
                            tok = np.int64(depth * (2 << 17)) * np.int64(MAXINT) + np.int64(f + 1)
                            if tok < zb[py, px]:
                                zb[py, px] = tok
                    py += 1
            px += 1
    fi = np.zeros((height, width), dtype=np.int32)
    bc = np.zeros((height, width, 3), dtype=np.float32)
    for py in range(height):
        for px in range(width):
            f = zb[py, px] % MAXINT
            if f == MAXINT - 1:
                continue
            fi[py, px] = f
            f -= 1
            a, b, c = F[f, 0], F[f, 1], F[f, 2]
            w0, w1, w2 = V[a, 3], V[b, 3], V[c, 3]
            x0 = (V[a, 0] / w0 * 0.5 + 0.5) * (width - 1) + 0.5
            y0 = (0.5 + 0.5 * V[a, 1] / w0) * (height - 1) + 0.5
            x1 = (V[b, 0] / w1 * 0.5 + 0.5) * (width - 1) + 0.5
            y1 = (0.5 + 0.5 * V[b, 1] / w1) * (height - 1) + 0.5
            x2 = (V[c, 0] / w2 * 0.5 + 0.5) * (width - 1) + 0.5
            y2 = (0.5 + 0.5 * V[c, 1] / w2) * (height - 1) + 0.5
            qx = px + 0.5
            qy = py + 0.5
            area = (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0)
            inv = 1.0 / area
            beta = ((x2 - x0) * (qy - y0) - (qx - x0) * (y2 - y0)) * inv
            gamma = ((qx - x0) * (y1 - y0) - (x1 - x0) * (qy - y0)) * inv
            alpha = 1.0 - beta - gamma
            a0 = alpha / w0
            a1 = beta / w1
            a2 = gamma / w2
            s = 1.0 / (a0 + a1 + a2)
            bc[py, px, 0] = a0 * s
            bc[py, px, 1] = a1 * s
            bc[py, px, 2] = a2 * s
    return fi, bc


def rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0):
    V = np.ascontiguousarray(pos[0].detach().cpu().numpy().astype(np.float32))
    F = np.ascontiguousarray(tri.detach().cpu().numpy().astype(np.int32))
    fi, bc = _raster(V, F, int(resolution[1]), int(resolution[0]))
    return torch.from_numpy(fi), torch.from_numpy(bc)


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
