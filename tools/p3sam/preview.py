"""Flat-shaded multi-view PNG of face labels (matplotlib, no GL). Used for tuned/ previews."""
import numpy as np


def _view(v, yaw, pitch):
    cy, sy, cp, sp_ = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp_], [0, sp_, cp]])
    return v @ Ry.T @ Rx.T


def palette(n):
    import colorsys
    return np.array([colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.65, 0.95) for i in range(n)])


def render(ax, v, f, colors, yaw, pitch=0.15, title=""):
    from matplotlib.collections import PolyCollection
    w = _view(v - v.mean(0), yaw, pitch)
    tri = w[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.45 + 0.55 * np.abs(n[:, 2])
    order = np.argsort(tri[:, :, 2].mean(1))
    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=np.clip(colors[order] * shade[order, None], 0, 1), edgecolors="none", linewidths=0))
    ax.set_xlim(w[:, 0].min(), w[:, 0].max())
    ax.set_ylim(w[:, 1].min(), w[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def grid_png(path, v, f, items, views=((0.0, "front"), (np.pi / 2, "side")), cols=None):
    """items: [(title, face_labels)] -> one row per item, one column per view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(items), len(views), figsize=(3.2 * len(views), 3.2 * len(items)), dpi=90, squeeze=False)
    for r, (title, fl) in enumerate(items):
        ids = np.unique(fl)
        col = palette(len(ids))[np.searchsorted(ids, fl)]
        for c, (yaw, name) in enumerate(views):
            render(axs[r][c], v, f, col, yaw, title=f"{title} - {len(ids)} parts - {name}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
