"""Plain-Python check of texproject's projection core: python3 tests/test_texproject.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app/src/main/assets/blender"))
import texproject as tp

def sphere(n=80, c=(0, 0, 0), r=1.0):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
    nrm = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    return nrm * r + np.array(c), nrm

P, N = sphere()
img = np.zeros((64, 64, 4)); img[..., 3] = 1
img[:, :32, 0] = 1  # image left half red
img[:, 32:, 2] = 1  # image right half blue
base = np.full((len(P), 3), 0.5)
rgb, cov = tp.blend(P, N, base, [("front", img)])
front_right = (P[:, 1] < -0.9) & (P[:, 0] > 0.2)   # character's left (+X) shows on the image's right
front_left = (P[:, 1] < -0.9) & (P[:, 0] < -0.2)
back = P[:, 1] > 0.5
assert (rgb[front_right, 2] > 0.9).all() and (rgb[front_left, 0] > 0.9).all(), "front colours misplaced"
assert np.allclose(rgb[back], 0.5) and (cov[back] == 0).all(), "back must keep base colour"

# occlusion: a small sphere behind the big one gets nothing from the front view
Q, M = sphere(40, c=(0, 2.5, 0), r=0.3)
occ = np.vstack([P, Q])
rgb2, cov2 = tp.blend(Q, M, np.full((len(Q), 3), 0.5), [("front", img)], occluders=occ)
assert (cov2 == 0).all(), "hidden sphere was painted"

# subject box: a photo with transparent margins maps the mesh onto the subject only
pad = np.zeros((100, 100, 4)); pad[20:80, 30:70] = img[::1, ::1][:60, :40] if False else 0
pad[20:80, 30:50, 0] = 1; pad[20:80, 50:70, 2] = 1; pad[20:80, 30:70, 3] = 1
rgb3, _ = tp.blend(P, N, base, [("front", pad)])
assert (rgb3[front_right, 2] > 0.9).all() and (rgb3[front_left, 0] > 0.9).all(), "subject box not used"
print("texproject core OK")
