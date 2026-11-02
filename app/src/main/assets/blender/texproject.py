"""Project view images onto a mesh and bake one texture: the StableGen idea re-implemented (not copied; StableGen
is GPL-3) for headless Blender 4.2 without Cycles/OSL.

  blender -b -P texproject.py -- --in model.glb --views views.json --out DIR [--res 2048] [--subdiv 1]

views.json: [{"image": "front.png", "view": "front"}, {"image": "back.png", "view": "back"}, ...]
  view: front | back | left | right | front_left | front_right | back_left | back_right | top
  Each image shows the whole character from that side (orthographic, as from render_views). RGBA images: alpha 0 =
  no data there. The photo given to TripoSR is the "front" view.

Per view and per surface point (StableGen's weighting): weight = cos(angle to the view)^exponent x edge feather,
zero past the angle limit or when occluded (z-buffer from the view). Colours blend by weight; points no view sees
keep the mesh's own colours (TripoSR vertex colours). The result goes through vcbake (vertex colours -> UV texture).
The numpy core has no bpy dependency: tests/test_texproject.py runs it in plain Python.
"""
import json
import os
import sys

import numpy as np

# Blender convention for the app: Z up, the character faces -Y (front camera sits at -Y looking +Y).
VIEWS = {
    "front": (0, 1, 0), "back": (0, -1, 0), "left": (-1, 0, 0), "right": (1, 0, 0),
    "front_left": (-0.7071, 0.7071, 0), "front_right": (0.7071, 0.7071, 0),
    "back_left": (-0.7071, -0.7071, 0), "back_right": (0.7071, -0.7071, 0), "top": (0, 0, -1),
}


FACING = -1  # the character faces -Y (app convention); TripoSR meshes face +Y (set by --facing +y)


def view_basis(name):
    """(forward, right, up) unit vectors of an orthographic camera looking along `forward`. "left" means the
    character's left side (+X), seen by a camera looking -X. Views turn with the character when it faces +Y."""
    f = np.array(VIEWS[name], float) * ([1, 1, 1] if FACING < 0 else [-1, -1, 1])
    f /= np.linalg.norm(f)
    up = np.array([0, 1.0, 0]) if abs(f[2]) > 0.9 else np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)


def project(points, f, r, u):
    """Image-plane coords (x right, y up) and depth along f."""
    return points @ r, points @ u, points @ f


def frame(points, f, r, u, margin=0.0):
    """Square-free framing: the image covers the mesh's projected bounding box (x0, x1, y0, y1)."""
    x, y, _ = project(points, f, r, u)
    return x.min() - margin, x.max() + margin, y.min() - margin, y.max() + margin


def zbuffer(points, f, r, u, box, res=None):
    """Nearest depth per pixel from splatted points. Resolution follows point density (about 2 px between
    neighbouring points) so the 1 px splat closes the gaps and the front surface hides what is behind it."""
    if res is None:
        res = int(np.clip(1.2 * np.sqrt(len(points)), 64, 1024))
    x, y, d = project(points, f, r, u)
    x0, x1, y0, y1 = box
    px = np.clip(((x - x0) / max(x1 - x0, 1e-9) * (res - 1)).round().astype(int), 0, res - 1)
    py = np.clip(((y - y0) / max(y1 - y0, 1e-9) * (res - 1)).round().astype(int), 0, res - 1)
    zb = np.full((res, res), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            qx, qy = np.clip(px + dx, 0, res - 1), np.clip(py + dy, 0, res - 1)
            np.minimum.at(zb, (qy, qx), d)
    return zb, px, py, d


def view_weights(points, normals, f, r, u, box, angle_limit=75.0, exponent=3.0, feather=0.08, depth_tol=None,
                 occluders=None):
    """Per-point weight and normalised image coords (s, t in 0..1, t up) for one view. `occluders` are extra points
    (the whole mesh) for the z-buffer when `points` are only some samples."""
    occ = points if occluders is None else occluders
    zb, _, _, _ = zbuffer(occ, f, r, u, box)
    res = zb.shape[0]
    x, y, d = project(points, f, r, u)
    x0, x1, y0, y1 = box
    s = (x - x0) / max(x1 - x0, 1e-9)
    t = (y - y0) / max(y1 - y0, 1e-9)
    px = np.clip((s * (res - 1)).round().astype(int), 0, res - 1)
    py = np.clip((t * (res - 1)).round().astype(int), 0, res - 1)
    if depth_tol is None:
        depth_tol = 0.01 * max(x1 - x0, y1 - y0)  # 1 % of the model: thin parts (fingers, ears) stay separate
    visible = d <= zb[py, px] + depth_tol
    cos = np.clip(-(normals @ f), 0, 1)  # f points from the camera into the scene; front faces face against it
    ok = visible & (cos > np.cos(np.radians(angle_limit)))
    edge = np.maximum(np.abs(2 * s - 1), np.abs(2 * t - 1))  # 0 centre .. 1 image border
    k = np.clip((edge - (1 - feather)) / feather, 0, 1)
    w = cos ** exponent * (1 - k * k * (3 - 2 * k))
    return np.where(ok, w, 0.0), s, t


def sample(img, s, t):
    """Bilinear RGBA sample of img (H, W, 4 float, row 0 = top) at s, t in 0..1 (t up)."""
    h, w = img.shape[:2]
    fx = np.clip(s, 0, 1) * (w - 1)
    fy = (1 - np.clip(t, 0, 1)) * (h - 1)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    jx, jy = np.minimum(ix + 1, w - 1), np.minimum(iy + 1, h - 1)
    ax, ay = (fx - ix)[:, None], (fy - iy)[:, None]
    return (img[iy, ix] * (1 - ax) * (1 - ay) + img[iy, jx] * ax * (1 - ay)
            + img[jy, ix] * (1 - ax) * ay + img[jy, jx] * ax * ay)


def subject_box(img, thresh=0.5):
    """(s0, s1, t0, t1) of the subject in an RGBA image (alpha), or the whole image if it has no alpha cut-out."""
    a = img[..., 3] > thresh
    if a.all() or not a.any():
        return 0.0, 1.0, 0.0, 1.0
    h, w = a.shape
    ys, xs = np.nonzero(a)
    return xs.min() / (w - 1), xs.max() / (w - 1), 1 - ys.max() / (h - 1), 1 - ys.min() / (h - 1)


def blend(points, normals, base_rgb, views, occluders=None, base_weight=1e-3, **kw):
    """views: [(name, img RGBA float)]. Returns (rgb, coverage) per point; coverage = summed view weight."""
    acc = base_rgb * base_weight
    tot = np.full(len(points), base_weight)
    occ = points if occluders is None else occluders
    for view in views:
        name, img = view[:2]
        gain = view[2] if len(view) > 2 else 1.0  # the source photo outranks AI-painted views where both see a point
        f, r, u = view_basis(name)
        box = frame(occ, f, r, u)
        w, s, t = view_weights(points, normals, f, r, u, box, occluders=occ, **kw)
        s0, s1, t0, t1 = subject_box(img)  # mesh box -> subject box in the picture (photos have margins)
        c = sample(img, s0 + s * (s1 - s0), t0 + t * (t1 - t0))
        w = w * (c[:, 3] > 0.5) * gain
        acc += c[:, :3] * w[:, None]
        tot += w
    return acc / tot[:, None], tot - base_weight


# ---------------------------------------------------------------------------------------------------- Blender part
def _load_rgba(path):
    import bpy
    im = bpy.data.images.load(path)
    w, h = im.size
    px = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(px)  # bottom row first, linear for 8-bit sRGB images
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[::-1]


def render_views(obj, names, out, res=512):
    """Orthographic Workbench renders (flat light, vertex colours, transparent background) framed exactly like
    project(): the mesh's projected box fills the square image's long side, so the alpha cut-out maps back 1:1.
    These are the init images the on-device painter (core TextureDiffusion) repaints."""
    import bpy
    from mathutils import Matrix
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    sc = bpy.context.scene
    r = sc.render
    r.engine = "BLENDER_WORKBENCH"
    r.resolution_x = r.resolution_y = res
    r.resolution_percentage = 100
    r.film_transparent = True
    r.image_settings.file_format, r.image_settings.color_mode = "PNG", "RGBA"
    sc.view_settings.view_transform = "Standard"
    sh = sc.display.shading
    sh.light, sh.color_type = "FLAT", ("VERTEX" if obj.data.color_attributes else "MATERIAL")
    cam = bpy.data.objects.new("TP_cam", bpy.data.cameras.new("TP_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    span = np.ptp(co, axis=0).max()
    for name in names:
        f, rt, u = view_basis(name)
        x0, x1, y0, y1 = frame(co, f, rt, u)
        _, _, d = project(co, f, rt, u)
        cam.data.ortho_scale = max(x1 - x0, y1 - y0)
        cam.data.clip_end = 10 * span + 10
        centre = rt * (x0 + x1) / 2 + u * (y0 + y1) / 2 + f * (d.min() - 2 * span - 1)
        cam.matrix_world = Matrix([[rt[0], u[0], -f[0], centre[0]], [rt[1], u[1], -f[1], centre[1]],
                                   [rt[2], u[2], -f[2], centre[2]], [0, 0, 0, 1]])
        r.filepath = os.path.join(out, f"render_{name}.png")
        bpy.ops.render.render(write_still=True)
        print(f"texproject: rendered {name}")
    json.dump(names, open(os.path.join(out, "renders.json"), "w"))


def main():
    import argparse
    import bpy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vcbake

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--views", help="views.json to project (project mode)")
    p.add_argument("--render", help="comma list of views to render from the mesh's own colours into --out (render mode)")
    p.add_argument("--out", required=True)
    p.add_argument("--res", type=int, default=2048)
    p.add_argument("--facing", default="-y", choices=["-y", "+y"], help="+y for raw TripoSR / refined photo meshes")
    p.add_argument("--subdiv", type=int, default=-1, help="simple subdivisions before sampling; -1 = once if under 60k faces")
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    global FACING
    FACING = 1 if a.facing == "+y" else -1
    os.makedirs(a.out, exist_ok=True)
    if not (a.views or a.render):
        raise SystemExit("give --views (project) or --render (render views)")
    spec = json.load(open(a.views)) if a.views else [{"view": v} for v in a.render.split(",")]
    base_dir = os.path.dirname(os.path.abspath(a.views)) if a.views else a.out
    for v in spec:
        if v["view"] not in VIEWS:
            raise SystemExit(f"unknown view {v['view']!r}; use one of {sorted(VIEWS)}")

    vcbake.import_glb(a.inp)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in " + a.inp)
    bpy.context.view_layer.objects.active = meshes[0]
    for o in meshes:
        o.select_set(True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if a.render:
        render_views(obj, [v["view"] for v in spec], a.out)
        return
    if a.subdiv < 0:
        a.subdiv = 1 if len(obj.data.polygons) < 60000 else 0  # TripoSR meshes are ~100k faces: phone RAM
    if a.subdiv:
        # shortcut: colours are sampled per face corner, so detail is limited by mesh density; render-to-UV
        # position maps would sample per texel; upgrade if faces look blurry at --res 2048.
        m = obj.modifiers.new("sub", "SUBSURF")
        m.subdivision_type, m.levels = "SIMPLE", a.subdiv
        bpy.ops.object.modifier_apply(modifier=m.name)

    me = obj.data
    nv, nl = len(me.vertices), len(me.loops)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    me.calc_normals_split() if hasattr(me, "calc_normals_split") else None
    ln = np.empty(nl * 3)
    me.loops.foreach_get("normal", ln)  # corner normals (smooth or flat as authored)
    ln = ln.reshape(-1, 3)

    base = np.full((nl, 3), 0.5)
    if me.color_attributes:
        col = me.color_attributes.active_color or me.color_attributes[0]
        c = np.empty(len(col.data) * 4, np.float32)
        col.data.foreach_get("color", c)
        c = c.reshape(-1, 4)[:, :3]
        base = c[lv] if col.domain == "POINT" else c

    views = [(v["view"], _load_rgba(os.path.join(base_dir, v["image"])), float(v.get("weight", 1))) for v in spec]
    rgb, cov = blend(co[lv], ln, base, views, occluders=co)
    for name, img, _ in views:
        print(f"texproject: view {name} {img.shape[1]}x{img.shape[0]}")
    print(f"texproject: {100 * (cov > 0).mean():.1f}% of corners covered by views")

    for ca in list(me.color_attributes):
        me.color_attributes.remove(ca)
    out = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    out.data.foreach_set("color", np.column_stack([rgb, np.ones(nl)]).astype(np.float32).ravel())
    me.color_attributes.active_color = out
    tex = vcbake.bake(obj, a.res)
    glb = vcbake.export_glb(os.path.join(a.out, "textured.glb"))
    json.dump({"script": "texproject", "vertices": len(obj.data.vertices), "triangles": sum(len(q.vertices) - 2 for q in obj.data.polygons), "views": [v["view"] for v in spec], "coverage": float((cov > 0).mean()),
               "texture": tex, "files": {"glb": glb}}, open(os.path.join(a.out, "generation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
