"""Shared helpers for the app's Blender scripts: vertex colours -> UV texture, GLB export, thumbnail."""
import math
import os

import bpy
import numpy as np


def bake(obj, res=1024, dilate=6):
    """Smart-UV-project obj and paint its active colour attribute into a res x res texture. The gpu draw module is
    off in background mode, so the mesh is laid flat in UV space and rendered with Workbench (flat, vertex colour,
    orthographic): render pixels = texture pixels. The material then samples the texture; the colour attribute stays
    for viewers that read COLOR_0. Returns the texture's info dict."""
    me = obj.data
    name = (me.color_attributes.active_color or me.color_attributes[0]).name
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o is obj)
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=4.0 / res)
    bpy.ops.object.mode_set(mode="OBJECT")

    me = obj.data  # edit mode rebuilt the mesh data: earlier attribute references are stale
    col = me.color_attributes[name]
    me.calc_loop_triangles()
    tri_loops = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tri_loops)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)[tri_loops]
    rgba = np.empty(len(col.data) * 4, np.float32)
    col.data.foreach_get("color", rgba)  # scene linear; Workbench + Standard view writes it back as sRGB
    rgba = rgba.reshape(-1, 4)
    if col.domain == "POINT":
        lv = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", lv)
        rgba = rgba[lv[tri_loops]]
    else:
        rgba = rgba[tri_loops]
    rgba[:, 3] = 1.0

    # Flat copy: one vertex per triangle corner at (u, v, 0), colours per corner.
    n = len(tri_loops)
    flat = bpy.data.meshes.new("BX_uvflat")
    flat.vertices.add(n)
    flat.vertices.foreach_set("co", np.column_stack([uv, np.zeros(n, np.float32)]).ravel())
    flat.loops.add(n)
    flat.loops.foreach_set("vertex_index", np.arange(n, dtype=np.int32))
    flat.polygons.add(n // 3)
    flat.polygons.foreach_set("loop_start", np.arange(0, n, 3, dtype=np.int32))
    flat.update(calc_edges=True)
    fc = flat.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    fc.data.foreach_set("color", rgba.ravel())
    flat.color_attributes.active_color = fc

    sc = bpy.data.scenes.new("BX_uvbake")
    fo = bpy.data.objects.new("BX_uvflat", flat)
    sc.collection.objects.link(fo)
    cam = bpy.data.objects.new("BX_uvcam", bpy.data.cameras.new("BX_uvcam"))
    cam.data.type, cam.data.ortho_scale = "ORTHO", 1.0
    cam.location = (0.5, 0.5, 1.0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    r = sc.render
    r.engine = "BLENDER_WORKBENCH"
    r.resolution_x = r.resolution_y = res
    r.resolution_percentage = 100
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    sc.display.render_aa = "OFF"  # no edge blending into the background; dilation pads the islands instead
    sc.view_settings.view_transform = "Standard"
    sh = sc.display.shading
    sh.light, sh.color_type = "FLAT", "VERTEX"
    sh.show_backface_culling = False
    tmp = os.path.join(bpy.app.tempdir or "/tmp", "bx_uvbake.png")
    r.filepath = tmp
    try:
        bpy.ops.render.render(write_still=True, scene=sc.name)
        shot = bpy.data.images.load(tmp, check_existing=False)
        px = np.array(shot.pixels[:], np.float32).reshape(res, res, 4)  # bottom-up rows = UV v up
        bpy.data.images.remove(shot)
    finally:
        bpy.data.objects.remove(fo)
        bpy.data.objects.remove(cam)
        bpy.data.meshes.remove(flat)
        bpy.data.scenes.remove(sc)
        if os.path.exists(tmp):
            os.remove(tmp)

    covered = float((px[..., 3] > 0.5).mean())
    # Grow islands outward so mip-mapping and bilinear filtering never pull in the empty background.
    for _ in range(dilate):
        empty = px[..., 3] < 0.5
        if not empty.any():
            break
        acc = np.zeros_like(px)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            acc += np.roll(px, (dy, dx), (0, 1)) * (np.roll(px[..., 3], (dy, dx), (0, 1)) > 0.5)[..., None]
        n = acc[..., 3:4]
        fill = empty & (n[..., 0] > 0)
        px[fill] = acc[fill] / n[fill]
        px[fill, 3] = 1.0
    px[..., 3] = 1.0

    img = bpy.data.images.new(obj.name + "_albedo", res, res, alpha=False)
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(px.ravel())
    img.pack()  # embedded in the GLB / .blend

    mat = bpy.data.materials.new(obj.name + "_Baked")
    mat.use_nodes = True
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    bsdf = nt.nodes["Principled BSDF"]
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    me.materials.clear()
    me.materials.append(mat)
    print(f"baked {res}px texture, islands cover {covered:.0%}")
    return {"size": res, "coverage": round(covered, 3)}


def import_glb(path):
    """Import a GLB into an empty scene, minus the bone-display shapes the glTF importer adds for skinned files."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    shapes = {pb.custom_shape for o in bpy.data.objects if o.type == "ARMATURE" for pb in o.pose.bones if pb.custom_shape}
    for o in shapes:
        bpy.data.objects.remove(o, do_unlink=True)


def export_glb(path, deform_only=False):
    """Visible objects -> GLB, keeping the colour attribute (COLOR_0) even when the material samples a texture."""
    op = bpy.ops.export_scene.gltf
    known = set(op.get_rna_type().properties.keys())
    kw = dict(filepath=path, export_format="GLB", use_selection=False, use_visible=True, export_yup=True,
              export_apply=False, export_skins=True, export_animations=False, export_materials="EXPORT",
              export_def_bones=deform_only,
              export_vertex_color="ACTIVE",  # 4.2+; older versions export colours via export_colors
              export_colors=True)
    op(**{k: v for k, v in kw.items() if k in known})
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise SystemExit(f"GLB export failed: {path}")
    return os.path.basename(path)
