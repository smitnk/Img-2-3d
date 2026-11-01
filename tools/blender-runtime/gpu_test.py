"""Runtime check: Blender features that need a GPU context (render engines, gpu module) work headless.

Run with the same environment the app sets (EGL_PLATFORM=surfaceless, LIBGL_ALWAYS_SOFTWARE=1).
"""
import os
import sys

import bpy

bpy.ops.wm.read_factory_settings()  # default scene: cube, camera, light
scene = bpy.context.scene
scene.render.resolution_x = scene.render.resolution_y = 160
for engine in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT"):
    scene.render.engine = engine
    if engine == "BLENDER_EEVEE_NEXT":
        scene.eevee.taa_render_samples = 4
    scene.render.filepath = f"/tmp/gpu_{engine}.png"
    bpy.ops.render.render(write_still=True)
    size = os.path.getsize(scene.render.filepath)
    assert size > 1000, f"{engine} wrote an empty image ({size} bytes)"
    print(f"GPUTEST render ok {engine} {size} bytes", flush=True)

import gpu  # noqa: E402  (imported after rendering so a GPU context already exists)

print("GPUTEST", gpu.platform.vendor_get(), "|", gpu.platform.renderer_get(), "|", gpu.platform.version_get(), flush=True)
sys.exit(0)
