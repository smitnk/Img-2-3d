Vendored from https://github.com/scenario-labs/skills (skills/dcc/blender/*/scripts: bx_audit, bx_review, bx_export, bx_uvbake, bx_rig, bx_retopo), MIT, Copyright (c) 2026 Scenario Inc. — see LICENSE.
Patched for Blender 4.2: `image_settings.media_type` (5.x-only) guarded with hasattr; `bx_export.export` drops operator
arguments the 4.2 exporter does not have. USD presets are unavailable (runtime built without USD).
- bx_retopo.target_bvh: falls back to BVHTree.FromPolygons when FromObject returns NULL (hidden reference).
