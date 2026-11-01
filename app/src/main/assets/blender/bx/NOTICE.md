Vendored from https://github.com/scenario-labs/skills (skills/dcc/blender/*/scripts), MIT, Copyright (c) 2026 Scenario Inc. — see LICENSE.
Patched for Blender 4.2: `image_settings.media_type` (5.x-only) guarded with hasattr; `bx_export.export` drops operator
arguments the 4.2 exporter does not have. USD presets are unavailable (runtime built without USD).
