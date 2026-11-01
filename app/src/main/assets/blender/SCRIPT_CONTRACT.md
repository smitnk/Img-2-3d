# Writing a generator script for Img-2-3D Studio

Paste this into any AI (Claude, ChatGPT, Gemini…) together with a description of your character,
then import the .py file it gives you in **Create → Python script**.

---
Write ONE Python file for Blender 4.2 (bpy) that builds a 3D character. Rules:

1. It runs headless (`blender -b --factory-startup`) inside an already-empty scene. No UI, no bpy.ops that need a
   viewport, no file dialogs, no network, no `pip`. Only the Python standard library, bpy, bmesh, mathutils and numpy.
2. Units are metres. Z is up, the character faces -Y, feet rest on z = 0, humanoids stand in a T-pose
   (arms straight out along ±X).
3. Do NOT export or save files: the app exports every mesh and armature to GLB afterwards.
4. Give every object a material (Principled BSDF base colour). Keep it under ~200k triangles.
5. Helpers you may import (they are on the path):
   - `import generate_model as gm`
   - `gm.material(name, "#RRGGBB", roughness)` → material
   - `gm.part(name, kind, size_xyz, location_xyz, material, bone_name, rot=(deg,deg,deg), taper=1.0,
     bevel=0.0, subsurf=0, solidify=0.0, cut=None, segments=24)` → mesh object; kind is
     "sphere" | "cyl" | "cone" | "cube"; geometry is a unit primitive scaled by size_xyz.
   - `gm.build_rig(parts, bones)` where bones = [(name, head_xyz, tail_xyz, parent_or_None), …]; parts with a
     bone_name follow that bone rigidly. Use the bone names root, pelvis, spine, neck, head,
     upper_arm.L/R, forearm.L/R, hand.L/R, thigh.L/R, shin.L/R, foot.L/R.
6. `FORGE_PARAMS` (a dict) is available as a global if the app passes parameters.
7. Every result goes through a quality gate (game profile). Build so it passes:
   - No floating parts: every accessory, eye, button or hair piece must touch or slightly sink into the body
     (overlap ~1-2 mm). Gaps read as errors.
   - No z-fighting: never place two faces in the same plane (e.g. a decal exactly on a surface); offset or sink them.
   - Normals point outward; no zero-area faces, no stray loose vertices or edges; apply object scale only via
     size_xyz/mesh data, never negative scales.
   - Joints where limbs bend (shoulders, elbows, hips, knees) need enough geometry: segments >= 16 on limbs.
Return only the Python code.
