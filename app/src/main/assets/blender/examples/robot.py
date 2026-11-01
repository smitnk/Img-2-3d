"""Example custom generator: a rigged boxy robot in T-pose. Copy this as a template."""
import generate_model as gm  # Model Forge helpers: material(), part(), build_rig()

steel = gm.material("Steel", "#9AA4B2", 0.35)
dark = gm.material("Dark", "#2B2F36", 0.6)
glow = gm.material("Glow", "#38BDF8", 0.2)

parts = [
    gm.part("Torso", "cube", (0.34, 0.22, 0.36), (0, 0, 0.62), steel, "spine", bevel=0.02),
    gm.part("Pelvis", "cube", (0.26, 0.18, 0.12), (0, 0, 0.40), dark, "pelvis", bevel=0.01),
    gm.part("Neck", "cyl", (0.08, 0.08, 0.08), (0, 0, 0.82), dark, "neck"),
    gm.part("Head", "cube", (0.24, 0.22, 0.2), (0, 0, 0.94), steel, "head", bevel=0.03),
    gm.part("Visor", "cube", (0.18, 0.02, 0.05), (0, -0.115, 0.95), glow, "head", bevel=0.005),
    gm.part("Antenna", "cyl", (0.015, 0.015, 0.12), (0, 0, 1.1), dark, "head"),
]
for s, x in (("L", 1), ("R", -1)):
    parts += [
        gm.part(f"UpperArm.{s}", "cyl", (0.08, 0.08, 0.2), (x * 0.28, 0, 0.74), dark, f"upper_arm.{s}", rot=(0, 90, 0)),
        gm.part(f"Forearm.{s}", "cyl", (0.075, 0.075, 0.18), (x * 0.47, 0, 0.74), steel, f"forearm.{s}", rot=(0, 90, 0)),
        gm.part(f"Hand.{s}", "cube", (0.08, 0.06, 0.07), (x * 0.6, 0, 0.74), dark, f"hand.{s}", bevel=0.01),
        gm.part(f"Thigh.{s}", "cyl", (0.1, 0.1, 0.18), (x * 0.08, 0, 0.29), dark, f"thigh.{s}"),
        gm.part(f"Shin.{s}", "cyl", (0.09, 0.09, 0.18), (x * 0.08, 0, 0.12), steel, f"shin.{s}"),
        gm.part(f"Foot.{s}", "cube", (0.11, 0.17, 0.05), (x * 0.08, -0.03, 0.025), dark, f"foot.{s}", bevel=0.01),
    ]

bones = [("root", (0, 0, 0), (0, 0, 0.1), None), ("pelvis", (0, 0, 0.36), (0, 0, 0.44), "root"),
         ("spine", (0, 0, 0.44), (0, 0, 0.8), "pelvis"), ("neck", (0, 0, 0.8), (0, 0, 0.84), "spine"),
         ("head", (0, 0, 0.84), (0, 0, 1.05), "neck")]
for s, x in (("L", 1), ("R", -1)):
    bones += [(f"upper_arm.{s}", (x * 0.17, 0, 0.74), (x * 0.38, 0, 0.74), "spine"),
              (f"forearm.{s}", (x * 0.38, 0, 0.74), (x * 0.56, 0, 0.74), f"upper_arm.{s}"),
              (f"hand.{s}", (x * 0.56, 0, 0.74), (x * 0.64, 0, 0.74), f"forearm.{s}"),
              (f"thigh.{s}", (x * 0.08, 0, 0.38), (x * 0.08, 0, 0.21), "pelvis"),
              (f"shin.{s}", (x * 0.08, 0, 0.21), (x * 0.08, 0, 0.05), f"thigh.{s}"),
              (f"foot.{s}", (x * 0.08, 0, 0.05), (x * 0.08, -0.12, 0.02), f"shin.{s}")]
gm.build_rig(parts, bones)
