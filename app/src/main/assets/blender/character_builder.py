"""Slider-driven character builder (runs inside headless Blender 4.2).

Builds a stylized humanoid from parametric parts in a T-pose, rigs it with the Model Forge 17-bone
skeleton, runs Model Forge's T-pose and deformation self-tests, and exports a GLB.
Reuses Model Forge's own helpers (scripts/generate_model.py, scripts/export_model.py).

usage: blender -b --factory-startup --python character_builder.py -- --out DIR [options]
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, "/opt/forge/scripts")
import generate_model as gm  # noqa: E402  (Model Forge: part(), material(), build_rig(), self-tests)
import export_model  # noqa: E402

STYLES = {
    "hair": ["none", "short", "spiky", "long", "bun", "mohawk"],
    "top": ["none", "tshirt", "hoodie", "tank", "jacket"],
    "bottom": ["shorts", "pants", "skirt"],
    "shoes": ["sneakers", "boots", "none"],
    "eyes": ["dot", "round", "anime"],
    "mouth": ["smile", "flat", "open"],
    "ears": ["round", "pointy", "animal", "none"],
    "tail": ["none", "cat", "lizard"],
    "accessory": ["none", "glasses", "hat", "cape", "crown"],
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="Custom Hero")
    p.add_argument("--height", type=float, default=1.2, help="total height in metres")
    for k, d in (("head", 1.0), ("body-width", 1.0), ("belly", 1.0), ("torso", 1.0), ("arm-length", 1.0),
                 ("arm-thickness", 1.0), ("leg-length", 1.0), ("leg-thickness", 1.0), ("neck", 1.0),
                 ("eye-size", 1.0), ("hand-size", 1.0), ("foot-size", 1.0)):
        p.add_argument(f"--{k}", type=float, default=d)
    for k, choices in STYLES.items():
        p.add_argument(f"--{k}", default=choices[0], choices=choices)
    for k, d in (("skin", "#F2C9A0"), ("hair-color", "#3B2417"), ("top-color", "#E8553D"), ("bottom-color", "#2F4A8A"),
                 ("shoe-color", "#F4F1EA"), ("eye-color", "#3A6EA5"), ("accent-color", "#F5C542")):
        p.add_argument(f"--{k}", default=d)
    p.add_argument("--smooth", type=int, default=1, choices=[0, 1, 2], help="subdivision level")
    p.add_argument("--no-rig", action="store_true")
    p.add_argument("--blend", action="store_true")
    return p.parse_args(argv)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build(a):
    # Layout is designed for a 1.0 m figure, then every coordinate is multiplied by H.
    H = a.height
    head_d = 0.24 * clamp(a.head, 0.6, 1.8)
    leg = 0.38 * clamp(a.leg_length, 0.6, 1.5)
    torso_h = 0.25 * clamp(a.torso, 0.7, 1.4)
    neck_h = 0.035 * clamp(a.neck, 0.3, 2.5)
    bw = clamp(a.body_width, 0.6, 1.6)
    belly = clamp(a.belly, 0.7, 1.8)
    arm_len = 0.34 * clamp(a.arm_length, 0.6, 1.5)
    arm_r = 0.032 * clamp(a.arm_thickness, 0.5, 2.0)
    leg_r = 0.045 * clamp(a.leg_thickness, 0.5, 2.0)
    sub = a.smooth

    foot_h = 0.05
    hip_z = foot_h + leg
    shoulder_z = hip_z + torso_h
    neck_top = shoulder_z + neck_h
    head_c = neck_top + head_d * 0.48
    shoulder_x = 0.11 * bw
    hip_x = 0.065 * bw
    arm_z = shoulder_z - 0.025
    knee_z = foot_h + leg * 0.5
    elbow_x = shoulder_x + arm_len * 0.45
    wrist_x = shoulder_x + arm_len * 0.85
    hand_x = shoulder_x + arm_len

    S = lambda *v: tuple(c * H for c in v)  # noqa: E731
    M = {
        "Skin": gm.material("Skin", a.skin, 0.55),
        "Hair": gm.material("Hair", a.hair_color, 0.45),
        "Top": gm.material("Top", a.top_color, 0.7),
        "Bottom": gm.material("Bottom", a.bottom_color, 0.7),
        "Shoe": gm.material("Shoe", a.shoe_color, 0.5),
        "Sole": gm.material("Sole", "#3A3A3A", 0.8),
        "EyeWhite": gm.material("EyeWhite", "#FAFAFA", 0.3),
        "Iris": gm.material("Iris", a.eye_color, 0.3),
        "Pupil": gm.material("Pupil", "#111111", 0.2),
        "Mouth": gm.material("Mouth", "#5A1E1E", 0.6),
        "Tongue": gm.material("Tongue", "#D9534F", 0.5),
        "Accent": gm.material("Accent", a.accent_color, 0.35),
    }
    parts = []

    def P(name, kind, size, loc, mat, bone, **kw):
        parts.append(gm.part(name, kind, S(*size), S(*loc), M[mat], bone, **kw))

    sleeve_upper = "Top" if a.top in ("tshirt", "hoodie", "jacket") else "Skin"
    sleeve_lower = "Top" if a.top in ("hoodie", "jacket") else "Skin"
    torso_mat = "Skin" if a.top == "none" else "Top"
    thigh_mat = "Skin" if a.bottom == "skirt" else "Bottom"
    shin_mat = "Bottom" if a.bottom == "pants" else "Skin"

    # Head and face (front is -Y).
    hw, hdp = head_d * 0.92, head_d * 0.88
    P("Head", "sphere", (hw, hdp, head_d), (0, 0, head_c), "Skin", "head", subsurf=sub)
    face_y = -hdp * 0.47
    eye_s = 0.035 * clamp(a.eye_size, 0.5, 2.0)
    for s, x in (("L", 1), ("R", -1)):
        ex, ez = x * hw * 0.2, head_c + head_d * 0.05
        if a.eyes == "dot":
            P(f"Eye.{s}", "sphere", (eye_s * 0.7, eye_s * 0.4, eye_s * 0.9), (ex, face_y, ez), "Pupil", "head")
        else:
            big = 1.5 if a.eyes == "anime" else 1.0
            P(f"EyeWhite.{s}", "sphere", (eye_s * 1.3 * big, eye_s * 0.5, eye_s * 1.5 * big), (ex, face_y, ez), "EyeWhite", "head")
            P(f"Iris.{s}", "sphere", (eye_s * 0.8 * big, eye_s * 0.3, eye_s * 0.95 * big), (ex, face_y - eye_s * 0.2, ez), "Iris", "head")
            P(f"Pupil.{s}", "sphere", (eye_s * 0.4 * big, eye_s * 0.2, eye_s * 0.5 * big), (ex, face_y - eye_s * 0.32, ez), "Pupil", "head")
    mz = head_c - head_d * 0.2
    if a.mouth == "smile":
        P("Mouth", "sphere", (hw * 0.28, 0.01, head_d * 0.05), (0, face_y + 0.005, mz), "Mouth", "head",
          cut=lambda v: v.z > 0.1)
    elif a.mouth == "open":
        P("Mouth", "sphere", (hw * 0.24, 0.012, head_d * 0.12), (0, face_y + 0.006, mz), "Mouth", "head")
        P("Tongue", "sphere", (hw * 0.14, 0.012, head_d * 0.05), (0, face_y + 0.002, mz - head_d * 0.03), "Tongue", "head")
    else:
        P("Mouth", "cube", (hw * 0.22, 0.01, head_d * 0.02), (0, face_y + 0.004, mz), "Mouth", "head", bevel=0.002)

    for s, x in (("L", 1), ("R", -1)):
        if a.ears == "round":
            P(f"Ear.{s}", "sphere", (head_d * 0.08, head_d * 0.14, head_d * 0.2), (x * hw * 0.5, 0, head_c), "Skin", "head")
        elif a.ears == "pointy":
            P(f"Ear.{s}", "cone", (head_d * 0.1, head_d * 0.06, head_d * 0.32), (x * hw * 0.55, 0, head_c + head_d * 0.08),
              "Skin", "head", rot=(0, -x * 60, 0))
        elif a.ears == "animal":
            P(f"Ear.{s}", "cone", (head_d * 0.22, head_d * 0.1, head_d * 0.3), (x * hw * 0.3, 0, head_c + head_d * 0.52),
              "Hair", "head", rot=(0, -x * 15, 0))

    cap_r = 1.06
    if a.hair in ("short", "long", "bun", "spiky"):
        P("HairCap", "sphere", (hw * cap_r, hdp * cap_r, head_d * cap_r), (0, head_d * 0.02, head_c + head_d * 0.03), "Hair", "head",
          cut=lambda v: v.z < -0.05 or (v.y < -0.3 and v.z < 0.25), subsurf=sub, solidify=0.004)
    if a.hair == "spiky":
        for i in range(7):
            ang = math.radians(-90 + i * 30)
            P(f"Spike.{i}", "cone", (head_d * 0.18, head_d * 0.18, head_d * 0.35),
              (math.cos(ang) * hw * 0.3, math.sin(ang) * hdp * 0.1 + head_d * 0.05, head_c + head_d * 0.5),
              "Hair", "head", rot=(math.degrees(math.sin(ang)) * 0.4, math.degrees(math.cos(ang)) * 0.5, 0))
    elif a.hair == "long":
        P("HairBack", "cube", (hw * 1.0, head_d * 0.18, head_d * 1.1), (0, hdp * 0.42, head_c - head_d * 0.35), "Hair", "head",
          bevel=0.01 * H, subsurf=sub)
    elif a.hair == "bun":
        P("HairBun", "sphere", (head_d * 0.4, head_d * 0.4, head_d * 0.4), (0, hdp * 0.35, head_c + head_d * 0.5), "Hair", "head", subsurf=sub)
    elif a.hair == "mohawk":
        P("Mohawk", "cube", (head_d * 0.12, hdp * 1.0, head_d * 0.25), (0, 0, head_c + head_d * 0.52), "Hair", "head",
          bevel=0.006 * H, subsurf=sub)

    # Body.
    P("Neck", "cyl", (0.05 * bw * 1.2, 0.05 * bw * 1.2, neck_h + 0.02), (0, 0, shoulder_z + neck_h / 2), "Skin", "neck")
    P("Torso", "cyl", (shoulder_x * 2.1, 0.15 * bw * belly, torso_h), (0, 0, hip_z + torso_h / 2), torso_mat, "spine",
      taper=0.92 / belly ** 0.3, subsurf=sub, bevel=0.01 * H)
    if belly > 1.15:
        P("Belly", "sphere", (shoulder_x * 1.6 * belly, 0.16 * bw * belly, torso_h * 0.75),
          (0, -0.02 * belly, hip_z + torso_h * 0.4), torso_mat, "spine", subsurf=sub)
    if a.top == "hoodie":
        P("Hood", "sphere", (head_d * 0.75, head_d * 0.35, head_d * 0.45), (0, hdp * 0.35, shoulder_z + 0.02), "Top", "spine", subsurf=sub)
    if a.top == "jacket":
        P("JacketTrim", "cube", (0.02, 0.01, torso_h * 0.95), (0, -0.15 * bw * belly / 2 - 0.003, hip_z + torso_h / 2), "Accent", "spine")
    P("Pelvis", "sphere", (hip_x * 3.2, 0.14 * bw * belly ** 0.5, 0.12), (0, 0, hip_z), "Bottom", "pelvis", subsurf=sub)
    if a.bottom == "skirt":
        P("Skirt", "cone", (hip_x * 5.0, 0.2 * bw, leg * 0.55), (0, 0, hip_z - leg * 0.22), "Bottom", "pelvis",
          cut=lambda v: v.z > 0.3)

    for s, x in (("L", 1), ("R", -1)):
        # Arms, horizontal (T-pose) along X.
        P(f"UpperArm.{s}", "cyl", (arm_r * 2, arm_r * 2, elbow_x - shoulder_x + 0.02), (x * (shoulder_x + elbow_x) / 2, 0, arm_z),
          sleeve_upper, f"upper_arm.{s}", rot=(0, 90, 0))
        P(f"Forearm.{s}", "cyl", (arm_r * 1.8, arm_r * 1.8, wrist_x - elbow_x + 0.02), (x * (elbow_x + wrist_x) / 2, 0, arm_z),
          sleeve_lower, f"forearm.{s}", rot=(0, 90, 0), taper=0.85)
        hs = 0.05 * clamp(a.hand_size, 0.5, 2.0)
        P(f"Hand.{s}", "sphere", (hs * 1.4, hs * 0.8, hs), (x * (wrist_x + hs * 0.6), 0, arm_z), "Skin", f"hand.{s}", subsurf=sub)
        P(f"Thumb.{s}", "sphere", (hs * 0.5, hs * 0.4, hs * 0.4), (x * (wrist_x + hs * 0.35), -hs * 0.45, arm_z), "Skin", f"hand.{s}")
        # Legs.
        P(f"Thigh.{s}", "cyl", (leg_r * 2, leg_r * 2, hip_z - knee_z + 0.02), (x * hip_x, 0, (hip_z + knee_z) / 2), thigh_mat, f"thigh.{s}")
        P(f"Shin.{s}", "cyl", (leg_r * 1.8, leg_r * 1.8, knee_z - foot_h + 0.02), (x * hip_x, 0, (knee_z + foot_h) / 2), shin_mat,
          f"shin.{s}", taper=0.85)
        fs = clamp(a.foot_size, 0.5, 2.0)
        if a.shoes == "none":
            P(f"Foot.{s}", "sphere", (leg_r * 1.9 * fs, 0.13 * fs, foot_h * 1.2), (x * hip_x, -0.03 * fs, foot_h / 2), "Skin", f"foot.{s}")
        else:
            boot = a.shoes == "boots"
            P(f"Shoe.{s}", "cube", (leg_r * 2.3 * fs, 0.15 * fs, foot_h), (x * hip_x, -0.035 * fs, foot_h / 2 + 0.006), "Shoe",
              f"foot.{s}", bevel=0.012 * H)
            P(f"Sole.{s}", "cube", (leg_r * 2.4 * fs, 0.155 * fs, 0.012), (x * hip_x, -0.035 * fs, 0.006), "Sole", f"foot.{s}", bevel=0.003 * H)
            if boot:
                P(f"BootShaft.{s}", "cyl", (leg_r * 2.2, leg_r * 2.2, leg * 0.3), (x * hip_x, 0, foot_h + leg * 0.15), "Shoe", f"shin.{s}")

    if a.tail == "cat":
        for i in range(5):
            P(f"Tail.{i}", "cyl", (0.025, 0.025, 0.09), (0, 0.1 * bw + i * 0.05, hip_z - 0.02 + i * 0.035 + (i * i) * 0.006),
              "Hair", "pelvis", rot=(-55 + i * 8, 0, 0))
    elif a.tail == "lizard":
        # cone axis after rot (-70 deg about X) = (0, sin70, cos70); put the wide base inside the pelvis back
        back = 0.07 * bw * belly ** 0.5
        P("Tail", "cone", (0.12 * bw, 0.12 * bw, 0.45), (0, back - 0.015 + 0.225 * 0.94, hip_z + 0.225 * 0.34), "Skin", "pelvis",
          rot=(-70, 0, 0), subsurf=sub)

    if a.accessory == "glasses":
        for s, x in (("L", 1), ("R", -1)):
            P(f"Lens.{s}", "cyl", (eye_s * 3.2, eye_s * 3.2, 0.006), (x * hw * 0.2, face_y - 0.02, head_c + head_d * 0.05),
              "Pupil", "head", rot=(90, 0, 0))  # closed disc: solidify would add an inward shell
        P("Bridge", "cube", (hw * 0.12, 0.006, 0.006), (0, face_y - 0.02, head_c + head_d * 0.07), "Pupil", "head")
    elif a.accessory == "hat":
        P("HatBrim", "cyl", (hw * 1.5, hdp * 1.5, 0.015), (0, 0, head_c + head_d * 0.36), "Accent", "head")
        P("HatTop", "cyl", (hw * 0.95, hdp * 0.95, head_d * 0.42), (0, 0, head_c + head_d * 0.56), "Accent", "head", bevel=0.006 * H)
    elif a.accessory == "crown":
        P("Crown", "cyl", (hw * 0.75, hdp * 0.75, head_d * 0.2), (0, 0, head_c + head_d * 0.5), "Accent", "head",
          cut=lambda v: v.z > 0.45, solidify=0.006)
    elif a.accessory == "cape":
        P("Cape", "cube", (shoulder_x * 2.6, 0.012, torso_h + leg * 0.7), (0, 0.15 * bw * belly / 2 - 0.002, shoulder_z - (torso_h + leg * 0.7) / 2),
          "Accent", "spine", bevel=0.004 * H)

    bones = [
        ("root", (0, 0, 0), (0, 0, 0.1 * H), None),
        ("pelvis", S(0, 0, hip_z - 0.04), S(0, 0, hip_z + 0.04), "root"),
        ("spine", S(0, 0, hip_z + 0.04), S(0, 0, shoulder_z), "pelvis"),
        ("neck", S(0, 0, shoulder_z), S(0, 0, neck_top), "spine"),
        ("head", S(0, 0, neck_top), S(0, 0, head_c + head_d * 0.5), "neck"),
    ]
    for s, x in (("L", 1), ("R", -1)):
        bones += [
            (f"upper_arm.{s}", S(x * shoulder_x * 0.9, 0, arm_z), S(x * elbow_x, 0, arm_z), "spine"),
            (f"forearm.{s}", S(x * elbow_x, 0, arm_z), S(x * wrist_x, 0, arm_z), f"upper_arm.{s}"),
            (f"hand.{s}", S(x * wrist_x, 0, arm_z), S(x * hand_x, 0, arm_z), f"forearm.{s}"),
            (f"thigh.{s}", S(x * hip_x, 0, hip_z - 0.02), S(x * hip_x, 0, knee_z), "pelvis"),
            (f"shin.{s}", S(x * hip_x, 0, knee_z), S(x * hip_x, 0, foot_h), f"thigh.{s}"),
            (f"foot.{s}", S(x * hip_x, 0, foot_h), S(x * hip_x, -0.12 * H, foot_h * 0.5), f"shin.{s}"),
        ]
    return parts, bones


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    gm.reset_scene()
    parts, bones = build(a)
    rig = None if a.no_rig else gm.build_rig(parts, bones)
    tpose = gm.tpose_selfcheck(parts)
    deform = gm.deformation_selftest(rig, parts) if rig else None
    slug = gm.slugify(a.name) or "character"
    files = {"glb": export_model.export_glb(os.path.join(a.out, f"{slug}.glb"))}
    if a.blend:
        files["blend"] = export_model.save_blend(os.path.join(a.out, f"{slug}.blend"))
    info = {
        "generator": "character_builder", "blender": bpy.app.version_string, "name": a.name, "slug": slug,
        "parts": len(parts), "triangles": sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in parts),
        "vertices": sum(len(o.data.vertices) for o in parts),
        "rig": {"bones": len(bones), "skinning": "rigid per-part"} if rig else None,
        "tpose_selfcheck": tpose, "deformation_selftest": deform, "files": files,
        "params": {k: v for k, v in vars(a).items() if k != "out"},
    }
    with open(os.path.join(a.out, "generation.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: info[k] for k in ("name", "triangles", "tpose_selfcheck", "deformation_selftest")}))
    if not tpose["pass"]:
        raise SystemExit(f"T-pose self-check failed: {tpose}")
    if deform and not deform["pass"]:
        raise SystemExit(f"Rig deformation self-test failed: {deform}")


if __name__ == "__main__":
    main()
