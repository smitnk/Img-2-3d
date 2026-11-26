"""CI check (plain python): the cleaned animated GLB has morph targets, twist bones with rotation channels and a baked 'weights' channel."""
import json
import struct
import sys

d = open(sys.argv[1], "rb").read()
j = json.loads(d[20:20 + struct.unpack("<I", d[12:16])[0]])
nodes = j["nodes"]
anims = j.get("animations", [])
paths = [(nodes[c["target"]["node"]]["name"], c["target"]["path"]) for a in anims for c in a["channels"]]
print("animations:", len(anims), "channels:", len(paths))
w = [p for p in paths if p[1] == "weights"]
tw = [p for p in paths if "twist" in p[0] and p[1] == "rotation"]
print("weights channels:", len(w), "twist rotation channels:", len(tw))
import math
# the baked twist must really twist: forearm_twist2.L copies 0.66 of the hand's 90 deg Y twist -> quaternion y = sin(0.66*45 deg) at the last key
binpos = 20 + struct.unpack("<I", d[12:16])[0]
bin_ = d[binpos + 8:]
def acc(i):
    a = j["accessors"][i]; bv = j["bufferViews"][a["bufferView"]]
    n = {"SCALAR": 1, "VEC4": 4}.get(a["type"], 1)
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    return [struct.unpack_from("<" + "f" * n, bin_, off + 4 * n * k) for k in range(a["count"])]
ang = {}
for a in anims:
    for c in a["channels"]:
        nm = nodes[c["target"]["node"]]["name"]
        if nm in ("hand.L", "forearm_twist1.L", "forearm_twist2.L") and c["target"]["path"] == "rotation":
            q = acc(a["samplers"][c["sampler"]]["output"])[-1]  # x y z w, last frame
            ang[nm] = math.degrees(2 * math.acos(min(1.0, abs(q[3]))))
print("last-frame rotation angle (deg):", {k: round(v, 1) for k, v in ang.items()}, "expected hand 90, twist1 ~29.7, twist2 ~59.4")
y = ang.get("forearm_twist2.L")
ok = y is not None and abs(y - 59.4) < 6 and abs(ang.get("forearm_twist1.L", -99) - 29.7) < 6 and len(anims) >= 1 and len(w) >= 1 and len(tw) >= 1
print("ANIM CHECK", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
