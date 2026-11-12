"""Standalone UniRig skeleton tokenizer (grammar + detokenize) for the articulation-xl 256 checkpoint: numpy only, no yaml/box/UniRig checkout.

Copied (MIT, VAST-AI-Research/UniRig @6793c66) from src/tokenizer/tokenizer_part.py + make_skeleton in spec.py, config inlined:
num_discrete 256, range [-1,1], cls {vroid, mixamo, articulationxl}, parts {body, hand}. Bone names: the model always starts with the
articulationxl class token, for which Order.make_names yields bone_0..bone_{n-1}, so the skeleton yamls are not needed.
"""
from collections import defaultdict, namedtuple

import numpy as np

ND = 256
BRANCH, BOS, EOS, PAD, SPRING = ND, ND + 1, ND + 2, ND + 3, ND + 4
PARTS = {"body": ND + 5, "hand": ND + 6}
CLS_NONE = ND + 7
CLS = {"vroid": ND + 8, "mixamo": ND + 9, "articulationxl": ND + 10}  # vocab size 267
RANGE = (-1.0, 1.0)

Detok = namedtuple("Detok", "tokens parents bones tails parts names")
Detok.joints = property(lambda s: s.bones[:, 3:])


class Tokenizer:
    bos, eos, pad, num_discrete = BOS, EOS, PAD, ND

    def next_posible_token(self, ids):
        if ids.ndim == 0 or ids.shape[0] == 0:
            return [BOS]
        state = "bos"
        for i in ids:
            if state == "bos":
                assert i == BOS, "ids do not start with bos"
                state = "cls_part_joint"
            elif state == "cls_part_joint":
                state = "j2" if i < ND else ("part_joint" if (i == CLS_NONE or i in CLS.values()) else "joint")
            elif state == "part_joint":
                state = "j2" if i < ND else "part_joint"
            elif state == "j2":
                state = "j3"
            elif state == "j3":
                state = "branch_part_joint"
            elif state == "branch_part_joint":
                state = "joint" if i == BRANCH else ("j2" if i < ND else "joint")
            elif state == "joint":
                state = "j2"
        joint = list(range(ND))
        part = [SPRING] + list(PARTS.values())
        return {"bos": [BOS], "cls_part_joint": [CLS_NONE] + list(CLS.values()) + part + joint, "part_joint": part + joint + [EOS],
                "j2": joint, "j3": joint, "branch_part_joint": joint + part + [BRANCH, EOS], "joint": joint}[state]

    def detokenize(self, ids):
        ids = np.asarray(ids)
        if ids[0] != BOS:
            raise ValueError("first token is not bos")
        pad = 0
        while pad < len(ids) and ids[-pad - 1] == PAD:
            pad += 1
        if ids[-1 - pad] != EOS:
            raise ValueError("last token is not eos")
        ids = ids[1:-1 - pad]
        joints, pj, tails, parts = [], [], {}, []
        i, branch, last, nb = 0, False, None, 0
        und = lambda t: (t.astype(np.float32) + 0.5) / ND * (RANGE[1] - RANGE[0]) + RANGE[0]  # undiscretize
        while i < len(ids):
            t = ids[i]
            if t < ND:
                if branch:
                    p, cur, i = und(ids[i:i + 3]), und(ids[i + 3:i + 6]), i + 6
                else:
                    cur = und(ids[i:i + 3])
                    p = cur if not pj else last
                    i += 3
                joints.append(cur)
                pj.append(p)
                if last is not None:
                    tails[nb - 1] = cur
                last, nb, branch = cur, nb + 1, False
            elif t == BRANCH:
                branch, last, i = True, None, i + 1
            elif t == SPRING or t in PARTS.values():
                parts.append(None if t == SPRING else next(k for k, v in PARTS.items() if v == t))
                i += 1
            elif t in CLS.values() or t == CLS_NONE:
                i += 1
            else:
                raise ValueError(f"unexpected token found: {t}")
        bones, tails, parents = make_skeleton(np.stack(joints), np.stack(pj), tails)
        return Detok(ids, parents, bones, tails, parts, [f"bone_{k}" for k in range(nb)])


def make_skeleton(joints, p_joints, tails_dict, extrude_scale=0.5):
    """spec.make_skeleton(convert_leaf_bones_to_tails=False, extrude_tail_for_leaf=True, extrude_tail_for_branch=True, strict=False)."""
    bones, parents = [], []
    for i, joint in enumerate(joints):
        if not bones:
            bones.append(np.concatenate([joint, joint]))
            parents.append(None)
            continue
        dis, pid = 999999, None
        for j in reversed(range(i)):
            d = ((bones[j][3:] - p_joints[i]) ** 2).sum()
            if d < dis:
                pid, dis = j, d
        bones.append(np.concatenate([joints[pid], joint]))
        parents.append(pid)
    bones = np.stack(bones)
    children = defaultdict(list)
    for i, pid in enumerate(parents):
        if pid is not None:
            children[pid].append(i)
    for i, pid in enumerate(parents):  # tail for a leaf
        if children[i]:
            continue
        d = bones[i, 3:] - bones[pid, 3:]
        if np.linalg.norm(d) <= 1e-9:
            d = np.array([0., 0., 1.])
        tails_dict[i] = bones[i, 3:] + d * extrude_scale
    for i, pid in enumerate(parents):  # tail for a joint with several children
        if len(children[i]) <= 1:
            continue
        if pid is None:  # (sic) UniRig adds the head twice here; kept so the tails match
            av = sum(np.linalg.norm(bones[i, 3:] - bones[c, 3:]) for c in children[i]) / len(children[i])
            d = bones[i, 3:] + np.array([0., 0., extrude_scale * av])
        else:
            d = bones[i, 3:] - bones[pid, 3:]
            if np.linalg.norm(d) <= 1e-9:
                d = np.array([0., 0., 1.])
        tails_dict[i] = bones[i, 3:] + d * extrude_scale
    for i in range(len(parents)):  # one child: the tail is that child's head
        if len(children[i]) == 1:
            tails_dict[i] = bones[children[i][0], 3:]
    return bones, np.stack([tails_dict[i] for i in range(len(bones))]), parents
