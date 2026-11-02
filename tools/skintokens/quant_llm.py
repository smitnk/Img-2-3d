"""Low-memory dynamic int8 quantization of llm.onnx (ORT's quantize_dynamic needed >34 GB on this 2.4 GB model).

python quant_llm.py llm.onnx llm_q.onnx
Each MatMul(x, W) with a constant 2-D W becomes the same graph ORT builds:
  DynamicQuantizeLinear(x) -> MatMulInteger(xq, Wq, xz, Wz) -> Cast float -> Mul(x_scale * W_scale)
W is read one tensor at a time straight from the external-data file (memmap), quantized per output column
(symmetric int8, zero point 0), and written to llm_q.onnx.data. Peak RAM ~ graph + one weight.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def load_weight(t, base):
    info = {e.key: e.value for e in t.external_data}
    shape = tuple(t.dims)
    if not info:
        return numpy_helper.to_array(t)
    mm = np.memmap(os.path.join(base, info["location"]), dtype=helper.tensor_dtype_to_np_dtype(t.data_type), mode="r",
                   offset=int(info.get("offset", 0)), shape=shape)
    return np.array(mm)


def quantize_col(w):
    """w [K,N] float32 -> int8 [K,N], scale [N] (per output column)."""
    s = np.abs(w).max(axis=0) / 63.0  # 7-bit like ORT reduce_range: x86 u8s8 MatMulInteger saturates on full int8
    s[s == 0] = 1.0
    return np.clip(np.round(w / s), -63, 63).astype(np.int8), s.astype(np.float32)


def main(src, dst):
    base = os.path.dirname(os.path.abspath(src))
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    out_dir = os.path.dirname(os.path.abspath(dst))
    data_name = os.path.basename(dst) + ".data"
    data_path = os.path.join(out_dir, data_name)
    if os.path.exists(data_path):
        os.remove(data_path)
    f = open(data_path, "ab")

    def ext(name, arr):  # append arr to the shared data file, return a TensorProto pointing at it
        arr = np.ascontiguousarray(arr)
        t = TensorProto(name=name, dims=list(arr.shape), data_type=helper.np_dtype_to_tensor_dtype(arr.dtype))
        off = f.tell()
        f.write(arr.tobytes())
        for k, v in (("location", data_name), ("offset", str(off)), ("length", str(arr.nbytes))):
            e = t.external_data.add()
            e.key, e.value = k, v
        t.data_location = TensorProto.EXTERNAL
        return t

    # without constant folding torch exports Linear as MatMul(x, Transpose(W)): follow that Transpose to W
    tposed = {n.output[0]: n for n in g.node if n.op_type == "Transpose" and n.input[0] in inits
              and list(next((a.ints for a in n.attribute if a.name == "perm"), [1, 0])) == [1, 0]}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    def weight_of(n):
        if n.op_type != "MatMul" or len(n.input) != 2:
            return None, False
        b = n.input[1]
        if b in inits:
            return inits[b], False
        if b in tposed and all(u.op_type == "MatMul" for u in users[b]):
            return inits[tposed[b].input[0]], True
        return None, False

    new_nodes, new_inits, quantized, done = [], [], 0, set()
    dropped = set()
    for n in g.node:
        w, t = weight_of(n)
        if w is None or len(w.dims) != 2 or w.data_type != TensorProto.FLOAT:
            new_nodes.append(n)
            continue
        if t:
            dropped.add(n.input[1])
        p = f"{n.name or n.output[0]}_q"
        if w.name not in done:
            arr = load_weight(w, base)
            q, s = quantize_col(arr.T if t else arr)
            new_inits += [ext(w.name + "_q", q), ext(w.name + "_s", s),
                          numpy_helper.from_array(np.zeros((), np.int8), w.name + "_z")]
            done.add(w.name)
        new_nodes += [
            helper.make_node("DynamicQuantizeLinear", [n.input[0]], [p + "_x", p + "_xs", p + "_xz"]),
            helper.make_node("MatMulInteger", [p + "_x", w.name + "_q", p + "_xz", w.name + "_z"], [p + "_i"]),
            helper.make_node("Cast", [p + "_i"], [p + "_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [p + "_xs", w.name + "_s"], [p + "_sc"]),
            helper.make_node("Mul", [p + "_f", p + "_sc"], [n.output[0]]),
        ]
        quantized += 1

    new_nodes = [n for n in new_nodes if not (n.op_type == "Transpose" and n.output[0] in dropped)]
    used = {i for n in new_nodes for i in n.input}
    keep = []
    for t in g.initializer:  # copy the untouched float initializers into the new data file
        if t.name in done or t.name not in used:
            continue
        keep.append(ext(t.name, load_weight(t, base)) if t.external_data else t)
    f.close()
    del g.node[:]
    g.node.extend(new_nodes)
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    onnx.save(m, dst)
    print(f"quantized {quantized} MatMuls, {len(done)} weights; data {os.path.getsize(data_path) / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
