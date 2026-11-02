"""Self-check for quant_llm.py on a tiny external-data model (runs in CI before the real export)."""
import os, subprocess, sys, tempfile
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto as T

d = tempfile.mkdtemp()
r = np.random.default_rng(0)
W1, W2 = r.standard_normal((64, 128)).astype(np.float32), r.standard_normal((128, 32)).astype(np.float32)
g = helper.make_graph([helper.make_node("MatMul", ["x", "W1"], ["h"], name="a"), helper.make_node("Relu", ["h"], ["r"]),
                       helper.make_node("Constant", [], ["W2t"], value=numpy_helper.from_array(np.ascontiguousarray(W2.T), "c")),
                       helper.make_node("Transpose", ["W2t"], ["W2x"], perm=[1, 0]), helper.make_node("MatMul", ["r", "W2x"], ["y"], name="b")], "g",
                      [helper.make_tensor_value_info("x", T.FLOAT, [1, "n", 64])], [helper.make_tensor_value_info("y", T.FLOAT, None)],
                      [numpy_helper.from_array(W1, "W1")])
onnx.save_model(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), f"{d}/t.onnx", save_as_external_data=True,
                all_tensors_to_one_file=True, location="t.onnx.data", size_threshold=0)
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "quant_llm.py"), f"{d}/t.onnx", f"{d}/tq.onnx"], check=True)
x = r.standard_normal((1, 5, 64)).astype(np.float32)
a = ort.InferenceSession(f"{d}/t.onnx").run(None, {"x": x})[0]
b = ort.InferenceSession(f"{d}/tq.onnx").run(None, {"x": x})[0]
err = np.abs(a - b).max() / np.abs(a).max()
q = onnx.load(f"{d}/tq.onnx", load_external_data=False)
assert sum(n.op_type == "MatMulInteger" for n in q.graph.node) == 2 and not any(n.op_type in ("Transpose", "Constant") for n in q.graph.node)
print("quant_llm self-test rel err", round(float(err), 4))
assert err < 0.05
