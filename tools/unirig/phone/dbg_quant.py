import collections, sys, onnx
m = onnx.load(sys.argv[1], load_external_data=False)
print("ops", dict(collections.Counter(n.op_type for n in m.graph.node)))
use = collections.defaultdict(list)
for n in m.graph.node:
    for i in n.input: use[i].append(n.op_type)
ini = sorted(m.graph.initializer, key=lambda t: -__import__("numpy").prod(t.dims))
tot = collections.Counter()
for t in m.graph.initializer: tot[onnx.TensorProto.DataType.Name(t.data_type)] += int(__import__("numpy").prod(t.dims))
print("elements by dtype", dict(tot))
for t in ini[:12]: print(t.name, list(t.dims), onnx.TensorProto.DataType.Name(t.data_type), use[t.name])
