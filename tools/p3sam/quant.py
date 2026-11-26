"""Make the int8 (dynamic, per-channel weights) and fp16 variants of work/enc.onnx and work/heads.onnx (names enc_int8.onnx ...)."""
import os
import shutil
import sys
import traceback

import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unirig"))
import onnx_skin as O  # noqa: E402


def int8(src, dst, ops=("MatMul", "Gemm")):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for f in (dst, dst + ".data"):
        if os.path.exists(f):
            os.remove(f)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=list(ops), use_external_data_format=True)


def fp16(src, dst):
    from onnxconverter_common import float16
    m = onnx.load(src, load_external_data=True)
    m = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False)
    tmp = dst + ".tmp.onnx"
    onnx.save_model(m, tmp, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(tmp) + ".data", size_threshold=1024)
    O.externalize(tmp, dst)
    for f in (tmp, tmp + ".data"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    w = sys.argv[1]
    for name in ("enc", "heads"):
        O.externalize(f"{w}/{name}.onnx", f"{w}/{name}_fp32.onnx")
    for name in ("enc", "heads"):
        for var, fn in (("int8", int8), ("fp16", fp16)):
            try:
                if var == "fp16" and name == "heads" and os.path.exists(f"{w}/heads_fp16.onnx"):
                    print("QUANT heads fp16: exported from torch (capture.py)", flush=True)
                    continue
                fn(f"{w}/{name}_fp32.onnx", f"{w}/{name}_{var}.onnx")
                sz = os.path.getsize(f"{w}/{name}_{var}.onnx") + os.path.getsize(f"{w}/{name}_{var}.onnx.data")
                print(f"QUANT {name} {var}: {sz / 2 ** 20:.1f} MB", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"QUANT {name} {var}: FAILED", flush=True)
