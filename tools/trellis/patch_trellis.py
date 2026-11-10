"""Patch a trellis.cpp checkout for CPU-only phones:  python patch_trellis.py /path/to/trellis.cpp

trellis.cpp skips ggml's repacked ARM matmul kernels on the CPU backend. Turning them on for real matmul
weights (2-D "*.weight" tensors) makes a DiT forward 1.4x faster with the same output. The 2-D rms-norm
gammas are not matmul weights: repacking them gave NaNs, so they are left alone.
"""
import sys

p = sys.argv[1] + "/src/trellis_model.cpp"
s = open(p).read()
a = "            reg = nullptr;\n"
b = "        if (ggml_n_dims(const_cast<ggml_tensor*>(t)) < 2) return false;\n"
assert a in s and b in s, "trellis.cpp changed: re-check the repack patch"
s = s.replace(a, "", 1).replace(b, """        if (ggml_n_dims(const_cast<ggml_tensor*>(t)) != 2) return false;
        { const char* n = t->name; size_t l = strlen(n); if (l < 7 || strcmp(n + l - 7, ".weight") != 0) return false; }
""", 1)
open(p, "w").write("#include <cstring>\n" + s)
print("patched", p)
