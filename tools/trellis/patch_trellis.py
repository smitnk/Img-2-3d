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

# Shape-only runs (--no-texture) wrote the raw dual-grid mesh: mixed winding, open sheets, floaters. Give them the
# same finish the textured path has: weld, fill holes, watertight narrow-band remesh, drop floaters, simplify to --faces.
p = sys.argv[1] + "/src/trellis_cli.cpp"
s = open(p).read()
a = "    if (pix) pixal3d_to_trellis_frame(mesh.verts);   // untextured GLB and the .ply\n"
assert a in s, "trellis.cpp changed: re-check the shape-only finish patch"
s = s.replace(a, """    if (!textured && pbr6.empty() && mesh.F() > 0) {
        const float weld_eps = 1.0f / ((float)so.res * 8.0f);
        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr, weld_eps);
        trellis::fill_small_holes(mesh.faces);
        trellis::TriBvh bvh = trellis::TriBvh::build(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F());
        trellis::Mesh rm = trellis::remesh_narrow_band_dc(mesh.verts.data(), mesh.V(), mesh.faces.data(), mesh.F(),
                                                          bvh, so.res, cfg.band > 0 ? cfg.band : std::max(1, so.res / 512));
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
            std::vector<float> dv; std::vector<int32_t> df;
            trellis::decimate_qem(rm.verts, rm.V(), rm.faces, rm.F(), trellis::qem_face_target(cfg), dv, df);
            trellis::weld_vertices(dv, df, nullptr, weld_eps);
            trellis::fill_small_holes(df);
            trellis::drop_small_components(dv, df, 0.03f);
            mesh.verts = dv; mesh.faces = df; colors.clear();
            printf("      shape-only finish: remeshed + simplified -> V=%d F=%d\\n", (int)mesh.V(), (int)mesh.F()); fflush(stdout);
        }
    }
""" + a, 1)
open(p, "w").write(s)
print("patched", p)
