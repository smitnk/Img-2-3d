// Host test for the mesh-repair pipeline (same mesh_repair.cpp the Android JNI library ships).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mesh_repair.h"

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } } while (0)

// Unit cube, 8 shared vertices, 12 outward triangles (2 per face).
static const float CUBE_V[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
static const int CUBE_T[12][3] = {
    {0,2,1},{0,3,2},   // z=0 (bottom, faces -z)
    {4,5,6},{4,6,7},   // z=1 (top)
    {0,1,5},{0,5,4},   // y=0
    {2,3,7},{2,7,6},   // y=1
    {0,4,7},{0,7,3},   // x=0
    {1,2,6},{1,6,5}};  // x=1

struct M { std::vector<float> p; std::vector<int> i; std::vector<float> c; };

static M cube(int skip0 = -1, int skip1 = -1) {
    M m;
    for (auto& v : CUBE_V) m.p.insert(m.p.end(), v, v + 3);
    for (int t = 0; t < 12; t++) { if (t == skip0 || t == skip1) continue; m.i.insert(m.i.end(), CUBE_T[t], CUBE_T[t] + 3); }
    for (int v = 0; v < 8; v++) { m.c.push_back(v / 8.f); m.c.push_back(0.5f); m.c.push_back(1.f - v / 8.f); }
    return m;
}

static void addTriangle(M& m, float ox, float oy, float oz, float s) {  // tiny floating open triangle, own vertices
    int b = (int)(m.p.size() / 3);
    float v[3][3] = {{ox, oy, oz}, {ox + s, oy, oz}, {ox, oy + s, oz}};
    for (auto& a : v) { m.p.insert(m.p.end(), a, a + 3); m.c.push_back(1); m.c.push_back(0); m.c.push_back(0); }
    m.i.insert(m.i.end(), {b, b + 1, b + 2});
}

static repair::Mesh go(const M& m, repair::Stats& st, double pct = 1.0) {
    repair::Params p; p.minPiecePct = pct;
    return repair::run(m.p.data(), m.p.size() / 3, m.i.data(), m.i.size(), m.c.empty() ? nullptr : m.c.data(), p, st);
}

static void report(const char* name, const repair::Stats& s) {
    std::printf("%s: tris %d -> %d, verts %d -> %d, welded %d, holes %d, degenerate %d, removed %d, watertight %s, vol %.4f, status '%s'\n",
                name, s.trisBefore, s.trisAfter, s.vertsBefore, s.vertsAfter, s.welded, s.holesFilled, s.degenerate,
                s.piecesRemoved, s.watertight ? "yes" : "no", s.volume, s.status.c_str());
}

int main() {
    {  // 1. cube missing one triangle (3-edge hole) + tiny floating triangle
        M m = cube(10);
        addTriangle(m, 5, 5, 5, 0.01f);
        repair::Stats s; repair::Mesh r = go(m, s); report("cube missing a triangle + floater", s);
        CHECK(s.watertight, "watertight");
        CHECK(s.piecesRemoved >= 1, "floater removed");
        CHECK(s.piecesKept == 1, "one piece kept");
        CHECK(std::fabs(s.volume - 1.0) < 1e-4, "cube volume 1");
        CHECK(s.holesFilled >= 1, "hole filled");
        CHECK(r.col.size() == r.pos.size(), "colours follow vertices");
        for (float v : r.pos) CHECK(v < 1.5f, "floater vertices gone");
    }
    {  // 2. cube missing a whole face (4-edge hole, centroid fan) + floater
        M m = cube(2, 3);
        addTriangle(m, -3, 0, 0, 0.02f);
        repair::Stats s; repair::Mesh r = go(m, s); report("cube missing a face + floater", s);
        CHECK(s.watertight, "watertight");
        CHECK(s.piecesRemoved >= 1, "floater removed");
        CHECK(std::fabs(s.volume - 1.0) < 1e-4, "cube volume 1");
        CHECK(r.idx.size() >= 36, "at least 12 triangles");
    }
    {  // 3. duplicate vertices: every triangle has its own 3 vertices
        M c = cube(), m;
        for (size_t t = 0; t < c.i.size(); t++) {
            m.p.insert(m.p.end(), &c.p[c.i[t] * 3], &c.p[c.i[t] * 3] + 3);
            m.i.push_back((int)t);
        }
        repair::Stats s; repair::Mesh r = go(m, s); report("duplicate-vertex cube (36 verts)", s);
        CHECK(s.vertsBefore == 36, "36 input vertices");
        CHECK(s.watertight, "watertight");
        CHECK(s.welded == 28, "28 vertices welded away");
        CHECK(r.pos.size() / 3 == 8, "8 vertices left");
        CHECK(s.trisAfter == 12, "12 triangles");
        CHECK(r.col.empty(), "no colours in, none out");
    }
    {  // 4. inside-out cube is flipped, not rejected
        M m = cube();
        for (size_t t = 0; t < m.i.size(); t += 3) std::swap(m.i[t + 1], m.i[t + 2]);
        repair::Stats s; repair::Mesh r = go(m, s); report("inside-out cube", s);
        CHECK(s.watertight, "watertight");
        CHECK(std::fabs(s.volume - 1.0) < 1e-4, "cube volume 1");
    }
    {  // 5. slider: a piece above the threshold survives, below it goes
        M m = cube();
        M big = cube();
        for (size_t k = 0; k < big.p.size(); k += 3) { big.p[k] = big.p[k] * 0.2f + 3; big.p[k + 1] *= 0.2f; big.p[k + 2] *= 0.2f; }  // volume 0.008
        int b = (int)(m.p.size() / 3);
        m.p.insert(m.p.end(), big.p.begin(), big.p.end());
        m.c.insert(m.c.end(), big.c.begin(), big.c.end());
        for (int v : big.i) m.i.push_back(v + b);
        repair::Stats a; go(m, a, 1.0); report("two cubes, min 1%", a);
        repair::Stats z; go(m, z, 0.5); report("two cubes, min 0.5%", z);
        CHECK(a.piecesRemoved == 1 && a.piecesKept == 1, "0.8% piece removed at 1%");
        CHECK(z.piecesRemoved == 0 && z.piecesKept == 2, "0.8% piece kept at 0.5%");
    }
    {  // 6. non-manifold (three triangles on one edge): reported not watertight, cleaned mesh still returned
        M m;
        m.p = {0,0,0, 1,0,0, 0,1,0, 0,-1,0, 0,0,1};
        m.i = {0,1,2, 1,0,3, 0,1,4};
        repair::Stats s; repair::Mesh r = go(m, s); report("non-manifold edge", s);
        CHECK(!s.watertight, "not watertight");
        CHECK(!r.idx.empty(), "cleaned mesh returned");
    }
    {  // 7. empty and garbage input do not crash
        M m; repair::Stats s; go(m, s);
        M g; g.p = {0,0,0, 1,0,0, 0,1,0}; g.i = {0,1,99, -1,0,1}; go(g, s);
        report("garbage indices", s);
        CHECK(true, "no crash");
    }
    std::printf(failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", failures);
    return failures ? 1 : 0;
}
