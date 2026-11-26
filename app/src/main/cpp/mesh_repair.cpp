#include "mesh_repair.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <unordered_map>
#include <string>

#include "manifold/manifold.h"

namespace repair {
namespace {

struct Cell {
    int64_t x, y, z;
    bool operator==(const Cell& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct CellHash {
    size_t operator()(const Cell& c) const {
        uint64_t h = (uint64_t)c.x * 73856093u ^ (uint64_t)c.y * 19349663u ^ (uint64_t)c.z * 83492791u;
        return (size_t)(h ^ (h >> 29));
    }
};

inline uint64_t ekey(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

struct Work {
    std::vector<float> pos, col;   // col empty when no colours
    std::vector<int> idx;
    bool hasCol() const { return !col.empty(); }
    int nv() const { return (int)(pos.size() / 3); }
    int addVert(float x, float y, float z, const float* c) {
        pos.push_back(x); pos.push_back(y); pos.push_back(z);
        if (hasCol()) { col.push_back(c[0]); col.push_back(c[1]); col.push_back(c[2]); }
        return nv() - 1;
    }
};

// Merge vertices closer than eps (grid hash, 27-cell search), then drop degenerate and duplicate triangles.
void weld(Work& w, double eps, Stats& st) {
    const int n = w.nv();
    std::vector<int> remap(n);
    std::unordered_map<Cell, std::vector<int>, CellHash> grid;
    grid.reserve((size_t)n * 2);
    std::vector<float> npos, ncol;
    const double inv = 1.0 / eps;
    for (int i = 0; i < n; i++) {
        const float* p = &w.pos[(size_t)i * 3];
        Cell c{(int64_t)std::floor(p[0] * inv), (int64_t)std::floor(p[1] * inv), (int64_t)std::floor(p[2] * inv)};
        int found = -1;
        for (int dx = -1; dx <= 1 && found < 0; dx++)
            for (int dy = -1; dy <= 1 && found < 0; dy++)
                for (int dz = -1; dz <= 1 && found < 0; dz++) {
                    auto it = grid.find(Cell{c.x + dx, c.y + dy, c.z + dz});
                    if (it == grid.end()) continue;
                    for (int r : it->second) {
                        const float* q = &npos[(size_t)r * 3];
                        double ex = p[0] - q[0], ey = p[1] - q[1], ez = p[2] - q[2];
                        if (ex * ex + ey * ey + ez * ez <= eps * eps) { found = r; break; }
                    }
                }
        if (found < 0) {
            found = (int)(npos.size() / 3);
            npos.insert(npos.end(), p, p + 3);
            if (w.hasCol()) ncol.insert(ncol.end(), &w.col[(size_t)i * 3], &w.col[(size_t)i * 3] + 3);
            grid[c].push_back(found);
        }
        remap[i] = found;
    }
    st.welded = n - (int)(npos.size() / 3);
    w.pos.swap(npos);
    w.col.swap(ncol);

    // Remap triangles; drop collapsed ones, repeats of the same oriented triangle, and cancel opposite-facing twins.
    struct Tri { int a, b, c; };
    std::vector<Tri> tris;
    for (size_t t = 0; t + 2 < w.idx.size(); t += 3) {
        int a = remap[w.idx[t]], b = remap[w.idx[t + 1]], c = remap[w.idx[t + 2]];
        if (a == b || b == c || a == c) { st.degenerate++; continue; }
        tris.push_back({a, b, c});
    }
    // Rotation-invariant key: smallest vertex first keeps the winding.
    auto canon = [](int a, int b, int c) {
        int m[3] = {a, b, c};
        int s = (a <= b && a <= c) ? 0 : (b <= c ? 1 : 2);
        return std::to_string(m[s]) + "," + std::to_string(m[(s + 1) % 3]) + "," + std::to_string(m[(s + 2) % 3]);
    };
    std::vector<char> drop(tris.size(), 0);
    std::unordered_map<std::string, size_t> first;
    for (size_t i = 0; i < tris.size(); i++) {
        std::string ks = canon(tris[i].a, tris[i].b, tris[i].c), fs = canon(tris[i].a, tris[i].c, tris[i].b);
        if (first.count(ks)) { drop[i] = 1; st.degenerate++; continue; }
        auto tw = first.find(fs);
        if (tw != first.end() && !drop[tw->second]) { drop[tw->second] = 1; drop[i] = 1; st.degenerate += 2; continue; }
        first[ks] = i;
    }
    w.idx.clear();
    for (size_t i = 0; i < tris.size(); i++)
        if (!drop[i]) { w.idx.push_back(tris[i].a); w.idx.push_back(tris[i].b); w.idx.push_back(tris[i].c); }
}

// Close every boundary loop: 3-loops with one triangle, longer ones with a fan around the loop centroid.
void fillHoles(Work& w, Stats& st) {
    std::unordered_map<uint64_t, int> he;
    for (size_t t = 0; t + 2 < w.idx.size(); t += 3)
        for (int k = 0; k < 3; k++) he[ekey(w.idx[t + k], w.idx[t + (k + 1) % 3])]++;
    std::unordered_map<int, std::vector<int>> out;  // boundary vertex -> boundary edge targets
    for (auto& kv : he) {
        int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
        if (!he.count(ekey(b, a))) out[a].push_back(b);
    }
    for (auto& kv : out) std::sort(kv.second.begin(), kv.second.end());  // deterministic
    std::vector<int> starts;
    for (auto& kv : out) starts.push_back(kv.first);
    std::sort(starts.begin(), starts.end());
    std::vector<std::vector<int>> loops;
    for (int s : starts) {
        std::vector<int> path{s};
        std::unordered_map<int, int> at{{s, 0}};
        while (true) {
            int head = path.back();
            auto it = out.find(head);
            if (it == out.end() || it->second.empty()) {
                if (path.size() == 1) break;
                break;  // open chain (non-manifold input): give up on it
            }
            int nxt = it->second.back();
            it->second.pop_back();
            auto p = at.find(nxt);
            if (p != at.end()) {
                std::vector<int> loop(path.begin() + p->second, path.end());
                for (size_t i = p->second + 1; i < path.size(); i++) at.erase(path[i]);
                path.resize(p->second + 1);
                if (loop.size() >= 3) loops.push_back(std::move(loop));
            } else {
                at[nxt] = (int)path.size();
                path.push_back(nxt);
            }
        }
    }
    for (auto& loop : loops) {
        const int n = (int)loop.size();
        if (n == 3) {
            w.idx.insert(w.idx.end(), {loop[1], loop[0], loop[2]});
        } else {
            double cx = 0, cy = 0, cz = 0, cr = 0, cg = 0, cb = 0;
            for (int v : loop) {
                cx += w.pos[(size_t)v * 3]; cy += w.pos[(size_t)v * 3 + 1]; cz += w.pos[(size_t)v * 3 + 2];
                if (w.hasCol()) { cr += w.col[(size_t)v * 3]; cg += w.col[(size_t)v * 3 + 1]; cb += w.col[(size_t)v * 3 + 2]; }
            }
            float c[3] = {(float)(cr / n), (float)(cg / n), (float)(cb / n)};
            int cv = w.addVert((float)(cx / n), (float)(cy / n), (float)(cz / n), c);
            for (int i = 0; i < n; i++) w.idx.insert(w.idx.end(), {loop[(i + 1) % n], loop[i], cv});
        }
        st.holesFilled++;
    }
}

const char* errName(manifold::Manifold::Error e) {
    using E = manifold::Manifold::Error;
    switch (e) {
        case E::NoError: return "ok";
        case E::NonFiniteVertex: return "not watertight: non-finite vertex";
        case E::NotManifold: return "not watertight: edges shared by more than two faces or inconsistent face directions";
        case E::VertexOutOfBounds: return "bad triangle index";
        default: return "not watertight: mesh rejected";
    }
}

// Connected pieces (shared vertices) among the triangles.
int countPieces(const Work& w) {
    std::vector<int> par(w.nv());
    for (int i = 0; i < w.nv(); i++) par[i] = i;
    auto find = [&](int x) { while (par[x] != x) x = par[x] = par[par[x]]; return x; };
    std::vector<char> used(w.nv(), 0);
    for (size_t t = 0; t + 2 < w.idx.size(); t += 3) {
        for (int k = 0; k < 3; k++) used[w.idx[t + k]] = 1;
        par[find(w.idx[t])] = find(w.idx[t + 1]);
        par[find(w.idx[t + 1])] = find(w.idx[t + 2]);
    }
    int n = 0;
    for (int i = 0; i < w.nv(); i++) if (used[i] && find(i) == i) n++;
    return n;
}

manifold::MeshGL toGL(const Work& w, bool flip) {
    manifold::MeshGL g;
    const int nv = w.nv();
    g.numProp = w.hasCol() ? 6 : 3;
    g.vertProperties.resize((size_t)nv * g.numProp);
    for (int i = 0; i < nv; i++) {
        float* d = &g.vertProperties[(size_t)i * g.numProp];
        std::memcpy(d, &w.pos[(size_t)i * 3], 12);
        if (w.hasCol()) std::memcpy(d + 3, &w.col[(size_t)i * 3], 12);
    }
    g.triVerts.resize(w.idx.size());
    for (size_t t = 0; t + 2 < w.idx.size(); t += 3) {
        g.triVerts[t] = (uint32_t)w.idx[t];
        g.triVerts[t + 1] = (uint32_t)w.idx[flip ? t + 2 : t + 1];
        g.triVerts[t + 2] = (uint32_t)w.idx[flip ? t + 1 : t + 2];
    }
    return g;
}

}  // namespace

Mesh run(const float* pos, size_t nVert, const int* idx, size_t nIdx, const float* col, const Params& p, Stats& st) {
    st = Stats();
    st.vertsBefore = (int)nVert;
    st.trisBefore = (int)(nIdx / 3);
    Work w;
    w.pos.assign(pos, pos + nVert * 3);
    if (col) w.col.assign(col, col + nVert * 3);
    for (size_t i = 0; i + 2 < nIdx; i += 3) {
        // Skip triangles that point outside the vertex array rather than crash.
        if (idx[i] < 0 || idx[i + 1] < 0 || idx[i + 2] < 0 || (size_t)idx[i] >= nVert || (size_t)idx[i + 1] >= nVert || (size_t)idx[i + 2] >= nVert) { st.degenerate++; continue; }
        w.idx.insert(w.idx.end(), {idx[i], idx[i + 1], idx[i + 2]});
    }

    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < nVert; i++)
        for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], pos[i * 3 + k]); hi[k] = std::max(hi[k], pos[i * 3 + k]); }
    double diag = nVert ? std::sqrt(std::pow(hi[0] - lo[0], 2.0) + std::pow(hi[1] - lo[1], 2.0) + std::pow(hi[2] - lo[2], 2.0)) : 0;
    double eps = std::max(diag * p.weldRel, 1e-12);

    weld(w, eps, st);
    fillHoles(w, st);

    Mesh res;
    auto cleaned = [&](const Work& x) { res.pos = x.pos; res.idx = x.idx; res.col = x.col; };
    try {
        const int pieces = countPieces(w);
        manifold::Manifold m(toGL(w, false));
        if (m.Status() == manifold::Manifold::Error::NoError && m.Volume() < 0) m = manifold::Manifold(toGL(w, true));  // inside-out input
        if (m.Status() != manifold::Manifold::Error::NoError) {
            st.status = errName(m.Status());
            cleaned(w);
        } else {
            std::vector<manifold::Manifold> parts = m.Decompose();
            std::vector<double> vol(parts.size());
            double total = 0;
            size_t biggest = 0;
            for (size_t i = 0; i < parts.size(); i++) {
                vol[i] = std::fabs(parts[i].Volume());
                total += vol[i];
                if (vol[i] > vol[biggest]) biggest = i;
            }
            const double cut = total > 0 ? total * p.minPiecePct / 100.0 : 0;
            res.col.clear();
            for (size_t i = 0; i < parts.size(); i++) {
                if (i != biggest && vol[i] < cut) { st.piecesRemoved++; continue; }
                st.piecesKept++;
                st.volume += vol[i];
                manifold::MeshGL g = parts[i].GetMeshGL();
                const int base = (int)(res.pos.size() / 3);
                const size_t nv = g.NumVert();
                for (size_t v = 0; v < nv; v++) {
                    const float* s = &g.vertProperties[v * g.numProp];
                    res.pos.insert(res.pos.end(), s, s + 3);
                    if (w.hasCol()) res.col.insert(res.col.end(), s + 3, s + 6);
                }
                for (uint32_t t : g.triVerts) res.idx.push_back(base + (int)t);
            }
            // Slivers with no volume (a floating triangle closed into a flat double face) vanish inside manifold itself.
            st.piecesRemoved = std::max(st.piecesRemoved, pieces - st.piecesKept);
            st.watertight = true;
            st.status = "ok";
        }
    } catch (const std::exception& e) {
        st.status = std::string("manifold error: ") + e.what();
        cleaned(w);
    } catch (...) {
        st.status = "manifold error";
        cleaned(w);
    }
    st.vertsAfter = (int)(res.pos.size() / 3);
    st.trisAfter = (int)(res.idx.size() / 3);
    return res;
}

}  // namespace repair
