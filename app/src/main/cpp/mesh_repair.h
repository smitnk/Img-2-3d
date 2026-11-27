// Mesh repair: weld duplicate vertices, fill holes, make watertight with manifold, drop tiny floating pieces.
// Plain C++17, no JNI: the Android library (jni_repair.cpp) and the host test (tests/native) share this file.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace repair {

struct Params {
    double weldRel = 1e-5;      // weld distance as a fraction of the bounding-box diagonal
    double minPiecePct = 1.0;   // remove pieces whose volume is below this % of the total volume
};

struct Stats {
    int trisBefore = 0, trisAfter = 0;
    int vertsBefore = 0, vertsAfter = 0;
    int welded = 0;             // vertices merged away
    int degenerate = 0;         // zero-area / duplicate triangles dropped
    int holesFilled = 0;
    int piecesRemoved = 0, piecesKept = 0;
    bool watertight = false;
    double volume = 0;          // of the result, when watertight
    std::string status;         // "ok" or why manifold refused the mesh
};

struct Mesh {
    std::vector<float> pos;     // xyz
    std::vector<int> idx;       // triangles
    std::vector<float> col;     // rgb per vertex, empty when the input had none
};

// col may be null. Never throws; on a mesh manifold cannot accept it returns the welded + hole-filled mesh with watertight=false.
Mesh run(const float* pos, size_t nVert, const int* idx, size_t nIdx, const float* col, const Params& p, Stats& st);

}  // namespace repair
