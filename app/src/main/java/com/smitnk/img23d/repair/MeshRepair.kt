package com.smitnk.img23d.repair

import com.smitnk.img23d.core.Mesh

/** Result of [MeshRepair.repair]: the cleaned mesh (glTF Y-up, same space as the input) plus before / after stats. */
class RepairResult(
    val positions: FloatArray, val indices: IntArray, val colors: FloatArray?,
    val trisBefore: Int, val trisAfter: Int, val vertsBefore: Int, val vertsAfter: Int,
    val watertight: Boolean, val piecesRemoved: Int, val holesFilled: Int, val welded: Int, val degenerate: Int,
    val volume: Double, val status: String,
) {
    fun toMesh() = Mesh(positions, indices, colors)
}

/**
 * Mesh repair in C++ (manifold, Apache-2.0): weld duplicate vertices, fill holes, build a watertight manifold, drop floating pieces
 * whose volume is under a percentage of the total. If the mesh cannot be made manifold the welded + hole-filled mesh comes back
 * with [RepairResult.watertight] = false.
 */
object MeshRepair {
    init { System.loadLibrary("meshrepair") }

    @JvmStatic // public: the JNI symbol is Java_com_smitnk_img23d_repair_MeshRepair_nativeRepair
    external fun nativeRepair(pos: FloatArray, idx: IntArray, col: FloatArray?, weldRel: Double, minPiecePct: Double): Array<Any?>

    /** [colors] is rgb per vertex or null. [minPiecePct] 0..100: pieces under this share of the total volume are removed (the biggest always stays). */
    fun repair(positions: FloatArray, indices: IntArray, colors: FloatArray?, minPiecePct: Double = 1.0, weldRel: Double = 1e-5): RepairResult {
        val o = nativeRepair(positions, indices, colors, weldRel, minPiecePct)
        val s = o[3] as IntArray
        return RepairResult(
            positions = o[0] as FloatArray, indices = o[1] as IntArray, colors = o[2] as FloatArray?,
            trisBefore = s[0], trisAfter = s[1], vertsBefore = s[2], vertsAfter = s[3], watertight = s[4] == 1,
            piecesRemoved = s[5], holesFilled = s[6], welded = s[7], degenerate = s[8],
            volume = o[5] as Double, status = o[4] as String,
        )
    }
}
