package com.smitnk.img23d.viewer

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Flattens any binary glTF 2.0 (TripoSR output, Blender exports with many meshes, node hierarchies, skins,
 * strided buffers, 8/16/32-bit indices) into one render-ready triangle soup in scene space.
 * Textures are not sampled: a primitive without COLOR_0 gets its material's baseColorFactor.
 */
class GltfMesh(val positions: FloatArray, val normals: FloatArray, val colors: FloatArray, val indices: IntArray) {
    constructor(glb: ByteArray) : this(GltfParser(glb))
    private constructor(p: GltfParser) : this(p.positions, p.normals, p.colors, p.indices)

    companion object {
        /** Smooth-shaded mesh from shared vertices + triangles (e.g. a live morph preview), one colour. */
        fun fromIndexed(pos: FloatArray, tris: IntArray, rgb: FloatArray) =
            GltfMesh(pos, faceNormals(pos, tris), FloatArray(pos.size) { rgb[it % 3] }, tris)
    }
}

private class GltfParser(glb: ByteArray) {
    val positions: FloatArray
    val normals: FloatArray
    val colors: FloatArray
    val indices: IntArray

    private val bin: ByteBuffer
    private val json: JSONObject

    init {
        val b = ByteBuffer.wrap(glb).order(ByteOrder.LITTLE_ENDIAN)
        require(glb.size >= 20 && b.getInt(0) == 0x46546C67) { "Not a GLB file" }
        require(b.getInt(4) == 2) { "Unsupported glTF version ${b.getInt(4)}" }
        var off = 12
        var j: JSONObject? = null
        var data: ByteBuffer = ByteBuffer.allocate(0)
        while (off + 8 <= glb.size) {
            val len = b.getInt(off)
            val type = b.getInt(off + 4)
            require(len >= 0 && off + 8 + len <= glb.size) { "Corrupt GLB chunk" }
            if (type == 0x4E4F534A) j = JSONObject(String(glb, off + 8, len, Charsets.UTF_8))
            if (type == 0x004E4942) data = ByteBuffer.wrap(glb, off + 8, len).slice().order(ByteOrder.LITTLE_ENDIAN)
            off += 8 + len + (4 - len % 4) % 4
        }
        json = j ?: throw IllegalArgumentException("GLB has no JSON chunk")
        bin = data

        val p = FloatList(); val n = FloatList(); val c = FloatList(); val idx = IntList()
        val nodes = json.optJSONArray("nodes") ?: JSONArray()
        val scene = json.optJSONArray("scenes")?.optJSONObject(json.optInt("scene", 0))
        val roots = scene?.optJSONArray("nodes")?.let { a -> (0 until a.length()).map { a.getInt(it) } }
            ?: (0 until nodes.length()).toList() // no scene: draw every node
        fun visit(i: Int, parent: FloatArray, depth: Int) {
            if (depth > 64) return
            val node = nodes.getJSONObject(i)
            val world = mul(parent, local(node))
            if (node.has("mesh")) {
                // glTF: a skinned mesh ignores its node transform; vertices are already in bind space.
                val m = if (node.has("skin")) IDENTITY else world
                addMesh(json.getJSONArray("meshes").getJSONObject(node.getInt("mesh")), m, p, n, c, idx)
            }
            node.optJSONArray("children")?.let { ch -> for (k in 0 until ch.length()) visit(ch.getInt(k), world, depth + 1) }
        }
        roots.forEach { visit(it, IDENTITY, 0) }
        require(idx.size > 0) { "GLB contains no triangles" }
        positions = p.toArray(); normals = n.toArray(); colors = c.toArray(); indices = idx.toArray()
    }

    private fun addMesh(mesh: JSONObject, m: FloatArray, p: FloatList, n: FloatList, c: FloatList, idx: IntList) {
        val prims = mesh.getJSONArray("primitives")
        for (k in 0 until prims.length()) {
            val prim = prims.getJSONObject(k)
            if (prim.optInt("mode", 4) != 4) continue // triangles only
            val attrs = prim.getJSONObject("attributes")
            if (!attrs.has("POSITION")) continue
            val pos = read(attrs.getInt("POSITION"))
            val count = pos.size / 3
            val base = p.size / 3
            val tri = if (prim.has("indices")) readIndices(prim.getInt("indices")) else IntArray(count) { it }
            val nor = if (attrs.has("NORMAL")) read(attrs.getInt("NORMAL")) else faceNormals(pos, tri)
            val col = if (attrs.has("COLOR_0")) read(attrs.getInt("COLOR_0")) else null
            val colComps = if (col != null && count > 0) col.size / count else 0
            val fallback = materialColor(prim)
            for (v in 0 until count) {
                val x = pos[v * 3]; val y = pos[v * 3 + 1]; val z = pos[v * 3 + 2]
                p.add(m[0] * x + m[4] * y + m[8] * z + m[12]); p.add(m[1] * x + m[5] * y + m[9] * z + m[13]); p.add(m[2] * x + m[6] * y + m[10] * z + m[14])
                val nx = nor[v * 3]; val ny = nor[v * 3 + 1]; val nz = nor[v * 3 + 2]
                var tx = m[0] * nx + m[4] * ny + m[8] * nz; var ty = m[1] * nx + m[5] * ny + m[9] * nz; var tz = m[2] * nx + m[6] * ny + m[10] * nz
                val l = sqrt(tx * tx + ty * ty + tz * tz).takeIf { it > 1e-12f } ?: 1f
                tx /= l; ty /= l; tz /= l
                n.add(tx); n.add(ty); n.add(tz)
                if (col != null) { c.add(col[v * colComps]); c.add(col[v * colComps + 1]); c.add(col[v * colComps + 2]) }
                else { c.add(fallback[0]); c.add(fallback[1]); c.add(fallback[2]) }
            }
            for (t in 0 until tri.size / 3 * 3) {
                val i = tri[t]
                require(i in 0 until count) { "Index $i out of range ($count vertices)" }
                idx.add(base + i)
            }
        }
    }

    /** baseColorFactor is linear; the viewer shades in display space, so convert to sRGB. */
    private fun materialColor(prim: JSONObject): FloatArray {
        val mat = prim.optInt("material", -1).takeIf { it >= 0 }?.let { json.optJSONArray("materials")?.optJSONObject(it) }
        val f = mat?.optJSONObject("pbrMetallicRoughness")?.optJSONArray("baseColorFactor") ?: return floatArrayOf(0.8f, 0.8f, 0.8f)
        return FloatArray(3) { i -> f.optDouble(i, 1.0).toFloat().coerceIn(0f, 1f).pow(1f / 2.2f) }
    }

    private fun local(node: JSONObject): FloatArray {
        node.optJSONArray("matrix")?.let { a -> return FloatArray(16) { a.getDouble(it).toFloat() } }
        val t = node.optJSONArray("translation"); val r = node.optJSONArray("rotation"); val s = node.optJSONArray("scale")
        val tx = t?.getDouble(0)?.toFloat() ?: 0f; val ty = t?.getDouble(1)?.toFloat() ?: 0f; val tz = t?.getDouble(2)?.toFloat() ?: 0f
        val qx = r?.getDouble(0)?.toFloat() ?: 0f; val qy = r?.getDouble(1)?.toFloat() ?: 0f
        val qz = r?.getDouble(2)?.toFloat() ?: 0f; val qw = r?.getDouble(3)?.toFloat() ?: 1f
        val sx = s?.getDouble(0)?.toFloat() ?: 1f; val sy = s?.getDouble(1)?.toFloat() ?: 1f; val sz = s?.getDouble(2)?.toFloat() ?: 1f
        // Column-major T * R * S.
        return floatArrayOf(
            (1 - 2 * (qy * qy + qz * qz)) * sx, (2 * (qx * qy + qz * qw)) * sx, (2 * (qx * qz - qy * qw)) * sx, 0f,
            (2 * (qx * qy - qz * qw)) * sy, (1 - 2 * (qx * qx + qz * qz)) * sy, (2 * (qy * qz + qx * qw)) * sy, 0f,
            (2 * (qx * qz + qy * qw)) * sz, (2 * (qy * qz - qx * qw)) * sz, (1 - 2 * (qx * qx + qy * qy)) * sz, 0f,
            tx, ty, tz, 1f,
        )
    }

    private fun accessorView(index: Int): Triple<JSONObject, Int, Int> {
        val acc = json.getJSONArray("accessors").getJSONObject(index)
        val comps = when (acc.getString("type")) { "SCALAR" -> 1; "VEC2" -> 2; "VEC3" -> 3; "VEC4" -> 4; else -> error("Unsupported accessor type") }
        return Triple(acc, comps, acc.getInt("count"))
    }

    /** Reads a float accessor (float or normalized integer components) honouring byteStride. */
    private fun read(index: Int): FloatArray {
        val (acc, comps, count) = accessorView(index)
        val out = FloatArray(count * comps)
        if (!acc.has("bufferView")) return out // all zeros per spec
        val view = json.getJSONArray("bufferViews").getJSONObject(acc.getInt("bufferView"))
        val ct = acc.getInt("componentType")
        val size = compSize(ct)
        val stride = view.optInt("byteStride", 0).takeIf { it > 0 } ?: (size * comps)
        val start = view.optInt("byteOffset", 0) + acc.optInt("byteOffset", 0)
        require(count == 0 || start + (count - 1) * stride + comps * size <= bin.limit()) { "Accessor $index exceeds the binary buffer" }
        val norm = acc.optBoolean("normalized", ct != FLOAT)
        for (i in 0 until count) for (k in 0 until comps) {
            val at = start + i * stride + k * size
            out[i * comps + k] = when (ct) {
                FLOAT -> bin.getFloat(at)
                UBYTE -> (bin.get(at).toInt() and 0xFF).let { if (norm) it / 255f else it.toFloat() }
                USHORT -> (bin.getShort(at).toInt() and 0xFFFF).let { if (norm) it / 65535f else it.toFloat() }
                BYTE -> bin.get(at).toFloat().let { if (norm) maxOf(it / 127f, -1f) else it }
                SHORT -> bin.getShort(at).toFloat().let { if (norm) maxOf(it / 32767f, -1f) else it }
                else -> error("Unsupported component type $ct")
            }
        }
        return out
    }

    private fun readIndices(index: Int): IntArray {
        val (acc, _, count) = accessorView(index)
        val view = json.getJSONArray("bufferViews").getJSONObject(acc.getInt("bufferView"))
        val ct = acc.getInt("componentType")
        val size = compSize(ct)
        val stride = view.optInt("byteStride", 0).takeIf { it > 0 } ?: size
        val start = view.optInt("byteOffset", 0) + acc.optInt("byteOffset", 0)
        require(count == 0 || start + (count - 1) * stride + size <= bin.limit()) { "Index accessor $index exceeds the binary buffer" }
        return IntArray(count) { i ->
            val at = start + i * stride
            when (ct) {
                UBYTE -> bin.get(at).toInt() and 0xFF
                USHORT -> bin.getShort(at).toInt() and 0xFFFF
                UINT -> bin.getInt(at)
                else -> error("Unsupported index type $ct")
            }
        }
    }


    private class FloatList { var d = FloatArray(1024); var size = 0
        fun add(v: Float) { if (size == d.size) d = d.copyOf(size * 2); d[size++] = v }
        fun toArray() = d.copyOf(size) }

    private class IntList { var d = IntArray(1024); var size = 0
        fun add(v: Int) { if (size == d.size) d = d.copyOf(size * 2); d[size++] = v }
        fun toArray() = d.copyOf(size) }

    companion object {
        private const val BYTE = 5120; private const val UBYTE = 5121; private const val SHORT = 5122
        private const val USHORT = 5123; private const val UINT = 5125; private const val FLOAT = 5126
        private val IDENTITY = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)
        private fun compSize(ct: Int) = when (ct) { BYTE, UBYTE -> 1; SHORT, USHORT -> 2; else -> 4 }

        /** Column-major 4x4 product a * b (plain Kotlin so it also runs in JVM unit tests). */
        private fun mul(a: FloatArray, b: FloatArray) = FloatArray(16) { i ->
            val col = i / 4; val row = i % 4
            a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] + a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3]
        }
    }
}

/** Area-weighted vertex normals (the shader normalizes). */
internal fun faceNormals(pos: FloatArray, tri: IntArray): FloatArray {
    val out = FloatArray(pos.size)
    for (f in 0 until tri.size / 3) {
        val a = tri[f * 3] * 3; val b = tri[f * 3 + 1] * 3; val c = tri[f * 3 + 2] * 3
        if (maxOf(a, b, c) + 2 >= pos.size) continue
        val ux = pos[b] - pos[a]; val uy = pos[b + 1] - pos[a + 1]; val uz = pos[b + 2] - pos[a + 2]
        val vx = pos[c] - pos[a]; val vy = pos[c + 1] - pos[a + 1]; val vz = pos[c + 2] - pos[a + 2]
        val nx = uy * vz - uz * vy; val ny = uz * vx - ux * vz; val nz = ux * vy - uy * vx
        for (v in intArrayOf(a, b, c)) { out[v] += nx; out[v + 1] += ny; out[v + 2] += nz }
    }
    return out
}
