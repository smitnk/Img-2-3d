package com.smitnk.img23d.viewer

import com.smitnk.img23d.core.Mesh
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GltfMeshTest {
    @Test fun readsTripoSrGlb() {
        val m = Mesh(floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f), intArrayOf(0, 1, 2), floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f))
        val g = GltfMesh(m.toGlb())
        assertEquals(9, g.positions.size)
        assertEquals(listOf(0, 1, 2), g.indices.toList())
        assertEquals(1f, g.colors[0], 1e-6f)
    }

    /** Blender-style file: node hierarchy with TRS, interleaved strided buffer, ushort indices, no normals, material colour, skinned node. */
    @Test fun readsHierarchyStridedUshortAndSkins() {
        // One triangle, interleaved position(12 bytes) + 4 padding bytes => stride 16.
        val verts = ByteBuffer.allocate(48 + 8).order(ByteOrder.LITTLE_ENDIAN)
        listOf(floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 0f, 0f), floatArrayOf(0f, 1f, 0f)).forEach { v -> v.forEach(verts::putFloat); verts.putFloat(9f) }
        listOf<Short>(0, 1, 2, 0).forEach(verts::putShort) // 3 indices + 2 bytes padding
        val json = """
        {"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,3]}],
         "nodes":[{"translation":[10,0,0],"children":[1]},{"scale":[2,2,2],"children":[2]},{"mesh":0,"rotation":[0,0,0,1]},{"mesh":0,"skin":0,"translation":[100,0,0]}],
         "skins":[{"joints":[0]}],
         "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,0,0,1]}}],
         "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"material":0}]}],
         "buffers":[{"byteLength":56}],
         "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48,"byteStride":16},{"buffer":0,"byteOffset":48,"byteLength":6}],
         "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}]}
        """.trimIndent()
        val g = GltfMesh(glb(json, verts.array()))
        assertEquals(6, g.indices.size) // two instances of the triangle
        // First instance: scale 2 then translate x+10 -> vertex 1 at (12,0,0).
        assertEquals(12f, g.positions[3], 1e-5f)
        // Skinned instance ignores its node transform (+100): vertex 1 stays at (1,0,0).
        assertEquals(1f, g.positions[12], 1e-5f)
        assertEquals(1f, g.colors[0], 1e-5f) // red material, sRGB 1.0
        assertEquals(0f, g.colors[1], 1e-5f)
        assertEquals(1f, g.normals[2], 1e-5f) // computed face normal +Z
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsTruncatedBuffer() {
        val json = """{"asset":{"version":"2.0"},"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
          "buffers":[{"byteLength":4}],"bufferViews":[{"buffer":0,"byteLength":4}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}]}"""
        GltfMesh(glb(json, ByteArray(4)))
    }

    /** Real Blender exports, copied in by CI (blender-scripts workflow) when available. */
    @Test fun readsEveryBlenderExport() {
        val dir = File(System.getProperty("blenderGlbDir") ?: "src/test/resources/blender")
        val files = dir.listFiles { f -> f.extension == "glb" }.orEmpty()
        files.forEach { f ->
            val g = GltfMesh(f.readBytes())
            assertTrue("${f.name} has no triangles", g.indices.size >= 3)
            assertTrue("${f.name} index out of range", g.indices.all { it in 0 until g.positions.size / 3 })
            println("${f.name}: ${g.positions.size / 3} verts, ${g.indices.size / 3} tris")
        }
    }

    private fun glb(json: String, bin: ByteArray): ByteArray {
        var j = json.toByteArray()
        j += ByteArray((4 - j.size % 4) % 4) { ' '.code.toByte() }
        val b = bin + ByteArray((4 - bin.size % 4) % 4)
        val out = ByteBuffer.allocate(12 + 8 + j.size + 8 + b.size).order(ByteOrder.LITTLE_ENDIAN)
        out.putInt(0x46546C67).putInt(2).putInt(out.capacity())
        out.putInt(j.size).putInt(0x4E4F534A).put(j)
        out.putInt(b.size).putInt(0x004E4942).put(b)
        return out.array()
    }
}
