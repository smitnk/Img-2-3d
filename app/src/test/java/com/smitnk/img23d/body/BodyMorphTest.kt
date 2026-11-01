package com.smitnk.img23d.body

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class BodyMorphTest {
    private fun npy(descr: String, shape: String, data: ByteArray): ByteArray {
        var header = "{'descr': '$descr', 'fortran_order': False, 'shape': $shape, }"
        while ((10 + header.length + 1) % 64 != 0) header += " "
        header += "\n"
        val b = ByteBuffer.allocate(10 + header.length + data.size).order(ByteOrder.LITTLE_ENDIAN)
        b.put(0x93.toByte()).put("NUMPY".toByteArray()).put(1).put(0).putShort(header.length.toShort()).put(header.toByteArray()).put(data)
        return b.array()
    }

    private fun le(n: Int, f: (ByteBuffer) -> Unit) = ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN).also(f).array()

    /** Two vertices; a macro target (male, full weight on vertex 0) and a slider target (vertex 1). */
    private fun pack(): File {
        val dir = Files.createTempDirectory("pack").toFile()
        ZipOutputStream(File(dir, "body.npz").outputStream()).use { z ->
            fun put(name: String, bytes: ByteArray) { z.putNextEntry(ZipEntry("$name.npy")); z.write(bytes); z.closeEntry() }
            put("verts", npy("<f4", "(4, 3)", le(48) { b -> repeat(12) { b.putFloat(0f) } }))
            put("f_body", npy("<i4", "(1, 4)", le(16) { b -> b.putInt(0).putInt(1).putInt(2).putInt(3) }))
            put("t_idx", npy("<i4", "(2,)", le(8) { b -> b.putInt(0).putInt(1) }))
            // deltas: target0 -> (1, 0, 0) on vertex 0; target1 -> (0, -2, 0) on vertex 1 (float16 0x3C00 = 1, 0xC000 = -2)
            put("t_delta", npy("<f2", "(2, 3)", le(12) { b -> b.putShort(0x3C00).putShort(0).putShort(0).putShort(0).putShort(0xC000.toShort()).putShort(0) }))
            put("t_off", npy("<i8", "(3,)", le(24) { b -> b.putLong(0).putLong(1).putLong(2) }))
        }
        File(dir, "targets.json").writeText("""[{"key": "macrodetails", "vars": ["male"]}, {"key": "nose-nose-width-incr", "vars": []}]""")
        File(dir, "modifiers.json").writeText("""[{"group": "nose", "modifiers": [{"target": "nose-width", "min": "decr", "max": "incr"}]}]""")
        return dir
    }

    @Test fun morphMatchesMakeHumanWeights() {
        val m = BodyMorph(pack())
        assertEquals(listOf(0, 1, 2, 0, 2, 3), m.triangles.toList())
        val v = m.morph(mapOf("gender" to 0.75f), mapOf("nose/nose-width-decr|incr" to 0.5f))
        assertEquals(0.75f, v[0], 1e-4f)   // macro target x male value
        assertEquals(-1.0f, v[4], 1e-4f)   // slider 0.5 -> right target at 0.5 x (-2)
        val left = m.morph(emptyMap(), mapOf("nose/nose-width-decr|incr" to -1f))
        assertEquals(0f, left[4], 1e-6f)   // negative value drives the (absent) decr target only
    }

    @Test fun macroValuesFollowMakeHuman() {
        val f = BodyMorph.macroValues(mapOf("age" to 0.25f, "weight" to 1f))
        assertEquals(0.2, f.getValue("young"), 1e-3)
        assertEquals(0.8, f.getValue("child"), 1e-3)
        assertEquals(1.0, f.getValue("maxweight"), 1e-9)
        assertEquals(0.0, f.getValue("averageweight"), 1e-9)
        assertEquals(1.0, f.getValue("african") + f.getValue("asian") + f.getValue("caucasian"), 1e-9)
    }
}
