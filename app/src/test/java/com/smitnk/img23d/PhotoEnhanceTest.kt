package com.smitnk.img23d

import com.smitnk.img23d.ai.PhotoEnhancer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class PhotoEnhanceTest {
    @Test fun fitLongSideOnlyShrinks() {
        assertEquals(128 to 64, PhotoEnhancer.fitLongSide(128, 64, 512))
        assertEquals(512 to 256, PhotoEnhancer.fitLongSide(2048, 1024, 512))
        assertEquals(256 to 512, PhotoEnhancer.fitLongSide(1500, 3000, 512))
        assertEquals(512 to 1, PhotoEnhancer.fitLongSide(5000, 1, 512))
    }

    @Test fun areaResizeAveragesTheBlocks() {
        // 4x2 image, two colours per row: halving each side averages 2x2 blocks
        val w = 4; val h = 2
        val src = ByteArray(w * h * 3)
        fun set(x: Int, y: Int, v: Int) { for (c in 0..2) src[(y * w + x) * 3 + c] = v.toByte() }
        set(0, 0, 0); set(1, 0, 100); set(0, 1, 100); set(1, 1, 200)
        set(2, 0, 255); set(3, 0, 255); set(2, 1, 255); set(3, 1, 255)
        val out = PhotoEnhancer.resizeArea(src, w, h, 2, 1)
        assertArrayEquals(byteArrayOf(100, 100, 100, -1, -1, -1), out)
    }

    @Test fun areaResizeKeepsAFlatImageFlat() {
        val src = ByteArray(7 * 5 * 3) { 77 }
        val out = PhotoEnhancer.resizeArea(src, 7, 5, 3, 2) // non-integer ratio
        assertEquals(3 * 2 * 3, out.size)
        out.forEach { assertEquals(77, it.toInt()) }
    }
}
