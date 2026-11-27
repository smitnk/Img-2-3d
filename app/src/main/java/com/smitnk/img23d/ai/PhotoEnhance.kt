package com.smitnk.img23d.ai

import android.graphics.Bitmap
import com.smitnk.img23d.body.EsrganPack
import java.io.File

/**
 * "Enhance photo": Real-ESRGAN x4 (realesrgan-x4plus-anime, ncnn) then a high-quality (area average) downscale back to
 * the size the 3D engines read ([ENGINE_SIZE] on the long side), so the engine sees a cleaner, sharper photo.
 */
class PhotoEnhancer(private val pack: EsrganPack, private val threads: Int) {
    var backend: String = ""
        private set

    /** [onProgress] 0..1 for this one image. Throws if the engine cannot run (library missing or ncnn failed). */
    fun enhance(src: Bitmap, onProgress: (Float) -> Unit): Bitmap {
        check(EsrganNative.available) { "Enhance photo engine library is missing in this build" }
        // never feed more than ENGINE_SIZE: the engines can't use more, and x4 of a big photo would not fit in RAM
        val fit = fitLongSide(src.width, src.height, ENGINE_SIZE)
        val small = if (fit.first == src.width && fit.second == src.height) src else Bitmap.createScaledBitmap(src, fit.first, fit.second, true)
        val w = small.width
        val h = small.height
        val px = IntArray(w * h).also { small.getPixels(it, 0, w, 0, 0, w, h) }
        val rgb = ByteArray(w * h * 3)
        var translucent = false
        for (i in px.indices) {
            val p = px[i]
            rgb[i * 3] = (p shr 16).toByte(); rgb[i * 3 + 1] = (p shr 8).toByte(); rgb[i * 3 + 2] = p.toByte()
            if ((p ushr 24) < 255) translucent = true
        }
        val info = arrayOfNulls<String>(1)
        val up = EsrganNative.upscale(File(pack.dir, PARAM).path, File(pack.dir, BIN).path, rgb, w, h, TILE, CPU_TILE, threads, true,
            EsrganNative.Progress { onProgress(it) }, info)
            ?: error("Photo enhance failed: ${info[0] ?: "unknown error"}")
        backend = info[0] ?: ""
        val (ow, oh) = fitLongSide(w * 4, h * 4, ENGINE_SIZE) // x4, then down to what the engine reads (never up)
        val down = resizeArea(up, w * 4, h * 4, ow, oh)
        val out = IntArray(ow * oh)
        // alpha is not part of the model: it is resized on its own (bilinear)
        val alpha = if (translucent) Bitmap.createScaledBitmap(small, ow, oh, true).let { a -> IntArray(ow * oh).also { a.getPixels(it, 0, ow, 0, 0, ow, oh) } } else null
        for (i in out.indices) {
            val a = if (alpha != null) alpha[i] ushr 24 else 255
            out[i] = (a shl 24) or ((down[i * 3].toInt() and 255) shl 16) or ((down[i * 3 + 1].toInt() and 255) shl 8) or (down[i * 3 + 2].toInt() and 255)
        }
        return Bitmap.createBitmap(out, ow, oh, Bitmap.Config.ARGB_8888)
    }

    companion object {
        const val ENGINE_SIZE = 512 // TripoSR.IMAGE_SIZE; the Hunyuan engines resize their crop the same way
        const val PARAM = "realesrgan-x4plus-anime.param"
        const val BIN = "realesrgan-x4plus-anime.bin"
        const val TILE = 200 // as Real-ESRGAN-ncnn-vulkan on a big GPU
        const val CPU_TILE = 100

        /** (w, h) scaled so the long side is at most [max] (never up). */
        fun fitLongSide(w: Int, h: Int, max: Int): Pair<Int, Int> {
            val m = maxOf(w, h)
            if (m <= max) return w to h
            return maxOf(1, Math.round(w.toLong() * max / m.toDouble()).toInt()) to maxOf(1, Math.round(h.toLong() * max / m.toDouble()).toInt())
        }

        /** Area-average downscale of interleaved RGB bytes (exact box filter for any ratio). */
        fun resizeArea(src: ByteArray, w: Int, h: Int, ow: Int, oh: Int): ByteArray {
            if (w == ow && h == oh) return src
            require(ow in 1..w && oh in 1..h) { "area resize only shrinks" }
            // horizontal pass into floats, then vertical
            val tmp = FloatArray(ow * h * 3)
            val sx = w.toDouble() / ow
            for (x in 0 until ow) {
                val a = x * sx
                val b = (x + 1) * sx
                val i0 = a.toInt()
                val i1 = minOf(w - 1, Math.ceil(b).toInt() - 1)
                for (y in 0 until h) for (c in 0 until 3) {
                    var acc = 0.0
                    for (i in i0..i1) acc += (src[(y * w + i) * 3 + c].toInt() and 255) * (minOf(b, i + 1.0) - maxOf(a, i.toDouble()))
                    tmp[(y * ow + x) * 3 + c] = (acc / sx).toFloat()
                }
            }
            val out = ByteArray(ow * oh * 3)
            val sy = h.toDouble() / oh
            for (y in 0 until oh) {
                val a = y * sy
                val b = (y + 1) * sy
                val j0 = a.toInt()
                val j1 = minOf(h - 1, Math.ceil(b).toInt() - 1)
                for (x in 0 until ow) for (c in 0 until 3) {
                    var acc = 0.0
                    for (j in j0..j1) acc += tmp[(j * ow + x) * 3 + c] * (minOf(b, j + 1.0) - maxOf(a, j.toDouble()))
                    out[(y * ow + x) * 3 + c] = Math.round(acc / sy).toInt().coerceIn(0, 255).toByte()
                }
            }
            return out
        }
    }
}
