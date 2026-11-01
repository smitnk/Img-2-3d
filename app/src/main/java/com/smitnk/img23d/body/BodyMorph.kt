package com.smitnk.img23d.body

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipFile
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * The phone-side twin of assets/blender/body/humanmorph.py: same target weights, so the live preview matches
 * what Blender builds. Base mesh + sum(weight * delta) over the active targets of body.npz.
 */
class BodyMorph(folder: File) {
    val base: FloatArray            // (N*3) decimetres, glTF axes (Y up, faces +Z)
    val triangles: IntArray          // body quads split into triangles
    private val tIdx: IntArray
    private val tDelta: ShortArray   // float16 bits, (M*3)
    private val tOff: LongArray
    private val keys: Array<String>
    private val vars: Array<Array<String>>
    val modifiers: JSONArray

    init {
        ZipFile(File(folder, "body.npz")).use { z ->
            fun arr(name: String) = z.getInputStream(z.getEntry("$name.npy")).use(::readNpy)
            base = arr("verts").floats()
            val quads = arr("f_body").ints()
            val tri = ArrayList<Int>(quads.size * 3 / 2)
            for (q in 0 until quads.size / 4) {
                val a = quads[q * 4]; val b = quads[q * 4 + 1]; val c = quads[q * 4 + 2]; val d = quads[q * 4 + 3]
                tri += listOf(a, b, c)
                if (d != c) tri += listOf(a, c, d)
            }
            triangles = tri.toIntArray()
            tIdx = arr("t_idx").ints()
            tDelta = arr("t_delta").shorts()
            tOff = arr("t_off").longs()
        }
        val index = JSONArray(File(folder, "targets.json").readText())
        keys = Array(index.length()) { index.getJSONObject(it).getString("key") }
        vars = Array(index.length()) { i -> index.getJSONObject(i).getJSONArray("vars").let { a -> Array(a.length()) { a.getString(it) } } }
        modifiers = JSONArray(File(folder, "modifiers.json").readText())
        require(tOff.size == keys.size + 1) { "targets.json does not match body.npz" }
    }

    /** Morphed vertex positions for [macros] (0..1 each) and [sliders] ("group/target-l|r" -> -1..1). */
    fun morph(macros: Map<String, Float>, sliders: Map<String, Float>): FloatArray {
        val mv = macroValues(macros)
        val sf = sliderFactors(sliders, modifiers)
        val v = base.copyOf()
        for (t in keys.indices) {
            val g = if (keys[t] in MACRO_GROUPS) 1.0 else sf[keys[t]] ?: 0.0
            if (g == 0.0) continue
            var w = g
            for (x in vars[t]) w *= mv[x] ?: 0.0
            if (abs(w) < 1e-6) continue
            val wf = w.toFloat()
            for (k in tOff[t].toInt() until tOff[t + 1].toInt()) {
                val o = tIdx[k] * 3
                v[o] += wf * HALF[tDelta[k * 3].toInt() and 0xFFFF]
                v[o + 1] += wf * HALF[tDelta[k * 3 + 1].toInt() and 0xFFFF]
                v[o + 2] += wf * HALF[tDelta[k * 3 + 2].toInt() and 0xFFFF]
            }
        }
        return v
    }

    companion object {
        val MACRO_GROUPS = setOf("macrodetails", "macrodetails-universal", "macrodetails-height", "macrodetails-proportions", "breast")
        val DEFAULT_MACROS = mapOf("gender" to 0.5f, "age" to 0.5f, "muscle" to 0.5f, "weight" to 0.5f, "height" to 0.5f,
            "proportions" to 0.5f, "african" to 1 / 3f, "asian" to 1 / 3f, "caucasian" to 1 / 3f, "cup" to 0.5f, "firmness" to 0.5f)

        /** float16 bits -> float, for every bit pattern (decoded once; 256 KB). */
        private val HALF = FloatArray(65536) { h ->
            val s = if (h and 0x8000 != 0) -1f else 1f
            val e = (h shr 10) and 0x1F
            val m = h and 0x3FF
            when (e) {
                0 -> s * m * 5.9604645e-8f // subnormal: m * 2^-24
                31 -> if (m == 0) s * Float.POSITIVE_INFINITY else Float.NaN
                else -> s * java.lang.Float.intBitsToFloat(((e - 15 + 127) shl 23) or (m shl 13))
            }
        }

        private fun tri(v: Double, lo: String, mid: String, hi: String): Map<String, Double> {
            val mx = max(0.0, v * 2 - 1); val mn = max(0.0, 1 - v * 2)
            return mapOf(hi to mx, lo to mn, mid to 1 - max(mx, mn))
        }

        /** MakeHuman's macro variables (human.py _set*Vals), identical to humanmorph.macro_values. */
        fun macroValues(input: Map<String, Float>): Map<String, Double> {
            val m = DEFAULT_MACROS + input
            fun c(k: String) = min(1.0, max(0.0, m.getValue(k).toDouble()))
            val f = HashMap<String, Double>()
            f["male"] = c("gender"); f["female"] = 1 - c("gender")
            val a = c("age")
            if (a < 0.5) {
                val young = max(0.0, (a - 0.1875) * 3.2)
                f["old"] = 0.0; f["baby"] = max(0.0, 1 - a * 5.333); f["young"] = young
                f["child"] = max(0.0, min(1.0, 5.333 * a) - young)
            } else {
                f["child"] = 0.0; f["baby"] = 0.0; f["old"] = max(0.0, a * 2 - 1); f["young"] = 1 - max(0.0, a * 2 - 1)
            }
            val race = doubleArrayOf(c("african"), c("asian"), c("caucasian"))
            val sum = race.sum()
            listOf("african", "asian", "caucasian").forEachIndexed { i, r -> f[r] = if (sum > 0) race[i] / sum else 1 / 3.0 }
            f += tri(c("muscle"), "minmuscle", "averagemuscle", "maxmuscle")
            f += tri(c("weight"), "minweight", "averageweight", "maxweight")
            f += tri(c("height"), "minheight", "averageheight", "maxheight")
            f += tri(c("cup"), "mincup", "averagecup", "maxcup")
            f += tri(c("firmness"), "minfirmness", "averagefirmness", "maxfirmness")
            f += tri(c("proportions"), "uncommonproportions", "regularproportions", "idealproportions")
            return f
        }

        /** Slider id ("group/target-min|max" or "group/target") of every MakeHuman modifier, with its target keys. */
        fun sliderTargets(modifiers: JSONArray): Map<String, Pair<String?, String>> {
            val out = LinkedHashMap<String, Pair<String?, String>>()
            for (i in 0 until modifiers.length()) {
                val g = modifiers.getJSONObject(i)
                val group = g.getString("group")
                val mods = g.getJSONArray("modifiers")
                for (k in 0 until mods.length()) {
                    val m: JSONObject = mods.getJSONObject(k)
                    if (!m.has("target")) continue
                    val base = "$group-${m.getString("target")}"
                    if (m.has("min")) out["$group/${m.getString("target")}-${m.getString("min")}|${m.getString("max")}"] =
                        "$base-${m.getString("min")}" to "$base-${m.getString("max")}"
                    else out["$group/${m.getString("target")}"] = null to base
                }
            }
            return out
        }

        fun sliderFactors(sliders: Map<String, Float>, modifiers: JSONArray): Map<String, Double> {
            val known = sliderTargets(modifiers)
            val out = HashMap<String, Double>()
            for ((name, value) in sliders) {
                val (left, right) = known[name] ?: continue
                val v = value.toDouble().coerceIn(if (left != null) -1.0 else 0.0, 1.0)
                if (left != null) out[left] = (out[left] ?: 0.0) + max(0.0, -v)
                out[right] = (out[right] ?: 0.0) + max(0.0, v)
            }
            return out
        }

        /** Minimal .npy reader (v1-v3, little-endian, C order): what numpy.savez writes. */
        internal fun readNpy(input: InputStream): Npy {
            val bytes = input.readBytes()
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(bytes.size > 10 && bytes[0] == 0x93.toByte() && String(bytes, 1, 5) == "NUMPY") { "Not an .npy array" }
            val major = bytes[6].toInt()
            val (hlen, start) = if (major == 1) (b.getShort(8).toInt() and 0xFFFF) to 10 else b.getInt(8) to 12
            val header = String(bytes, start, hlen, Charsets.ISO_8859_1)
            val descr = Regex("'descr':\\s*'([^']+)'").find(header)!!.groupValues[1]
            require(!header.contains("'fortran_order': True")) { "Fortran-order arrays are not supported" }
            require(descr.startsWith("<") || descr.startsWith("|")) { "Big-endian .npy not supported: $descr" }
            return Npy(descr.drop(1), ByteBuffer.wrap(bytes, start + hlen, bytes.size - start - hlen).slice().order(ByteOrder.LITTLE_ENDIAN))
        }
    }

    internal class Npy(val type: String, val data: ByteBuffer) {
        fun floats(): FloatArray { require(type == "f4") { type }; return FloatArray(data.remaining() / 4).also { data.asFloatBuffer().get(it) } }
        fun ints(): IntArray { require(type == "i4") { type }; return IntArray(data.remaining() / 4).also { data.asIntBuffer().get(it) } }
        fun shorts(): ShortArray { require(type == "f2") { type }; return ShortArray(data.remaining() / 2).also { data.asShortBuffer().get(it) } }
        fun longs(): LongArray { require(type == "i8") { type }; return LongArray(data.remaining() / 8).also { data.asLongBuffer().get(it) } }
    }
}
