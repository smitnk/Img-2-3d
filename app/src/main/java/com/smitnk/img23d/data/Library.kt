package com.smitnk.img23d.data

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.File

/** Where a model came from. PHOTO is System B (TripoSR); everything else ran through Blender (System A). */
enum class Source(val label: String) {
    PHOTO("Photo → 3D"), CHARACTER("Photo → Character"), REFINED("Refined"), BUILDER("Builder"),
    SCRIPT("Python script"), TEXT("Text to 3D"), PRESET("Preset"), BODY("Body Creator"),
}

data class Model(
    val id: String,
    val dir: File,
    val name: String,
    val source: Source,
    val createdAt: Long,
    val vertices: Int,
    val faces: Int,
    val bones: Int,
    val resolution: Int,
    val timingsMs: Map<String, Long>,
    val peakMemMb: Int,
    val detail: String,
    val quality: Quality? = null,
    /** Extra metadata saved with the model (Blender info; "repair" = Repair Mesh before/after stats). */
    val extra: JSONObject? = null,
) {
    val glb get() = File(dir, "model.glb")
    val blend get() = File(dir, "model.blend").takeIf { it.isFile }
    /** Auto Split: one closed GLB per part, zipped. */
    val partsZip get() = File(dir, "parts.zip").takeIf { it.isFile }
    val thumbnail get() = File(dir, "preprocessed.png").takeIf { it.isFile }
    val script get() = File(dir, "script.py").takeIf { it.isFile }
    /** The cut-out source photo (TripoSR models and what was made from them): the front view for Paint texture. */
    val photo get() = File(dir, "photo.png").takeIf { it.isFile }
    /** Every saved source view as (label, file), front first, then left / back / right: what Hunyuan Paint gets as its references. */
    val photos: List<Pair<String, File>> get() = PHOTO_FILES.mapNotNull { (label, name) -> File(dir, name).takeIf { it.isFile }?.let { label to it } }
    val totalMs get() = timingsMs.values.sum()
    /** Before / after stats when this model came out of Repair Mesh. */
    val repair get() = extra?.optJSONObject("repair")

    companion object {
        /** View label -> file name in the model folder. Hunyuan3D-2mv view slots: 0 front (photo.png), 1 left, 2 back, 3 right. */
        val PHOTO_FILES = listOf("front" to "photo.png", "left" to "photo_left.png", "back" to "photo_back.png", "right" to "photo_right.png")
        val VIEW_LABELS = listOf("front", "left", "back", "right")
    }

    /** Copies every saved source view of this model into [to] (a model made from it keeps its photos). */
    fun copyPhotosTo(to: Model) = photos.forEach { (label, f) -> f.copyTo(File(to.dir, PHOTO_FILES.first { it.first == label }.second), overwrite = true) }
}

/** bx_audit.deliverable result for a Blender model (game profile). */
data class Quality(val pass: Boolean, val errors: List<String>, val warnings: List<String>) {
    companion object {
        fun from(j: JSONObject?): Quality? = j?.let {
            fun list(k: String) = it.optJSONArray(k)?.let { a -> (0 until a.length()).map(a::getString) }.orEmpty()
            Quality(it.optBoolean("pass"), list("errors"), list("warnings"))
        }
    }
}

/** One shared model library for both engines. Reads System B's v1 project metadata unchanged. */
class Library(context: Context) {
    private val root = File(context.filesDir, "projects").apply { mkdirs() }

    fun list(): List<Model> = root.listFiles()?.mapNotNull(::load)?.sortedByDescending { it.createdAt } ?: emptyList()

    fun get(id: String) = load(File(root, id))

    fun add(
        name: String,
        source: Source,
        writeGlb: (File) -> Unit,
        vertices: Int,
        faces: Int,
        bones: Int = 0,
        resolution: Int = 0,
        timings: Map<String, Long>,
        peakMemMb: Int,
        detail: String = "",
        thumbnail: Bitmap? = null,
        thumbnailFrom: File? = null,
        blend: File? = null,
        script: String? = null,
        extra: JSONObject? = null,
        quality: JSONObject? = null,
    ): Model {
        val now = System.currentTimeMillis()
        val dir = File(root, now.toString()).apply { mkdirs() }
        writeGlb(File(dir, "model.glb"))
        val thumb = File(dir, "preprocessed.png")
        thumbnail?.let { b -> thumb.outputStream().use { b.compress(Bitmap.CompressFormat.PNG, 100, it) } }
        thumbnailFrom?.takeIf { it.isFile && thumbnail == null }?.copyTo(thumb)
        blend?.copyTo(File(dir, "model.blend"))
        script?.let { File(dir, "script.py").writeText(it) }
        val meta = JSONObject()
            .put("name", name).put("source", source.name).put("createdAt", now)
            .put("vertices", vertices).put("faces", faces).put("bones", bones).put("resolution", resolution)
            .put("peakPssMb", peakMemMb).put("detail", detail)
            .put("timingsMs", JSONObject(timings as Map<*, *>))
        extra?.let { meta.put("extra", it) }
        quality?.let { meta.put("quality", it) }
        File(dir, "meta.json").writeText(meta.toString(2))
        return load(dir)!!
    }

    fun rename(m: Model, name: String) {
        val f = File(m.dir, "meta.json")
        f.writeText(JSONObject(f.readText()).put("name", name.trim()).toString(2))
    }

    fun delete(m: Model) = m.dir.deleteRecursively()

    fun clear() = root.listFiles()?.forEach { it.deleteRecursively() }

    fun bytesUsed(): Long = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    private fun load(dir: File): Model? = runCatching {
        val j = JSONObject(File(dir, "meta.json").readText())
        val t = j.optJSONObject("timingsMs") ?: JSONObject()
        val source = runCatching { Source.valueOf(j.optString("source", "PHOTO")) }.getOrDefault(Source.PHOTO)
        Model(
            id = dir.name, dir = dir,
            name = j.optString("name", "").ifBlank { source.label },
            source = source, createdAt = j.getLong("createdAt"),
            vertices = j.optInt("vertices"), faces = j.optInt("faces"), bones = j.optInt("bones"),
            resolution = j.optInt("resolution"),
            timingsMs = t.keys().asSequence().associateWith { t.getLong(it) },
            peakMemMb = j.optInt("peakPssMb"), detail = j.optString("detail"),
            quality = Quality.from(j.optJSONObject("quality")),
            extra = j.optJSONObject("extra"),
        )
    }.getOrNull()
}
