package com.smitnk.img23d.data

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.File

data class Project(
    val id: String,
    val dir: File,
    val createdAt: Long,
    val vertices: Int,
    val faces: Int,
    val resolution: Int,
    val timingsMs: Map<String, Long>,
    val peakPssMb: Int,
) {
    val glb get() = File(dir, "model.glb")
    val thumbnail get() = File(dir, "preprocessed.png")
    val totalMs get() = timingsMs.values.sum()
}

class ProjectStore(context: Context) {
    private val root = File(context.filesDir, "projects").apply { mkdirs() }

    fun list(): List<Project> =
        root.listFiles()?.mapNotNull { load(it) }?.sortedByDescending { it.createdAt } ?: emptyList()

    fun get(id: String): Project? = load(File(root, id))

    fun create(
        glb: (File) -> Unit,
        preview: Bitmap,
        vertices: Int,
        faces: Int,
        resolution: Int,
        timings: Map<String, Long>,
        peakPssMb: Int,
    ): Project {
        val now = System.currentTimeMillis()
        val dir = File(root, now.toString()).apply { mkdirs() }
        glb(File(dir, "model.glb"))
        File(dir, "preprocessed.png").outputStream().use { preview.compress(Bitmap.CompressFormat.PNG, 100, it) }
        val meta = JSONObject()
            .put("createdAt", now).put("vertices", vertices).put("faces", faces)
            .put("resolution", resolution).put("peakPssMb", peakPssMb)
            .put("timingsMs", JSONObject(timings as Map<*, *>))
        File(dir, "meta.json").writeText(meta.toString(2))
        return load(dir)!!
    }

    fun delete(p: Project) = p.dir.deleteRecursively()

    private fun load(dir: File): Project? = runCatching {
        val j = JSONObject(File(dir, "meta.json").readText())
        val t = j.getJSONObject("timingsMs")
        Project(
            id = dir.name, dir = dir, createdAt = j.getLong("createdAt"),
            vertices = j.getInt("vertices"), faces = j.getInt("faces"), resolution = j.getInt("resolution"),
            timingsMs = t.keys().asSequence().associateWith { t.getLong(it) },
            peakPssMb = j.getInt("peakPssMb"),
        )
    }.getOrNull()
}
