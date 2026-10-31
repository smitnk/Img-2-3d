package com.smitnk.forge3d.engine

import android.content.Context
import org.json.JSONObject
import java.io.File

data class Creation(
    val id: String,
    val dir: File,
    val title: String,
    val generator: String,
    val createdAt: Long,
    val triangles: Int,
    val bones: Int,
    val seconds: Double,
    val peakRssMb: Int,
    val blender: String,
) {
    val glb get() = File(dir, "model.glb")
    val blend get() = File(dir, "model.blend").takeIf { it.isFile }
}

class Library(context: Context) {
    private val root = File(context.filesDir, "library").apply { mkdirs() }

    fun list(): List<Creation> = root.listFiles()?.mapNotNull(::load)?.sortedByDescending { it.createdAt } ?: emptyList()

    fun get(id: String) = load(File(root, id))

    fun add(req: GenRequest, res: GenResult, peakRssMb: Int): Creation {
        val now = System.currentTimeMillis()
        val dir = File(root, now.toString()).apply { mkdirs() }
        res.glb.copyTo(File(dir, "model.glb"))
        res.blend?.copyTo(File(dir, "model.blend"))
        val i = res.info
        val meta = JSONObject()
            .put("title", i.optString("name", req.generator.title))
            .put("generator", req.generator.name)
            .put("createdAt", now)
            .put("triangles", i.optInt("triangles"))
            .put("bones", i.optJSONObject("rig")?.optInt("bones") ?: 0)
            .put("seconds", res.seconds)
            .put("peakRssMb", peakRssMb)
            .put("blender", i.optString("blender", "4.2.3"))
            .put("generation", i)
        File(dir, "meta.json").writeText(meta.toString(2))
        return load(dir)!!
    }

    fun delete(c: Creation) = c.dir.deleteRecursively()

    private fun load(dir: File): Creation? = runCatching {
        val j = JSONObject(File(dir, "meta.json").readText())
        Creation(dir.name, dir, j.getString("title"), j.getString("generator"), j.getLong("createdAt"),
            j.optInt("triangles"), j.optInt("bones"), j.optDouble("seconds"), j.optInt("peakRssMb"), j.optString("blender"))
    }.getOrNull()
}
