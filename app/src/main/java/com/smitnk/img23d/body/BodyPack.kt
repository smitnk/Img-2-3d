package com.smitnk.img23d.body

import android.content.Context
import com.smitnk.img23d.BuildConfig
import com.smitnk.img23d.blender.BlenderRuntime
import com.smitnk.img23d.blender.download
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException

sealed interface PackStatus {
    data object NotInstalled : PackStatus
    data class Downloading(val done: Long, val total: Long) : PackStatus
    data object Ready : PackStatus
    data class Failed(val message: String) : PackStatus
}

/**
 * A downloadable asset pack published as a GitHub release: [infoName] lists every file with bytes + sha256, files are
 * verified after download, and a ".ready" copy of the info marks a complete install of at least [requiredVersion].
 */
open class AssetPack(val dir: File, private val baseUrl: String, private val infoName: String, private val requiredVersion: Int,
                     private val mustExist: String) {
    private val info = File(dir, infoName)
    private val ready = File(dir, ".ready")

    init { dir.mkdirs() }

    fun isReady() = ready.isFile && File(dir, mustExist).isFile &&
        runCatching { JSONObject(ready.readText()).optInt("version") >= requiredVersion }.getOrDefault(false)

    /** An older pack is installed. */
    fun needsUpdate() = ready.isFile && !isReady()

    suspend fun install(onStatus: (PackStatus) -> Unit) = withContext(Dispatchers.IO) {
        ready.delete()
        download(baseUrl + infoName, info, null) { _, _ -> }
        val files = JSONObject(info.readText()).getJSONObject("files")
        val total = files.keys().asSequence().sumOf { files.getJSONObject(it).getLong("bytes") }
        var before = 0L
        for (name in files.keys()) {
            val meta = files.getJSONObject(name)
            require(name.matches(Regex("[A-Za-z0-9_.-]+"))) { "Bad file name in pack info: $name" } // trust boundary
            val target = File(dir, name)
            if (target.length() != meta.getLong("bytes") || BlenderRuntime.sha256(target) != meta.getString("sha256")) {
                download(baseUrl + name, target, meta.getLong("bytes")) { d, _ -> onStatus(PackStatus.Downloading(before + d, total)) }
                if (BlenderRuntime.sha256(target) != meta.getString("sha256")) {
                    target.delete()
                    throw IOException("Checksum mismatch for $name; deleted, please retry")
                }
            }
            before += meta.getLong("bytes")
        }
        ready.writeText(info.readText())
        onStatus(PackStatus.Ready)
    }

    fun uninstall() = dir.deleteRecursively().also { dir.mkdirs() }
}

/** Body Creator assets (MakeHuman base mesh, morph targets, skeleton, weights; CC0), release body-assets-v1 (body.yml). */
class BodyPack(context: Context) : AssetPack(File(context.filesDir, "bodypack"), BuildConfig.BODY_BASE_URL, "body_info.json",
    REQUIRED_VERSION, "body.npz") {
    companion object { const val REQUIRED_VERSION = 5 } // v2: hair/clothes/skins, v3: style morphs, v4: chibi/SD + toon eyes, v5: toon head shapes
}

/** On-device texture painter (LCM Dreamshaper v7 ONNX, ~1 GB), release texgen-v1 (texgen.yml). */
class TexPack(context: Context) : AssetPack(File(context.filesDir, "texgen"), BuildConfig.TEXGEN_BASE_URL, "texgen_info.json",
    1, "prompts.bin") {
    val unet: String get() = runCatching { JSONObject(File(dir, "texgen_info.json").readText()).getString("unet") }.getOrDefault("unet_q.onnx")
}

/** Hunyuan3D-2mini Turbo shape generator (ONNX int8/4-bit, Tencent Hunyuan 3D 2.0 Community License), release hunyuan-v1. */
class HunyuanPack(context: Context) : AssetPack(File(context.filesDir, "hunyuan"), BuildConfig.HUNYUAN_BASE_URL, "hunyuan_info.json",
    1, "dit.onnx")

/** One wearable or look from the pack's assets.json (MakeHuman system assets). */
data class BodyAsset(val kind: String, val name: String, val label: String, val thumb: File?)

/** Catalog of the installed pack, by kind: hair, eyebrows, eyelashes, clothes, skins, eyes. Empty for an old pack. */
fun BodyPack.catalog(): Map<String, List<BodyAsset>> {
    val f = File(dir, "assets.json")
    if (!f.isFile) return emptyMap()
    val j = JSONObject(f.readText())
    return j.keys().asSequence().associateWith { kind ->
        val a = j.getJSONArray(kind)
        (0 until a.length()).map { i ->
            val o = a.getJSONObject(i)
            BodyAsset(kind, o.getString("name"), o.optString("label", o.getString("name")),
                o.optString("thumb", "").takeIf { it.isNotBlank() && it != "null" }?.let { File(dir, it) }?.takeIf { it.isFile })
        }
    }
}
