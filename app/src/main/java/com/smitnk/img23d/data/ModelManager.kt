package com.smitnk.img23d.data

import android.content.Context
import com.smitnk.img23d.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

data class ModelFile(val name: String, val bytes: Long, val sha256: String)

sealed interface ModelStatus {
    data object Checking : ModelStatus
    data class Missing(val dir: String) : ModelStatus
    data class Downloading(val file: String, val done: Long, val total: Long) : ModelStatus
    data class Verifying(val file: String) : ModelStatus
    data object Ready : ModelStatus
    data class Failed(val message: String) : ModelStatus
}

/**
 * TripoSR ONNX weights live in app-specific external storage so they can also be
 * pushed with adb: /sdcard/Android/data/com.smitnk.img23d/files/models/
 */
class ModelManager(context: Context) {
    val dir: File = File(context.getExternalFilesDir(null) ?: context.filesDir, "models").apply { mkdirs() }
    private val info = File(dir, "model_info.json")

    fun manifest(): List<ModelFile>? = manifest(info)

    private fun manifest(info: File): List<ModelFile>? {
        if (!info.isFile) return null
        val files = JSONObject(info.readText()).getJSONObject("files")
        return files.keys().asSequence().map {
            val f = files.getJSONObject(it)
            ModelFile(it, f.getLong("bytes"), f.getString("sha256"))
        }.toList()
    }

    /** Size check only; full hashes are verified once after download. */
    fun isReady(): Boolean {
        val m = manifest() ?: return false
        return m.all { File(dir, it.name).length() == it.bytes } && File(dir, VERIFIED).isFile
    }

    /** After a restore from a backup folder (PackFolder verified every sha256). */
    fun markVerified() = File(dir, VERIFIED).writeText("ok")

    fun unmarkVerified() { File(dir, VERIFIED).delete() }

    /** Changed files are downloaded and verified in .staging first; the old models stay usable until the swap. */
    suspend fun download(onStatus: (ModelStatus) -> Unit) = withContext(Dispatchers.IO) {
        val stage = File(dir, ".staging").apply { mkdirs() }
        val stagedInfo = File(stage, "model_info.json")
        fetch("model_info.json", stagedInfo, null) { _, _ -> }
        val files = manifest(stagedInfo) ?: throw IOException("Invalid model_info.json")
        val fresh = mutableListOf<String>()
        for (f in files) {
            require(f.name.matches(Regex("[A-Za-z0-9_.-]+"))) { "Bad file name in model info: ${f.name}" } // trust boundary
            onStatus(ModelStatus.Verifying(f.name))
            val live = File(dir, f.name)
            if (live.length() == f.bytes && sha256(live) == f.sha256) continue
            val staged = File(stage, f.name)
            if (staged.length() != f.bytes || sha256(staged) != f.sha256) {
                fetch(f.name, staged, f.bytes) { done, total -> onStatus(ModelStatus.Downloading(f.name, done, total)) }
                onStatus(ModelStatus.Verifying(f.name))
                if (sha256(staged) != f.sha256) {
                    staged.delete()
                    throw IOException("Checksum mismatch for ${f.name}; deleted, please retry")
                }
            }
            fresh += f.name
        }
        File(dir, VERIFIED).delete() // verified: swap in
        for (name in fresh) {
            val live = File(dir, name)
            live.delete()
            if (!File(stage, name).renameTo(live)) throw IOException("Cannot move $name into place")
        }
        info.delete()
        if (!stagedInfo.renameTo(info)) throw IOException("Cannot move model_info.json into place")
        stage.deleteRecursively()
        File(dir, VERIFIED).writeText("ok")
        onStatus(ModelStatus.Ready)
    }

    private suspend fun fetch(name: String, target: File, expected: Long?, progress: (Long, Long) -> Unit) =
        com.smitnk.img23d.blender.download(BuildConfig.MODEL_BASE_URL + name, target, expected, progress) // retries + resume


    fun delete() {
        dir.listFiles()?.forEach { it.deleteRecursively() }
    }

    companion object {
        private const val VERIFIED = ".verified"

        fun sha256(f: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { s ->
                val buf = ByteArray(1 shl 20)
                while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

/** For files pushed manually with adb: verify hashes and mark ready. */
suspend fun ModelManager.verifyLocal(onStatus: (ModelStatus) -> Unit) = withContext(Dispatchers.IO) {
    val files = manifest() ?: throw IOException("model_info.json not found in ${dir.path}")
    for (f in files) {
        val target = File(dir, f.name)
        if (target.length() != f.bytes) throw IOException("${f.name}: expected ${f.bytes} bytes, found ${target.length()}")
        onStatus(ModelStatus.Verifying(f.name))
        if (ModelManager.sha256(target) != f.sha256) throw IOException("Checksum mismatch for ${f.name}")
    }
    FileOutputStream(File(dir, ".verified")).use { it.write("ok".toByteArray()) }
    onStatus(ModelStatus.Ready)
}
