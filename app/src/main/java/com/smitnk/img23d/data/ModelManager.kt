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

    fun manifest(): List<ModelFile>? {
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

    suspend fun download(onStatus: (ModelStatus) -> Unit) = withContext(Dispatchers.IO) {
        File(dir, VERIFIED).delete()
        fetch("model_info.json", info, null) { _, _ -> }
        val files = manifest() ?: throw IOException("Invalid model_info.json")
        for (f in files) {
            val target = File(dir, f.name)
            if (target.length() != f.bytes) {
                fetch(f.name, target, f.bytes) { done, total -> onStatus(ModelStatus.Downloading(f.name, done, total)) }
            }
            onStatus(ModelStatus.Verifying(f.name))
            if (sha256(target) != f.sha256) {
                target.delete()
                throw IOException("Checksum mismatch for ${f.name}; deleted, please retry")
            }
        }
        File(dir, VERIFIED).writeText("ok")
        onStatus(ModelStatus.Ready)
    }

    private suspend fun fetch(name: String, target: File, expected: Long?, progress: (Long, Long) -> Unit) {
        val part = File(target.path + ".part")
        val start = if (expected != null) part.length() else 0L
        val conn = (URL(BuildConfig.MODEL_BASE_URL + name).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = 20_000
            readTimeout = 60_000
            if (start > 0) setRequestProperty("Range", "bytes=$start-")
        }
        try {
            val code = conn.responseCode
            if (code !in 200..299) throw IOException("HTTP $code downloading $name")
            val resumed = code == 206
            val total = expected ?: conn.contentLengthLong
            RandomAccessFile(part, "rw").use { raf ->
                if (!resumed) raf.setLength(0)
                raf.seek(if (resumed) start else 0)
                var done = if (resumed) start else 0L
                val buf = ByteArray(1 shl 16)
                conn.inputStream.use { input ->
                    var last = 0L
                    while (true) {
                        coroutineContext.ensureActive()
                        val n = input.read(buf)
                        if (n < 0) break
                        raf.write(buf, 0, n)
                        done += n
                        if (done - last > (4 shl 20)) { progress(done, total); last = done }
                    }
                }
                progress(done, total)
            }
            if (expected != null && part.length() != expected) throw IOException("Incomplete download of $name")
            target.delete()
            if (!part.renameTo(target)) throw IOException("Cannot move $name into place")
        } finally {
            conn.disconnect()
        }
    }

    fun delete() {
        dir.listFiles()?.forEach { it.delete() }
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
