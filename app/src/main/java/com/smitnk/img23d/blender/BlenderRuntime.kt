package com.smitnk.img23d.blender

import android.content.Context
import android.system.Os
import com.smitnk.img23d.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.File
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import kotlin.coroutines.coroutineContext

sealed interface RuntimeStatus {
    data object NotInstalled : RuntimeStatus
    data class Downloading(val file: String, val done: Long, val total: Long) : RuntimeStatus
    data class Verifying(val file: String) : RuntimeStatus
    data class Extracting(val done: Long, val total: Long) : RuntimeStatus
    data object Ready : RuntimeStatus
    data class Failed(val message: String) : RuntimeStatus
}

/**
 * Headless Blender 4.2.3 (linux-aarch64) inside a minimal Ubuntu userland, executed through Termux's proot.
 * Built and tested by .github/workflows/blender-runtime.yml, published as release blender-runtime-v1.
 */
class BlenderRuntime(context: Context) {
    val dir = File(context.filesDir, "runtime").apply { mkdirs() }
    val rootfs = File(dir, "rootfs")
    val bin = File(dir, "bin")
    private val downloads = File(context.cacheDir, "runtime-dl").apply { mkdirs() }
    private val info = File(dir, "runtime_info.json")
    private val ready = File(dir, ".ready")

    fun isReady() = ready.isFile && File(rootfs, "opt/blender/blender").exists() && File(bin, "proot").canExecute() &&
        runCatching { JSONObject(ready.readText()).optInt("version") >= REQUIRED_VERSION }.getOrDefault(false)

    /** True when an older runtime is installed and must be updated (e.g. v1 lacked libandroid-shmem). */
    fun needsUpdate() = ready.isFile && !isReady()

    suspend fun install(onStatus: (RuntimeStatus) -> Unit) = withContext(Dispatchers.IO) {
        ready.delete()
        fetch("runtime_info.json", info, null) { _, _ -> }
        val files = JSONObject(info.readText()).getJSONObject("files")
        for (name in files.keys()) {
            val meta = files.getJSONObject(name)
            val bytes = meta.getLong("bytes")
            val target = if (name.endsWith(".tar.gz")) File(downloads, name) else File(bin.apply { mkdirs() }, name)
            if (target.length() != bytes || sha256(target) != meta.getString("sha256")) {
                fetch(name, target, bytes) { d, t -> onStatus(RuntimeStatus.Downloading(name, d, t)) }
                onStatus(RuntimeStatus.Verifying(name))
                if (sha256(target) != meta.getString("sha256")) {
                    target.delete()
                    throw IOException("Checksum mismatch for $name; deleted, please retry")
                }
            }
        }
        for (f in bin.listFiles().orEmpty()) Os.chmod(f.path, "755".toInt(8))
        val tarball = File(downloads, "blender-runtime-arm64.tar.gz")
        rootfs.deleteRecursively()
        extract(tarball, rootfs) { d, t -> onStatus(RuntimeStatus.Extracting(d, t)) }
        tarball.delete() // ~200 MB that is no longer needed once unpacked
        ready.writeText(info.readText())
        onStatus(RuntimeStatus.Ready)
    }

    fun uninstall() {
        dir.deleteRecursively()
        downloads.deleteRecursively()
    }

    private suspend fun extract(tarball: File, dest: File, progress: (Long, Long) -> Unit) {
        dest.mkdirs()
        val total = tarball.length()
        val counter = CountingStream(BufferedInputStream(tarball.inputStream(), 1 shl 16))
        TarArchiveInputStream(GZIPInputStream(counter, 1 shl 16)).use { tar ->
            var last = 0L
            while (true) {
                coroutineContext.ensureActive()
                val e = tar.nextEntry ?: break
                // Reject entries escaping the rootfs (trust boundary: downloaded archive). Checked on the name, not
                // canonicalPath, because absolute guest symlinks would otherwise resolve against the Android root.
                if (e.name.split('/').any { it == ".." } || e.name.startsWith("/")) throw IOException("Unsafe path in runtime archive: ${e.name}")
                val out = File(dest, e.name)
                when {
                    e.isDirectory -> { out.mkdirs(); Os.chmod(out.path, (e.mode and "7777".toInt(8)) or "700".toInt(8)) }
                    e.isSymbolicLink -> {
                        out.parentFile?.mkdirs()
                        out.delete()
                        // Absolute targets are resolved by proot against the guest root, so keep them verbatim.
                        Os.symlink(e.linkName, out.path)
                    }
                    e.isLink -> {
                        out.parentFile?.mkdirs()
                        File(dest, e.linkName).copyTo(out, overwrite = true)
                        Os.chmod(out.path, e.mode and "7777".toInt(8))
                    }
                    e.isFile -> {
                        out.parentFile?.mkdirs()
                        out.outputStream().buffered(1 shl 16).use { tar.copyTo(it) }
                        Os.chmod(out.path, (e.mode and "7777".toInt(8)) or "600".toInt(8))
                    }
                    else -> Unit // device nodes/fifos: proot binds the host /dev instead
                }
                if (counter.count - last > (2 shl 20)) { progress(counter.count, total); last = counter.count }
            }
        }
        progress(total, total)
    }

    private suspend fun fetch(name: String, target: File, expected: Long?, progress: (Long, Long) -> Unit) =
        download(BuildConfig.RUNTIME_BASE_URL + name, target, expected, progress)

    private class CountingStream(input: InputStream) : FilterInputStream(input) {
        var count = 0L
        override fun read(): Int = super.read().also { if (it >= 0) count++ }
        override fun read(b: ByteArray, off: Int, len: Int): Int = super.read(b, off, len).also { if (it > 0) count += it }
    }

    companion object {
        /** v2: proot's full shared-library closure. v3: Mesa software EGL (scripts that render / use gpu). */
        const val REQUIRED_VERSION = 3

        fun sha256(f: File): String {
            if (!f.isFile) return ""
            val md = MessageDigest.getInstance("SHA-256")
            f.inputStream().use { s ->
                val buf = ByteArray(1 shl 20)
                while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

/** Resumable HTTP download into [target] (via target.part); [expected] bytes when known. */
internal suspend fun download(url: String, target: File, expected: Long?, progress: (Long, Long) -> Unit) {
    val name = target.name
    val part = File(target.path + ".part")
    val start = if (expected != null) part.length() else 0L
    val conn = (URL(url).openConnection() as HttpURLConnection).apply {
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
            var last = 0L
            conn.inputStream.use { input ->
                while (true) {
                    coroutineContext.ensureActive()
                    val n = input.read(buf)
                    if (n < 0) break
                    raf.write(buf, 0, n)
                    done += n
                    if (done - last > (2 shl 20)) { progress(done, total); last = done }
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
