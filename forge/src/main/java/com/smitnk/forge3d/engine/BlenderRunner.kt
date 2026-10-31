package com.smitnk.forge3d.engine

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.coroutineContext

class GenResult(val glb: File, val blend: File?, val info: JSONObject, val seconds: Double, val log: List<String>)

/**
 * Runs `blender -b --factory-startup --python scripts/generate_model.py -- <args>` inside the runtime via proot.
 * One job at a time; the guest output dir is read directly from app storage afterwards.
 */
class BlenderRunner(private val context: Context, private val runtime: BlenderRuntime) {
    @Volatile private var process: Process? = null

    fun cancel() { process?.destroyForcibly() }

    suspend fun generate(req: GenRequest, onLine: (String) -> Unit): GenResult = withContext(Dispatchers.IO) {
        check(runtime.isReady()) { "Blender runtime is not installed" }
        val guestOut = "/tmp/forge-job"
        val hostOut = File(runtime.rootfs, guestOut.removePrefix("/"))
        hostOut.deleteRecursively()
        hostOut.mkdirs()
        val tmp = File(context.cacheDir, "proot-tmp").apply { deleteRecursively(); mkdirs() }

        val cmd = listOf(
            File(runtime.bin, "proot").path,
            "--kill-on-exit", "--link2symlink", "-0",
            "-r", runtime.rootfs.path,
            "-b", "/dev", "-b", "/proc", "-b", "/sys",
            "-w", "/opt/forge",
            "/opt/blender/blender", "-b", "--factory-startup", "-t", "${Runtime.getRuntime().availableProcessors().coerceIn(2, 6)}",
            "--python-exit-code", "1", "--python", "scripts/generate_model.py", "--",
        ) + req.scriptArgs(guestOut)

        val pb = ProcessBuilder(cmd).redirectErrorStream(true).directory(runtime.dir)
        pb.environment().apply {
            clear() // never leak Android's LD_PRELOAD/BOOTCLASSPATH etc. into the guest
            put("PROOT_TMP_DIR", tmp.path)
            put("PROOT_LOADER", File(runtime.bin, "proot-loader").path)
            put("LD_LIBRARY_PATH", runtime.bin.path) // libtalloc.so.2 for the bionic proot binary
            put("HOME", "/root")
            put("TMPDIR", "/tmp")
            put("LANG", "C.UTF-8")
            put("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        }
        val log = ArrayDeque<String>()
        val t0 = System.nanoTime()
        val p = pb.start().also { process = it }
        try {
            p.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    coroutineContext.ensureActive()
                    if (log.size >= 400) log.removeFirst()
                    log.addLast(line)
                    onLine(line)
                }
            }
            p.waitFor(30, TimeUnit.SECONDS)
            coroutineContext.ensureActive()
            val code = if (p.isAlive) -1 else p.exitValue()
            val seconds = (System.nanoTime() - t0) / 1e9
            val glb = hostOut.listFiles { f -> f.extension == "glb" }?.firstOrNull()
            if (code != 0 || glb == null || glb.length() == 0L) {
                throw IOException("Blender exited with code $code\n" + log.takeLast(25).joinToString("\n"))
            }
            val infoFile = File(hostOut, "generation.json")
            val info = if (infoFile.isFile) JSONObject(infoFile.readText()) else JSONObject()
            GenResult(glb, hostOut.listFiles { f -> f.extension == "blend" }?.firstOrNull(), info, seconds, log.toList())
        } finally {
            if (p.isAlive) p.destroyForcibly()
            process = null
        }
    }
}
