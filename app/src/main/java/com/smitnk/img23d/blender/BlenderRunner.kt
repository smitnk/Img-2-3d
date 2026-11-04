package com.smitnk.img23d.blender

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.coroutineContext

class BlenderResult(
    val glb: File, val blend: File?, val info: JSONObject, val seconds: Double, val log: List<String>,
    /** postprocess.py output: {"quality": {pass, errors, warnings, ...}, "thumbnail": ...}; null if it didn't run. */
    val quality: JSONObject? = null, val thumb: File? = null,
)

class BlenderException(message: String, val log: List<String>) : IOException(message)

/**
 * Runs headless Blender inside the runtime via proot. Guest paths:
 *   /opt/forge/scripts  Model Forge generators (part of the runtime)
 *   /opt/app            this app's scripts (assets/blender, bind-mounted, so no runtime rebuild needed)
 *   /opt/job            per-job inputs (input GLB, user script), bind-mounted
 *   /tmp/forge-job      output, read back from app storage
 * One job at a time.
 */
class BlenderRunner(private val context: Context, private val runtime: BlenderRuntime) {
    @Volatile private var process: Process? = null
    val jobIn = File(context.cacheDir, "blender-job-in")

    fun cancel() { process?.destroyForcibly() }

    /** Runs [script]; then, unless [postprocess] is false, postprocess.py (quality gate + thumbnail) on its GLB. */
    suspend fun run(script: String, args: List<String>, threads: Int, postprocess: Boolean = true, onLine: (String) -> Unit): BlenderResult {
        val r = exec(script, args, threads, "glb", onLine)
        if (!postprocess) return r
        onLine("Checking quality and rendering a thumbnail…")
        // Best-effort: a model is never lost because its quality check failed.
        return runCatching {
            exec("postprocess.py", listOf("--in", "/tmp/forge-job/${r.glb.name}"), threads, "glb", onLine, clearOut = false)
            val q = File(r.glb.parentFile, "quality.json")
            BlenderResult(r.glb, r.blend, r.info, r.seconds, r.log,
                q.takeIf { it.isFile }?.let { JSONObject(it.readText()) }, File(r.glb.parentFile, "thumb.png").takeIf { it.isFile })
        }.getOrElse { if (it is kotlinx.coroutines.CancellationException) throw it; r }
    }

    /** texproject.py --render: writes render_<view>.png files; returns the output folder. */
    suspend fun renderViews(args: List<String>, threads: Int, onLine: (String) -> Unit): File =
        exec("texproject.py", args, threads, "png", onLine).glb.parentFile!!

    /** Re-exports [glb] with a bx_export preset; returns the exported file (fbx/glb/stl). */
    suspend fun export(glb: File, preset: String, name: String, threads: Int, onLine: (String) -> Unit): File {
        glb.copyTo(File(jobIn.apply { mkdirs() }, "export_in.glb"), overwrite = true)
        val ext = if (preset == "print") "stl" else if (preset.startsWith("unity") || preset.startsWith("unreal") || preset == "roblox") "fbx" else "glb"
        return exec("export_preset.py", listOf("--in", "/opt/job/export_in.glb", "--preset", preset, "--name", name), threads, ext, onLine).glb
    }

    private suspend fun exec(
        script: String, args: List<String>, threads: Int, outExt: String, onLine: (String) -> Unit, clearOut: Boolean = true,
    ): BlenderResult = withContext(Dispatchers.IO) {
        check(runtime.isReady()) { "Blender engine is not installed" }
        val scripts = ScriptAssets.install(context)
        val guestOut = "/tmp/forge-job"
        val hostOut = File(runtime.rootfs, guestOut.removePrefix("/"))
        if (clearOut) hostOut.deleteRecursively()
        hostOut.mkdirs()
        jobIn.mkdirs()
        val tmp = File(context.cacheDir, "proot-tmp").apply { deleteRecursively(); mkdirs() }

        val cmd = listOf(
            File(runtime.bin, "proot").path,
            "--kill-on-exit", "--link2symlink", "-0",
            "-r", runtime.rootfs.path,
            "-b", "/dev", "-b", "/proc", "-b", "/sys",
            "-b", "${scripts.path}:/opt/app",
            "-b", "${jobIn.path}:/opt/job",
            "-b", "${File(context.filesDir, "bodypack").apply { mkdirs() }.path}:/opt/bodypack", // Body Creator assets
            "-b", "${File(context.filesDir, "rigpack").apply { mkdirs() }.path}:/opt/rigpack", // Rig with AI (SkinTokens)
            "-w", "/opt/app",
            "/opt/blender/blender", "-b", "--factory-startup", "-t", threads.toString(),
            "--python-exit-code", "1", "--python", script, "--",
        ) + args + listOf("--out", guestOut)

        val pb = ProcessBuilder(cmd).redirectErrorStream(true).directory(runtime.dir)
        pb.environment().apply {
            clear() // never leak Android's LD_PRELOAD/BOOTCLASSPATH etc. into the guest
            put("PROOT_TMP_DIR", tmp.path)
            put("PROOT_LOADER", File(runtime.bin, "proot-loader").path)
            put("LD_LIBRARY_PATH", runtime.bin.path) // proot's bionic shared libraries
            put("HOME", "/root")
            put("TMPDIR", "/tmp")
            put("LANG", "C.UTF-8")
            put("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
            // Software OpenGL (Mesa llvmpipe) for scripts that render or use the gpu module; no GPU driver is reachable.
            put("EGL_PLATFORM", "surfaceless")
            put("LIBGL_ALWAYS_SOFTWARE", "1")
            put("GALLIUM_DRIVER", "llvmpipe")
            put("XDG_CACHE_HOME", "/tmp/.cache") // Mesa shader cache; always writable
        }
        val log = ArrayDeque<String>()
        val t0 = System.nanoTime()
        val p = pb.start().also { process = it }
        try {
            p.inputStream.bufferedReader().useLines { lines ->
                for (line in lines) {
                    coroutineContext.ensureActive()
                    if (log.size >= 600) log.removeFirst()
                    log.addLast(line)
                    onLine(line)
                }
            }
            p.waitFor(30, TimeUnit.SECONDS)
            coroutineContext.ensureActive()
            val code = if (p.isAlive) -1 else p.exitValue()
            val glb = hostOut.listFiles { f -> f.extension == outExt }?.firstOrNull()
            if (code != 0 || glb == null || glb.length() == 0L) {
                throw BlenderException("Blender exited with code $code\n" + log.takeLast(25).joinToString("\n"), log.toList())
            }
            val infoFile = File(hostOut, "generation.json")
            val info = if (infoFile.isFile) JSONObject(infoFile.readText()) else JSONObject()
            BlenderResult(glb, hostOut.listFiles { f -> f.extension == "blend" }?.firstOrNull(), info, (System.nanoTime() - t0) / 1e9, log.toList())
        } finally {
            if (p.isAlive) p.destroyForcibly()
            if (process === p) process = null // a newer run may already own the field
        }
    }
}

/** Copies assets/blender into app storage once per app version (bind-mounted at /opt/app). */
object ScriptAssets {
    fun install(context: Context): File {
        val dir = File(context.filesDir, "blender-scripts")
        val stamp = File(dir, ".version")
        val version = context.packageManager.getPackageInfo(context.packageName, 0).let {
            "${it.versionName}-${it.lastUpdateTime}"
        }
        if (stamp.isFile && stamp.readText() == version) return dir
        dir.deleteRecursively()
        copy(context, "blender", dir)
        stamp.writeText(version)
        return dir
    }

    fun read(context: Context, path: String): String = context.assets.open("blender/$path").bufferedReader().use { it.readText() }

    private fun copy(context: Context, asset: String, dest: File) {
        val children = context.assets.list(asset).orEmpty()
        if (children.isEmpty()) {
            dest.parentFile?.mkdirs()
            context.assets.open(asset).use { i -> dest.outputStream().use { i.copyTo(it) } }
        } else {
            dest.mkdirs()
            children.forEach { copy(context, "$asset/$it", File(dest, it)) }
        }
    }
}
