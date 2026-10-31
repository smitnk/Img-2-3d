package com.smitnk.img23d

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Debug
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.smitnk.img23d.core.TripoSR
import com.smitnk.img23d.data.ModelManager
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Project
import com.smitnk.img23d.data.ProjectStore
import com.smitnk.img23d.data.verifyLocal
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStream

sealed interface Screen {
    data object Home : Screen
    data object Create : Screen
    data object Generating : Screen
    data class Result(val projectId: String) : Screen
    data object History : Screen
    data object Models : Screen
}

data class GenSettings(
    val resolution: Int = 256,
    val removeBackground: Boolean = true,
    val vertexColors: Boolean = true,
    val foregroundRatio: Float = TripoSR.FOREGROUND_RATIO,
)

data class Progress(
    val stage: TripoSR.Stage = TripoSR.Stage.BACKGROUND,
    val fraction: Float = 0f,
    val elapsedMs: Long = 0,
    val pssMb: Int = 0,
    val peakPssMb: Int = 0,
)

data class UiState(
    val backStack: List<Screen> = listOf(Screen.Home),
    val modelStatus: ModelStatus = ModelStatus.Checking,
    val image: Bitmap? = null,
    val settings: GenSettings = GenSettings(),
    val progress: Progress = Progress(),
    val error: String? = null,
    val projects: List<Project> = emptyList(),
) {
    val screen: Screen get() = backStack.last()
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val models = ModelManager(app)
    private val store = ProjectStore(app)
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var job: Job? = null
    private var pipeline: TripoSR? = null

    init {
        refreshModels()
        refreshProjects()
    }

    fun navigate(s: Screen) = _state.update { it.copy(backStack = it.backStack + s, error = null) }

    /** @return false when already at the root so the activity can finish. */
    fun back(): Boolean {
        val st = _state.value
        if (st.screen == Screen.Generating) { cancel(); return true }
        if (st.backStack.size <= 1) return false
        _state.update { it.copy(backStack = it.backStack.dropLast(1), error = null) }
        return true
    }

    fun dismissError() = _state.update { it.copy(error = null) }

    fun refreshModels() {
        val cur = _state.value.modelStatus
        if (cur is ModelStatus.Downloading || cur is ModelStatus.Verifying) return
        _state.update { it.copy(modelStatus = if (models.isReady()) ModelStatus.Ready else ModelStatus.Missing(models.dir.path)) }
    }

    fun refreshProjects() = viewModelScope.launch(Dispatchers.IO) {
        val list = store.list()
        _state.update { it.copy(projects = list) }
    }

    fun downloadModels() = viewModelScope.launch {
        runCatching { models.download { s -> _state.update { it.copy(modelStatus = s) } } }
            .onFailure { e -> if (e !is CancellationException) _state.update { it.copy(modelStatus = ModelStatus.Failed(e.message ?: e.toString())) } }
    }

    fun verifyPushedModels() = viewModelScope.launch {
        runCatching { models.verifyLocal { s -> _state.update { it.copy(modelStatus = s) } } }
            .onFailure { e -> _state.update { it.copy(modelStatus = ModelStatus.Failed(e.message ?: e.toString())) } }
    }

    fun deleteModels() {
        models.delete()
        refreshModels()
    }

    fun pickImage(uri: Uri) = viewModelScope.launch {
        runCatching { withContext(Dispatchers.IO) { decode(uri) } }
            .onSuccess { bmp -> _state.update { it.copy(image = bmp) }; navigate(Screen.Create) }
            .onFailure { e -> _state.update { it.copy(error = "Could not open image: ${e.message}") } }
    }

    fun updateSettings(f: (GenSettings) -> GenSettings) = _state.update { it.copy(settings = f(it.settings)) }

    fun generate() {
        val st = _state.value
        val bmp = st.image ?: return
        if (job?.isActive == true) return // one inference job at a time
        if (st.modelStatus != ModelStatus.Ready) { navigate(Screen.Models); return }
        _state.update { it.copy(progress = Progress(), error = null) }
        navigate(Screen.Generating)
        val pipe = TripoSR(models.dir).also { pipeline = it }
        job = viewModelScope.launch {
            val start = System.currentTimeMillis()
            var peak = 0
            val monitor = launch(Dispatchers.Default) {
                while (isActive) {
                    val pss = (Debug.getPss() / 1024).toInt()
                    peak = maxOf(peak, pss)
                    _state.update { it.copy(progress = it.progress.copy(elapsedMs = System.currentTimeMillis() - start, pssMb = pss, peakPssMb = peak)) }
                    delay(500)
                }
            }
            try {
                val project = withContext(Dispatchers.Default) {
                    val w = bmp.width
                    val h = bmp.height
                    val argb = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
                    val rgba = IntArray(w * h * 4)
                    val alpha = bmp.hasAlpha()
                    for (i in argb.indices) {
                        val p = argb[i]
                        rgba[i * 4] = (p shr 16) and 255
                        rgba[i * 4 + 1] = (p shr 8) and 255
                        rgba[i * 4 + 2] = p and 255
                        rgba[i * 4 + 3] = if (alpha) p ushr 24 else 255
                    }
                    val opt = TripoSR.Options().apply {
                        resolution = st.settings.resolution
                        removeBackground = st.settings.removeBackground
                        vertexColors = st.settings.vertexColors
                        foregroundRatio = st.settings.foregroundRatio
                        threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
                    }
                    val res = pipe.run(rgba, w, h, opt) { stage, f ->
                        _state.update { it.copy(progress = it.progress.copy(stage = stage, fraction = f)) }
                    }
                    peak = maxOf(peak, (Debug.getPss() / 1024).toInt())
                    store.create(
                        glb = { file -> file.outputStream().buffered().use { res.mesh.writeGlb(it) } },
                        preview = previewBitmap(res.preprocessed),
                        vertices = res.mesh.vertexCount(), faces = res.mesh.faceCount(),
                        resolution = opt.resolution, timings = res.timingsMs, peakPssMb = peak,
                    )
                }
                refreshProjects()
                _state.update { it.copy(backStack = it.backStack.dropLast(1) + Screen.Result(project.id)) }
            } catch (e: CancellationException) {
                _state.update { it.copy(backStack = it.backStack.dropLast(1)) }
            } catch (e: Throwable) {
                val msg = when (e) {
                    is OutOfMemoryError -> "Out of memory. Try a lower mesh resolution."
                    else -> e.message ?: e.toString()
                }
                _state.update { it.copy(backStack = it.backStack.dropLast(1), error = "Generation failed: $msg") }
            } finally {
                monitor.cancel()
                pipeline = null
                System.gc()
            }
        }
    }

    fun cancel() {
        pipeline?.cancel()
        job?.cancel()
    }

    fun project(id: String): Project? = _state.value.projects.firstOrNull { it.id == id } ?: store.get(id)

    fun deleteProject(p: Project) = viewModelScope.launch(Dispatchers.IO) {
        store.delete(p)
        refreshProjects()
    }

    fun exportGlb(p: Project, out: OutputStream) = viewModelScope.launch(Dispatchers.IO) {
        runCatching { out.use { o -> p.glb.inputStream().use { it.copyTo(o) } } }
            .onFailure { e -> _state.update { it.copy(error = "Export failed: ${e.message}") } }
    }

    private fun decode(uri: Uri): Bitmap {
        val cr = getApplication<Application>().contentResolver
        // shortcut: inputs are capped at MAX_SIDE px to bound heap use; TripoSR itself only sees 512x512.
        return if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(cr, uri)) { d, info, _ ->
                d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val s = info.size
                val scale = MAX_SIDE.toFloat() / maxOf(s.width, s.height)
                if (scale < 1f) d.setTargetSize((s.width * scale).toInt().coerceAtLeast(1), (s.height * scale).toInt().coerceAtLeast(1))
            }
        } else {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / sample > MAX_SIDE) sample *= 2
            cr.openInputStream(uri).use {
                BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
            } ?: error("Unsupported image")
        }
    }

    private fun previewBitmap(img: FloatArray): Bitmap {
        val s = TripoSR.IMAGE_SIZE
        val px = IntArray(s * s) { i ->
            val r = (img[i * 3] * 255f).toInt().coerceIn(0, 255)
            val g = (img[i * 3 + 1] * 255f).toInt().coerceIn(0, 255)
            val b = (img[i * 3 + 2] * 255f).toInt().coerceIn(0, 255)
            (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, s, s, Bitmap.Config.ARGB_8888)
    }

    override fun onCleared() = cancel()

    companion object {
        const val MAX_SIDE = 1536
    }
}
