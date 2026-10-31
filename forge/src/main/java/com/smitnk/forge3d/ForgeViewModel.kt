package com.smitnk.forge3d

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.smitnk.forge3d.engine.BlenderRunner
import com.smitnk.forge3d.engine.BlenderRuntime
import com.smitnk.forge3d.engine.ColorParam
import com.smitnk.forge3d.engine.Creation
import com.smitnk.forge3d.engine.GenRequest
import com.smitnk.forge3d.engine.Generator
import com.smitnk.forge3d.engine.Library
import com.smitnk.forge3d.engine.MemoryProbe
import com.smitnk.forge3d.engine.RuntimeStatus
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
import java.io.OutputStream

sealed interface Dest {
    data object Home : Dest
    data class Configure(val generator: Generator) : Dest
    data object Running : Dest
    data class Result(val id: String) : Dest
    data object Library : Dest
    data object Runtime : Dest
}

data class Draft(
    val name: String = "",
    val colors: Map<ColorParam, String> = emptyMap(),
    val rig: Boolean = true,
    val saveBlend: Boolean = false,
)

data class RunState(
    val generator: Generator? = null,
    val lines: List<String> = emptyList(),
    val elapsedMs: Long = 0,
    val rssMb: Int = 0,
    val peakRssMb: Int = 0,
)

data class ForgeState(
    val stack: List<Dest> = listOf(Dest.Home),
    val runtime: RuntimeStatus = RuntimeStatus.NotInstalled,
    val draft: Draft = Draft(),
    val run: RunState = RunState(),
    val creations: List<Creation> = emptyList(),
    val error: String? = null,
) {
    val dest get() = stack.last()
}

class ForgeViewModel(app: Application) : AndroidViewModel(app) {
    val runtime = BlenderRuntime(app)
    private val runner = BlenderRunner(app, runtime)
    private val library = Library(app)
    private val _state = MutableStateFlow(ForgeState(runtime = if (runtime.isReady()) RuntimeStatus.Ready else RuntimeStatus.NotInstalled))
    val state: StateFlow<ForgeState> = _state.asStateFlow()
    private var job: Job? = null

    init { refreshLibrary() }

    fun go(d: Dest) = _state.update { it.copy(stack = it.stack + d, error = null) }

    fun back(): Boolean {
        val s = _state.value
        if (s.dest == Dest.Running) { cancel(); return true }
        if (s.stack.size <= 1) return false
        _state.update { it.copy(stack = it.stack.dropLast(1)) }
        return true
    }

    fun dismissError() = _state.update { it.copy(error = null) }

    fun openGenerator(g: Generator) {
        _state.update { it.copy(draft = Draft(colors = g.defaults)) }
        go(Dest.Configure(g))
    }

    fun editDraft(f: (Draft) -> Draft) = _state.update { it.copy(draft = f(it.draft)) }

    fun refreshLibrary() = viewModelScope.launch(Dispatchers.IO) {
        val list = library.list()
        _state.update { it.copy(creations = list) }
    }

    fun installRuntime() = viewModelScope.launch {
        runCatching { runtime.install { s -> _state.update { it.copy(runtime = s) } } }
            .onFailure { e -> if (e !is CancellationException) _state.update { it.copy(runtime = RuntimeStatus.Failed(e.message ?: e.toString())) } }
    }

    fun uninstallRuntime() {
        runtime.uninstall()
        _state.update { it.copy(runtime = RuntimeStatus.NotInstalled) }
    }

    fun generate(g: Generator) {
        if (job?.isActive == true) return
        if (!runtime.isReady()) { go(Dest.Runtime); return }
        val d = _state.value.draft
        val req = GenRequest(g, d.name, d.colors, d.rig, d.saveBlend)
        _state.update { it.copy(run = RunState(generator = g)) }
        go(Dest.Running)
        job = viewModelScope.launch {
            val start = System.currentTimeMillis()
            var peak = 0
            val monitor = launch(Dispatchers.Default) {
                while (isActive) {
                    // Blender runs in child processes, so Debug.getPss() would miss it; sum RSS over our UID.
                    val pss = MemoryProbe.appTreeRssMb()
                    peak = maxOf(peak, pss)
                    _state.update { it.copy(run = it.run.copy(elapsedMs = System.currentTimeMillis() - start, rssMb = pss, peakRssMb = peak)) }
                    delay(500)
                }
            }
            try {
                val res = runner.generate(req) { line ->
                    _state.update { s -> s.copy(run = s.run.copy(lines = (s.run.lines + line).takeLast(200))) }
                }
                val c = library.add(req, res, peak)
                refreshLibrary()
                _state.update { it.copy(stack = it.stack.dropLast(1) + Dest.Result(c.id)) }
            } catch (e: CancellationException) {
                _state.update { it.copy(stack = it.stack.dropLast(1)) }
            } catch (e: Throwable) {
                _state.update { it.copy(stack = it.stack.dropLast(1), error = e.message ?: e.toString()) }
            } finally {
                monitor.cancel()
            }
        }
    }

    fun cancel() {
        runner.cancel()
        job?.cancel()
    }

    fun creation(id: String) = _state.value.creations.firstOrNull { it.id == id } ?: library.get(id)

    fun delete(c: Creation) = viewModelScope.launch(Dispatchers.IO) { library.delete(c); refreshLibrary() }

    fun export(c: Creation, out: OutputStream) = viewModelScope.launch(Dispatchers.IO) {
        runCatching { out.use { o -> c.glb.inputStream().use { it.copyTo(o) } } }
            .onFailure { e -> _state.update { it.copy(error = "Export failed: ${e.message}") } }
    }

    override fun onCleared() = cancel()
}
