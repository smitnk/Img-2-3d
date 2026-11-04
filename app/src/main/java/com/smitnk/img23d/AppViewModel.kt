package com.smitnk.img23d

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.smitnk.img23d.ai.ScriptWriter
import com.smitnk.img23d.blender.BlenderException
import com.smitnk.img23d.blender.BlenderResult
import com.smitnk.img23d.blender.BlenderRunner
import com.smitnk.img23d.blender.BlenderRuntime
import com.smitnk.img23d.blender.BuilderSpec
import com.smitnk.img23d.body.BodyMorph
import com.smitnk.img23d.body.BodyPack
import com.smitnk.img23d.body.TexPack
import com.smitnk.img23d.body.HunyuanPack
import com.smitnk.img23d.body.RigPack
import com.smitnk.img23d.core.HunyuanShape
import com.smitnk.img23d.core.TextureDiffusion
import com.smitnk.img23d.body.BodySpec
import com.smitnk.img23d.body.PackStatus
import com.smitnk.img23d.viewer.GltfMesh
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext
import com.smitnk.img23d.blender.ExportPreset
import com.smitnk.img23d.blender.MemoryProbe
import com.smitnk.img23d.blender.Preset
import com.smitnk.img23d.blender.RuntimeStatus
import com.smitnk.img23d.blender.ScriptAssets
import com.smitnk.img23d.core.TripoSR
import com.smitnk.img23d.data.Library
import com.smitnk.img23d.data.Model
import com.smitnk.img23d.data.ModelManager
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Settings
import com.smitnk.img23d.data.SettingsStore
import com.smitnk.img23d.data.Source
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
import org.json.JSONObject
import java.io.File
import java.io.OutputStream

enum class PhotoMode { MESH, CHARACTER }

sealed interface Dest {
    data object Home : Dest
    data object Library : Dest
    data object Settings : Dest
    data class Photo(val mode: PhotoMode) : Dest
    data object Builder : Dest
    data object Script : Dest
    data object Text : Dest
    data object Presets : Dest
    data object Body : Dest
    data class PresetConfig(val preset: Preset) : Dest
    data object Running : Dest
    data class Result(val id: String) : Dest
}

data class RunState(
    val title: String = "",
    val engine: String = "",
    val stage: String = "",
    val fraction: Float? = null,
    val step: Int = 1,
    val steps: Int = 1,
    val lines: List<String> = emptyList(),
    val elapsedMs: Long = 0,
    val memMb: Int = 0,
    val peakMb: Int = 0,
)

data class ScriptState(val fileName: String = "", val code: String = "", val params: String = "{}")

data class TextState(
    val description: String = "",
    val style: String = "Stylized cartoon",
    val detail: String = "medium",
    val draft: ScriptWriter.Draft? = null,
    val writing: Boolean = false,
    val lastLog: String = "",
)

data class Failure(val title: String, val message: String, val canRepair: Boolean = false)

data class UiState(
    val stack: List<Dest> = listOf(Dest.Home),
    val modelStatus: ModelStatus = ModelStatus.Checking,
    val runtime: RuntimeStatus = RuntimeStatus.NotInstalled,
    val image: Bitmap? = null,
    val builder: BuilderSpec = BuilderSpec(),
    val presetName: String = "",
    val presetColors: Map<String, String> = emptyMap(),
    val script: ScriptState = ScriptState(),
    val text: TextState = TextState(),
    val run: RunState = RunState(),
    val models: List<Model> = emptyList(),
    val filter: Source? = null,
    val query: String = "",
    val failure: Failure? = null,
    val storageMb: Long = 0,
    /** Preset export in progress (its label), and a finished one waiting for a save location (file, model name). */
    val exporting: String? = null,
    val exportReady: Pair<File, String>? = null,
    /** Builder live preview: the last quick build (no rig, no postprocess) and whether one is running. */
    val builderPreview: File? = null,
    val builderPreviewBusy: Boolean = false,
    val builderPreviewVersion: Int = 0,
    val body: BodySpec = BodySpec(),
    val bodyPack: PackStatus = PackStatus.NotInstalled,
    val texPack: PackStatus = PackStatus.NotInstalled,
    val hunyuanPack: PackStatus = PackStatus.NotInstalled,
    val rigPack: PackStatus = PackStatus.NotInstalled,
    /** Backup / restore of packs to a user folder: progress text, or null when idle. */
    val packFolder: String? = null,
    /** Live morph of [body] (no Blender), rebuilt on every edit. */
    val bodyPreview: GltfMesh? = null,
) {
    val dest: Dest get() = stack.last()
    val visibleModels get() = models.filter { (filter == null || it.source == filter) && (query.isBlank() || it.name.contains(query, true)) }
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val models = ModelManager(app)
    val runtime = BlenderRuntime(app)
    private val runner = BlenderRunner(app, runtime)
    private val library = Library(app)
    private val settingsStore = SettingsStore(app)
    val settings: StateFlow<Settings> = settingsStore.state
    private val writer by lazy { ScriptWriter(ScriptAssets.read(app, "SCRIPT_CONTRACT.md"), ScriptAssets.read(app, "examples/robot.py")) }
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var job: Job? = null
    private var tripo: TripoSR? = null

    init {
        refreshEngines()
        refreshLibrary()
    }

    private val s get() = settings.value

    // ---------- navigation ----------
    fun go(d: Dest) = _state.update { it.copy(stack = it.stack + d, failure = null) }

    fun tab(d: Dest) = _state.update { it.copy(stack = listOf(Dest.Home) + if (d == Dest.Home) emptyList() else listOf(d)) }

    /** @return false when already at the root so the activity can finish. */
    fun back(): Boolean {
        val st = _state.value
        if (st.dest == Dest.Running) { cancel(); return true }
        if (st.stack.size <= 1) return false
        _state.update { it.copy(stack = it.stack.dropLast(1)) }
        return true
    }

    fun dismissFailure() = _state.update { it.copy(failure = null) }

    fun updateSettings(f: (Settings) -> Settings) = settingsStore.update(f)

    // ---------- engines ----------
    fun refreshEngines() {
        val st = _state.value
        val busyB = st.modelStatus is ModelStatus.Downloading || st.modelStatus is ModelStatus.Verifying
        val busyA = st.runtime is RuntimeStatus.Downloading || st.runtime is RuntimeStatus.Verifying || st.runtime is RuntimeStatus.Extracting
        _state.update {
            it.copy(
                modelStatus = if (busyB) it.modelStatus else if (models.isReady()) ModelStatus.Ready else ModelStatus.Missing(models.dir.path),
                runtime = if (busyA) it.runtime else if (runtime.isReady()) RuntimeStatus.Ready else RuntimeStatus.NotInstalled,
            )
        }
    }

    fun downloadModels() = viewModelScope.launch {
        runCatching { models.download { m -> _state.update { it.copy(modelStatus = m) } } }
            .onFailure { e -> if (e !is CancellationException) _state.update { it.copy(modelStatus = ModelStatus.Failed(e.message ?: e.toString())) } }
    }

    fun verifyPushedModels() = viewModelScope.launch {
        runCatching { models.verifyLocal { m -> _state.update { it.copy(modelStatus = m) } } }
            .onFailure { e -> _state.update { it.copy(modelStatus = ModelStatus.Failed(e.message ?: e.toString())) } }
    }

    fun deleteModels() { models.delete(); refreshEngines() }

    fun installRuntime() = viewModelScope.launch {
        runCatching { runtime.install { r -> _state.update { it.copy(runtime = r) } } }
            .onFailure { e -> if (e !is CancellationException) _state.update { it.copy(runtime = RuntimeStatus.Failed(e.message ?: e.toString())) } }
    }

    fun uninstallRuntime() { runtime.uninstall(); refreshEngines() }

    // ---------- library ----------
    fun refreshLibrary() = viewModelScope.launch(Dispatchers.IO) {
        val list = library.list()
        val mb = library.bytesUsed() / 1_000_000
        _state.update { it.copy(models = list, storageMb = mb) }
    }

    fun setFilter(f: Source?) = _state.update { it.copy(filter = f) }
    fun setQuery(q: String) = _state.update { it.copy(query = q) }
    fun model(id: String) = _state.value.models.firstOrNull { it.id == id } ?: library.get(id)
    fun rename(m: Model, name: String) = viewModelScope.launch(Dispatchers.IO) { library.rename(m, name); refreshLibrary() }
    fun delete(m: Model) = viewModelScope.launch(Dispatchers.IO) { library.delete(m); refreshLibrary() }
    fun clearLibrary() = viewModelScope.launch(Dispatchers.IO) { library.clear(); refreshLibrary() }

    fun saveFile(file: File, out: OutputStream) = viewModelScope.launch(Dispatchers.IO) {
        runCatching { out.use { o -> file.inputStream().use { it.copyTo(o) } } }
            .onFailure { e -> fail("Export failed", e.message ?: e.toString()) }
    }

    /** Converts a model for an engine (bx_export preset) in Blender, then hands the file to the save dialog. */
    fun exportPreset(m: Model, preset: ExportPreset) {
        if (!runtime.isReady()) { go(Dest.Settings); return }
        if (job?.isActive == true) return
        stopPreview()
        _state.update { it.copy(exporting = preset.label) }
        job = viewModelScope.launch {
            runCatching {
                val f = runner.export(m.glb, preset.id, m.name, s.threadCount()) {}
                // Keep a copy outside the job dir, which the next Blender run wipes.
                f.copyTo(File(getApplication<Application>().cacheDir, "export/${f.name}"), overwrite = true)
            }.onSuccess { f -> _state.update { it.copy(exporting = null, exportReady = f to m.name) } }
                .onFailure { e ->
                    _state.update { it.copy(exporting = null) }
                    if (e !is CancellationException) fail("${preset.label} export failed", e.message ?: e.toString())
                }
        }
    }

    fun exportHandled() = _state.update { it.copy(exportReady = null) }

    /** Text-to-3D model flagged by the quality gate: ask Claude to fix the causes, then run it again. */
    fun fixQualityWithAi(m: Model) {
        val script = m.script?.readText() ?: return
        val issues = m.quality?.let { it.errors + it.warnings }.orEmpty()
        if (issues.isEmpty() || s.apiKey.isBlank()) return
        editText { it.copy(description = it.description.ifBlank { m.detail }, writing = true) }
        go(Dest.Text)
        viewModelScope.launch {
            runCatching { writer.fixQuality(s.apiKey, s.apiModel, _state.value.text.description, script, issues) }
                .onSuccess { nd -> editText { it.copy(writing = false, draft = nd) }; runText() }
                .onFailure { e -> editText { it.copy(writing = false) }; fail("AI fix failed", e.message ?: e.toString()) }
        }
    }

    // ---------- inputs ----------
    fun pickImage(uri: Uri, mode: PhotoMode) = viewModelScope.launch {
        runCatching { withContext(Dispatchers.IO) { decode(uri) } }
            .onSuccess { bmp -> _state.update { it.copy(image = bmp) }; if (_state.value.dest !is Dest.Photo) go(Dest.Photo(mode)) }
            .onFailure { e -> fail("Could not open image", e.message ?: e.toString()) }
    }

    fun editBuilder(f: (BuilderSpec) -> BuilderSpec) {
        _state.update { it.copy(builder = f(it.builder)) }
        schedulePreview()
    }

    private var previewJob: Job? = null

    // ---------- Body Creator ----------
    val bodyPack = BodyPack(app)
    val texPack = TexPack(app)
    val hunyuanPack = HunyuanPack(app)
    val rigPack = RigPack(app)
    private var hunyuan: HunyuanShape? = null
    private var morph: BodyMorph? = null
    private var morphJob: Job? = null

    init { // after bodyPack is initialized (Kotlin runs initializers in declaration order)
        if (bodyPack.isReady()) _state.update { it.copy(bodyPack = PackStatus.Ready) }
        if (texPack.isReady()) _state.update { it.copy(texPack = PackStatus.Ready) }
        if (hunyuanPack.isReady()) _state.update { it.copy(hunyuanPack = PackStatus.Ready) }
        if (rigPack.isReady()) _state.update { it.copy(rigPack = PackStatus.Ready) }
    }

    private fun packs() = listOf(bodyPack, texPack, hunyuanPack, rigPack)

    fun uninstallRigPack() { rigPack.uninstall(); _state.update { it.copy(rigPack = PackStatus.NotInstalled) } }

    fun installRigPack() = viewModelScope.launch {
        runCatching { rigPack.install { st -> _state.update { it.copy(rigPack = st) } } }
            .onFailure { e -> _state.update { it.copy(rigPack = PackStatus.Failed(e.message ?: e.toString())) } }
    }

    /** Rig with AI: SkinTokens predicts bones + skin weights for this exact mesh (rig_ai.py in the phone's Blender). */
    fun rigAi(m: Model) {
        if (!runtime.isReady() || !rigPack.isReady()) { go(Dest.Settings); return }
        launchJob("Rig with AI", "Blender", 1) { report ->
            m.glb.copyTo(File(runner.jobIn.apply { mkdirs() }, "input.glb"), overwrite = true)
            val b = blender("rig_ai.py", listOf("--in", "/opt/job/input.glb", "--pack", "/opt/rigpack", "--name", "${m.name} rigged",
                "--threads", s.threadCount().toString()), report)
            saveBlender(b, "${m.name} rigged", Source.REFINED, thumbFile = m.thumbnail, detail = "AI rig (SkinTokens) of ${m.name}")
                .also { n -> m.photo?.copyTo(File(n.dir, "photo.png"), overwrite = true) }
        }
    }

    fun backupPacks(tree: android.net.Uri) = viewModelScope.launch {
        runCatching {
            val show: (String) -> Unit = { t -> _state.update { it.copy(packFolder = t) } }
            com.smitnk.img23d.body.PackFolder.backup(getApplication(), tree, models.dir, models.isReady(), show)
            for (p in packs()) com.smitnk.img23d.body.PackFolder.backup(getApplication(), tree, p, show)
        }.onFailure { e -> fail("Backup failed", e.message ?: e.toString()) }
        _state.update { it.copy(packFolder = null) }
    }

    fun restorePacks(tree: android.net.Uri) = viewModelScope.launch {
        val found = mutableListOf<String>()
        runCatching {
            val show: (String) -> Unit = { t -> _state.update { it.copy(packFolder = t) } }
            if (com.smitnk.img23d.body.PackFolder.restore(getApplication(), tree, models.dir, "model_info.json", { models.markVerified() }, show)) {
                found += "models"
                _state.update { it.copy(modelStatus = ModelStatus.Ready) }
            }
            for (p in packs()) if (com.smitnk.img23d.body.PackFolder.restore(getApplication(), tree, p, show)) found += p.dir.name
        }.onFailure { e -> fail("Restore failed", e.message ?: e.toString()) }
        _state.update {
            it.copy(packFolder = null,
                bodyPack = if (bodyPack.isReady()) PackStatus.Ready else it.bodyPack,
                texPack = if (texPack.isReady()) PackStatus.Ready else it.texPack,
                hunyuanPack = if (hunyuanPack.isReady()) PackStatus.Ready else it.hunyuanPack,
                rigPack = if (rigPack.isReady()) PackStatus.Ready else it.rigPack)
        }
        if (found.isEmpty()) fail("Nothing to restore", "That folder has no pack backup. Pick the folder you backed up to (it holds bodypack / texgen / hunyuan).")
    }

    /** One engine's pack to / from a folder: key = its folder name (models, texgen, hunyuan, bodypack). */
    fun backupOne(key: String, tree: android.net.Uri) = viewModelScope.launch {
        val show: (String) -> Unit = { t -> _state.update { it.copy(packFolder = t) } }
        runCatching {
            if (key == "models") {
                if (!models.isReady()) error("TripoSR isn't fully installed yet")
                com.smitnk.img23d.body.PackFolder.backup(getApplication(), tree, models.dir, true, show)
            } else {
                val p = packs().first { it.dir.name == key }
                if (!p.isReady()) error("This pack isn't fully installed yet")
                com.smitnk.img23d.body.PackFolder.backup(getApplication(), tree, p, show)
            }
        }.onFailure { e -> fail("Save failed", e.message ?: e.toString()) }
        _state.update { it.copy(packFolder = null) }
    }

    fun restoreOne(key: String, tree: android.net.Uri) = viewModelScope.launch {
        val show: (String) -> Unit = { t -> _state.update { it.copy(packFolder = t) } }
        runCatching {
            val ok = if (key == "models") com.smitnk.img23d.body.PackFolder.restore(getApplication(), tree, models.dir, "model_info.json", { models.markVerified() }, show)
            else com.smitnk.img23d.body.PackFolder.restore(getApplication(), tree, packs().first { it.dir.name == key }, show)
            if (!ok) error("No saved \"$key\" pack there. Pick the folder you saved to, or its \"$key\" folder.")
        }.onFailure { e -> fail("Load failed", e.message ?: e.toString()) }
        _state.update {
            it.copy(packFolder = null, modelStatus = if (models.isReady()) ModelStatus.Ready else it.modelStatus,
                texPack = if (texPack.isReady()) PackStatus.Ready else it.texPack,
                hunyuanPack = if (hunyuanPack.isReady()) PackStatus.Ready else it.hunyuanPack,
                rigPack = if (rigPack.isReady()) PackStatus.Ready else it.rigPack)
        }
    }

    fun uninstallHunyuanPack() { hunyuanPack.uninstall(); _state.update { it.copy(hunyuanPack = PackStatus.NotInstalled) } }

    fun installHunyuanPack() = viewModelScope.launch {
        runCatching { hunyuanPack.install { st -> _state.update { it.copy(hunyuanPack = st) } } }
            .onFailure { e -> _state.update { it.copy(hunyuanPack = PackStatus.Failed(e.message ?: e.toString())) } }
    }

    fun uninstallTexPack() { texPack.uninstall(); _state.update { it.copy(texPack = PackStatus.NotInstalled) } }

    fun installTexPack() = viewModelScope.launch {
        runCatching { texPack.install { st -> _state.update { it.copy(texPack = st) } } }
            .onFailure { e -> _state.update { it.copy(texPack = PackStatus.Failed(e.message ?: e.toString())) } }
    }

    /**
     * Paint texture, all on the phone: Blender renders back/left/right from the model's own colours, the texture AI
     * (LCM img2img, core TextureDiffusion) repaints each render, then texproject.py projects the repaints plus the
     * source photo (front) onto the mesh and bakes one UV texture.
     */
    fun paintTexture(m: Model) {
        if (!runtime.isReady()) { go(Dest.Settings); return }
        val ai = s.paintAi
        if (!ai && m.photo == null) { fail("Paint texture", "This model has no saved photo (made before this version). Turn on \"AI paints the sides\" in Settings, or make the model again."); return }
        if (ai && !texPack.isReady()) { installTexPack(); return }
        launchJob("Paint texture", "Blender", if (ai) 3 else 1) { report ->
            val tex = File(runner.jobIn.apply { mkdirs() }, "tex").apply { deleteRecursively(); mkdirs() }
            m.glb.copyTo(File(runner.jobIn, "input.glb"), overwrite = true)
            val views = if (ai) PAINT_VIEWS else emptyList()
            // TripoSR output (and refines of it) faces +Y in Blender; Blender-built characters face -Y.
            val facing = if (m.source in setOf(Source.PHOTO, Source.CHARACTER, Source.REFINED)) "+y" else "-y"
            val renders = if (views.isEmpty()) tex else runner.renderViews(listOf("--in", "/opt/job/input.glb", "--render", views.joinToString(","), "--facing", facing), s.threadCount()) { report.line(it) }
            if (ai) report.step(2, "Texture AI")
            val td by lazy { TextureDiffusion(texPack.dir, texPack.unet, s.threadCount()) }
            val entries = org.json.JSONArray()
            m.photo?.let { it.copyTo(File(tex, "front.png"), overwrite = true); entries.put(JSONObject().put("image", "front.png").put("view", "front").put("weight", 8)) } // weight 8: the real photo beats AI views
            views.forEachIndexed { i, v ->
                coroutineContext.ensureActive()
                report.stage("Painting $v view (${i + 1}/${views.size})", i.toFloat() / views.size)
                val bmp = android.graphics.BitmapFactory.decodeFile(File(renders, "render_$v.png").path)
                    ?: throw IllegalStateException("Blender did not render the $v view")
                val alpha = FloatArray(TextureDiffusion.SIZE * TextureDiffusion.SIZE)
                val init = TextureDiffusion.toChw(rgbaOf(bmp), bmp.width, bmp.height, alpha)
                val out = td.paint(init, TextureDiffusion.VIEWS.indexOf(v), 4, PAINT_STRENGTH, TextureDiffusion.seeded(v.hashCode().toLong()), null)
                bitmapOf(TextureDiffusion.toRgba(out, alpha), TextureDiffusion.SIZE).compressTo(File(tex, "$v.png"))
                entries.put(JSONObject().put("image", "$v.png").put("view", v))
            }
            File(tex, "views.json").writeText(entries.toString())
            if (ai) report.step(3, "Blender")
            val b = blender("texproject.py", listOf("--in", "/opt/job/input.glb", "--views", "/opt/job/tex/views.json", "--res", "2048", "--facing", facing), report)
            saveBlender(b, "${m.name} painted", m.source, thumbFile = m.thumbnail, detail = "AI-painted texture of ${m.name}")
                .also { n -> m.photo?.copyTo(File(n.dir, "photo.png"), overwrite = true) }
        }
    }

    fun installBodyPack() = viewModelScope.launch {
        runCatching { bodyPack.install { st -> _state.update { it.copy(bodyPack = st) } } }
            .onFailure { e -> _state.update { it.copy(bodyPack = PackStatus.Failed(e.message ?: e.toString())) } }
        if (bodyPack.isReady()) refreshBodyPreview()
    }

    fun editBody(f: (BodySpec) -> BodySpec) {
        _state.update { it.copy(body = f(it.body)) }
        refreshBodyPreview()
    }

    /** Morphs on a background thread; a newer edit cancels the older morph, so dragging stays responsive. */
    fun refreshBodyPreview() {
        if (!bodyPack.isReady()) return
        morphJob?.cancel()
        morphJob = viewModelScope.launch(Dispatchers.Default) {
            val m = morph ?: BodyMorph(bodyPack.dir).also { morph = it }
            val spec = _state.value.body
            val v = m.morph(spec.macros, spec.sliders)
            ensureActive()
            val c = parseHex(spec.skin)
            _state.update { it.copy(bodyPreview = GltfMesh.fromIndexed(v, m.triangles, c)) }
        }
    }

    fun bodySliders(): Map<String, Pair<String?, String>> = morph?.let { BodyMorph.sliderTargets(it.modifiers) }.orEmpty()

    fun runBody() = blenderJob(if (_state.value.body.fit) "Fit character to model" else "Body Creator") { report ->
        val spec = _state.value.body
        val json = JSONObject(spec.toJson())
        spec.reference?.let { id ->
            val refModel = library.get(id)
            val ref = refModel?.glb ?: throw IllegalStateException("The reference model was deleted; remove it in Body Creator")
            ref.copyTo(File(runner.jobIn.apply { mkdirs() }, "ref.glb"), overwrite = true)
            if (refModel.source in setOf(Source.PHOTO, Source.CHARACTER, Source.REFINED)) json.put("ref_facing", "+y")
            refModel.photo?.let { it.copyTo(File(runner.jobIn, "ref_photo.png"), overwrite = true); json.put("ref_photo", "/opt/job/ref_photo.png") }
        }
        File(runner.jobIn.apply { mkdirs() }, "body_spec.json").writeText(json.toString())
        val b = blender("body_creator.py", listOf("--pack", "/opt/bodypack", "--spec", "/opt/job/body_spec.json"), report)
        // Keep editing from what was built (after a fit: the fitted sliders).
        b.info.optJSONObject("spec")?.let { j -> _state.update { it.copy(body = BodySpec.fromJson(j)) }; refreshBodyPreview() }
        saveBlender(b, spec.name, Source.BODY, detail = "Body Creator · ${b.info.optDouble("height_m", 0.0).let { "%.2f m".format(it) }}" +
            if (spec.reference != null) " · fitted to ${spec.referenceName}" else "")
    }

    /** Body Creator fitted to any library model (TripoSR photo mesh, refined, imported...): sliders, then wrap + colours. */
    fun fitBodyTo(m: Model) {
        if (!bodyPack.isReady()) { go(Dest.Body); return }
        val photo = m.source == Source.PHOTO || m.source == Source.CHARACTER || m.source == Source.REFINED
        _state.update {
            it.copy(body = it.body.copy(name = "${m.name} (Body)", reference = m.id, referenceName = m.name, fit = true, wrap = 0.3f,
                refColors = photo, clothes = if (photo) emptyList() else it.body.clothes, hair = if (photo) null else it.body.hair))
        }
        runBody()
    }

    /** Re-open a Body Creator model in the editor with the exact settings it was built from. */
    fun openBodyEditor(m: Model) {
        val spec = runCatching { JSONObject(File(m.dir, "meta.json").readText()).getJSONObject("extra").getJSONObject("spec") }.getOrNull() ?: return
        _state.update { it.copy(body = BodySpec.fromJson(spec)) }
        refreshBodyPreview()
        go(Dest.Body)
    }

    private fun parseHex(h: String): FloatArray = runCatching {
        val c = android.graphics.Color.parseColor(h)
        floatArrayOf(((c shr 16) and 255) / 255f, ((c shr 8) and 255) / 255f, (c and 255) / 255f)
    }.getOrDefault(floatArrayOf(0.8f, 0.65f, 0.55f))

    /** Quick unrigged build after the user stops editing, so the Builder shows the real character, not just options. */
    fun schedulePreview(delayMs: Long = 1200) {
        if (!runtime.isReady() || job?.isActive == true) return
        stopPreview()
        previewJob = viewModelScope.launch {
            delay(delayMs)
            _state.update { it.copy(builderPreviewBusy = true) }
            try {
                val spec = _state.value.builder
                val b = runner.run("character_builder.py", spec.copy(smooth = minOf(spec.smooth, 1)).args(rig = false, blend = false),
                    s.threadCount(), postprocess = false) {}
                val f = File(getApplication<Application>().cacheDir, "builder-preview.glb")
                b.glb.copyTo(f, overwrite = true)
                _state.update { it.copy(builderPreview = f, builderPreviewVersion = it.builderPreviewVersion + 1) }
            } catch (e: CancellationException) {
                throw e
            } catch (_: Exception) {
                // a failed preview is not an error worth a dialog; the real build reports it
            } finally {
                _state.update { it.copy(builderPreviewBusy = false) }
            }
        }
    }

    fun importScript(uri: Uri) = viewModelScope.launch {
        runCatching {
            withContext(Dispatchers.IO) {
                val cr = getApplication<Application>().contentResolver
                val name = uri.lastPathSegment?.substringAfterLast('/') ?: "script.py"
                val code = cr.openInputStream(uri)!!.use { it.readBytes() }
                require(code.size < 1_000_000) { "Script is larger than 1 MB" }
                name to String(code)
            }
        }.onSuccess { (n, c) -> _state.update { it.copy(script = it.script.copy(fileName = n, code = c)) } }
            .onFailure { e -> fail("Could not read script", e.message ?: e.toString()) }
    }

    fun loadExampleScript() = _state.update {
        it.copy(script = it.script.copy(fileName = "robot.py", code = ScriptAssets.read(getApplication(), "examples/robot.py")))
    }

    fun setScriptParams(p: String) = _state.update { it.copy(script = it.script.copy(params = p)) }

    fun scriptContract() = ScriptAssets.read(getApplication(), "SCRIPT_CONTRACT.md")

    fun editText(f: (TextState) -> TextState) = _state.update { it.copy(text = f(it.text)) }

    fun setPreset(name: String, colors: Map<String, String>) = _state.update { it.copy(presetName = name, presetColors = colors) }

    // ---------- AI script writing (optional, cloud) ----------
    fun writeTextScript() {
        val t = _state.value.text
        if (t.description.isBlank() || t.writing) return
        editText { it.copy(writing = true, draft = null) }
        viewModelScope.launch {
            runCatching { writer.write(s.apiKey, s.apiModel, t.description, t.style, t.detail) }
                .onSuccess { d -> editText { it.copy(writing = false, draft = d, lastLog = "") } }
                .onFailure { e -> editText { it.copy(writing = false) }; fail("Script writing failed", e.message ?: e.toString()) }
        }
    }

    fun repairTextScript() {
        val t = _state.value.text
        val d = t.draft ?: return
        dismissFailure()
        editText { it.copy(writing = true) }
        viewModelScope.launch {
            runCatching { writer.repair(s.apiKey, s.apiModel, t.description, d.script, t.lastLog) }
                .onSuccess { nd -> editText { it.copy(writing = false, draft = nd) }; runText() }
                .onFailure { e -> editText { it.copy(writing = false) }; fail("Repair failed", e.message ?: e.toString()) }
        }
    }

    // ---------- jobs ----------
    fun runPhoto(mode: PhotoMode) {
        val bmp = _state.value.image ?: return
        val useHunyuan = s.hunyuan && hunyuanPack.isReady()
        if (!useHunyuan && _state.value.modelStatus != ModelStatus.Ready) { go(Dest.Settings); return }
        if (mode == PhotoMode.CHARACTER && !runtime.isReady()) { go(Dest.Settings); return }
        val steps = if (mode == PhotoMode.CHARACTER) 2 else 1
        launchJob(if (mode == PhotoMode.CHARACTER) "Photo → Character" else "Photo → 3D", if (useHunyuan) "Hunyuan3D" else "TripoSR", steps) { report ->
            val (res, opt) = if (useHunyuan) hunyuan3d(bmp, report) else tripoSR(bmp, report)
            val engine = if (useHunyuan) "Hunyuan3D-2mini ${s.hunyuanOctree}³" else "TripoSR ${opt.resolution}³"
            val thumb = previewBitmap(res.preprocessed)
            if (mode == PhotoMode.MESH) {
                library.add("Photo model", Source.PHOTO, { f -> f.outputStream().buffered().use { res.mesh.writeGlb(it) } },
                    res.mesh.vertexCount(), res.mesh.faceCount(), resolution = opt.resolution, timings = res.timingsMs,
                    peakMemMb = 0, detail = engine, thumbnail = thumb).also { savePhoto(it, res) }
            } else {
                val input = File(runner.jobIn.apply { mkdirs() }, "input.glb")
                input.outputStream().buffered().use { res.mesh.writeGlb(it) }
                report.step(2, "Blender")
                val b = blender("refine_mesh.py", refineArgs("Photo character"), report)
                saveBlender(b, "Photo character", Source.CHARACTER, res.timingsMs, thumb = thumb, detail = "$engine + Blender refine")
                    .also { savePhoto(it, res) }
            }
        }
    }

    fun refine(m: Model) {
        if (!runtime.isReady()) { go(Dest.Settings); return }
        launchJob("Refine in Blender", "Blender", 1) { report ->
            m.glb.copyTo(File(runner.jobIn.apply { mkdirs() }, "input.glb"), overwrite = true)
            val b = blender("refine_mesh.py", refineArgs("${m.name} refined"), report)
            saveBlender(b, "${m.name} refined", Source.REFINED, thumbFile = m.thumbnail, detail = "Blender refine of ${m.name}")
                .also { n -> m.photo?.copyTo(File(n.dir, "photo.png"), overwrite = true) }
        }
    }

    fun runBuilder() = buildCharacter()

    /** The runner does one Blender process at a time: kill a running preview now (cancel alone waits for its next log line). */
    private fun stopPreview() {
        if (previewJob?.isActive == true) {
            previewJob?.cancel()
            if (_state.value.builderPreviewBusy) runner.cancel()
        }
    }

    private fun buildCharacter() = blenderJob("Character Builder") { report ->
        val spec = _state.value.builder
        val b = blender("character_builder.py", spec.args(s.rig, s.saveBlend), report)
        saveBlender(b, spec.name, Source.BUILDER, detail = "Builder · ${spec.picks.values.joinToString(" · ")}")
    }

    fun runScript() {
        val sc = _state.value.script
        if (sc.code.isBlank()) return
        blenderJob(sc.fileName.ifBlank { "Python script" }) { report ->
            val file = File(runner.jobIn.apply { mkdirs() }, "user_script.py").apply { writeText(sc.code) }
            JSONObject(sc.params.ifBlank { "{}" }) // validate before spending a Blender run
            val name = sc.fileName.removeSuffix(".py").ifBlank { "Custom model" }
            val b = blender("run_custom.py", listOf("--script", "/opt/job/${file.name}", "--name", name, "--params", sc.params.ifBlank { "{}" }) +
                if (s.saveBlend) listOf("--blend") else emptyList(), report)
            saveBlender(b, name, Source.SCRIPT, script = sc.code, detail = "Script ${sc.fileName}")
        }
    }

    fun runText() {
        val t = _state.value.text
        val d = t.draft ?: return
        blenderJob("Text to 3D") { report ->
            val file = File(runner.jobIn.apply { mkdirs() }, "ai_script.py").apply { writeText(d.script) }
            val name = t.description.take(32).trim().ifBlank { "AI character" }
            try {
                val b = blender("run_custom.py", listOf("--script", "/opt/job/${file.name}", "--name", name), report)
                saveBlender(b, name, Source.TEXT, script = d.script, detail = t.description.take(200))
            } catch (e: BlenderException) {
                editText { it.copy(lastLog = e.log.takeLast(80).joinToString("\n")) }
                throw e
            }
        }
    }

    fun runPreset(p: Preset) = blenderJob(p.title) { report ->
        val st = _state.value
        val b = blender("/opt/forge/scripts/generate_model.py", p.args(st.presetName, st.presetColors, s.rig, s.saveBlend), report)
        saveBlender(b, st.presetName.ifBlank { p.title }, Source.PRESET, detail = "Model Forge · ${p.title}")
    }

    fun cancel() {
        tripo?.cancel()
        hunyuan?.cancel()
        runner.cancel()
        job?.cancel()
    }

    // ---------- internals ----------
    inner class Report(private val start: Long) {
        fun step(n: Int, engine: String) = _state.update { it.copy(run = it.run.copy(step = n, engine = engine, stage = "", fraction = null, lines = emptyList())) }
        fun stage(text: String, fraction: Float?) = _state.update { it.copy(run = it.run.copy(stage = text, fraction = fraction)) }
        fun line(l: String) = _state.update { it.copy(run = it.run.copy(lines = (it.run.lines + l).takeLast(300), stage = l.take(90))) }
        fun peak() = _state.value.run.peakMb
        val elapsed get() = System.currentTimeMillis() - start
    }

    private fun blenderJob(title: String, block: suspend (Report) -> Model) {
        if (!runtime.isReady()) { go(Dest.Settings); return }
        launchJob(title, "Blender", 1, block)
    }

    private fun launchJob(title: String, engine: String, steps: Int, block: suspend (Report) -> Model) {
        if (job?.isActive == true) return // one job at a time, both engines
        stopPreview()
        _state.update { it.copy(run = RunState(title = title, engine = engine, steps = steps), failure = null) }
        go(Dest.Running)
        job = viewModelScope.launch {
            val report = Report(System.currentTimeMillis())
            val monitor = launch(Dispatchers.Default) {
                while (isActive) {
                    val mem = MemoryProbe.appTreeRssMb() // app + proot/Blender children
                    _state.update { it.copy(run = it.run.copy(elapsedMs = report.elapsed, memMb = mem, peakMb = maxOf(it.run.peakMb, mem))) }
                    delay(500)
                }
            }
            try {
                val m = withContext(Dispatchers.Default) { block(report) }
                withContext(Dispatchers.IO) { patchPeak(m, report.peak()) }
                refreshLibrary()
                _state.update { it.copy(stack = it.stack.dropLast(1) + Dest.Result(m.id)) }
            } catch (e: CancellationException) {
                _state.update { it.copy(stack = it.stack.dropLast(1)) }
            } catch (e: Throwable) {
                val msg = if (e is OutOfMemoryError) "Out of memory. Lower the mesh resolution in Settings." else e.message ?: e.toString()
                val repair = e is BlenderException && _state.value.run.title == "Text to 3D" && s.apiKey.isNotBlank()
                _state.update { it.copy(stack = it.stack.dropLast(1), failure = Failure("$title failed", msg, repair)) }
            } finally {
                monitor.cancel()
                tripo = null
                hunyuan = null
                System.gc()
            }
        }
    }

    private fun patchPeak(m: Model, peak: Int) {
        val f = File(m.dir, "meta.json")
        f.writeText(JSONObject(f.readText()).put("peakPssMb", peak).toString(2))
    }

    private suspend fun blender(script: String, args: List<String>, report: Report): BlenderResult =
        runner.run(script, args, s.threadCount()) { report.line(it) }

    private fun refineArgs(name: String) = buildList {
        addAll(listOf("--in", "/opt/job/input.glb", "--name", name,
            "--height", s.refineHeight.toString(), "--voxel", s.refineVoxel.toString(),
            "--smooth", s.refineSmooth.toString(), "--decimate", s.refineDecimate.toString()))
        if (!s.vertexColors) add("--no-colors")
        if (s.refineRig) add("--rig")
        if (s.vertexColors) addAll(listOf("--texture", "1024")) // UV texture for engines that ignore vertex colours
        if (s.saveBlend) add("--blend")
    }

    private fun saveBlender(
        b: BlenderResult, name: String, source: Source, extraTimings: Map<String, Long> = emptyMap(),
        thumb: Bitmap? = null, thumbFile: File? = null, script: String? = null, detail: String = "",
    ): Model {
        val i = b.info
        return library.add(
            name = i.optString("name", name).ifBlank { name }, source = source, writeGlb = { b.glb.copyTo(it, overwrite = true) },
            vertices = i.optInt("vertices"), faces = i.optInt("triangles"), bones = i.optJSONObject("rig")?.optInt("bones") ?: 0,
            timings = extraTimings + ("blender" to (b.seconds * 1000).toLong()), peakMemMb = 0, detail = detail,
            // Blender's own 3/4 render beats a copied photo thumbnail; fall back when postprocess didn't run.
            thumbnail = thumb.takeIf { b.thumb == null }, thumbnailFrom = b.thumb ?: thumbFile, blend = b.blend, script = script,
            extra = i, quality = b.quality?.optJSONObject("quality"),
        )
    }

    private fun tripoSR(bmp: Bitmap, report: Report): Pair<TripoSR.Result, TripoSR.Options> {
        val w = bmp.width
        val h = bmp.height
        val rgba = rgbaOf(bmp)
        val opt = TripoSR.Options().apply {
            resolution = s.resolution
            removeBackground = s.removeBackground
            vertexColors = s.vertexColors
            foregroundRatio = s.foregroundRatio
            threads = s.threadCount()
        }
        val pipe = TripoSR(models.dir).also { tripo = it }
        val res = pipe.run(rgba, w, h, opt) { stage, f -> report.stage(STAGES[stage] ?: stage.name, if (f > 0f && f < 1f) f else null) }
        return res to opt
    }

    /** Photo → shape with Hunyuan3D-2mini; background removal reuses TripoSR's u2net code with the pack's own u2net.onnx. */
    private fun hunyuan3d(bmp: Bitmap, report: Report): Pair<TripoSR.Result, TripoSR.Options> {
        val w = bmp.width
        val h = bmp.height
        var rgba = rgbaOf(bmp)
        val opt = TripoSR.Options().apply { threads = s.threadCount(); resolution = s.hunyuanOctree }
        val res = TripoSR.Result()
        val tp = TripoSR(hunyuanPack.dir).also { tripo = it } // only its u2net background removal, shipped in the Hunyuan pack
        if (s.removeBackground && (3 until rgba.size step 4).none { rgba[it] < 255 }) {
            report.stage("Removing background", null)
            rgba = tp.removeBackground(rgba, w, h, opt.threads)
        }
        val size = IntArray(1)
        res.foreground = com.smitnk.img23d.core.ImageOps.resizeForeground(rgba, w, h, s.foregroundRatio, size)
        res.foregroundSize = size[0]
        res.preprocessed = com.smitnk.img23d.core.ImageOps.resize(
            com.smitnk.img23d.core.ImageOps.compositeGray(res.foreground, size[0] * size[0]), size[0], size[0], 3, TripoSR.IMAGE_SIZE, TripoSR.IMAGE_SIZE)
        System.setProperty("hunyuan.arena", "false") // the ORT memory pool force-closed an 8 GB phone even for the shape steps only
        val hs = HunyuanShape.fromInfo(hunyuanPack.dir, File(hunyuanPack.dir, "hunyuan_info.json").readText(), opt.threads).also { hunyuan = it }
        res.mesh = hs.run(rgba, w, h, s.hunyuanOctree, System.nanoTime()) { stage, f -> report.stage(stage, if (f > 0f && f < 1f) f else null) }
        res.timingsMs.putAll(hs.timingsMs)
        return res to opt
    }

    private fun rgbaOf(bmp: Bitmap): IntArray {
        val argb = IntArray(bmp.width * bmp.height).also { bmp.getPixels(it, 0, bmp.width, 0, 0, bmp.width, bmp.height) }
        val alpha = bmp.hasAlpha()
        return IntArray(argb.size * 4).also { o ->
            for (i in argb.indices) {
                val p = argb[i]
                o[i * 4] = (p shr 16) and 255; o[i * 4 + 1] = (p shr 8) and 255; o[i * 4 + 2] = p and 255
                o[i * 4 + 3] = if (alpha) p ushr 24 else 255
            }
        }
    }

    private fun bitmapOf(rgba: IntArray, size: Int): Bitmap = Bitmap.createBitmap(
        IntArray(size * size) { i -> (rgba[i * 4 + 3] shl 24) or (rgba[i * 4] shl 16) or (rgba[i * 4 + 1] shl 8) or rgba[i * 4 + 2] },
        size, size, Bitmap.Config.ARGB_8888)

    private fun Bitmap.compressTo(f: File) = f.outputStream().use { compress(Bitmap.CompressFormat.PNG, 100, it) }

    private fun savePhoto(m: Model, res: TripoSR.Result) {
        if (res.foreground != null) bitmapOf(res.foreground, res.foregroundSize).compressTo(File(m.dir, "photo.png"))
    }

    private fun fail(title: String, msg: String) = _state.update { it.copy(failure = Failure(title, msg)) }

    private fun decode(uri: Uri): Bitmap {
        val cr = getApplication<Application>().contentResolver
        // shortcut: inputs are capped at MAX_SIDE px to bound heap use; TripoSR itself only sees 512x512.
        return if (Build.VERSION.SDK_INT >= 28) {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(cr, uri)) { d, info, _ ->
                d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val sz = info.size
                val scale = MAX_SIDE.toFloat() / maxOf(sz.width, sz.height)
                if (scale < 1f) d.setTargetSize((sz.width * scale).toInt().coerceAtLeast(1), (sz.height * scale).toInt().coerceAtLeast(1))
            }
        } else {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / sample > MAX_SIDE) sample *= 2
            cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample }) }
                ?: error("Unsupported image")
        }
    }

    private fun previewBitmap(img: FloatArray): Bitmap {
        val sz = TripoSR.IMAGE_SIZE
        val px = IntArray(sz * sz) { i ->
            val r = (img[i * 3] * 255f).toInt().coerceIn(0, 255)
            val g = (img[i * 3 + 1] * 255f).toInt().coerceIn(0, 255)
            val b = (img[i * 3 + 2] * 255f).toInt().coerceIn(0, 255)
            (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, sz, sz, Bitmap.Config.ARGB_8888)
    }

    override fun onCleared() = cancel()

    companion object {
        const val MAX_SIDE = 1536
        val PAINT_VIEWS = listOf("back", "left", "right")
        const val PAINT_STRENGTH = 0.35f // how far the AI may move from the model's own colours (texgen CI checks this value)
        val STAGES = mapOf(
            TripoSR.Stage.BACKGROUND to "Removing background",
            TripoSR.Stage.PREPROCESS to "Preparing image",
            TripoSR.Stage.ENCODER_LOAD to "Loading TripoSR weights",
            TripoSR.Stage.ENCODER to "Running TripoSR network",
            TripoSR.Stage.FIELD to "Evaluating density field",
            TripoSR.Stage.MARCHING_CUBES to "Marching cubes",
            TripoSR.Stage.COLORS to "Sampling vertex colours",
            TripoSR.Stage.DONE to "Saving",
        )
    }
}
