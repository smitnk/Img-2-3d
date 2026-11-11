package com.smitnk.img23d.ui

import android.text.format.DateUtils
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Switch
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.CloudDownload
import androidx.compose.material.icons.rounded.Collections
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material.icons.rounded.DeleteSweep
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material.icons.rounded.Visibility
import androidx.compose.material.icons.rounded.VisibilityOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.BuildConfig
import com.smitnk.img23d.Dest
import com.smitnk.img23d.UiState
import com.smitnk.img23d.blender.RuntimeStatus
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Settings
import com.smitnk.img23d.data.Source

@Composable
fun LibraryScreen(state: UiState, vm: AppViewModel) {
    LaunchedEffect(Unit) { vm.refreshLibrary() }
    val list = state.visibleModels
    LazyVerticalGrid(GridCells.Adaptive(150.dp), contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(14.dp), modifier = Modifier.fillMaxSize()) {
        item(span = { GridItemSpan(maxLineSpan) }) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Library", style = MaterialTheme.typography.displaySmall)
                Text("${state.models.size} models · ${state.storageMb} MB", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                OutlinedTextField(state.query, vm::setQuery, leadingIcon = { Icon(Icons.Rounded.Search, null) }, placeholder = { Text("Search models") },
                    singleLine = true, modifier = Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.extraLarge)
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    item { FilterChip(selected = state.filter == null, onClick = { vm.setFilter(null) }, label = { Text("All") }) }
                    items(Source.entries) { s ->
                        val n = state.models.count { it.source == s }
                        if (n > 0) FilterChip(selected = state.filter == s, onClick = { vm.setFilter(if (state.filter == s) null else s) }, label = { Text("${s.label} · $n") })
                    }
                }
            }
        }
        if (list.isEmpty()) item(span = { GridItemSpan(maxLineSpan) }) {
            Column(Modifier.fillMaxWidth().padding(vertical = 48.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Rounded.Collections, null, Modifier.size(56.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(8.dp))
                Text(if (state.models.isEmpty()) "Nothing here yet — create something in Studio." else "No models match", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        itemsIndexed(list, key = { _, m -> m.id }) { i, m ->
            Column(Modifier.staggerIn(i.coerceAtMost(8)).clip(MaterialTheme.shapes.large).clickable(onClickLabel = "Open ${m.name}") { vm.go(Dest.Result(m.id)) }) {
                ModelThumb(m, Modifier.fillMaxWidth().aspectRatio(1f))
                Text(m.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 6.dp))
                Text("${m.source.label} · ${DateUtils.getRelativeTimeSpanString(m.createdAt)}", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("%,d faces".format(m.faces), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.tertiary)
            }
        }
    }
}

@Composable
fun SettingsScreen(state: UiState, s: Settings, vm: AppViewModel, cb: AppCallbacks) {
    var confirmClear by remember { mutableStateOf(false) }
    var showKey by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Settings", style = MaterialTheme.typography.displaySmall)

        SectionHeader("Engines")
        EngineCard(
            title = "TripoSR · image to 3D", sub = "Original stabilityai/TripoSR weights (ONNX) + rembg u2net · ≈1.9 GB",
            ready = state.modelStatus == ModelStatus.Ready,
            progress = when (val m = state.modelStatus) {
                is ModelStatus.Downloading -> "Downloading ${m.file} · ${m.done shr 20} / ${m.total shr 20} MB" to (if (m.total > 0) m.done.toFloat() / m.total else null)
                is ModelStatus.Verifying -> "Verifying ${m.file}" to null
                else -> null
            },
            error = (state.modelStatus as? ModelStatus.Failed)?.message,
            onInstall = vm::downloadModels, onRemove = vm::deleteModels,
            onSave = { cb.pickFolder { vm.backupOne("models", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("models", it) } },
            extra = { if (state.modelStatus != ModelStatus.Ready) TextButton(onClick = vm::verifyPushedModels) { Text("Verify files copied with adb") } },
        )
        EngineCard(
            title = "Blender 4.2.3 LTS · procedural", sub = "Headless Blender in an ARM64 Linux userland via proot · ≈126 MB download",
            ready = state.runtime == RuntimeStatus.Ready,
            progress = when (val r = state.runtime) {
                is RuntimeStatus.Downloading -> "Downloading ${r.file} · ${r.done shr 20} / ${r.total shr 20} MB" to (if (r.total > 0) r.done.toFloat() / r.total else null)
                is RuntimeStatus.Verifying -> "Verifying ${r.file}" to null
                is RuntimeStatus.Extracting -> "Unpacking" to (if (r.total > 0) r.done.toFloat() / r.total else null)
                else -> null
            },
            error = (state.runtime as? RuntimeStatus.Failed)?.message ?: if (vm.runtime.needsUpdate()) "Engine update available (adds software GPU for scripts that render). Tap Install." else null,
            onInstall = vm::installRuntime, onRemove = vm::uninstallRuntime,
        )

        EngineCard(
            title = "Texture AI · paint texture", sub = "LCM Dreamshaper (SD1.5) int8 on ONNX Runtime, runs on the phone · ≈1.2 GB. " +
                "Use: open a model in Library → Paint texture",
            ready = state.texPack == com.smitnk.img23d.body.PackStatus.Ready,
            progress = (state.texPack as? com.smitnk.img23d.body.PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
            error = (state.texPack as? com.smitnk.img23d.body.PackStatus.Failed)?.message,
            onInstall = vm::installTexPack, onRemove = vm::uninstallTexPack,
            onSave = { cb.pickFolder { vm.backupOne("texgen", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("texgen", it) } },
            extra = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("AI paints the sides (off: photo on the front, TripoSR colours elsewhere)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    Switch(s.paintAi, { on -> vm.updateSettings { it.copy(paintAi = on) } })
                }
            },
        )

        EngineCard(
            title = "Hunyuan3D-2mini · detailed photo → 3D", sub = "Tencent Hunyuan3D-2mini Turbo on ONNX Runtime · ≈2 GB download. " +
                "Separate limbs, armour and cloth folds; slower than TripoSR (expect 15–30 min). No colours: use Paint texture after. " +
                "Licence: Tencent Hunyuan 3D 2.0 Community License (not for the EU, UK or South Korea).",
            ready = state.hunyuanPack == com.smitnk.img23d.body.PackStatus.Ready,
            progress = (state.hunyuanPack as? com.smitnk.img23d.body.PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
            error = (state.hunyuanPack as? com.smitnk.img23d.body.PackStatus.Failed)?.message
                ?: if (vm.hunyuanPack.needsUpdate()) "Update available (adds its own background removal, ~170 MB; no TripoSR needed). Tap Install." else null,
            onInstall = vm::installHunyuanPack, onRemove = vm::uninstallHunyuanPack,
            onSave = { cb.pickFolder { vm.backupOne("hunyuan", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("hunyuan", it) } },
            extra = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Use for Photo → 3D (also selectable on the photo screen)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    Switch(s.hunyuan, { on -> vm.updateSettings { it.copy(engine = if (on) "hunyuan" else "triposr") } })
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Detail: ${s.hunyuanOctree}³ — on = 256 (detailed), off = 128 (faster)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    Switch(s.hunyuanOctree >= 256, { on -> vm.updateSettings { it.copy(hunyuanOctree = if (on) 256 else 128) } })
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Fast mode — fewer surface checks, 4 threads (off = current quality path)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    Switch(s.hunyuanFast, { on -> vm.updateSettings { it.copy(hunyuanFast = on) } })
                }
            },
        )

        EngineCard(
            title = "Hunyuan3D-2mv · several photos → 3D", sub = "Tencent Hunyuan3D-2mv Turbo on ONNX Runtime · ≈3.6 GB download. " +
                "Give it a front photo plus back / left / right views of the same pose, so the hidden sides are real instead of guessed. " +
                "Twice the shape work of 2mini (expect 45–60 min with Fast mode). Uses the Detail and Fast mode switches of the 2mini card. " +
                "No colours: use Paint texture after. Licence: Tencent Hunyuan 3D 2.0 Community License (not for the EU, UK or South Korea).",
            ready = state.hunyuanMvPack == com.smitnk.img23d.body.PackStatus.Ready,
            progress = (state.hunyuanMvPack as? com.smitnk.img23d.body.PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
            error = (state.hunyuanMvPack as? com.smitnk.img23d.body.PackStatus.Failed)?.message,
            onInstall = vm::installHunyuanMvPack, onRemove = vm::uninstallHunyuanMvPack,
            onSave = { cb.pickFolder { vm.backupOne("hunyuanmv", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("hunyuanmv", it) } },
            extra = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Use for Photo → 3D (also selectable on the photo screen)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    Switch(s.hunyuanMv, { on -> vm.updateSettings { it.copy(engine = if (on) "hunyuanmv" else "triposr") } })
                }
            },
        )

        EngineCard(
            title = "TRELLIS.2 · best shape, slowest", sub = "Microsoft TRELLIS.2 (4B) on trellis.cpp, 4-bit weights, CPU · ≈2.8 GB download. " +
                "Needs the Blender engine installed (it runs inside its Linux runtime). Expect 1–2 hours per model on an 8 GB phone; " +
                "keep the app open and the phone charging. Shape only: use Paint texture after.",
            ready = state.trellisPack == com.smitnk.img23d.body.PackStatus.Ready,
            progress = (state.trellisPack as? com.smitnk.img23d.body.PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
            error = (state.trellisPack as? com.smitnk.img23d.body.PackStatus.Failed)?.message,
            onInstall = vm::installTrellisPack, onRemove = vm::uninstallTrellisPack,
            onSave = { cb.pickFolder { vm.backupOne("trellis", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("trellis", it) } },
            extra = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Use for Photo → 3D (also selectable on the photo screen)", Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    Switch(s.trellis, { on -> vm.updateSettings { it.copy(engine = if (on) "trellis" else "triposr") } })
                }
            },
        )

        SectionHeader("Engine packs on your storage")
        Panel {
            Text("Save all installed packs at once (each engine card above also has its own Save / Load). Pick a folder you choose, e.g. Download/Img23d. " +
                "After reinstalling the app, restore them from that folder instead of downloading again.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            state.packFolder?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { cb.pickFolder(vm::backupPacks) }, enabled = state.packFolder == null, modifier = Modifier.weight(1f)) { Text("Back up to folder") }
                OutlinedButton(onClick = { cb.pickFolder(vm::restorePacks) }, enabled = state.packFolder == null, modifier = Modifier.weight(1f)) { Text("Restore from folder") }
            }
        }

        SectionHeader("Photo → 3D defaults")
        Panel {
            ResolutionPicker(s.resolution) { r -> vm.updateSettings { it.copy(resolution = r) } }
            ToggleRow("Remove background", null, s.removeBackground) { v -> vm.updateSettings { it.copy(removeBackground = v) } }
            ToggleRow("Vertex colours", null, s.vertexColors) { v -> vm.updateSettings { it.copy(vertexColors = v) } }
        }

        SectionHeader("Blender output")
        Panel {
            ToggleRow("Armature rig", "Builder, presets: 17 bones + deformation self-test", s.rig) { v -> vm.updateSettings { it.copy(rig = v) } }
            ToggleRow("Also save .blend", "Editable Blender file next to every GLB", s.saveBlend) { v -> vm.updateSettings { it.copy(saveBlend = v) } }
            LabeledSlider("Refine height (m)", s.refineHeight, 0.3f..2.2f) { v -> vm.updateSettings { it.copy(refineHeight = v) } }
            LabeledSlider("Refine remesh voxel", s.refineVoxel, 0.004f..0.03f, "%.3f") { v -> vm.updateSettings { it.copy(refineVoxel = v) } }
        }

        SectionHeader("Performance")
        Panel {
            LabeledSlider("CPU threads (0 = auto: ${Settings().copy(threads = 0).threadCount()})", s.threads.toFloat(), 0f..8f, "%.0f", steps = 7) { v ->
                vm.updateSettings { it.copy(threads = v.toInt()) }
            }
            ToggleRow("Keep screen on while generating", null, s.keepScreenOn) { v -> vm.updateSettings { it.copy(keepScreenOn = v) } }
        }

        SectionHeader("Text to 3D (optional)")
        Panel {
            Text("Uses Claude to write the Blender script; the model is still built on your phone. Your key is stored only on this device.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedTextField(s.apiKey, { v -> vm.updateSettings { it.copy(apiKey = v.trim()) } }, label = { Text("Anthropic API key") }, singleLine = true,
                visualTransformation = if (showKey) VisualTransformation.None else PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                trailingIcon = { IconButton(onClick = { showKey = !showKey }) { Icon(if (showKey) Icons.Rounded.VisibilityOff else Icons.Rounded.Visibility, if (showKey) "Hide key" else "Show key") } },
                modifier = Modifier.fillMaxWidth())
            OutlinedTextField(s.apiModel, { v -> vm.updateSettings { it.copy(apiModel = v.trim()) } }, label = { Text("Model") }, singleLine = true, modifier = Modifier.fillMaxWidth())
        }

        SectionHeader("Appearance")
        Panel {
            ToggleRow("Wallpaper colours", "Material You dynamic colour instead of the Studio palette", s.dynamicColor) { v -> vm.updateSettings { it.copy(dynamicColor = v) } }
            ToggleRow("Turntable in viewer", "Models slowly spin when opened", s.autoRotate) { v -> vm.updateSettings { it.copy(autoRotate = v) } }
            ToggleRow("Reduce motion", "Disable entrance and loading animations", s.reduceMotion) { v -> vm.updateSettings { it.copy(reduceMotion = v) } }
        }

        SectionHeader("Storage")
        Panel {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.Memory, null, tint = MaterialTheme.colorScheme.tertiary)
                Spacer(Modifier.width(10.dp))
                Text("Library: ${state.models.size} models, ${state.storageMb} MB", Modifier.weight(1f))
            }
            OutlinedButton(onClick = { confirmClear = true }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp), enabled = state.models.isNotEmpty()) {
                Icon(Icons.Rounded.DeleteSweep, null); Spacer(Modifier.width(8.dp)); Text("Delete all models")
            }
        }

        SectionHeader("About")
        Panel {
            Text("Img-2-3D Studio ${BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.titleSmall)
            Text("System B: TripoSR (MIT) on ONNX Runtime. System A: Blender 4.2.3 LTS (GPL) running Model Forge generators. Both run on-device.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            TextButton(onClick = { cb.copy("Script rules", vm.scriptContract()) }) {
                Icon(Icons.Rounded.ContentCopy, null); Spacer(Modifier.width(6.dp)); Text("Copy AI script rules")
            }
        }
        Spacer(Modifier.height(24.dp))
    }
    if (confirmClear) AlertDialog(
        onDismissRequest = { confirmClear = false }, title = { Text("Delete all ${state.models.size} models?") },
        text = { Text("This cannot be undone. Engines stay installed.") },
        confirmButton = { TextButton(onClick = { confirmClear = false; vm.clearLibrary() }) { Text("Delete all", color = MaterialTheme.colorScheme.error) } },
        dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } },
    )
}

@Composable
private fun EngineCard(
    title: String, sub: String, ready: Boolean, progress: Pair<String, Float?>?, error: String?,
    onInstall: () -> Unit, onRemove: () -> Unit, extra: @Composable () -> Unit = {},
    onSave: (() -> Unit)? = null, onLoad: (() -> Unit)? = null,
) {
    var confirm by remember { mutableStateOf(false) }
    Panel {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(sub, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (ready) Icon(Icons.Rounded.CheckCircle, "Installed", tint = MaterialTheme.colorScheme.tertiary)
        }
        if (progress != null) {
            Text(progress.first, style = MaterialTheme.typography.labelSmall)
            val f = progress.second
            if (f != null) LinearProgressIndicator(progress = { f }, modifier = Modifier.fillMaxWidth()) else LinearProgressIndicator(Modifier.fillMaxWidth())
        }
        if (error != null) Text(error, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        if (progress == null) {
            if (!ready) Button(onClick = onInstall, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.CloudDownload, null); Spacer(Modifier.width(8.dp)); Text("Install")
            } else OutlinedButton(onClick = { confirm = true }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Remove") }
        }
        extra()
        if (onSave != null || onLoad != null) Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (onSave != null && ready) OutlinedButton(onClick = onSave, modifier = Modifier.weight(1f)) { Text("Save to folder") }
            if (onLoad != null && !ready) OutlinedButton(onClick = onLoad, modifier = Modifier.weight(1f)) { Text("Load from folder") }
        }
    }
    if (confirm) AlertDialog(
        onDismissRequest = { confirm = false }, title = { Text("Remove engine?") },
        text = { Text("You can reinstall it later; your models stay.") },
        confirmButton = { TextButton(onClick = { confirm = false; onRemove() }) { Text("Remove") } },
        dismissButton = { TextButton(onClick = { confirm = false }) { Text("Cancel") } },
    )
}
