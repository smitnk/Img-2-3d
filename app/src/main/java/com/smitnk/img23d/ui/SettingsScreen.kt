package com.smitnk.img23d.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.background
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AutoFixHigh
import androidx.compose.material.icons.rounded.Brush
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.CloudDownload
import androidx.compose.material.icons.rounded.Construction
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material.icons.rounded.DeleteOutline
import androidx.compose.material.icons.rounded.DeleteSweep
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.ExpandLess
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.Layers
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.MoreVert
import androidx.compose.material.icons.rounded.PhotoLibrary
import androidx.compose.material.icons.rounded.RadioButtonUnchecked
import androidx.compose.material.icons.rounded.Save
import androidx.compose.material.icons.rounded.Sync
import androidx.compose.material.icons.rounded.ViewInAr
import androidx.compose.material.icons.rounded.Visibility
import androidx.compose.material.icons.rounded.VisibilityOff
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.BuildConfig
import com.smitnk.img23d.UiState
import com.smitnk.img23d.blender.RuntimeStatus
import com.smitnk.img23d.body.PackStatus
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Settings

@Composable
fun SettingsScreen(state: UiState, s: Settings, vm: AppViewModel, cb: AppCallbacks) {
    var confirmClear by remember { mutableStateOf(false) }
    var showKey by remember { mutableStateOf(false) }
    val cs = MaterialTheme.colorScheme
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
        Column(
            Modifier.widthIn(max = Space.maxContent).fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = Space.lg, vertical = Space.lg),
            verticalArrangement = Arrangement.spacedBy(Space.md),
        ) {
            Text("Settings", style = MaterialTheme.typography.displaySmall, modifier = Modifier.semantics { heading() })

            SectionHeader("3D generation")
            EngineCard(
                icon = Icons.Rounded.ViewInAr, title = "TripoSR", summary = "Fast photo to 3D mesh", size = "≈1.9 GB",
                details = "Original stabilityai/TripoSR weights (ONNX) + rembg u2net background removal. Runs fully on the phone.",
                ready = state.modelStatus == ModelStatus.Ready,
                progress = when (val m = state.modelStatus) {
                    is ModelStatus.Downloading -> "Downloading ${m.file} · ${m.done shr 20} / ${m.total shr 20} MB" to (if (m.total > 0) m.done.toFloat() / m.total else null)
                    is ModelStatus.Verifying -> "Verifying ${m.file}" to null
                    else -> null
                },
                error = (state.modelStatus as? ModelStatus.Failed)?.message, failed = state.modelStatus is ModelStatus.Failed,
                onInstall = vm::downloadModels, onRemove = vm::deleteModels,
                onSave = { cb.pickFolder { vm.backupOne("models", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("models", it) } },
                extra = { if (state.modelStatus != ModelStatus.Ready) TextButton(onClick = vm::verifyPushedModels, modifier = Modifier.padding(horizontal = Space.sm).heightIn(min = 48.dp)) { Text("Verify files copied with adb") } },
            )
            EngineCard(
                icon = Icons.Rounded.Layers, title = "Hunyuan3D-2mini", summary = "Detailed photo to 3D", size = "≈2 GB",
                details = "Tencent Hunyuan3D-2mini Turbo on ONNX Runtime. Separate limbs, armour and cloth folds; slower than TripoSR (expect 15–30 min). " +
                    "No colours: use Paint texture after.\nLicence: Tencent Hunyuan 3D 2.0 Community License (not for the EU, UK or South Korea).",
                ready = state.hunyuanPack == PackStatus.Ready,
                progress = (state.hunyuanPack as? PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
                error = (state.hunyuanPack as? PackStatus.Failed)?.message
                    ?: if (vm.hunyuanPack.needsUpdate()) "Update available (adds its own background removal, ~170 MB; no TripoSR needed). Tap Install." else null,
                failed = state.hunyuanPack is PackStatus.Failed,
                onInstall = vm::installHunyuanPack, onRemove = vm::uninstallHunyuanPack,
                onSave = { cb.pickFolder { vm.backupOne("hunyuan", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("hunyuan", it) } },
                extra = {
                    ToggleRow("Use for Photo → 3D", "Also selectable on the photo screen", s.hunyuan) { on -> vm.updateSettings { it.copy(engine = if (on) "hunyuan" else "triposr") } }
                    ToggleRow("High detail (256³)", "Off = 128³, faster. Shared with 2mv", s.hunyuanOctree >= 256) { on -> vm.updateSettings { it.copy(hunyuanOctree = if (on) 256 else 128) } }
                    ToggleRow("Fast mode", "Fewer surface checks, 4 threads. Off = current quality path", s.hunyuanFast) { on -> vm.updateSettings { it.copy(hunyuanFast = on) } }
                },
            )
            EngineCard(
                icon = Icons.Rounded.PhotoLibrary, title = "Hunyuan3D-2mv", summary = "Several photos to 3D", size = "≈3.6 GB",
                details = "Tencent Hunyuan3D-2mv Turbo on ONNX Runtime. Give it a front photo plus back / left / right views of the same pose, so the hidden sides are real instead of guessed. " +
                    "Twice the shape work of 2mini (expect 45–60 min with Fast mode). Uses the High detail and Fast mode switches of the 2mini card. No colours: use Paint texture after.\n" +
                    "Licence: Tencent Hunyuan 3D 2.0 Community License (not for the EU, UK or South Korea).",
                ready = state.hunyuanMvPack == PackStatus.Ready,
                progress = (state.hunyuanMvPack as? PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
                error = (state.hunyuanMvPack as? PackStatus.Failed)?.message, failed = state.hunyuanMvPack is PackStatus.Failed,
                onInstall = vm::installHunyuanMvPack, onRemove = vm::uninstallHunyuanMvPack,
                onSave = { cb.pickFolder { vm.backupOne("hunyuanmv", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("hunyuanmv", it) } },
                extra = {
                    ToggleRow("Use for Photo → 3D", "Also selectable on the photo screen", s.hunyuanMv) { on -> vm.updateSettings { it.copy(engine = if (on) "hunyuanmv" else "triposr") } }
                },
            )

            SectionHeader("Photo")
            EngineCard(
                icon = Icons.Rounded.AutoFixHigh, title = "Enhance photo", summary = "Sharper, cleaner photo before 3D", size = "≈9 MB",
                details = "Real-ESRGAN x4plus-anime on ncnn (BSD-3). Uses the phone's GPU (Vulkan) and falls back to the CPU with smaller tiles. " +
                    "When on, the photo (and every extra view for 2mv) is upscaled 4x and scaled back down to the size the 3D engine reads, before TripoSR or Hunyuan runs.",
                ready = state.esrganPack == PackStatus.Ready,
                progress = (state.esrganPack as? PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
                error = (state.esrganPack as? PackStatus.Failed)?.message, failed = state.esrganPack is PackStatus.Failed,
                onInstall = vm::installEsrganPack, onRemove = vm::uninstallEsrganPack,
                onSave = { cb.pickFolder { vm.backupOne("esrgan", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("esrgan", it) } },
                extra = {
                    ToggleRow("Enhance photo before 3D", "Also on the photo screen. Slower when the phone has no usable GPU", s.enhancePhoto) { on -> vm.updateSettings { it.copy(enhancePhoto = on) } }
                },
            )

            SectionHeader("Texture")
            EngineCard(
                icon = Icons.Rounded.Brush, title = "Texture AI", summary = "Paint textures on your models", size = "≈1.2 GB",
                details = "LCM Dreamshaper (SD1.5) int8 on ONNX Runtime, runs on the phone. Use: open a model in Library → Paint texture.",
                ready = state.texPack == PackStatus.Ready,
                progress = (state.texPack as? PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
                error = (state.texPack as? PackStatus.Failed)?.message, failed = state.texPack is PackStatus.Failed,
                onInstall = vm::installTexPack, onRemove = vm::uninstallTexPack,
                onSave = { cb.pickFolder { vm.backupOne("texgen", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("texgen", it) } },
                extra = {
                    ToggleRow("AI paints the sides", "Off: photo on the front, TripoSR colours elsewhere", s.paintAi) { on -> vm.updateSettings { it.copy(paintAi = on) } }
                },
            )

            EngineCard(
                icon = Icons.Rounded.Brush, title = "Hunyuan Paint", summary = "Paints all sides from your photos", size = "≈2.0 GB",
                details = "Tencent Hunyuan3D-2 Paint (turbo), int8. Needs about 2 GB free RAM, 15–40 min. Licence: Tencent Hunyuan 3D 2.0 Community License (not for EU, UK or South Korea).",
                ready = state.hunyuanPaintPack == PackStatus.Ready,
                progress = (state.hunyuanPaintPack as? PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
                error = (state.hunyuanPaintPack as? PackStatus.Failed)?.message, failed = state.hunyuanPaintPack is PackStatus.Failed,
                onInstall = vm::installHunyuanPaintPack, onRemove = vm::uninstallHunyuanPaintPack,
                onSave = { cb.pickFolder { vm.backupOne("hunyuanpaint", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("hunyuanpaint", it) } },
            )

            SectionHeader("Rigging")
            EngineCard(
                icon = Icons.Rounded.AutoFixHigh, title = "UniRig", summary = "Auto-rig: skeleton + skin weights", size = "≈2.5 GB",
                details = "MIT licence. Runs inside Blender on the phone. Needs about 2 GB free RAM.",
                ready = state.unirigPack == PackStatus.Ready,
                progress = (state.unirigPack as? PackStatus.Downloading)?.let { "Downloading ${it.done shr 20} / ${it.total shr 20} MB" to (if (it.total > 0) it.done.toFloat() / it.total else null) },
                error = (state.unirigPack as? PackStatus.Failed)?.message, failed = state.unirigPack is PackStatus.Failed,
                onInstall = vm::installUnirigPack, onRemove = vm::uninstallUnirigPack,
                onSave = { cb.pickFolder { vm.backupOne("unirig", it) } }, onLoad = { cb.pickFolder { vm.restoreOne("unirig", it) } },
            )

            SectionHeader("Runtime")
            EngineCard(
                icon = Icons.Rounded.Construction, title = "Blender 4.2.3 LTS", summary = "Procedural models, rigs and clean-up", size = "≈126 MB",
                details = "Headless Blender in an ARM64 Linux userland via proot.",
                ready = state.runtime == RuntimeStatus.Ready,
                progress = when (val r = state.runtime) {
                    is RuntimeStatus.Downloading -> "Downloading ${r.file} · ${r.done shr 20} / ${r.total shr 20} MB" to (if (r.total > 0) r.done.toFloat() / r.total else null)
                    is RuntimeStatus.Verifying -> "Verifying ${r.file}" to null
                    is RuntimeStatus.Extracting -> "Unpacking" to (if (r.total > 0) r.done.toFloat() / r.total else null)
                    else -> null
                },
                error = (state.runtime as? RuntimeStatus.Failed)?.message ?: if (vm.runtime.needsUpdate()) "Engine update available (adds software GPU for scripts that render). Tap Install." else null,
                failed = state.runtime is RuntimeStatus.Failed,
                onInstall = vm::installRuntime, onRemove = vm::uninstallRuntime,
            )

            SectionHeader("Backup")
            Panel {
                Text("Save every installed pack to a folder you choose (e.g. Download/Img23d), then restore from it after reinstalling the app instead of downloading again.",
                    style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant)
                state.packFolder?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
                Row(horizontalArrangement = Arrangement.spacedBy(Space.sm)) {
                    FilledTonalButton(onClick = { cb.pickFolder(vm::backupPacks) }, enabled = state.packFolder == null, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) {
                        Icon(Icons.Rounded.Save, null, Modifier.size(18.dp)); Spacer(Modifier.width(Space.sm)); Text("Back up")
                    }
                    FilledTonalButton(onClick = { cb.pickFolder(vm::restorePacks) }, enabled = state.packFolder == null, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) {
                        Icon(Icons.Rounded.FolderOpen, null, Modifier.size(18.dp)); Spacer(Modifier.width(Space.sm)); Text("Restore")
                    }
                }
            }

            SectionHeader("Photo → 3D defaults")
            Panel(flush = true) {
                Box(Modifier.panelPad()) { ResolutionPicker(s.resolution) { r -> vm.updateSettings { it.copy(resolution = r) } } }
                ToggleRow("Remove background", "Cut the subject out of the photo first", s.removeBackground) { v -> vm.updateSettings { it.copy(removeBackground = v) } }
                ToggleRow("Vertex colours", "Keep colour per vertex on the mesh", s.vertexColors) { v -> vm.updateSettings { it.copy(vertexColors = v) } }
            }

            SectionHeader("Blender output")
            Panel(flush = true) {
                ToggleRow("Armature rig", "Builder, presets: 17 bones + deformation self-test", s.rig) { v -> vm.updateSettings { it.copy(rig = v) } }
                ToggleRow("Also save .blend", "Editable Blender file next to every GLB", s.saveBlend) { v -> vm.updateSettings { it.copy(saveBlend = v) } }
                Box(Modifier.panelPad()) { LabeledSlider("Refine height (m)", s.refineHeight, 0.3f..2.2f) { v -> vm.updateSettings { it.copy(refineHeight = v) } } }
                Box(Modifier.panelPad()) { LabeledSlider("Refine remesh voxel", s.refineVoxel, 0.004f..0.03f, "%.3f") { v -> vm.updateSettings { it.copy(refineVoxel = v) } } }
            }

            SectionHeader("Performance")
            Panel(flush = true) {
                Box(Modifier.panelPad()) {
                    LabeledSlider("CPU threads (0 = auto: ${Settings().copy(threads = 0).threadCount()})", s.threads.toFloat(), 0f..8f, "%.0f", steps = 7) { v ->
                        vm.updateSettings { it.copy(threads = v.toInt()) }
                    }
                }
                ToggleRow("Keep screen on while generating", null, s.keepScreenOn) { v -> vm.updateSettings { it.copy(keepScreenOn = v) } }
            }

            SectionHeader("Text to 3D (optional)")
            Panel {
                Text("Uses Claude to write the Blender script; the model is still built on your phone. Your key is stored only on this device.",
                    style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant)
                OutlinedTextField(s.apiKey, { v -> vm.updateSettings { it.copy(apiKey = v.trim()) } }, label = { Text("Anthropic API key") }, singleLine = true,
                    visualTransformation = if (showKey) VisualTransformation.None else PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    trailingIcon = { IconButton(onClick = { showKey = !showKey }) { Icon(if (showKey) Icons.Rounded.VisibilityOff else Icons.Rounded.Visibility, if (showKey) "Hide key" else "Show key") } },
                    modifier = Modifier.fillMaxWidth())
                OutlinedTextField(s.apiModel, { v -> vm.updateSettings { it.copy(apiModel = v.trim()) } }, label = { Text("Model") }, singleLine = true, modifier = Modifier.fillMaxWidth())
            }

            SectionHeader("Appearance")
            Panel(flush = true) {
                ToggleRow("Wallpaper colours", "Material You dynamic colour instead of the Studio palette", s.dynamicColor) { v -> vm.updateSettings { it.copy(dynamicColor = v) } }
                ToggleRow("Turntable in viewer", "Models slowly spin when opened", s.autoRotate) { v -> vm.updateSettings { it.copy(autoRotate = v) } }
                ToggleRow("Reduce motion", "Disable entrance and loading animations", s.reduceMotion) { v -> vm.updateSettings { it.copy(reduceMotion = v) } }
            }

            SectionHeader("Storage")
            Panel {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.Memory, null, tint = cs.tertiary)
                    Spacer(Modifier.width(10.dp))
                    Text("Library: ${state.models.size} models, ${state.storageMb} MB", Modifier.weight(1f))
                }
                OutlinedButton(onClick = { confirmClear = true }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp), enabled = state.models.isNotEmpty()) {
                    Icon(Icons.Rounded.DeleteSweep, null); Spacer(Modifier.width(Space.sm)); Text("Delete all models")
                }
            }

            SectionHeader("About")
            Panel {
                Text("Img-2-3D Studio ${BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.titleSmall)
                Text("System B: TripoSR (MIT) on ONNX Runtime. System A: Blender 4.2.3 LTS (GPL) running Model Forge generators. Both run on-device.",
                    style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant)
                TextButton(onClick = { cb.copy("Script rules", vm.scriptContract()) }, modifier = Modifier.heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.ContentCopy, null); Spacer(Modifier.width(6.dp)); Text("Copy AI script rules")
                }
            }
            Spacer(Modifier.height(Space.xl))
        }
    }
    if (confirmClear) AlertDialog(
        onDismissRequest = { confirmClear = false }, title = { Text("Delete all ${state.models.size} models?") },
        text = { Text("This cannot be undone. Engines stay installed.") },
        confirmButton = { TextButton(onClick = { confirmClear = false; vm.clearLibrary() }) { Text("Delete all", color = MaterialTheme.colorScheme.error) } },
        dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } },
    )
}

/**
 * Compact engine card: icon, name, one-line summary, size + status chips, then one action row
 * (Install / Retry as the primary action, overflow menu for Remove / Save / Load, "Details" for the long text).
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun EngineCard(
    icon: ImageVector, title: String, summary: String, size: String, details: String?,
    ready: Boolean, progress: Pair<String, Float?>?, error: String?, failed: Boolean,
    onInstall: () -> Unit, onRemove: () -> Unit, extra: @Composable () -> Unit = {},
    onSave: (() -> Unit)? = null, onLoad: (() -> Unit)? = null,
) {
    val cs = MaterialTheme.colorScheme
    var confirm by remember { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }
    var showDetails by remember { mutableStateOf(false) }
    val busy = progress != null
    val statusText = when {
        busy -> progress!!.first.substringBefore(' ') + (progress.second?.let { " ${(it * 100).toInt()}%" } ?: "")
        failed -> "Failed"
        ready -> "Installed"
        else -> "Not installed"
    }
    val statusIcon = when { busy -> Icons.Rounded.Sync; failed -> Icons.Rounded.ErrorOutline; ready -> Icons.Rounded.CheckCircle; else -> Icons.Rounded.RadioButtonUnchecked }
    val statusFg = when { busy -> cs.onPrimaryContainer; failed -> cs.onErrorContainer; ready -> cs.onTertiaryContainer; else -> cs.onSurfaceVariant }
    val statusBg = when { busy -> cs.primaryContainer; failed -> cs.errorContainer; ready -> cs.tertiaryContainer; else -> cs.surfaceContainerHigh }
    val canSave = onSave != null && ready
    val canLoad = onLoad != null && !ready
    val hasMenu = ready || canLoad

    Panel(flush = true) {
        Column(Modifier.padding(horizontal = Space.lg, vertical = Space.sm), verticalArrangement = Arrangement.spacedBy(Space.md)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(44.dp).clip(RoundedCornerShape(12.dp)).background(cs.primaryContainer), contentAlignment = Alignment.Center) {
                    Icon(icon, null, Modifier.size(24.dp), tint = cs.onPrimaryContainer)
                }
                Spacer(Modifier.width(Space.md))
                Column(Modifier.weight(1f)) {
                    Text(title, style = MaterialTheme.typography.titleMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
                    Text(summary, style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(Space.sm), verticalArrangement = Arrangement.spacedBy(Space.xs)) {
                InfoChip(size)
                InfoChip(statusText, icon = statusIcon, tint = statusFg, container = statusBg, modifier = Modifier.semantics { contentDescription = "Status: $statusText" })
            }
            if (progress != null) {
                Text(progress.first, style = MaterialTheme.typography.labelSmall, color = cs.onSurfaceVariant)
                val f = progress.second
                if (f != null) LinearProgressIndicator(progress = { f }, modifier = Modifier.fillMaxWidth()) else LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            if (error != null) Text(error, style = MaterialTheme.typography.bodySmall, color = if (failed) cs.error else cs.tertiary)
            if (showDetails && details != null) Text(details, style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant)
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (details != null) TextButton(onClick = { showDetails = !showDetails }, modifier = Modifier.heightIn(min = 48.dp)) {
                    Text("Details"); Icon(if (showDetails) Icons.Rounded.ExpandLess else Icons.Rounded.ExpandMore, if (showDetails) "Hide details" else "Show details", Modifier.size(18.dp))
                }
                Spacer(Modifier.weight(1f))
                if (!ready && !busy) Button(onClick = onInstall, modifier = Modifier.heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.CloudDownload, null, Modifier.size(18.dp)); Spacer(Modifier.width(Space.sm)); Text(if (failed) "Retry" else "Install")
                }
                if (hasMenu) Box {
                    IconButton(onClick = { menu = true }) { Icon(Icons.Rounded.MoreVert, "More actions for $title") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        if (canSave) DropdownMenuItem(text = { Text("Save to folder") }, leadingIcon = { Icon(Icons.Rounded.Save, null) }, onClick = { menu = false; onSave?.invoke() })
                        if (canLoad) DropdownMenuItem(text = { Text("Load from folder") }, leadingIcon = { Icon(Icons.Rounded.FolderOpen, null) }, onClick = { menu = false; onLoad?.invoke() })
                        if (ready) DropdownMenuItem(
                            text = { Text("Remove", color = cs.error) }, leadingIcon = { Icon(Icons.Rounded.DeleteOutline, null, tint = cs.error) },
                            enabled = !busy, onClick = { menu = false; confirm = true },
                        )
                    }
                }
            }
        }
        extra()
    }
    if (confirm) AlertDialog(
        onDismissRequest = { confirm = false }, title = { Text("Remove $title?") },
        text = { Text("You can reinstall it later; your models stay.") },
        confirmButton = { TextButton(onClick = { confirm = false; onRemove() }) { Text("Remove") } },
        dismissButton = { TextButton(onClick = { confirm = false }) { Text("Cancel") } },
    )
}
