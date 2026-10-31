package com.smitnk.img23d.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AddPhotoAlternate
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Casino
import androidx.compose.material.icons.rounded.Code
import androidx.compose.material.icons.rounded.ContentCopy
import androidx.compose.material.icons.rounded.FileOpen
import androidx.compose.material.icons.rounded.Key
import androidx.compose.material.icons.rounded.PhotoLibrary
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.RestartAlt
import androidx.compose.material.icons.rounded.Widgets
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryScrollableTabRow
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.Dest
import com.smitnk.img23d.PhotoMode
import com.smitnk.img23d.UiState
import com.smitnk.img23d.blender.Builder
import com.smitnk.img23d.blender.Preset
import com.smitnk.img23d.blender.RuntimeStatus
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Settings
import com.smitnk.img23d.data.Source

/** Scrollable form with a floating primary action that never covers the last field. */
@Composable
private fun FormWithAction(
    actionText: String, actionIcon: androidx.compose.ui.graphics.vector.ImageVector, enabled: Boolean, onAction: () -> Unit,
    content: @Composable () -> Unit,
) {
    Box(Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            content()
            Spacer(Modifier.height(88.dp))
        }
        ExtendedFloatingActionButton(
            onClick = { if (enabled) onAction() }, icon = { Icon(actionIcon, null) }, text = { Text(actionText) },
            containerColor = if (enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerHighest,
            contentColor = if (enabled) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
        )
    }
}

@Composable
private fun EngineWarning(text: String, onFix: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer), shape = MaterialTheme.shapes.large, onClick = onFix) {
        Text(text, Modifier.padding(16.dp), color = MaterialTheme.colorScheme.onErrorContainer, style = MaterialTheme.typography.bodyMedium)
    }
}

// ---------------------------------------------------------------- Photo (System B, optionally + A)

@Composable
fun PhotoScreen(mode: PhotoMode, state: UiState, settings: Settings, vm: AppViewModel, cb: AppCallbacks) {
    val character = mode == PhotoMode.CHARACTER
    val tripoReady = state.modelStatus == ModelStatus.Ready
    val blenderReady = state.runtime == RuntimeStatus.Ready
    val ready = tripoReady && (!character || blenderReady)
    FormWithAction(if (character) "Make character" else "Generate 3D", Icons.Rounded.AutoAwesome, ready && state.image != null, { vm.runPhoto(mode) }) {
        if (!tripoReady) EngineWarning("TripoSR model is not installed. Tap to open Settings → Engines.") { vm.tab(Dest.Settings) }
        if (character && !blenderReady) EngineWarning("Blender engine is not installed. Tap to open Settings → Engines.") { vm.tab(Dest.Settings) }
        val img = state.image
        Box(
            Modifier.fillMaxWidth().aspectRatio(1f).clip(MaterialTheme.shapes.extraLarge).background(MaterialTheme.colorScheme.surfaceContainer),
            contentAlignment = Alignment.Center,
        ) {
            if (img != null) Image(img.asImageBitmap(), "Selected photo", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            else Icon(Icons.Rounded.AddPhotoAlternate, null, Modifier.size(64.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        OutlinedButton(onClick = { cb.pickImage(mode) }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
            Icon(Icons.Rounded.PhotoLibrary, null); Spacer(Modifier.width(8.dp)); Text(if (img == null) "Choose photo" else "Change photo")
        }
        PipelineSteps(character)
        Panel {
            SectionHeader("TripoSR")
            ResolutionPicker(settings.resolution) { r -> vm.updateSettings { it.copy(resolution = r) } }
            ToggleRow("Remove background", "rembg u2net; skipped for transparent images", settings.removeBackground) { v -> vm.updateSettings { it.copy(removeBackground = v) } }
            ToggleRow("Vertex colours", "Colour from TripoSR's field", settings.vertexColors) { v -> vm.updateSettings { it.copy(vertexColors = v) } }
            LabeledSlider("Foreground ratio", settings.foregroundRatio, 0.5f..1f) { v -> vm.updateSettings { it.copy(foregroundRatio = v) } }
        }
        if (character) Panel {
            SectionHeader("Blender refine")
            LabeledSlider("Height (m)", settings.refineHeight, 0.3f..2.2f) { v -> vm.updateSettings { it.copy(refineHeight = v) } }
            LabeledSlider("Remesh detail (smaller = finer)", settings.refineVoxel, 0.004f..0.03f, "%.3f") { v -> vm.updateSettings { it.copy(refineVoxel = v) } }
            LabeledSlider("Smoothing", settings.refineSmooth.toFloat(), 0f..12f, "%.0f", steps = 11) { v -> vm.updateSettings { it.copy(refineSmooth = v.toInt()) } }
            LabeledSlider("Keep faces", settings.refineDecimate, 0.05f..1f, "%.2f") { v -> vm.updateSettings { it.copy(refineDecimate = v) } }
            ToggleRow("Auto-rig (experimental)", "17-bone humanoid; best when the photo is a T/A-pose", settings.refineRig) { v -> vm.updateSettings { it.copy(refineRig = v) } }
        }
    }
}

@Composable
private fun PipelineSteps(character: Boolean) {
    val steps = if (character) listOf("Photo", "TripoSR mesh", "Blender remesh + colours", "GLB") else listOf("Photo", "TripoSR mesh", "GLB")
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
        steps.forEachIndexed { i, s ->
            Pill(null, s)
            if (i < steps.lastIndex) Text("→", color = MaterialTheme.colorScheme.primary)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ResolutionPicker(value: Int, onPick: (Int) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Mesh resolution", style = MaterialTheme.typography.titleSmall)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(128 to "Fast", 192 to "Balanced", 256 to "Detailed").forEach { (r, l) ->
                FilterChip(selected = value == r, onClick = { onPick(r) }, label = { Text("$r³ · $l") })
            }
        }
    }
}

// ---------------------------------------------------------------- Character builder (System A)

@Composable
fun BuilderScreen(state: UiState, vm: AppViewModel) {
    val spec = state.builder
    var tab by remember { mutableIntStateOf(0) }
    val tabs = listOf("Body", "Limbs", "Face", "Style", "Colours")
    Column(Modifier.fillMaxSize()) {
        Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(spec.name, { v -> vm.editBuilder { it.copy(name = v.take(40)) } }, label = { Text("Name") }, singleLine = true, modifier = Modifier.weight(1f))
            FilledTonalButton(onClick = { vm.editBuilder { it.randomized() } }, modifier = Modifier.heightIn(min = 56.dp)) {
                Icon(Icons.Rounded.Casino, "Randomize character")
            }
            OutlinedButton(onClick = { vm.editBuilder { com.smitnk.img23d.blender.BuilderSpec(name = it.name) } }, modifier = Modifier.heightIn(min = 56.dp)) {
                Icon(Icons.Rounded.RestartAlt, "Reset sliders")
            }
        }
        PrimaryScrollableTabRow(selectedTabIndex = tab, edgePadding = 16.dp, containerColor = MaterialTheme.colorScheme.surface) {
            tabs.forEachIndexed { i, t -> Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) }) }
        }
        Box(Modifier.weight(1f)) {
            FormWithAction("Build character", Icons.Rounded.AutoAwesome, state.runtime == RuntimeStatus.Ready, vm::runBuilder) {
                if (state.runtime != RuntimeStatus.Ready) EngineWarning("Blender engine is not installed. Tap to set it up.") { vm.tab(Dest.Settings) }
                when (tabs[tab]) {
                    "Body", "Limbs", "Face" -> Panel {
                        Builder.sliders.filter { it.group == tabs[tab] }.forEach { sl ->
                            LabeledSlider(sl.label, spec.values[sl.arg] ?: sl.default, sl.min..sl.max) { v ->
                                vm.editBuilder { it.copy(values = it.values + (sl.arg to v)) }
                            }
                        }
                        if (tabs[tab] == "Face") Builder.choices.filter { it.arg in setOf("eyes", "mouth", "ears") }.forEach { c ->
                            ChoiceChips(c.label, c.options, spec.picks[c.arg] ?: c.options.first()) { v -> vm.editBuilder { it.copy(picks = it.picks + (c.arg to v)) } }
                        }
                        if (tabs[tab] == "Body") ChoiceChips("Smoothness", listOf("0", "1", "2"), spec.smooth.toString()) { v -> vm.editBuilder { it.copy(smooth = v.toInt()) } }
                    }
                    "Style" -> Panel {
                        Builder.choices.filter { it.arg !in setOf("eyes", "mouth", "ears") }.forEach { c ->
                            ChoiceChips(c.label, c.options, spec.picks[c.arg] ?: c.options.first()) { v -> vm.editBuilder { it.copy(picks = it.picks + (c.arg to v)) } }
                        }
                    }
                    else -> Panel {
                        Builder.colors.forEach { (k, label) ->
                            ColorPicker(label, spec.colors[k], allowDefault = false) { v -> if (v != null) vm.editBuilder { it.copy(colors = it.colors + (k to v)) } }
                        }
                    }
                }
                Text("T-pose, 17-bone rig and Model Forge's deformation self-test are applied automatically.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

// ---------------------------------------------------------------- Python script (System A)

@Composable
fun ScriptScreen(state: UiState, vm: AppViewModel, cb: AppCallbacks) {
    val sc = state.script
    FormWithAction("Run in Blender", Icons.Rounded.PlayArrow, sc.code.isNotBlank() && state.runtime == RuntimeStatus.Ready, vm::runScript) {
        if (state.runtime != RuntimeStatus.Ready) EngineWarning("Blender engine is not installed. Tap to set it up.") { vm.tab(Dest.Settings) }
        Panel {
            Text("Generate a Blender script with any AI, then import the .py file here. Copy the instructions below into the AI first so the script fits the app.",
                style = MaterialTheme.typography.bodyMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = { cb.copy("AI instructions", vm.scriptContract()) }, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.ContentCopy, null); Spacer(Modifier.width(6.dp)); Text("Copy AI rules")
                }
                OutlinedButton(onClick = vm::loadExampleScript, modifier = Modifier.weight(1f).heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.Code, null); Spacer(Modifier.width(6.dp)); Text("Example")
                }
            }
            OutlinedButton(onClick = cb.pickScript, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.FileOpen, null); Spacer(Modifier.width(8.dp)); Text(if (sc.fileName.isBlank()) "Import .py file" else "Replace ${sc.fileName}")
            }
        }
        if (sc.code.isNotBlank()) CodeView(sc.fileName, sc.code)
        OutlinedTextField(sc.params, vm::setScriptParams, label = { Text("FORGE_PARAMS (JSON, optional)") },
            supportingText = { Text("Passed to the script as the FORGE_PARAMS dict") }, modifier = Modifier.fillMaxWidth(), minLines = 2)
    }
}

@Composable
fun CodeView(title: String, code: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest), shape = MaterialTheme.shapes.medium) {
        Column(Modifier.padding(12.dp)) {
            Text("$title · ${code.lines().size} lines", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.tertiary)
            Spacer(Modifier.height(8.dp))
            SelectionContainer {
                Text(code.take(20_000), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.heightIn(max = 320.dp).verticalScroll(rememberScrollState()).horizontalScroll(rememberScrollState()), softWrap = false)
            }
        }
    }
}

// ---------------------------------------------------------------- Text to 3D (optional, cloud script writer)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun TextScreen(state: UiState, settings: Settings, vm: AppViewModel) {
    val t = state.text
    val hasKey = settings.apiKey.isNotBlank()
    val canRun = t.draft != null && state.runtime == RuntimeStatus.Ready && !t.writing
    FormWithAction(if (t.draft == null) "Write script" else "Build in Blender", if (t.draft == null) Icons.Rounded.AutoAwesome else Icons.Rounded.PlayArrow,
        enabled = hasKey && !t.writing && (t.draft == null && t.description.isNotBlank() || canRun),
        onAction = { if (t.draft == null) vm.writeTextScript() else vm.runText() }) {
        if (!hasKey) Card(onClick = { vm.tab(Dest.Settings) }, shape = MaterialTheme.shapes.large,
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer)) {
            Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.Key, null, tint = MaterialTheme.colorScheme.onTertiaryContainer)
                Spacer(Modifier.width(12.dp))
                Text("Optional feature: add an Anthropic API key in Settings. Claude writes the Blender script in the cloud; the model is built on your phone.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onTertiaryContainer)
            }
        }
        OutlinedTextField(t.description, { v -> vm.editText { it.copy(description = v.take(1500), draft = null) } },
            label = { Text("Describe your character") }, placeholder = { Text("A chubby orange fox mechanic with goggles, overalls and a big wrench") },
            modifier = Modifier.fillMaxWidth(), minLines = 4)
        Panel {
            ChoiceChips("Style", listOf("Stylized cartoon", "Chibi", "Low-poly", "Realistic proportions", "Robot / mech", "Creature"), t.style) { v ->
                vm.editText { it.copy(style = v, draft = null) }
            }
            ChoiceChips("Detail", listOf("low", "medium", "high"), t.detail) { v -> vm.editText { it.copy(detail = v, draft = null) } }
        }
        if (t.writing) Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp); Spacer(Modifier.width(12.dp)); Text("Claude is writing the Blender script…")
        }
        t.draft?.let { d ->
            if (d.notes.isNotBlank()) Text(d.notes, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            CodeView("generated script", d.script)
            OutlinedButton(onClick = vm::writeTextScript, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Rewrite script") }
        }
        Text("Model: ${settings.apiModel}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

// ---------------------------------------------------------------- Model Forge presets (System A)

@Composable
fun PresetsScreen(vm: AppViewModel) {
    LazyVerticalGrid(GridCells.Adaptive(160.dp), contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        itemsIndexed(Preset.entries) { i, p ->
            Card(onClick = { vm.setPreset("", emptyMap()); vm.go(Dest.PresetConfig(p)) }, shape = MaterialTheme.shapes.large,
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer), modifier = Modifier.staggerIn(i)) {
                Box(Modifier.fillMaxWidth().aspectRatio(1.3f).background(Brush.linearGradient(Source.PRESET.art().let { if (i % 2 == 0) it else it.reversed() })),
                    contentAlignment = Alignment.Center) { Icon(Icons.Rounded.Widgets, null, Modifier.size(44.dp)) }
                Column(Modifier.padding(12.dp)) {
                    Text(p.title, style = MaterialTheme.typography.titleSmall)
                    Text(p.tagline, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PresetConfigScreen(p: Preset, state: UiState, vm: AppViewModel) {
    FormWithAction("Forge it", Icons.Rounded.AutoAwesome, state.runtime == RuntimeStatus.Ready, { vm.runPreset(p) }) {
        if (state.runtime != RuntimeStatus.Ready) EngineWarning("Blender engine is not installed. Tap to set it up.") { vm.tab(Dest.Settings) }
        Text("generate_model.py --model-type ${p.modelType}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.tertiary)
        OutlinedTextField(state.presetName, { vm.setPreset(it.take(40), state.presetColors) }, label = { Text("Name") }, placeholder = { Text(p.title) },
            singleLine = true, modifier = Modifier.fillMaxWidth())
        if (p.colors.isNotEmpty()) Panel {
            p.colors.forEach { c ->
                ColorPicker(c.replaceFirstChar { it.uppercase() }, state.presetColors[c], allowDefault = true) { v ->
                    vm.setPreset(state.presetName, if (v == null) state.presetColors - c else state.presetColors + (c to v))
                }
            }
        } else Panel { Text("Shape and colours come from this generator's reference turnaround sheet.", style = MaterialTheme.typography.bodyMedium) }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            AssistChip(onClick = {}, label = { Text("T-pose") })
            AssistChip(onClick = {}, label = { Text("Rig self-test") })
            if (p.referenceDriven) AssistChip(onClick = {}, label = { Text("Reference-driven") })
        }
    }
}
