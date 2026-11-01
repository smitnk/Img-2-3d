package com.smitnk.img23d.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Accessibility
import androidx.compose.material.icons.rounded.Casino
import androidx.compose.material.icons.rounded.CloudDownload
import androidx.compose.material.icons.rounded.ExpandLess
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.RestartAlt
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryScrollableTabRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.Dest
import com.smitnk.img23d.UiState
import com.smitnk.img23d.blender.RuntimeStatus
import com.smitnk.img23d.body.BodySpec
import com.smitnk.img23d.body.PackStatus
import com.smitnk.img23d.viewer.ModelView
import kotlin.random.Random

private val TABS = listOf("Body", "Face", "Torso", "Limbs", "Skin")
private val TAB_GROUPS = mapOf(
    "Face" to listOf("head", "forehead", "eyebrows", "eyes", "nose", "mouth", "ears", "chin", "cheek", "neck"),
    "Torso" to listOf("torso", "breast", "stomach", "hip", "buttocks", "pelvis"),
    "Limbs" to listOf("armslegs"),
)

/** "nose/nose-trans-in|out" -> "Trans · in ↔ out"; "eyes/l-eye-scale-decr|incr" -> "Eye scale" (sides merged when symmetric). */
private fun sliderLabel(id: String, symmetric: Boolean): String {
    val (group, rest) = id.split("/", limit = 2).let { it[0] to it[1] }
    var t = rest.substringBefore("-decr|incr").removePrefix("$group-")
    if (symmetric) t = t.removePrefix("l-").removePrefix("r-") else t = t.replaceFirst(Regex("^l-"), "left ").replaceFirst(Regex("^r-"), "right ")
    val range = if ('|' in t) t.substringAfterLast('-').replace("|", " ↔ ") else ""
    if ('|' in t) t = t.substringBeforeLast('-')
    val words = t.replace('-', ' ').replaceFirstChar { it.uppercase() }
    return if (range.isEmpty()) words else "$words · $range"
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun BodyScreen(state: UiState, vm: AppViewModel) {
    val spec = state.body
    var tab by remember { mutableIntStateOf(0) }
    LaunchedEffect(state.bodyPack) { if (state.bodyPack == PackStatus.Ready && state.bodyPreview == null) vm.refreshBodyPreview() }

    if (state.bodyPack != PackStatus.Ready) { PackSetup(state.bodyPack, vm); return }

    Column(Modifier.fillMaxSize()) {
        val cs = MaterialTheme.colorScheme
        Box(Modifier.fillMaxWidth().weight(0.44f).padding(horizontal = 12.dp).clip(MaterialTheme.shapes.extraLarge)
            .background(Brush.verticalGradient(listOf(cs.surfaceContainerHigh, cs.surfaceContainerLowest)))) {
            val mesh = state.bodyPreview
            if (mesh != null) AndroidView(factory = { ModelView(it) }, update = { it.load(mesh, keepCamera = true) }, modifier = Modifier.fillMaxSize())
            Text("Live preview · drag to turn", Modifier.align(Alignment.BottomCenter).padding(8.dp),
                style = MaterialTheme.typography.labelSmall, color = cs.onSurfaceVariant)
        }
        Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(spec.name, { v -> vm.editBody { it.copy(name = v.take(40)) } }, label = { Text("Name") }, singleLine = true, modifier = Modifier.weight(1f))
            FilledTonalButton(onClick = { vm.editBody { randomize(it) } }, modifier = Modifier.heightIn(min = 56.dp)) { Icon(Icons.Rounded.Casino, "Randomize") }
            OutlinedButton(onClick = { vm.editBody { BodySpec(name = it.name) } }, modifier = Modifier.heightIn(min = 56.dp)) { Icon(Icons.Rounded.RestartAlt, "Reset") }
        }
        PrimaryScrollableTabRow(selectedTabIndex = tab, edgePadding = 16.dp, containerColor = cs.surface) {
            TABS.forEachIndexed { i, t -> Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) }) }
        }
        Box(Modifier.weight(0.56f)) {
            FormWithAction("Build character", Icons.Rounded.Accessibility, state.runtime == RuntimeStatus.Ready, vm::runBody) {
                if (state.runtime != RuntimeStatus.Ready) EngineWarning("The Blender engine builds the final rigged model. Tap to install it.") { vm.tab(Dest.Settings) }
                when (TABS[tab]) {
                    "Body" -> {
                        Panel {
                            Text("Start from", style = MaterialTheme.typography.titleSmall)
                            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                BodySpec.PRESETS.forEach { (label, p) ->
                                    FilterChip(selected = spec.name == p.name && spec.macros == p.macros, onClick = { vm.editBody { p.copy(skin = it.skin, subdiv = it.subdiv) } },
                                        label = { Text(label) })
                                }
                            }
                        }
                        Panel {
                            BodySpec.MACROS.forEach { (k, label) ->
                                LabeledSlider(label, spec.macro(k), 0f..1f) { v -> vm.editBody { it.copy(macros = it.macros + (k to v)) } }
                            }
                        }
                        Panel {
                            Text("Heritage mix (face and body shape)", style = MaterialTheme.typography.titleSmall)
                            BodySpec.ETHNIC.forEach { (k, label) ->
                                LabeledSlider(label, spec.macro(k), 0f..1f) { v -> vm.editBody { it.copy(macros = it.macros + (k to v)) } }
                            }
                        }
                    }
                    "Skin" -> {
                        Panel {
                            Text("Skin tone", style = MaterialTheme.typography.titleSmall)
                            FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                                BodySpec.SKINS.forEach { hex ->
                                    val on = hex.equals(spec.skin, true)
                                    Box(Modifier.size(44.dp).clip(CircleShape).background(parseHex(hex))
                                        .border(if (on) 3.dp else 1.dp, if (on) cs.primary else cs.outlineVariant, CircleShape)
                                        .clickable { vm.editBody { it.copy(skin = hex) } }
                                        .semantics { contentDescription = "Skin tone $hex" + if (on) ", selected" else "" })
                                }
                            }
                            ColorPicker("Custom", spec.skin, allowDefault = false) { v -> if (v != null) vm.editBody { it.copy(skin = v) } }
                        }
                        Panel {
                            ChoiceChips("Surface detail", listOf("Draft", "Smooth", "Ultra"), listOf("Draft", "Smooth", "Ultra")[spec.subdiv.coerceIn(0, 2)]) { v ->
                                vm.editBody { it.copy(subdiv = listOf("Draft", "Smooth", "Ultra").indexOf(v)) }
                            }
                            Text("Draft ≈ 27k triangles (games, fast), Smooth ≈ 110k, Ultra ≈ 440k (renders).",
                                style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant)
                        }
                    }
                    else -> {
                        Panel {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("Mirror left and right", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                                Switch(spec.symmetric, { on -> vm.editBody { it.copy(symmetric = on) } })
                            }
                        }
                        val all = remember(state.bodyPack) { vm.bodySliders().keys.toList() }
                        TAB_GROUPS.getValue(TABS[tab]).forEach { g ->
                            val ids = all.filter { it.startsWith("$g/") }
                                .filter { !spec.symmetric || !it.substringAfter('/').startsWith("r-") }
                            if (ids.isNotEmpty()) SliderGroup(g, ids, spec, vm)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SliderGroup(group: String, ids: List<String>, spec: BodySpec, vm: AppViewModel) {
    var open by remember { mutableStateOf(false) }
    val changed = ids.count { (spec.sliders[it] ?: 0f) != 0f }
    Card(shape = MaterialTheme.shapes.large, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow)) {
        Row(Modifier.fillMaxWidth().clickable { open = !open }.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(group.replaceFirstChar { it.uppercase() }.replace("Armslegs", "Arms, legs, hands & feet"), style = MaterialTheme.typography.titleMedium)
                Text("${ids.size} sliders" + if (changed > 0) " · $changed changed" else "", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Icon(if (open) Icons.Rounded.ExpandLess else Icons.Rounded.ExpandMore, if (open) "Collapse" else "Expand")
        }
        if (open) Column(Modifier.padding(start = 16.dp, end = 16.dp, bottom = 16.dp)) {
            ids.forEach { id ->
                val twoSided = '|' in id
                LabeledSlider(sliderLabel(id, spec.symmetric), spec.sliders[id] ?: 0f, if (twoSided) -1f..1f else 0f..1f) { v ->
                    vm.editBody { it.withSlider(id, v) }
                }
            }
        }
    }
}

@Composable
private fun PackSetup(st: PackStatus, vm: AppViewModel) {
    Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally) {
        GradientBadge(com.smitnk.img23d.data.Source.BODY.art(), Icons.Rounded.Accessibility, size = 72)
        Text("Body Creator", style = MaterialTheme.typography.headlineSmall)
        Text("A professionally modelled human base (MakeHuman, CC0) with 238 face and body sliders, a 163-bone rig with fingers, " +
            "and a live preview. One-time download of about 34 MB.", style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        when (st) {
            is PackStatus.Downloading -> {
                LinearProgressIndicator(progress = { if (st.total > 0) st.done.toFloat() / st.total else 0f }, modifier = Modifier.fillMaxWidth())
                Text("%.1f / %.1f MB".format(st.done / 1e6, st.total / 1e6), style = MaterialTheme.typography.labelMedium)
            }
            else -> {
                if (st is PackStatus.Failed) Text(st.message, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                Button(onClick = vm::installBodyPack, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                    Icon(Icons.Rounded.CloudDownload, null); Spacer(Modifier.width(8.dp)); Text(if (st is PackStatus.Failed) "Retry download" else "Download body assets")
                }
            }
        }
    }
}

private fun randomize(s: BodySpec, rnd: Random = Random): BodySpec = s.copy(
    macros = BodySpec.MACROS.associate { (k, _) -> k to (0.2f + rnd.nextFloat() * 0.6f) } +
        BodySpec.ETHNIC.associate { (k, _) -> k to rnd.nextFloat() },
    sliders = emptyMap(),
    skin = BodySpec.SKINS.random(rnd),
)
