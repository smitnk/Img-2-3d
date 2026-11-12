package com.smitnk.img23d.ui

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AutoFixHigh
import androidx.compose.material.icons.rounded.Palette
import com.smitnk.img23d.body.PackStatus
import androidx.compose.material.icons.rounded.Code
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Verified
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.runtime.collectAsState
import com.smitnk.img23d.blender.ExportPreset
import com.smitnk.img23d.data.Quality
import androidx.compose.material.icons.rounded.Edit
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.SaveAlt
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.Timer
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.ViewInAr
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilledTonalIconButton
import androidx.compose.material3.Icon
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
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.UiState
import com.smitnk.img23d.data.Model
import com.smitnk.img23d.data.Settings
import com.smitnk.img23d.data.Source
import com.smitnk.img23d.viewer.GltfMesh
import com.smitnk.img23d.viewer.ModelView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
fun RunningScreen(state: UiState, vm: AppViewModel) {
    val r = state.run
    val cs = MaterialTheme.colorScheme
    val reduce = LocalReducedMotion.current
    val spin by rememberInfiniteTransition(label = "orb").animateFloat(0f, 360f, infiniteRepeatable(tween(3200, easing = LinearEasing), RepeatMode.Restart), label = "spin")
    val list = rememberLazyListState()
    LaunchedEffect(r.lines.size) { if (r.lines.isNotEmpty()) list.scrollToItem(r.lines.size - 1) }
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(84.dp).rotate(if (reduce) 0f else spin).clip(CircleShape)
                    .background(Brush.sweepGradient(listOf(cs.primary, cs.tertiary, cs.primaryContainer, cs.primary))),
                contentAlignment = Alignment.Center,
            ) { Box(Modifier.size(66.dp).clip(CircleShape).background(cs.surface)) }
            Spacer(Modifier.width(16.dp))
            Column {
                Text(r.title, style = MaterialTheme.typography.headlineSmall)
                Text("${r.engine}${if (r.steps > 1) " · step ${r.step} of ${r.steps}" else ""}", style = MaterialTheme.typography.labelMedium, color = cs.primary)
            }
        }
        Text(r.stage.ifBlank { "Starting…" }, style = MaterialTheme.typography.bodyMedium, maxLines = 2, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.semantics { contentDescription = "Current stage: ${r.stage}" })
        val f = r.fraction
        if (f != null) LinearProgressIndicator(progress = { f }, modifier = Modifier.fillMaxWidth()) else LinearProgressIndicator(Modifier.fillMaxWidth())
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Pill(Icons.Rounded.Timer, "%.1f s".format(r.elapsedMs / 1000f), Modifier.weight(1f))
            Pill(Icons.Rounded.Memory, "${r.memMb} MB · peak ${r.peakMb}", Modifier.weight(1.4f))
        }
        Card(Modifier.fillMaxWidth().weight(1f), shape = MaterialTheme.shapes.medium,
            colors = CardDefaults.cardColors(containerColor = cs.surfaceContainerLowest), border = BorderStroke(1.dp, cs.outlineVariant)) {
            if (r.lines.isEmpty()) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(if (r.engine == "TripoSR" || r.engine == "Hunyuan3D") "Neural network running on CPU" else "Waiting for Blender output…",
                    style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant)
            } else LazyColumn(state = list, contentPadding = PaddingValues(12.dp), modifier = Modifier.semantics { contentDescription = "Engine log" }) {
                items(r.lines) { Text(it, style = MaterialTheme.typography.labelSmall, color = cs.onSurfaceVariant) }
            }
        }
        OutlinedButton(onClick = vm::cancel, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Cancel") }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ResultScreen(m: Model, settings: Settings, vm: AppViewModel, cb: AppCallbacks) {
    var confirmDelete by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf(false) }
    var showScript by remember { mutableStateOf(false) }
    var spin by remember { mutableStateOf(settings.autoRotate) }
    var exportMenu by remember { mutableStateOf(false) }
    val exporting = vm.state.collectAsState().value.exporting
    // Parse off the main thread; a broken or unsupported file shows a message instead of crashing the app.
    val mesh by produceState<Result<GltfMesh>?>(null, m.id) { value = withContext(Dispatchers.Default) { runCatching { GltfMesh(m.glb.readBytes()) } } }
    val view = remember { mutableStateOf<ModelView?>(null) }
    LaunchedEffect(spin, view.value) { view.value?.setAutoRotate(spin) }
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().weight(1f).padding(horizontal = 12.dp).clip(MaterialTheme.shapes.extraLarge)
            .background(Brush.verticalGradient(listOf(MaterialTheme.colorScheme.surfaceContainerHigh, MaterialTheme.colorScheme.surfaceContainerLowest)))) {
            val loaded = mesh
            when {
                loaded == null -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                loaded.isFailure -> Text("Preview unavailable: ${loaded.exceptionOrNull()?.message}\nThe GLB can still be shared or saved.",
                    Modifier.align(Alignment.Center).padding(24.dp), style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                else -> AndroidView(factory = { ModelView(it).apply { load(loaded.getOrThrow()); view.value = this } }, modifier = Modifier.fillMaxSize())
            }
            Row(Modifier.align(Alignment.TopEnd).padding(10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalIconButton(onClick = { spin = !spin }) { Icon(if (spin) Icons.Rounded.Pause else Icons.Rounded.PlayArrow, if (spin) "Stop turntable" else "Turntable") }
            }
            Text("drag · pinch · two-finger pan · double-tap reset", Modifier.align(Alignment.BottomCenter).padding(8.dp),
                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Column(Modifier.verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                GradientBadge(m.source.art(), Icons.Rounded.ViewInAr, size = 40)
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(m.name, style = MaterialTheme.typography.titleLarge, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(m.source.label + if (m.detail.isNotBlank()) " · ${m.detail}" else "", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis)
                }
                FilledTonalIconButton(onClick = { renaming = true }) { Icon(Icons.Rounded.Edit, "Rename") }
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Pill(Icons.Rounded.ViewInAr, "%,d faces".format(m.faces))
                if (m.bones > 0) Pill(Icons.Rounded.Tune, "${m.bones} bones")
                Pill(Icons.Rounded.Timer, "%.1f s".format(m.totalMs / 1000f))
                if (m.peakMemMb > 0) Pill(Icons.Rounded.Memory, "peak ${m.peakMemMb} MB")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(onClick = { cb.share(m) }, Modifier.weight(1f).heightIn(min = 48.dp)) { Icon(Icons.Rounded.Share, null); Spacer(Modifier.width(6.dp)); Text("Share") }
                FilledTonalButton(onClick = { cb.export(m, false) }, Modifier.weight(1f).heightIn(min = 48.dp)) { Icon(Icons.Rounded.SaveAlt, null); Spacer(Modifier.width(6.dp)); Text("Save GLB") }
            }
            m.quality?.let { q ->
                QualityCard(q, canFix = m.source == Source.TEXT && m.script != null && settings.apiKey.isNotBlank() && !q.pass) { vm.fixQualityWithAi(m) }
            }
            Box {
                OutlinedButton(onClick = { exportMenu = true }, enabled = exporting == null, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                    if (exporting != null) { CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp); Spacer(Modifier.width(8.dp)); Text("Exporting for $exporting…") }
                    else { Icon(Icons.Rounded.SaveAlt, null); Spacer(Modifier.width(8.dp)); Text("Export for Unity, Unreal, Roblox, print…") }
                }
                DropdownMenu(exportMenu, { exportMenu = false }) {
                    ExportPreset.entries.filter { m.bones > 0 || !it.id.endsWith("_skeletal") }.forEach { p ->
                        DropdownMenuItem(text = { Column { Text(p.label); Text(p.note, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) } },
                            onClick = { exportMenu = false; vm.exportPreset(m, p) })
                    }
                }
            }
            if (m.source == Source.BODY) FilledTonalButton(onClick = { vm.openBodyEditor(m) }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.Edit, null); Spacer(Modifier.width(8.dp)); Text("Edit in Body Creator")
            } else OutlinedButton(onClick = { vm.fitBodyTo(m) }, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.AutoFixHigh, null); Spacer(Modifier.width(8.dp)); Text("Fit an editable human body to it (Body Creator)")
            }
            if (m.source != Source.BODY) {
                val tex = vm.state.collectAsState().value.texPack
                OutlinedButton(onClick = { vm.paintTexture(m) }, enabled = tex !is PackStatus.Downloading,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.Palette, null); Spacer(Modifier.width(8.dp))
                    Text(when {
                        !settings.paintAi -> "Paint texture from photo"
                        tex is PackStatus.Downloading -> "Downloading texture AI… ${tex.done * 100 / maxOf(tex.total, 1)}%"
                        tex is PackStatus.Failed -> "Texture AI download failed: tap to retry"
                        tex == PackStatus.Ready -> if (m.photo != null) "Paint texture (photo + AI sides)" else "Paint texture with AI"
                        else -> "Paint texture: download AI (~1 GB, runs on phone)"
                    })
                }
            }
            if (m.source == Source.PHOTO || m.source == Source.CHARACTER) OutlinedButton(onClick = { vm.refine(m) }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.AutoFixHigh, null); Spacer(Modifier.width(8.dp)); Text("Refine in Blender")
            }
            if (m.source == Source.PHOTO || m.source == Source.CHARACTER || m.source == Source.REFINED) Column {
                var open by androidx.compose.runtime.saveable.rememberSaveable { androidx.compose.runtime.mutableStateOf(false) }
                OutlinedButton(onClick = { open = true }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.AutoFixHigh, null); Spacer(Modifier.width(8.dp)); Text("Smart Mesh: clean quads for games / animation")
                }
                androidx.compose.material3.DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                    listOf(Triple("GAME", 20000, "Game · 20k quads (~1 min)"), Triple("LOW_POLY", 5000, "Low poly · 5k quads (mobile games)"),
                        Triple("ANIMATION", 10000, "Animation · 10k quads (check joints after)")).forEach { (use, faces, label) ->
                        androidx.compose.material3.DropdownMenuItem(text = { Text(label) }, onClick = { open = false; vm.smartMesh(m, use, faces) })
                    }
                }
            }
            if (m.source == Source.PHOTO || m.source == Source.CHARACTER || m.source == Source.REFINED) OutlinedButton(onClick = { vm.rigUnirig(m) }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.AutoFixHigh, null); Spacer(Modifier.width(8.dp)); Text("Rig with UniRig (AI skeleton + weights, ~20 min)")
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                if (m.blend != null) OutlinedButton(onClick = { cb.export(m, true) }, Modifier.weight(1f).heightIn(min = 48.dp)) { Text("Save .blend") }
                if (m.script != null) OutlinedButton(onClick = { showScript = true }, Modifier.weight(1f).heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.Code, null); Spacer(Modifier.width(6.dp)); Text("Script")
                }
                OutlinedButton(onClick = { confirmDelete = true }, Modifier.weight(1f).heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.Delete, null, tint = MaterialTheme.colorScheme.error); Spacer(Modifier.width(6.dp)); Text("Delete")
                }
            }
        }
    }
    if (confirmDelete) AlertDialog(
        onDismissRequest = { confirmDelete = false }, title = { Text("Delete ${m.name}?") },
        text = { Text("The GLB and its files are removed from this phone.") },
        confirmButton = { TextButton(onClick = { confirmDelete = false; vm.delete(m); vm.back() }) { Text("Delete", color = MaterialTheme.colorScheme.error) } },
        dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Keep") } },
    )
    if (renaming) {
        var name by remember { mutableStateOf(m.name) }
        AlertDialog(
            onDismissRequest = { renaming = false }, title = { Text("Rename") },
            text = { OutlinedTextField(name, { name = it.take(60) }, singleLine = true) },
            confirmButton = { TextButton(onClick = { renaming = false; if (name.isNotBlank()) vm.rename(m, name) }) { Text("Save") } },
            dismissButton = { TextButton(onClick = { renaming = false }) { Text("Cancel") } },
        )
    }
    if (showScript) m.script?.let { f ->
        val code = remember(f) { f.readText() }
        AlertDialog(
            onDismissRequest = { showScript = false },
            title = { Text("Generator script") },
            text = { CodeView(f.name, code) },
            confirmButton = { TextButton(onClick = { cb.copy("Script", code); showScript = false }) { Text("Copy") } },
            dismissButton = { TextButton(onClick = { showScript = false }) { Text("Close") } },
        )
    }
}

/** bx_audit result: what an engine or printer would trip over (floating parts, z-fighting, flipped normals…). */
@Composable
private fun QualityCard(q: Quality, canFix: Boolean, onFix: () -> Unit) {
    val cs = MaterialTheme.colorScheme
    var open by remember { mutableStateOf(false) }
    val issues = q.errors + q.warnings
    Card(onClick = { open = !open }, enabled = issues.isNotEmpty(), modifier = Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(containerColor = if (q.pass) cs.secondaryContainer else cs.errorContainer,
            disabledContainerColor = cs.secondaryContainer)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(if (q.pass) Icons.Rounded.Verified else Icons.Rounded.Warning, null,
                    tint = if (q.pass) cs.onSecondaryContainer else cs.onErrorContainer)
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(if (q.pass) "Game-ready check passed" else "${q.errors.size} issue${if (q.errors.size == 1) "" else "s"} to fix",
                        style = MaterialTheme.typography.titleSmall)
                    Text(buildString {
                        if (q.warnings.isNotEmpty()) append("${q.warnings.size} warning${if (q.warnings.size == 1) "" else "s"} · ")
                        append(if (issues.isEmpty()) "no floating parts, z-fighting or bad normals" else if (open) "tap to hide" else "tap for details")
                    }, style = MaterialTheme.typography.bodySmall)
                }
            }
            if (open) {
                q.errors.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) }
                q.warnings.forEach { Text("◦ $it", style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant) }
            }
            if (canFix) FilledTonalButton(onClick = onFix, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.AutoFixHigh, null); Spacer(Modifier.width(8.dp)); Text("Fix issues with AI")
            }
        }
    }
}
