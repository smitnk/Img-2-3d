package com.smitnk.forge3d.ui

import android.text.format.DateUtils
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.CloudDownload
import androidx.compose.material.icons.rounded.Collections
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.PhotoLibrary
import androidx.compose.material.icons.rounded.SaveAlt
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.Terminal
import androidx.compose.material.icons.rounded.Timer
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.ViewInAr
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.smitnk.forge3d.Dest
import com.smitnk.forge3d.ForgeState
import com.smitnk.forge3d.ForgeViewModel
import com.smitnk.forge3d.engine.ColorParam
import com.smitnk.forge3d.engine.Creation
import com.smitnk.forge3d.engine.Generator
import com.smitnk.forge3d.engine.RuntimeStatus
import com.smitnk.img23d.viewer.ModelView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.compose.runtime.produceState

class ForgeActions(val export: (Creation) -> Unit, val share: (Creation) -> Unit)

private fun parseHex(h: String): Color = runCatching { Color(android.graphics.Color.parseColor(h)) }.getOrDefault(Color.Gray)

/** Two-stop art gradient per generator, derived from its look, used for gallery tiles. */
private fun Generator.art(): Pair<Color, Color> = when (this) {
    Generator.FORGE_KID -> Color(0xFFE8553D) to Color(0xFF2F4A8A)
    Generator.CLAY_CHIBI -> Color(0xFFC9A27E) to Color(0xFF6B4A35)
    Generator.SNOW_BUDDY -> Color(0xFFE6F1FF) to Color(0xFF7DA7D9)
    Generator.ROUND_BUDDY -> Color(0xFFF2B880) to Color(0xFFD9604C)
    Generator.GORILLA -> Color(0xFF6E6A8F) to Color(0xFF2B2840)
    Generator.LIZARD_WARRIOR -> Color(0xFF7FB069) to Color(0xFF3A5A40)
    Generator.LIZARD_REFINED -> Color(0xFF9CC5A1) to Color(0xFF49654A)
}

@Composable
fun ForgeApp(s: ForgeState, vm: ForgeViewModel, actions: ForgeActions) {
    val top = s.dest in listOf(Dest.Home, Dest.Library, Dest.Runtime)
    Scaffold(
        containerColor = MaterialTheme.colorScheme.surface,
        topBar = { if (!top) DetailBar(s.dest, vm) },
        bottomBar = { if (top) BottomNav(s, vm) },
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            when (val d = s.dest) {
                Dest.Home -> HomeScreen(s, vm)
                is Dest.Configure -> ConfigureScreen(d.generator, s, vm)
                Dest.Running -> RunningScreen(s, vm)
                is Dest.Result -> vm.creation(d.id)?.let { ResultScreen(it, vm, actions) }
                Dest.Library -> LibraryScreen(s, vm)
                Dest.Runtime -> RuntimeScreen(s, vm)
            }
        }
    }
    s.error?.let { msg ->
        AlertDialog(
            onDismissRequest = vm::dismissError,
            icon = { Icon(Icons.Rounded.Terminal, null) },
            title = { Text("Generation failed") },
            text = { Text(msg, style = MaterialTheme.typography.labelSmall, modifier = Modifier.heightIn(max = 320.dp).verticalScroll(rememberScrollState())) },
            confirmButton = { TextButton(onClick = vm::dismissError) { Text("OK") } },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DetailBar(d: Dest, vm: ForgeViewModel) {
    val title = when (d) {
        is Dest.Configure -> d.generator.title
        Dest.Running -> "Forging in Blender"
        is Dest.Result -> "Your character"
        else -> ""
    }
    TopAppBar(
        title = { Text(title, style = MaterialTheme.typography.titleLarge) },
        navigationIcon = { IconButton(onClick = { vm.back() }) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, "Back") } },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
    )
}

@Composable
private fun BottomNav(s: ForgeState, vm: ForgeViewModel) {
    NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainer) {
        listOf(
            Triple(Dest.Home, Icons.Rounded.AutoAwesome, "Create"),
            Triple(Dest.Library, Icons.Rounded.Collections, "Library"),
            Triple(Dest.Runtime, Icons.Rounded.Memory, "Engine"),
        ).forEach { (d, icon, label) ->
            NavigationBarItem(
                selected = s.dest == d,
                onClick = { if (s.dest != d) { vm.back(); if (d != Dest.Home) vm.go(d) } },
                icon = { Icon(icon, null) },
                label = { Text(label) },
            )
        }
    }
}

@Composable
private fun SectionTitle(text: String) =
    Text(text, style = MaterialTheme.typography.titleMedium, modifier = Modifier.semantics { heading() })

@Composable
private fun HomeScreen(s: ForgeState, vm: ForgeViewModel) {
    val cs = MaterialTheme.colorScheme
    LazyVerticalGrid(
        columns = GridCells.Adaptive(160.dp),
        contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxSize(),
    ) {
        item(span = { GridItemSpan(maxLineSpan) }) {
            Box(
                Modifier.fillMaxWidth().clip(MaterialTheme.shapes.extraLarge)
                    .background(Brush.linearGradient(listOf(cs.primaryContainer, cs.secondaryContainer, cs.tertiaryContainer)))
                    .padding(24.dp),
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("SYSTEM A · PROCEDURAL", style = MaterialTheme.typography.labelMedium, color = cs.onPrimaryContainer)
                    Text("Forge characters with real Blender", style = MaterialTheme.typography.displaySmall, color = cs.onPrimaryContainer)
                    Text(
                        "Blender 4.2.3 runs headless on this phone and executes the Model Forge Python generators. Rigged, T-posed GLBs.",
                        style = MaterialTheme.typography.bodyMedium, color = cs.onPrimaryContainer,
                    )
                }
            }
        }
        if (s.runtime != RuntimeStatus.Ready) item(span = { GridItemSpan(maxLineSpan) }) { EngineBanner(s.runtime) { vm.go(Dest.Runtime) } }
        item(span = { GridItemSpan(maxLineSpan) }) { SectionTitle("Generators") }
        items(Generator.entries, key = { it.name }) { g -> GeneratorTile(g) { vm.openGenerator(g) } }
    }
}

@Composable
private fun EngineBanner(st: RuntimeStatus, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh),
        shape = MaterialTheme.shapes.large,
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.CloudDownload, null, tint = MaterialTheme.colorScheme.primary)
            Spacer(Modifier.width(16.dp))
            Column(Modifier.weight(1f)) {
                Text("Install the Blender engine", style = MaterialTheme.typography.titleMedium)
                Text(
                    if (st is RuntimeStatus.Downloading || st is RuntimeStatus.Extracting) "Installing…" else "One-time download, then fully offline.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun GeneratorTile(g: Generator, onClick: () -> Unit) {
    val (a, b) = g.art()
    ElevatedCard(onClick = onClick, shape = MaterialTheme.shapes.large, modifier = Modifier.semantics(mergeDescendants = true) {}) {
        Box(Modifier.fillMaxWidth().aspectRatio(1.25f).background(Brush.linearGradient(listOf(a, b))), contentAlignment = Alignment.Center) {
            Icon(Icons.Rounded.ViewInAr, null, Modifier.size(48.dp), tint = Color.White.copy(alpha = 0.9f))
            if (g.colors.isNotEmpty()) {
                Row(Modifier.align(Alignment.BottomStart).padding(10.dp), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    g.colors.take(5).forEach { p ->
                        Box(Modifier.size(12.dp).clip(CircleShape).background(g.defaults[p]?.let(::parseHex) ?: Color.White.copy(alpha = 0.6f))
                            .border(1.dp, Color.White.copy(alpha = 0.8f), CircleShape))
                    }
                }
            }
        }
        Column(Modifier.padding(12.dp)) {
            Text(g.title, style = MaterialTheme.typography.titleMedium)
            Text(g.tagline, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2)
        }
    }
}

private val Palette = listOf(
    "#F2C9A0", "#C68B59", "#8D5524", "#3B2A20", "#F7F7F2", "#E8553D", "#F59E0B", "#22C55E",
    "#2F4A8A", "#7C3AED", "#EC4899", "#14B8A6", "#111827", "#9CA3AF",
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ConfigureScreen(g: Generator, s: ForgeState, vm: ForgeViewModel) {
    val d = s.draft
    val (a, b) = g.art()
    Box(Modifier.fillMaxSize()) {
        Column(
            Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Box(
                Modifier.fillMaxWidth().height(140.dp).clip(MaterialTheme.shapes.extraLarge).background(Brush.linearGradient(listOf(a, b))).padding(20.dp),
                contentAlignment = Alignment.BottomStart,
            ) {
                Column {
                    Text(g.tagline, style = MaterialTheme.typography.titleMedium, color = Color.White)
                    Text("generate_model.py --model-type ${g.modelType}", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.85f))
                }
            }
            OutlinedTextField(
                value = d.name, onValueChange = { v -> vm.editDraft { it.copy(name = v.take(40)) } },
                label = { Text("Character name") }, placeholder = { Text(g.title) }, singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            if (g.colors.isNotEmpty()) {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow), shape = MaterialTheme.shapes.large) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        SectionTitle("Colours")
                        g.colors.forEach { p -> ColorRow(p, d.colors[p]) { v -> vm.editDraft { it.copy(colors = if (v == null) it.colors - p else it.colors + (p to v)) } } }
                    }
                }
            } else {
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow), shape = MaterialTheme.shapes.large) {
                    Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Rounded.PhotoLibrary, null, tint = MaterialTheme.colorScheme.tertiary)
                        Spacer(Modifier.width(12.dp))
                        Text(
                            "Shape and colours come from this generator's reference turnaround sheet, so it has no colour options.",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow), shape = MaterialTheme.shapes.large) {
                Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                    ToggleRow("Armature rig", "Skinned bones, verified by the generator's deformation self-test", d.rig) { v -> vm.editDraft { it.copy(rig = v) } }
                    ToggleRow("Also save .blend", "Editable Blender file next to the GLB", d.saveBlend) { v -> vm.editDraft { it.copy(saveBlend = v) } }
                }
            }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AssistChip(onClick = {}, label = { Text("T-pose") }, leadingIcon = { Icon(Icons.Rounded.Check, null, Modifier.size(18.dp)) })
                AssistChip(onClick = {}, label = { Text("GLB export") }, leadingIcon = { Icon(Icons.Rounded.Check, null, Modifier.size(18.dp)) })
                if (g.referenceDriven) AssistChip(onClick = {}, label = { Text("Reference-driven") }, leadingIcon = { Icon(Icons.Rounded.PhotoLibrary, null, Modifier.size(18.dp)) })
            }
            Spacer(Modifier.height(72.dp))
        }
        ExtendedFloatingActionButton(
            onClick = { vm.generate(g) },
            icon = { Icon(Icons.Rounded.AutoAwesome, null) },
            text = { Text(if (s.runtime == RuntimeStatus.Ready) "Forge it" else "Install engine first") },
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ColorRow(p: ColorParam, value: String?, onPick: (String?) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(p.label, style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
            Text(value ?: "generator default", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = value == null, onClick = { onPick(null) }, label = { Text("Default") })
            Palette.forEach { hex ->
                val sel = value.equals(hex, ignoreCase = true)
                Box(
                    Modifier.size(40.dp).clip(CircleShape).background(parseHex(hex))
                        .border(if (sel) 3.dp else 1.dp, if (sel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant, CircleShape)
                        .clickable(onClickLabel = "Use $hex for ${p.label}") { onPick(hex) }
                        .semantics { contentDescription = "${p.label} $hex" + if (sel) ", selected" else "" },
                    contentAlignment = Alignment.Center,
                ) { if (sel) Icon(Icons.Rounded.Check, null, tint = if (parseHex(hex).luminance() > 0.5f) Color.Black else Color.White) }
            }
        }
    }
}

private fun Color.luminance() = 0.2126f * red + 0.7152f * green + 0.0722f * blue

@Composable
private fun ToggleRow(title: String, sub: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clickable { onChange(!checked) }.padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(sub, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun RunningScreen(s: ForgeState, vm: ForgeViewModel) {
    val r = s.run
    val cs = MaterialTheme.colorScheme
    val reduce = LocalReducedMotion.current
    val spin by rememberInfiniteTransition(label = "orb").animateFloat(0f, 360f, infiniteRepeatable(tween(4000, easing = LinearEasing), RepeatMode.Restart), label = "spin")
    val listState = rememberLazyListState()
    LaunchedEffect(r.lines.size) { if (r.lines.isNotEmpty()) listState.scrollToItem(r.lines.size - 1) }
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(72.dp).rotate(if (reduce) 0f else spin).clip(CircleShape)
                    .background(Brush.sweepGradient(listOf(cs.primary, cs.tertiary, cs.secondary, cs.primary))),
                contentAlignment = Alignment.Center,
            ) { Box(Modifier.size(56.dp).clip(CircleShape).background(cs.surface)) }
            Spacer(Modifier.width(16.dp))
            Column {
                Text(r.generator?.title ?: "", style = MaterialTheme.typography.headlineSmall)
                Text("blender -b --python generate_model.py", style = MaterialTheme.typography.labelSmall, color = cs.onSurfaceVariant)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            StatPill(Icons.Rounded.Timer, "%.1f s".format(r.elapsedMs / 1000f), Modifier.weight(1f))
            StatPill(Icons.Rounded.Memory, "${r.rssMb} MB · peak ${r.peakRssMb}", Modifier.weight(1f))
        }
        LinearProgressIndicator(Modifier.fillMaxWidth())
        Card(
            Modifier.fillMaxWidth().weight(1f),
            colors = CardDefaults.cardColors(containerColor = cs.surfaceContainerLowest),
            shape = MaterialTheme.shapes.medium,
            border = BorderStroke(1.dp, cs.outlineVariant),
        ) {
            LazyColumn(state = listState, contentPadding = PaddingValues(12.dp), modifier = Modifier.semantics { contentDescription = "Blender output log" }) {
                items(r.lines) { line -> Text(line, style = MaterialTheme.typography.labelSmall, color = cs.onSurfaceVariant) }
            }
        }
        OutlinedButton(onClick = vm::cancel, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Cancel") }
    }
}

@Composable
private fun StatPill(icon: ImageVector, text: String, modifier: Modifier = Modifier) {
    Row(
        modifier.clip(MaterialTheme.shapes.extraLarge).background(MaterialTheme.colorScheme.surfaceContainerHigh).padding(horizontal = 14.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.width(8.dp))
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultScreen(c: Creation, vm: ForgeViewModel, actions: ForgeActions) {
    var confirm by remember { mutableStateOf(false) }
    val glb by produceState<ByteArray?>(null, c.id) { value = withContext(Dispatchers.IO) { c.glb.readBytes() } }
    Column(Modifier.fillMaxSize()) {
        Box(
            Modifier.fillMaxWidth().weight(1f).padding(horizontal = 16.dp).clip(MaterialTheme.shapes.extraLarge)
                .background(MaterialTheme.colorScheme.surfaceContainerLowest),
        ) {
            val bytes = glb
            if (bytes == null) CircularProgressIndicator(Modifier.align(Alignment.Center))
            else AndroidView(factory = { ModelView(it).apply { load(bytes) } }, modifier = Modifier.fillMaxSize())
        }
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(c.title, style = MaterialTheme.typography.headlineSmall)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                StatPill(Icons.Rounded.ViewInAr, "%,d tris".format(c.triangles))
                StatPill(Icons.Rounded.Tune, "${c.bones} bones")
                StatPill(Icons.Rounded.Timer, "%.1f s".format(c.seconds))
                StatPill(Icons.Rounded.Memory, "peak ${c.peakRssMb} MB")
            }
            Text("Blender ${c.blender}" + if (c.blend != null) " · .blend saved" else "", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Button(onClick = { actions.share(c) }, Modifier.weight(1f).heightIn(min = 48.dp)) { Icon(Icons.Rounded.Share, null); Spacer(Modifier.width(8.dp)); Text("Share") }
                FilledTonalButton(onClick = { actions.export(c) }, Modifier.weight(1f).heightIn(min = 48.dp)) { Icon(Icons.Rounded.SaveAlt, null); Spacer(Modifier.width(8.dp)); Text("Save GLB") }
                IconButton(onClick = { confirm = true }, Modifier.size(48.dp)) { Icon(Icons.Rounded.Delete, "Delete", tint = MaterialTheme.colorScheme.error) }
            }
        }
    }
    if (confirm) AlertDialog(
        onDismissRequest = { confirm = false },
        title = { Text("Delete ${c.title}?") },
        confirmButton = { TextButton(onClick = { confirm = false; vm.delete(c); vm.back() }) { Text("Delete") } },
        dismissButton = { TextButton(onClick = { confirm = false }) { Text("Keep") } },
    )
}

@Composable
private fun LibraryScreen(s: ForgeState, vm: ForgeViewModel) {
    LaunchedEffect(Unit) { vm.refreshLibrary() }
    if (s.creations.isEmpty()) {
        Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Rounded.Collections, null, Modifier.size(56.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(12.dp))
            Text("No characters yet", style = MaterialTheme.typography.titleMedium)
            Text("Pick a generator on Create.", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        return
    }
    LazyVerticalGrid(GridCells.Adaptive(160.dp), contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item(span = { GridItemSpan(maxLineSpan) }) { SectionTitle("Library") }
        items(s.creations, key = { it.id }) { c ->
            val g = runCatching { Generator.valueOf(c.generator) }.getOrNull()
            val (a, b) = g?.art() ?: (Color.Gray to Color.DarkGray)
            ElevatedCard(onClick = { vm.go(Dest.Result(c.id)) }, shape = MaterialTheme.shapes.large) {
                Box(Modifier.fillMaxWidth().aspectRatio(1.4f).background(Brush.linearGradient(listOf(a, b))), contentAlignment = Alignment.Center) {
                    Icon(Icons.Rounded.ViewInAr, null, Modifier.size(40.dp), tint = Color.White)
                }
                Column(Modifier.padding(12.dp)) {
                    Text(c.title, style = MaterialTheme.typography.titleMedium, maxLines = 1)
                    Text(DateUtils.getRelativeTimeSpanString(c.createdAt).toString() + " · %,d tris".format(c.triangles),
                        style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun RuntimeScreen(s: ForgeState, vm: ForgeViewModel) {
    val st = s.runtime
    val busy = st is RuntimeStatus.Downloading || st is RuntimeStatus.Verifying || st is RuntimeStatus.Extracting
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        SectionTitle("Blender engine")
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow), shape = MaterialTheme.shapes.large) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Blender 4.2.3 LTS · headless · linux-arm64", style = MaterialTheme.typography.titleMedium)
                Text("Runs inside a minimal Ubuntu userland through proot, with the bpy API, bmesh, OpenSubdiv, OpenVDB remesh and the official glTF exporter. No UI, no viewport.",
                    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(vm.runtime.dir.path, style = MaterialTheme.typography.labelSmall)
            }
        }
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer), shape = MaterialTheme.shapes.large) {
            Column(Modifier.padding(16.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                when (st) {
                    RuntimeStatus.Ready -> Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Rounded.Check, null, tint = MaterialTheme.colorScheme.primary); Spacer(Modifier.width(8.dp)); Text("Installed and verified")
                    }
                    is RuntimeStatus.Downloading -> {
                        Text("Downloading ${st.file}")
                        LinearProgressIndicator(progress = { if (st.total > 0) st.done.toFloat() / st.total else 0f }, Modifier.fillMaxWidth())
                        Text("%.0f / %.0f MB".format(st.done / 1e6, st.total / 1e6), style = MaterialTheme.typography.labelSmall)
                    }
                    is RuntimeStatus.Verifying -> { Text("Verifying ${st.file}"); LinearProgressIndicator(Modifier.fillMaxWidth()) }
                    is RuntimeStatus.Extracting -> {
                        Text("Unpacking Blender")
                        LinearProgressIndicator(progress = { if (st.total > 0) st.done.toFloat() / st.total else 0f }, Modifier.fillMaxWidth())
                    }
                    is RuntimeStatus.Failed -> Text(st.message, color = MaterialTheme.colorScheme.error)
                    RuntimeStatus.NotInstalled -> Text("Not installed. One-time download, Wi-Fi recommended.")
                }
            }
        }
        if (st != RuntimeStatus.Ready) {
            Button(onClick = vm::installRuntime, enabled = !busy, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.CloudDownload, null); Spacer(Modifier.width(8.dp)); Text("Install Blender engine")
            }
        } else {
            OutlinedButton(onClick = vm::uninstallRuntime, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Remove engine") }
        }
    }
}
