package com.smitnk.img23d.ui

import android.graphics.BitmapFactory
import android.text.format.DateUtils
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.AddPhotoAlternate
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.CloudDownload
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.RadioButtonUnchecked
import androidx.compose.material.icons.rounded.SaveAlt
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.ViewInAr
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.Screen
import com.smitnk.img23d.UiState
import com.smitnk.img23d.core.TripoSR
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Project
import com.smitnk.img23d.viewer.ModelView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class AppCallbacks(
    val pickImage: () -> Unit,
    val export: (Project) -> Unit,
    val share: (Project) -> Unit,
)

@Composable
fun AppRoot(state: UiState, vm: AppViewModel, cb: AppCallbacks) {
    val s = state.screen
    val title = when (s) {
        Screen.Home -> "Img-2-3D"
        Screen.Create -> "New model"
        Screen.Generating -> "Generating"
        is Screen.Result -> "Result"
        Screen.History -> "History"
        Screen.Models -> "TripoSR model"
    }
    Scaffold(
        topBar = { AppBar(title, showBack = s != Screen.Home, onBack = { vm.back() }, onHistory = { vm.navigate(Screen.History) }.takeIf { s == Screen.Home }) },
        containerColor = MaterialTheme.colorScheme.background,
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            when (s) {
                Screen.Home -> HomeScreen(state, vm, cb)
                Screen.Create -> CreateScreen(state, vm)
                Screen.Generating -> GeneratingScreen(state, vm)
                is Screen.Result -> vm.project(s.projectId)?.let { ResultScreen(it, vm, cb) }
                Screen.History -> HistoryScreen(state, vm)
                Screen.Models -> ModelsScreen(state, vm)
            }
        }
    }
    state.error?.let { msg ->
        AlertDialog(
            onDismissRequest = vm::dismissError,
            confirmButton = { TextButton(onClick = vm::dismissError) { Text("OK") } },
            title = { Text("Something went wrong") },
            text = { Text(msg) },
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppBar(title: String, showBack: Boolean, onBack: () -> Unit, onHistory: (() -> Unit)?) {
    TopAppBar(
        title = { Text(title, style = MaterialTheme.typography.titleLarge) },
        navigationIcon = {
            if (showBack) IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, "Back") }
        },
        actions = {
            if (onHistory != null) IconButton(onClick = onHistory) { Icon(Icons.Rounded.History, "History") }
        },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
    )
}

@Composable
private fun SectionCard(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Palette.Card),
        border = BorderStroke(1.dp, Palette.Muted),
    ) { Column(Modifier.padding(16.dp)) { content() } }
}

@Composable
private fun HomeScreen(state: UiState, vm: AppViewModel, cb: AppCallbacks) {
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Turn one photo into a 3D model", style = MaterialTheme.typography.headlineMedium)
        Text(
            "Runs the original TripoSR network fully on this phone. No uploads.",
            color = Palette.MutedText,
        )
        val ready = state.modelStatus == ModelStatus.Ready
        if (!ready) {
            SectionCard {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.CloudDownload, null, tint = Palette.Accent)
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Text("Model not installed", style = MaterialTheme.typography.titleMedium)
                        Text("TripoSR weights (~1.9 GB) are needed once.", color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium)
                    }
                }
                Spacer(Modifier.height(12.dp))
                Button(onClick = { vm.navigate(Screen.Models) }, Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Set up model") }
            }
        }
        Box(
            Modifier.fillMaxWidth().aspectRatio(1.4f).clip(RoundedCornerShape(20.dp))
                .background(Palette.Card).clickable(onClickLabel = "Choose image from gallery", onClick = cb.pickImage),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(Icons.Rounded.AddPhotoAlternate, null, Modifier.size(56.dp), tint = Palette.Accent)
                Spacer(Modifier.height(12.dp))
                Text("Choose an image", style = MaterialTheme.typography.titleMedium)
                Text("Single object, clear subject works best", color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium)
            }
        }
        Steps()
        if (state.projects.isNotEmpty()) {
            Text("Recent", style = MaterialTheme.typography.titleMedium)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                items(state.projects.take(10), key = { it.id }) { p ->
                    Thumb(p, Modifier.size(120.dp).clickable(onClickLabel = "Open model") { vm.navigate(Screen.Result(p.id)) })
                }
            }
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun Steps() {
    SectionCard {
        listOf("1" to "Pick a photo", "2" to "TripoSR reconstructs a triplane on-device", "3" to "Mesh is extracted and exported as GLB").forEach { (n, t) ->
            Row(Modifier.padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(28.dp).clip(RoundedCornerShape(8.dp)).background(Palette.Muted), contentAlignment = Alignment.Center) {
                    Text(n, color = Palette.Accent, style = MaterialTheme.typography.labelMedium)
                }
                Spacer(Modifier.width(12.dp))
                Text(t, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

@Composable
private fun Thumb(p: Project, modifier: Modifier) {
    val bmp by produceState<android.graphics.Bitmap?>(null, p.id) {
        value = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(p.thumbnail.path, BitmapFactory.Options().apply { inSampleSize = 2 }) }
    }
    Box(modifier.clip(RoundedCornerShape(12.dp)).background(Palette.Muted)) {
        bmp?.let { Image(it.asImageBitmap(), "Model input", Modifier.fillMaxSize(), contentScale = ContentScale.Crop) }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CreateScreen(state: UiState, vm: AppViewModel) {
    val img = state.image ?: return
    val set = state.settings
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Image(
            img.asImageBitmap(), "Selected image",
            Modifier.fillMaxWidth().heightIn(max = 360.dp).clip(RoundedCornerShape(16.dp)).background(Palette.Card),
            contentScale = ContentScale.Fit,
        )
        SectionCard {
            Text("Mesh resolution", style = MaterialTheme.typography.titleMedium)
            Text("Marching-cubes grid. Higher = more detail, more time and RAM.", color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(8.dp))
            val opts = listOf(128, 192, 256)
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                opts.forEachIndexed { i, r ->
                    SegmentedButton(
                        selected = set.resolution == r,
                        onClick = { vm.updateSettings { it.copy(resolution = r) } },
                        shape = SegmentedButtonDefaults.itemShape(i, opts.size),
                    ) { Text("$r³") }
                }
            }
            Spacer(Modifier.height(16.dp))
            ToggleRow("Remove background", "rembg u2net. Skipped automatically for images with transparency.", set.removeBackground) { v ->
                vm.updateSettings { it.copy(removeBackground = v) }
            }
            ToggleRow("Vertex colors", "Sample TripoSR's color field at each vertex.", set.vertexColors) { v ->
                vm.updateSettings { it.copy(vertexColors = v) }
            }
            Spacer(Modifier.height(8.dp))
            Text("Foreground ratio: %.2f".format(set.foregroundRatio), style = MaterialTheme.typography.titleMedium)
            Slider(
                value = set.foregroundRatio,
                onValueChange = { v -> vm.updateSettings { it.copy(foregroundRatio = v) } },
                valueRange = 0.5f..1f,
                modifier = Modifier.semantics { contentDescription = "Foreground ratio" },
            )
        }
        Button(
            onClick = vm::generate,
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.OnAccent),
        ) {
            Icon(Icons.Rounded.ViewInAr, null)
            Spacer(Modifier.width(8.dp))
            Text(if (state.modelStatus == ModelStatus.Ready) "Generate 3D" else "Install model first", style = MaterialTheme.typography.titleMedium)
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
private fun ToggleRow(title: String, sub: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable { onChange(!checked) }.padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(sub, color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

private val stageLabels = linkedMapOf(
    TripoSR.Stage.BACKGROUND to "Removing background",
    TripoSR.Stage.PREPROCESS to "Preparing image",
    TripoSR.Stage.ENCODER_LOAD to "Loading TripoSR weights",
    TripoSR.Stage.ENCODER to "Running TripoSR network",
    TripoSR.Stage.FIELD to "Evaluating density field",
    TripoSR.Stage.MARCHING_CUBES to "Marching cubes",
    TripoSR.Stage.COLORS to "Sampling vertex colors",
)

@Composable
private fun GeneratingScreen(state: UiState, vm: AppViewModel) {
    val p = state.progress
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        SectionCard {
            stageLabels.forEach { (stage, label) ->
                val done = stage.ordinal < p.stage.ordinal
                val active = stage == p.stage
                Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    when {
                        done -> Icon(Icons.Rounded.CheckCircle, "Done", tint = Palette.Accent)
                        active -> CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 3.dp)
                        else -> Icon(Icons.Rounded.RadioButtonUnchecked, "Pending", tint = Palette.Border)
                    }
                    Spacer(Modifier.width(12.dp))
                    Text(label, color = if (done || active) Palette.Text else Palette.MutedText)
                }
                if (active && (stage == TripoSR.Stage.FIELD || stage == TripoSR.Stage.COLORS)) {
                    LinearProgressIndicator(progress = { p.fraction }, modifier = Modifier.fillMaxWidth().padding(start = 36.dp, bottom = 4.dp))
                }
            }
        }
        SectionCard {
            Stat(Icons.Rounded.ViewInAr, "Elapsed", "%.1f s".format(p.elapsedMs / 1000f))
            Stat(Icons.Rounded.Memory, "App memory (PSS)", "${p.pssMb} MB · peak ${p.peakPssMb} MB")
        }
        Text(
            "The network step can take a minute or more on CPU. Keep the app open.",
            color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedButton(onClick = vm::cancel, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Cancel") }
    }
}

@Composable
private fun Stat(icon: ImageVector, label: String, value: String) {
    Row(Modifier.padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(icon, null, tint = Palette.MutedText, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(8.dp))
        Text(label, Modifier.weight(1f), color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.labelMedium)
    }
}

@Composable
private fun ResultScreen(p: Project, vm: AppViewModel, cb: AppCallbacks) {
    var confirmDelete by remember { mutableStateOf(false) }
    val glb by produceState<ByteArray?>(null, p.id) { value = withContext(Dispatchers.IO) { p.glb.readBytes() } }
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().weight(1f).background(Palette.Background)) {
            val bytes = glb
            if (bytes == null) CircularProgressIndicator(Modifier.align(Alignment.Center))
            else AndroidView(factory = { ModelView(it).apply { load(bytes) } }, modifier = Modifier.fillMaxSize())
            Text(
                "Drag to rotate · pinch to zoom · two fingers to pan · double-tap to reset",
                Modifier.align(Alignment.BottomCenter).padding(8.dp),
                color = Palette.MutedText, style = MaterialTheme.typography.labelMedium, textAlign = TextAlign.Center,
            )
        }
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SectionCard {
                Stat(Icons.Rounded.ViewInAr, "Geometry", "%,d verts · %,d faces · %d³".format(p.vertices, p.faces, p.resolution))
                Stat(Icons.Rounded.Memory, "Peak memory (PSS)", "${p.peakPssMb} MB")
                val t = p.timingsMs
                Stat(Icons.Rounded.History, "Network (load + infer)", "%.1f s + %.1f s".format((t["encoder_load"] ?: 0) / 1000f, (t["encoder"] ?: 0) / 1000f))
                Stat(Icons.Rounded.History, "Mesh (field + MC + color)", "%.1f s".format(((t["field"] ?: 0) + (t["marching_cubes"] ?: 0) + (t["colors"] ?: 0)) / 1000f))
                Stat(Icons.Rounded.History, "Total", "%.1f s".format(p.totalMs / 1000f))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = { cb.share(p) }, Modifier.weight(1f).heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.Share, null); Spacer(Modifier.width(6.dp)); Text("Share")
                }
                OutlinedButton(onClick = { cb.export(p) }, Modifier.weight(1f).heightIn(min = 48.dp)) {
                    Icon(Icons.Rounded.SaveAlt, null); Spacer(Modifier.width(6.dp)); Text("Save GLB")
                }
                IconButton(onClick = { confirmDelete = true }, Modifier.size(48.dp)) { Icon(Icons.Rounded.Delete, "Delete model", tint = Palette.Danger) }
            }
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete this model?") },
            text = { Text("The GLB and its input preview will be removed from this phone.") },
            confirmButton = { TextButton(onClick = { confirmDelete = false; vm.deleteProject(p); vm.back() }) { Text("Delete", color = Palette.Danger) } },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Keep") } },
        )
    }
}

@Composable
private fun HistoryScreen(state: UiState, vm: AppViewModel) {
    LaunchedEffect(Unit) { vm.refreshProjects() }
    if (state.projects.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No models yet", color = Palette.MutedText) }
        return
    }
    LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items(state.projects, key = { it.id }) { p ->
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(Palette.Card)
                    .clickable(onClickLabel = "Open model") { vm.navigate(Screen.Result(p.id)) }.padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Thumb(p, Modifier.size(64.dp))
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(DateUtils.getRelativeTimeSpanString(p.createdAt).toString(), style = MaterialTheme.typography.titleMedium)
                    Text("%,d faces · %d³ · %.0f s".format(p.faces, p.resolution, p.totalMs / 1000f), color = Palette.MutedText, style = MaterialTheme.typography.labelMedium)
                }
            }
        }
    }
}

@Composable
private fun ModelsScreen(state: UiState, vm: AppViewModel) {
    val st = state.modelStatus
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        SectionCard {
            Text("Original TripoSR weights (stabilityai/TripoSR), exported to ONNX without modification, plus rembg u2net for background removal.", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(8.dp))
            Text("Stored at", color = Palette.MutedText, style = MaterialTheme.typography.bodyMedium)
            Text(vm.models.dir.path, style = MaterialTheme.typography.labelMedium)
        }
        SectionCard {
            when (st) {
                ModelStatus.Ready -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.CheckCircle, null, tint = Palette.Accent); Spacer(Modifier.width(8.dp)); Text("Installed and verified")
                }
                is ModelStatus.Downloading -> {
                    Text("Downloading ${st.file}")
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(progress = { if (st.total > 0) st.done.toFloat() / st.total else 0f }, Modifier.fillMaxWidth())
                    Text("%.0f / %.0f MB".format(st.done / 1e6, st.total / 1e6), style = MaterialTheme.typography.labelMedium)
                }
                is ModelStatus.Verifying -> Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp); Spacer(Modifier.width(8.dp)); Text("Verifying ${st.file}")
                }
                is ModelStatus.Failed -> Text(st.message, color = Palette.Danger)
                is ModelStatus.Missing, ModelStatus.Checking -> Text("Not installed (one-time download ≈1.9 GB, Wi-Fi recommended)")
            }
        }
        val busy = st is ModelStatus.Downloading || st is ModelStatus.Verifying
        if (st != ModelStatus.Ready) {
            Button(onClick = vm::downloadModels, enabled = !busy, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Icon(Icons.Rounded.CloudDownload, null); Spacer(Modifier.width(8.dp)); Text("Download model")
            }
            OutlinedButton(onClick = vm::verifyPushedModels, enabled = !busy, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) {
                Text("I copied the files with adb — verify")
            }
        } else {
            OutlinedButton(onClick = vm::deleteModels, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp)) { Text("Delete model files") }
        }
    }
}
