package com.smitnk.img23d.ui

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Collections
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.Dest
import com.smitnk.img23d.PhotoMode
import com.smitnk.img23d.UiState
import com.smitnk.img23d.data.Model
import com.smitnk.img23d.data.Settings

class AppCallbacks(
    val pickImage: (PhotoMode) -> Unit,
    val pickScript: () -> Unit,
    val export: (Model, Boolean) -> Unit,
    val share: (Model) -> Unit,
    val copy: (String, String) -> Unit,
)

private val tabs = listOf(Dest.Home, Dest.Library, Dest.Settings)

@Composable
fun AppRoot(state: UiState, settings: Settings, vm: AppViewModel, cb: AppCallbacks) {
    val d = state.dest
    val isTab = d in tabs
    val fade = if (LocalReducedMotion.current) 0 else 220
    Scaffold(
        containerColor = MaterialTheme.colorScheme.surface,
        topBar = { if (!isTab) DetailBar(titleFor(d), onBack = { vm.back() }) },
        bottomBar = {
            if (isTab) NavigationBar(containerColor = MaterialTheme.colorScheme.surfaceContainerLow) {
                listOf(Triple(Dest.Home, Icons.Rounded.Home, "Studio"), Triple(Dest.Library, Icons.Rounded.Collections, "Library"),
                    Triple(Dest.Settings, Icons.Rounded.Settings, "Settings")).forEach { (t, icon, label) ->
                    NavigationBarItem(selected = d == t, onClick = { vm.tab(t) }, icon = { Icon(icon, null) }, label = { Text(label) })
                }
            }
        },
    ) { pad ->
        AnimatedContent(
            targetState = d,
            transitionSpec = { fadeIn(tween(fade)) togetherWith fadeOut(tween(fade / 2)) },
            modifier = Modifier.padding(pad).fillMaxSize(),
            label = "dest",
        ) { dest ->
            Box(Modifier.fillMaxSize()) {
                when (dest) {
                    Dest.Home -> HomeScreen(state, vm, cb)
                    Dest.Library -> LibraryScreen(state, vm)
                    Dest.Settings -> SettingsScreen(state, settings, vm, cb)
                    is Dest.Photo -> PhotoScreen(dest.mode, state, settings, vm, cb)
                    Dest.Builder -> BuilderScreen(state, vm)
                    Dest.Body -> BodyScreen(state, vm)
                    Dest.Script -> ScriptScreen(state, vm, cb)
                    Dest.Text -> TextScreen(state, settings, vm)
                    Dest.Presets -> PresetsScreen(vm)
                    is Dest.PresetConfig -> PresetConfigScreen(dest.preset, state, vm)
                    Dest.Running -> RunningScreen(state, vm)
                    is Dest.Result -> vm.model(dest.id)?.let { ResultScreen(it, settings, vm, cb) }
                }
            }
        }
    }
    state.failure?.let { f ->
        AlertDialog(
            onDismissRequest = vm::dismissFailure,
            icon = { Icon(Icons.Rounded.ErrorOutline, null, tint = MaterialTheme.colorScheme.error) },
            title = { Text(f.title) },
            text = {
                Text(f.message, style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.heightIn(max = 340.dp).verticalScroll(rememberScrollState()))
            },
            confirmButton = {
                if (f.canRepair) TextButton(onClick = vm::repairTextScript) {
                    Icon(Icons.Rounded.AutoAwesome, null); Text("  Fix with AI")
                } else TextButton(onClick = vm::dismissFailure) { Text("OK") }
            },
            dismissButton = {
                Box {
                    TextButton(onClick = { cb.copy("Error log", f.title + "\n" + f.message) }) { Text("Copy log") }
                }
            },
        )
    }
}

private fun titleFor(d: Dest) = when (d) {
    is Dest.Photo -> if (d.mode == PhotoMode.CHARACTER) "Photo → Character" else "Photo → 3D"
    Dest.Builder -> "Character Builder"
    Dest.Body -> "Body Creator"
    Dest.Script -> "Python script"
    Dest.Text -> "Text to 3D"
    Dest.Presets -> "Model Forge presets"
    is Dest.PresetConfig -> d.preset.title
    Dest.Running -> "Working"
    is Dest.Result -> "Model"
    else -> ""
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DetailBar(title: String, onBack: () -> Unit) {
    TopAppBar(
        title = { Text(title, style = MaterialTheme.typography.titleLarge) },
        navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, "Back") } },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface),
    )
}
