package com.smitnk.img23d.ui

import android.text.format.DateUtils
import androidx.compose.foundation.background
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
    LazyVerticalGrid(GridCells.Adaptive(150.dp), contentPadding = PaddingValues(Space.lg),
        horizontalArrangement = Arrangement.spacedBy(Space.md), verticalArrangement = Arrangement.spacedBy(Space.md), modifier = Modifier.fillMaxSize()) {
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
            val cs = MaterialTheme.colorScheme
            Column(
                Modifier.staggerIn(i.coerceAtMost(8)).clip(MaterialTheme.shapes.large).background(cs.surfaceContainerLow)
                    .clickable(onClickLabel = "Open ${m.name}") { vm.go(Dest.Result(m.id)) },
            ) {
                ModelThumb(m, Modifier.fillMaxWidth().aspectRatio(1f))
                Column(Modifier.padding(horizontal = Space.md, vertical = Space.sm), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(m.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text("${m.source.label} · ${DateUtils.getRelativeTimeSpanString(m.createdAt)}", style = MaterialTheme.typography.bodySmall,
                        color = cs.onSurfaceVariant, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text("%,d faces".format(m.faces), style = MaterialTheme.typography.labelSmall, color = cs.tertiary)
                }
            }
        }
    }
}
