package com.smitnk.img23d.ui

import android.graphics.BitmapFactory
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AddPhotoAlternate
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Code
import androidx.compose.material.icons.rounded.Face
import androidx.compose.material.icons.rounded.Memory
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.Tune
import androidx.compose.material.icons.rounded.ViewInAr
import androidx.compose.material.icons.rounded.Widgets
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.smitnk.img23d.AppViewModel
import com.smitnk.img23d.Dest
import com.smitnk.img23d.PhotoMode
import com.smitnk.img23d.UiState
import com.smitnk.img23d.blender.RuntimeStatus
import com.smitnk.img23d.data.Model
import com.smitnk.img23d.data.ModelStatus
import com.smitnk.img23d.data.Source
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
fun HomeScreen(state: UiState, vm: AppViewModel, cb: AppCallbacks) {
    val cs = MaterialTheme.colorScheme
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Hero(Modifier.staggerIn(0))
        EngineStrip(state, Modifier.staggerIn(1)) { vm.tab(Dest.Settings) }

        SectionHeader("Create")
        // Asymmetric bento: one tall feature tile beside two stacked tiles, then a wide row.
        Row(Modifier.fillMaxWidth().height(260.dp).staggerIn(2), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            BentoTile(
                "Photo → Character", "TripoSR shape, Blender clean-up, rig", Icons.Rounded.Face, Source.CHARACTER.art(),
                Modifier.weight(1.15f).fillMaxSize(), tag = "BOTH ENGINES",
            ) { cb.pickImage(PhotoMode.CHARACTER) }
            Column(Modifier.weight(1f).fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                BentoTile("Photo → 3D", "Raw TripoSR mesh", Icons.Rounded.AddPhotoAlternate, Source.PHOTO.art(), Modifier.weight(1f).fillMaxWidth(), compact = true) {
                    cb.pickImage(PhotoMode.MESH)
                }
                BentoTile("Builder", "Sliders & styles", Icons.Rounded.Tune, Source.BUILDER.art(), Modifier.weight(1f).fillMaxWidth(), compact = true) {
                    vm.go(Dest.Builder)
                }
            }
        }
        Row(Modifier.fillMaxWidth().height(132.dp).staggerIn(3), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            BentoTile("Python script", "Any AI's bpy code", Icons.Rounded.Code, Source.SCRIPT.art(), Modifier.weight(1f).fillMaxSize(), compact = true) {
                vm.go(Dest.Script)
            }
            BentoTile("Text to 3D", "Optional · API key", Icons.Rounded.AutoAwesome, Source.TEXT.art(), Modifier.weight(1f).fillMaxSize(), compact = true) {
                vm.go(Dest.Text)
            }
            BentoTile("Presets", "7 Forge models", Icons.Rounded.Widgets, Source.PRESET.art(), Modifier.weight(1f).fillMaxSize(), compact = true) {
                vm.go(Dest.Presets)
            }
        }

        if (state.models.isNotEmpty()) {
            SectionHeader("Recent") {
                Text("See all", style = MaterialTheme.typography.labelLarge, color = cs.tertiary,
                    modifier = Modifier.clip(CircleShape).clickable { vm.tab(Dest.Library) }.padding(8.dp))
            }
            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.staggerIn(4)) {
                items(state.models.take(12), key = { it.id }) { m -> RecentCard(m) { vm.go(Dest.Result(m.id)) } }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun Hero(modifier: Modifier) {
    val cs = MaterialTheme.colorScheme
    val reduce = LocalReducedMotion.current
    val t by rememberInfiniteTransition(label = "hero").animateFloat(0f, 1f, infiniteRepeatable(tween(9000, easing = LinearEasing), RepeatMode.Reverse), label = "t")
    val shift = if (reduce) 0.5f else t
    Box(
        modifier.fillMaxWidth().padding(top = 12.dp).clip(MaterialTheme.shapes.extraLarge)
            .background(Brush.linearGradient(listOf(cs.primaryContainer, cs.surfaceContainerHigh, cs.tertiaryContainer),
                start = Offset(shift * 600f, 0f), end = Offset(900f - shift * 300f, 700f)))
            .padding(24.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("IMG-2-3D STUDIO", style = MaterialTheme.typography.labelMedium, color = cs.onPrimaryContainer)
            Text("Make 3D.\nOn your phone.", style = MaterialTheme.typography.displayMedium, color = cs.onPrimaryContainer)
            Text("TripoSR and real Blender 4.2, both running locally.", style = MaterialTheme.typography.bodyMedium, color = cs.onPrimaryContainer.copy(alpha = 0.85f))
        }
    }
}

@Composable
private fun EngineStrip(state: UiState, modifier: Modifier, onClick: () -> Unit) {
    Row(modifier.fillMaxWidth().clickable(onClickLabel = "Manage engines", onClick = onClick), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        EngineDot("TripoSR", state.modelStatus == ModelStatus.Ready, Modifier.weight(1f))
        EngineDot("Blender 4.2", state.runtime == RuntimeStatus.Ready, Modifier.weight(1f))
    }
}

@Composable
private fun EngineDot(name: String, ready: Boolean, modifier: Modifier) {
    val cs = MaterialTheme.colorScheme
    Row(
        modifier.clip(CircleShape).background(cs.surfaceContainer).padding(horizontal = 14.dp, vertical = 10.dp).semantics(mergeDescendants = true) {},
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(10.dp).clip(CircleShape).background(if (ready) cs.tertiary else cs.outline))
        Spacer(Modifier.width(8.dp))
        Icon(Icons.Rounded.Memory, null, Modifier.size(16.dp), tint = cs.onSurfaceVariant)
        Spacer(Modifier.width(6.dp))
        Text("$name · ${if (ready) "ready" else "set up"}", style = MaterialTheme.typography.labelLarge, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun BentoTile(
    title: String, sub: String, icon: ImageVector, colors: List<Color>, modifier: Modifier,
    compact: Boolean = false, tag: String? = null, onClick: () -> Unit,
) {
    val cs = MaterialTheme.colorScheme
    Box(
        modifier.clip(MaterialTheme.shapes.large).background(cs.surfaceContainer)
            .clickable(onClickLabel = title, onClick = onClick).semantics(mergeDescendants = true) {},
    ) {
        // Soft gradient glow in the corner gives each mode its own identity.
        Box(Modifier.matchParentSizeGlow(colors))
        Column(Modifier.padding(14.dp).fillMaxSize(), verticalArrangement = Arrangement.SpaceBetween) {
            GradientBadge(colors, icon, size = if (compact) 36 else 52)
            Column {
                if (tag != null) Text(tag, style = MaterialTheme.typography.labelMedium, color = colors.first())
                Text(title, style = if (compact) MaterialTheme.typography.titleSmall else MaterialTheme.typography.headlineSmall,
                    maxLines = 2, overflow = TextOverflow.Ellipsis)
                Text(sub, style = MaterialTheme.typography.bodySmall, color = cs.onSurfaceVariant, maxLines = 2, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

private fun Modifier.matchParentSizeGlow(colors: List<Color>) = this.fillMaxSize()
    .background(Brush.linearGradient(listOf(Color.Transparent, Color.Transparent, colors.last().copy(alpha = 0.32f))))

@Composable
fun RecentCard(m: Model, onClick: () -> Unit) {
    Column(Modifier.width(132.dp).clip(MaterialTheme.shapes.medium).clickable(onClickLabel = "Open ${m.name}", onClick = onClick)) {
        ModelThumb(m, Modifier.fillMaxWidth().aspectRatio(1f))
        Text(m.name, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(top = 6.dp))
        Text(m.source.label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 1)
    }
}

@Composable
fun ModelThumb(m: Model, modifier: Modifier) {
    val bmp by produceState<android.graphics.Bitmap?>(null, m.id) {
        value = m.thumbnail?.let { f -> withContext(Dispatchers.IO) { BitmapFactory.decodeFile(f.path, BitmapFactory.Options().apply { inSampleSize = 2 }) } }
    }
    Box(modifier.clip(MaterialTheme.shapes.medium).background(Brush.linearGradient(m.source.art())), contentAlignment = Alignment.Center) {
        val b = bmp
        if (b != null) Image(b.asImageBitmap(), m.name, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        else Icon(if (m.source == Source.BUILDER || m.source == Source.PRESET) Icons.Rounded.Person else Icons.Rounded.ViewInAr, null,
            Modifier.size(40.dp), tint = Color(0xFF120A0F))
    }
}
