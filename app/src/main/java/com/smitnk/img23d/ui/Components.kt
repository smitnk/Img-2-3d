package com.smitnk.img23d.ui

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
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
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.smitnk.img23d.data.Source

fun parseHex(h: String): Color = runCatching { Color(android.graphics.Color.parseColor(h)) }.getOrDefault(Color.Gray)

/** Art gradient per model source, so library tiles without a photo still read at a glance. */
fun Source.art(): List<Color> = when (this) {
    Source.PHOTO -> listOf(Color(0xFF5ED6F2), Color(0xFF2B6CB0))
    Source.CHARACTER -> listOf(Color(0xFFFF8AC4), Color(0xFF5ED6F2))
    Source.REFINED -> listOf(Color(0xFF7DD3A8), Color(0xFF2F855A))
    Source.BUILDER -> listOf(Color(0xFFFFB86B), Color(0xFFE5486B))
    Source.SCRIPT -> listOf(Color(0xFFB794F4), Color(0xFF4C51BF))
    Source.TEXT -> listOf(Color(0xFFF6E05E), Color(0xFFDD6B20))
    Source.PRESET -> listOf(Color(0xFFE2BDCD), Color(0xFF8A1F5A))
    Source.BODY -> listOf(Color(0xFFF2C9A0), Color(0xFFB4533C))
}

/** Entrance animation: fade + rise, staggered by [index]; instant under reduced motion. */
@Composable
fun Modifier.staggerIn(index: Int): Modifier {
    val reduce = LocalReducedMotion.current
    val a = remember { Animatable(if (reduce) 1f else 0f) }
    LaunchedEffect(Unit) { if (!reduce) { kotlinx.coroutines.delay(40L * index); a.animateTo(1f, tween(380)) } }
    return graphicsLayer { alpha = a.value; translationY = (1f - a.value) * 36f }
}

@Composable
fun SectionHeader(text: String, modifier: Modifier = Modifier, trailing: @Composable (() -> Unit)? = null) {
    Row(modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(text.uppercase(), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.weight(1f).semantics { heading() })
        trailing?.invoke()
    }
}

@Composable
fun Pill(icon: ImageVector?, text: String, modifier: Modifier = Modifier, tint: Color = MaterialTheme.colorScheme.primary) {
    Row(
        modifier.clip(CircleShape).background(MaterialTheme.colorScheme.surfaceContainerHigh).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) { Icon(icon, null, Modifier.size(16.dp), tint = tint); Spacer(Modifier.width(6.dp)) }
        Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
fun Panel(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(
        modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.large,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
    ) { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) { content() } }
}

@Composable
fun GradientBadge(colors: List<Color>, icon: ImageVector, modifier: Modifier = Modifier, size: Int = 44) {
    Box(modifier.size(size.dp).clip(MaterialTheme.shapes.medium).background(Brush.linearGradient(colors)), contentAlignment = Alignment.Center) {
        Icon(icon, null, Modifier.size((size * 0.5).dp), tint = Color(0xFF120A0F))
    }
}

@Composable
fun ToggleRow(title: String, sub: String?, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clickable { onChange(!checked) }.padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleSmall)
            if (sub != null) Text(sub, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, format: String = "%.2f", steps: Int = 0, onChange: (Float) -> Unit) {
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
            Text(format.format(value), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.tertiary)
        }
        Slider(value = value, onValueChange = onChange, valueRange = range, steps = steps,
            modifier = Modifier.semantics { contentDescription = label; stateDescription = format.format(value) })
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ChoiceChips(label: String, options: List<String>, selected: String, onPick: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(label, style = MaterialTheme.typography.titleSmall)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            options.forEach { o -> FilterChip(selected = o == selected, onClick = { onPick(o) }, label = { Text(o.replaceFirstChar { it.uppercase() }) }) }
        }
    }
}

val SwatchPalette = listOf(
    "#F2C9A0", "#C68B59", "#8D5524", "#3B2417", "#F7F7F2", "#E8553D", "#F59E0B", "#F5C542",
    "#22C55E", "#14B8A6", "#3A6EA5", "#2F4A8A", "#7C3AED", "#EC4899", "#111827", "#9CA3AF",
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ColorPicker(label: String, value: String?, allowDefault: Boolean, onPick: (String?) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
            Text(value ?: "generator default", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (allowDefault) FilterChip(selected = value == null, onClick = { onPick(null) }, label = { Text("Default") })
            SwatchPalette.forEach { hex ->
                val sel = value.equals(hex, ignoreCase = true)
                val c = parseHex(hex)
                Box(
                    Modifier.size(36.dp).clip(CircleShape).background(c)
                        .border(if (sel) 3.dp else 1.dp, if (sel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant, CircleShape)
                        .clickable(onClickLabel = "Use $hex for $label") { onPick(hex) }
                        .semantics { contentDescription = "$label $hex" + if (sel) ", selected" else "" },
                    contentAlignment = Alignment.Center,
                ) {
                    if (sel) Icon(Icons.Rounded.Check, null, Modifier.size(18.dp),
                        tint = if (0.2126f * c.red + 0.7152f * c.green + 0.0722f * c.blue > 0.55f) Color.Black else Color.White)
                }
            }
        }
    }
}
