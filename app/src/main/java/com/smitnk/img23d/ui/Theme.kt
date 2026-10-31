package com.smitnk.img23d.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

object Palette {
    val Background = Color(0xFF0F172A)
    val Card = Color(0xFF1B2336)
    val Muted = Color(0xFF272F42)
    val MutedText = Color(0xFF94A3B8)
    val Border = Color(0xFF475569)
    val Accent = Color(0xFF22C55E)
    val OnAccent = Color(0xFF0F172A)
    val Text = Color(0xFFF8FAFC)
    val Danger = Color(0xFFEF4444)
}

private val scheme = darkColorScheme(
    primary = Palette.Accent,
    onPrimary = Palette.OnAccent,
    secondary = Color(0xFF334155),
    onSecondary = Palette.Text,
    background = Palette.Background,
    onBackground = Palette.Text,
    surface = Palette.Background,
    onSurface = Palette.Text,
    surfaceVariant = Palette.Card,
    onSurfaceVariant = Palette.MutedText,
    surfaceContainer = Palette.Card,
    surfaceContainerHigh = Palette.Muted,
    outline = Palette.Border,
    error = Palette.Danger,
    onError = Color.Black,
)

private val typography = Typography(
    headlineMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 26.sp, lineHeight = 32.sp),
    titleLarge = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 20.sp, lineHeight = 26.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.Medium, fontSize = 16.sp, lineHeight = 22.sp),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelMedium = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 16.sp),
)

@Composable
fun Img23DTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = scheme, typography = typography, content = content)
