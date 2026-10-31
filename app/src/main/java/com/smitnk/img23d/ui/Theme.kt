package com.smitnk.img23d.ui

import android.os.Build
import android.provider.Settings
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Studio brand: hot pink (primary) + electric cyan (tertiary) on ink, built as M3 roles.
private val Studio = darkColorScheme(
    primary = Color(0xFFFF8AC4), onPrimary = Color(0xFF5C0034),
    primaryContainer = Color(0xFF8A1F5A), onPrimaryContainer = Color(0xFFFFD8E8),
    secondary = Color(0xFFE2BDCD), onSecondary = Color(0xFF422935),
    secondaryContainer = Color(0xFF5B3F4C), onSecondaryContainer = Color(0xFFFFD8E8),
    tertiary = Color(0xFF5ED6F2), onTertiary = Color(0xFF00363F),
    tertiaryContainer = Color(0xFF004E5B), onTertiaryContainer = Color(0xFFA6EEFF),
    background = Color(0xFF0E0B10), onBackground = Color(0xFFECE0E4),
    surface = Color(0xFF0E0B10), onSurface = Color(0xFFECE0E4),
    surfaceVariant = Color(0xFF514348), onSurfaceVariant = Color(0xFFD5C2C8),
    surfaceContainerLowest = Color(0xFF09070A), surfaceContainerLow = Color(0xFF181316),
    surfaceContainer = Color(0xFF1E171B), surfaceContainerHigh = Color(0xFF292125),
    surfaceContainerHighest = Color(0xFF342B30),
    outline = Color(0xFF9E8C92), outlineVariant = Color(0xFF514348),
    error = Color(0xFFFFB4AB), onError = Color(0xFF690005), errorContainer = Color(0xFF93000A), onErrorContainer = Color(0xFFFFDAD6),
)

private val StudioType = Typography(
    displayMedium = TextStyle(fontWeight = FontWeight.Black, fontSize = 44.sp, lineHeight = 46.sp, letterSpacing = (-1.5).sp),
    displaySmall = TextStyle(fontWeight = FontWeight.ExtraBold, fontSize = 34.sp, lineHeight = 38.sp, letterSpacing = (-1).sp),
    headlineMedium = TextStyle(fontWeight = FontWeight.ExtraBold, fontSize = 28.sp, lineHeight = 32.sp, letterSpacing = (-0.5).sp),
    headlineSmall = TextStyle(fontWeight = FontWeight.Bold, fontSize = 22.sp, lineHeight = 28.sp),
    titleLarge = TextStyle(fontWeight = FontWeight.Bold, fontSize = 20.sp, lineHeight = 26.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp, lineHeight = 22.sp),
    titleSmall = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 14.sp, lineHeight = 20.sp),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    bodySmall = TextStyle(fontSize = 12.sp, lineHeight = 16.sp),
    labelLarge = TextStyle(fontWeight = FontWeight.Bold, fontSize = 14.sp, lineHeight = 20.sp, letterSpacing = 0.2.sp),
    labelMedium = TextStyle(fontWeight = FontWeight.Bold, fontSize = 12.sp, lineHeight = 16.sp, letterSpacing = 1.2.sp),
    labelSmall = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp),
)

private val StudioShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp), small = RoundedCornerShape(10.dp), medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(24.dp), extraLarge = RoundedCornerShape(32.dp),
)

val LocalReducedMotion = staticCompositionLocalOf { false }

@Composable
fun StudioTheme(dynamicColor: Boolean, reduceMotion: Boolean, content: @Composable () -> Unit) {
    val ctx = LocalContext.current
    val scheme = if (dynamicColor && Build.VERSION.SDK_INT >= 31) dynamicDarkColorScheme(ctx) else Studio
    val systemReduce = remember { Settings.Global.getFloat(ctx.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f }
    CompositionLocalProvider(LocalReducedMotion provides (reduceMotion || systemReduce)) {
        MaterialTheme(colorScheme = scheme, typography = StudioType, shapes = StudioShapes, content = content)
    }
}
