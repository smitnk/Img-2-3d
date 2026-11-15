package com.smitnk.img23d.ui

import android.os.Build
import android.provider.Settings
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
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

// Studio palette: calm violet (primary) + mint teal (tertiary) on blue-tinted ink; light mirrors it on a warm paper tone.
private val StudioDark = darkColorScheme(
    primary = Color(0xFFBFB3FF), onPrimary = Color(0xFF2A1B6B),
    primaryContainer = Color(0xFF41338A), onPrimaryContainer = Color(0xFFE4DEFF),
    secondary = Color(0xFFC9C3DC), onSecondary = Color(0xFF312E41),
    secondaryContainer = Color(0xFF484459), onSecondaryContainer = Color(0xFFE5DFF9),
    tertiary = Color(0xFF5FD6C4), onTertiary = Color(0xFF00382F),
    tertiaryContainer = Color(0xFF005045), onTertiaryContainer = Color(0xFF7DF3DF),
    background = Color(0xFF0F0E14), onBackground = Color(0xFFE6E1EA),
    surface = Color(0xFF0F0E14), onSurface = Color(0xFFE6E1EA),
    surfaceVariant = Color(0xFF47464F), onSurfaceVariant = Color(0xFFC8C5D0),
    surfaceContainerLowest = Color(0xFF0A0A0F), surfaceContainerLow = Color(0xFF16151C),
    surfaceContainer = Color(0xFF1B1A22), surfaceContainerHigh = Color(0xFF26242D),
    surfaceContainerHighest = Color(0xFF312F38),
    outline = Color(0xFF928F9A), outlineVariant = Color(0xFF47464F),
    error = Color(0xFFFFB4AB), onError = Color(0xFF690005), errorContainer = Color(0xFF93000A), onErrorContainer = Color(0xFFFFDAD6),
)

private val StudioLight = lightColorScheme(
    primary = Color(0xFF5B4BC4), onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFE4DEFF), onPrimaryContainer = Color(0xFF170065),
    secondary = Color(0xFF5F5B71), onSecondary = Color(0xFFFFFFFF),
    secondaryContainer = Color(0xFFE5DFF9), onSecondaryContainer = Color(0xFF1C192B),
    tertiary = Color(0xFF006B5D), onTertiary = Color(0xFFFFFFFF),
    tertiaryContainer = Color(0xFF7DF3DF), onTertiaryContainer = Color(0xFF00201B),
    background = Color(0xFFFDF8FF), onBackground = Color(0xFF1C1B20),
    surface = Color(0xFFFDF8FF), onSurface = Color(0xFF1C1B20),
    surfaceVariant = Color(0xFFE5E0EC), onSurfaceVariant = Color(0xFF47464F),
    surfaceContainerLowest = Color(0xFFFFFFFF), surfaceContainerLow = Color(0xFFF7F2FA),
    surfaceContainer = Color(0xFFF1ECF4), surfaceContainerHigh = Color(0xFFECE6EE),
    surfaceContainerHighest = Color(0xFFE6E1E9),
    outline = Color(0xFF77757F), outlineVariant = Color(0xFFC8C5D0),
    error = Color(0xFFBA1A1A), onError = Color(0xFFFFFFFF), errorContainer = Color(0xFFFFDAD6), onErrorContainer = Color(0xFF410002),
)

/** Spacing tokens on the 4/8dp grid; use these instead of ad-hoc dp literals. */
object Space {
    val xs = 4.dp
    val sm = 8.dp
    val md = 12.dp
    val lg = 16.dp
    val xl = 24.dp
    /** Cap for content width on tablets / unfolded foldables. */
    val maxContent = 720.dp
}

private val StudioType = Typography(
    displayMedium = TextStyle(fontWeight = FontWeight.Bold, fontSize = 40.sp, lineHeight = 44.sp, letterSpacing = (-1).sp),
    displaySmall = TextStyle(fontWeight = FontWeight.Bold, fontSize = 32.sp, lineHeight = 38.sp, letterSpacing = (-0.5).sp),
    headlineMedium = TextStyle(fontWeight = FontWeight.Bold, fontSize = 28.sp, lineHeight = 32.sp, letterSpacing = (-0.5).sp),
    headlineSmall = TextStyle(fontWeight = FontWeight.Bold, fontSize = 22.sp, lineHeight = 28.sp),
    titleLarge = TextStyle(fontWeight = FontWeight.Bold, fontSize = 20.sp, lineHeight = 26.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp, lineHeight = 22.sp),
    titleSmall = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 14.sp, lineHeight = 20.sp),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    bodySmall = TextStyle(fontSize = 12.sp, lineHeight = 16.sp),
    labelLarge = TextStyle(fontWeight = FontWeight.Bold, fontSize = 14.sp, lineHeight = 20.sp, letterSpacing = 0.2.sp),
    labelMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 12.sp, lineHeight = 16.sp, letterSpacing = 1.0.sp),
    labelSmall = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp),
)

private val StudioShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp), small = RoundedCornerShape(10.dp), medium = RoundedCornerShape(14.dp),
    large = RoundedCornerShape(20.dp), extraLarge = RoundedCornerShape(28.dp),
)

val LocalReducedMotion = staticCompositionLocalOf { false }

@Composable
fun StudioTheme(dynamicColor: Boolean, reduceMotion: Boolean, darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    val ctx = LocalContext.current
    val scheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= 31 -> if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        darkTheme -> StudioDark
        else -> StudioLight
    }
    val systemReduce = remember { Settings.Global.getFloat(ctx.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f }
    CompositionLocalProvider(LocalReducedMotion provides (reduceMotion || systemReduce)) {
        MaterialTheme(colorScheme = scheme, typography = StudioType, shapes = StudioShapes, content = content)
    }
}
