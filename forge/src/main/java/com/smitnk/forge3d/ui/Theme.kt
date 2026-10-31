package com.smitnk.forge3d.ui

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

// Brand fallback (forge: ember amber on deep violet) for devices without dynamic colour.
private val ForgeDark = darkColorScheme(
    primary = Color(0xFFFFB95C), onPrimary = Color(0xFF462A00),
    primaryContainer = Color(0xFF653E00), onPrimaryContainer = Color(0xFFFFDDB6),
    secondary = Color(0xFFCDBDFF), onSecondary = Color(0xFF341B80),
    secondaryContainer = Color(0xFF4B3598), onSecondaryContainer = Color(0xFFE8DDFF),
    tertiary = Color(0xFF7BD8C4), onTertiary = Color(0xFF00382F),
    tertiaryContainer = Color(0xFF005045), onTertiaryContainer = Color(0xFF98F5E0),
    background = Color(0xFF140F24), onBackground = Color(0xFFE8E0F2),
    surface = Color(0xFF140F24), onSurface = Color(0xFFE8E0F2),
    surfaceVariant = Color(0xFF4A4458), onSurfaceVariant = Color(0xFFCBC3D9),
    surfaceContainerLowest = Color(0xFF0F0B1C), surfaceContainerLow = Color(0xFF1C162D),
    surfaceContainer = Color(0xFF211A33), surfaceContainerHigh = Color(0xFF2B243E),
    surfaceContainerHighest = Color(0xFF362E4A),
    outline = Color(0xFF958EA3), outlineVariant = Color(0xFF4A4458),
    error = Color(0xFFFFB4AB), onError = Color(0xFF690005),
)

private val ForgeType = Typography(
    displaySmall = TextStyle(fontWeight = FontWeight.Bold, fontSize = 34.sp, lineHeight = 40.sp, letterSpacing = (-0.5).sp),
    headlineSmall = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 24.sp, lineHeight = 30.sp),
    titleLarge = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 20.sp, lineHeight = 26.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp, lineHeight = 22.sp),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 14.sp, lineHeight = 20.sp),
    labelMedium = TextStyle(fontWeight = FontWeight.Medium, fontSize = 12.sp, lineHeight = 16.sp),
    labelSmall = TextStyle(fontFamily = FontFamily.Monospace, fontSize = 11.sp, lineHeight = 15.sp),
)

private val ForgeShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp), small = RoundedCornerShape(8.dp), medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(24.dp), extraLarge = RoundedCornerShape(32.dp),
)

val LocalReducedMotion = staticCompositionLocalOf { false }

@Composable
fun ForgeTheme(content: @Composable () -> Unit) {
    val ctx = LocalContext.current
    val scheme = if (Build.VERSION.SDK_INT >= 31) dynamicDarkColorScheme(ctx) else ForgeDark
    val reduce = remember { Settings.Global.getFloat(ctx.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f }
    CompositionLocalProvider(LocalReducedMotion provides reduce) {
        MaterialTheme(colorScheme = scheme, typography = ForgeType, shapes = ForgeShapes, content = content)
    }
}
