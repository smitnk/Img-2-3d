package com.smitnk.img23d.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class Settings(
    // System B: TripoSR
    val resolution: Int = 256,
    val removeBackground: Boolean = true,
    val vertexColors: Boolean = true,
    val foregroundRatio: Float = 0.85f,
    // System A: Blender
    val rig: Boolean = true,
    val saveBlend: Boolean = false,
    val refineHeight: Float = 1.0f,
    val refineVoxel: Float = 0.008f,
    val refineSmooth: Int = 4,
    val paintAi: Boolean = false, // AI-painted sides: slow and often worse than TripoSR's colours
    val engine: String = "triposr", // Photo → 3D engine: triposr | hunyuan (2mini) | hunyuanmv (2mv, multi-view)
    val hunyuanOctree: Int = 256,
    val hunyuanFast: Boolean = false, // fewer surface queries + big cores only
    val refineDecimate: Float = 1.0f,
    val refineRig: Boolean = false,
    // Performance
    val threads: Int = 0, // 0 = auto
    val keepScreenOn: Boolean = true,
    // Text to 3D (optional, cloud)
    val apiKey: String = "",
    val apiModel: String = "claude-sonnet-5-5",
    // Look
    val dynamicColor: Boolean = false,
    val autoRotate: Boolean = true,
    val reduceMotion: Boolean = false,
) {
    val hunyuan get() = engine == "hunyuan"
    val hunyuanMv get() = engine == "hunyuanmv"
    fun threadCount() = if (threads > 0) threads else Runtime.getRuntime().availableProcessors().coerceIn(2, 6)
}

/**
 * SharedPreferences-backed settings. The API key stays in app-private storage (allowBackup=false) and is only
 * ever sent to api.anthropic.com.
 */
class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(load())
    val state: StateFlow<Settings> = _state.asStateFlow()

    fun update(f: (Settings) -> Settings) {
        val s = f(_state.value)
        prefs.edit()
            .putInt("resolution", s.resolution).putBoolean("removeBackground", s.removeBackground)
            .putBoolean("vertexColors", s.vertexColors).putFloat("foregroundRatio", s.foregroundRatio)
            .putBoolean("rig", s.rig).putBoolean("saveBlend", s.saveBlend)
            .putFloat("refineHeight", s.refineHeight).putFloat("refineVoxel", s.refineVoxel)
            .putInt("refineSmooth", s.refineSmooth).putBoolean("paintAi", s.paintAi).putString("engine", s.engine).putInt("hunyuanOctree", s.hunyuanOctree).putBoolean("hunyuanFast", s.hunyuanFast).putFloat("refineDecimate", s.refineDecimate)
            .putBoolean("refineRig", s.refineRig).putInt("threads", s.threads).putBoolean("keepScreenOn", s.keepScreenOn)
            .putString("apiKey", s.apiKey).putString("apiModel", s.apiModel)
            .putBoolean("dynamicColor", s.dynamicColor).putBoolean("autoRotate", s.autoRotate)
            .putBoolean("reduceMotion", s.reduceMotion)
            .apply()
        _state.value = s
    }

    private fun load(): Settings {
        val d = Settings()
        return Settings(
            resolution = prefs.getInt("resolution", d.resolution),
            removeBackground = prefs.getBoolean("removeBackground", d.removeBackground),
            vertexColors = prefs.getBoolean("vertexColors", d.vertexColors),
            foregroundRatio = prefs.getFloat("foregroundRatio", d.foregroundRatio),
            rig = prefs.getBoolean("rig", d.rig),
            saveBlend = prefs.getBoolean("saveBlend", d.saveBlend),
            refineHeight = prefs.getFloat("refineHeight", d.refineHeight),
            refineVoxel = prefs.getFloat("refineVoxel", d.refineVoxel),
            refineSmooth = prefs.getInt("refineSmooth", d.refineSmooth),
            paintAi = prefs.getBoolean("paintAi", d.paintAi),
            // older builds stored two switches
            engine = prefs.getString("engine", null)?.takeIf { it != "trellis" } ?: if (prefs.getBoolean("hunyuan", false)) "hunyuan" else d.engine,
            hunyuanOctree = prefs.getInt("hunyuanOctree", d.hunyuanOctree),
            hunyuanFast = prefs.getBoolean("hunyuanFast", d.hunyuanFast),
            refineDecimate = prefs.getFloat("refineDecimate", d.refineDecimate),
            refineRig = prefs.getBoolean("refineRig", d.refineRig),
            threads = prefs.getInt("threads", d.threads),
            keepScreenOn = prefs.getBoolean("keepScreenOn", d.keepScreenOn),
            apiKey = prefs.getString("apiKey", d.apiKey) ?: "",
            apiModel = prefs.getString("apiModel", d.apiModel) ?: d.apiModel,
            dynamicColor = prefs.getBoolean("dynamicColor", d.dynamicColor),
            autoRotate = prefs.getBoolean("autoRotate", d.autoRotate),
            reduceMotion = prefs.getBoolean("reduceMotion", d.reduceMotion),
        )
    }
}
