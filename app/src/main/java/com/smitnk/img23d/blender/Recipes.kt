package com.smitnk.img23d.blender

/** Model Forge preset generators (runtime /opt/forge/scripts/generate_model.py). */
enum class Preset(val modelType: String, val title: String, val tagline: String, val colors: List<String>, val referenceDriven: Boolean) {
    FORGE_KID("chibi_humanoid", "Forge Kid", "Dressed chibi · 17-bone rig", listOf("skin", "hair", "shirt", "shorts", "shoes"), false),
    CLAY_CHIBI("clay_chibi", "Clay Chibi", "Metaball sculpt base", listOf("skin"), false),
    SNOW_BUDDY("snow_buddy", "Snow Buddy", "Round cartoon", listOf("skin"), false),
    ROUND_BUDDY("round_buddy", "Round Buddy", "24-view turnaround", listOf("skin"), true),
    GORILLA("gorilla", "Gorilla", "Visual hull + texture", emptyList(), true),
    LIZARD_WARRIOR("lizard_warrior", "Lizard Warrior", "12-view carve", emptyList(), true),
    LIZARD_REFINED("lizard_refined", "Lizard Refined", "Armour + loincloth", emptyList(), true);

    fun args(name: String, colors: Map<String, String>, rig: Boolean, blend: Boolean) = buildList {
        add("--model-type"); add(modelType)
        if (name.isNotBlank()) { add("--name"); add(name.trim()) }
        for (c in this@Preset.colors) colors[c]?.let { add("--$c"); add(it) }
        if (!rig) add("--no-rig")
        if (blend) add("--blend")
    }
}

/** One slider of the character builder; [arg] is the character_builder.py flag. */
data class Slider(val arg: String, val label: String, val min: Float, val max: Float, val default: Float, val group: String)

/** One style choice of the character builder; [options] must match STYLES in character_builder.py. */
data class Choice(val arg: String, val label: String, val options: List<String>)

object Builder {
    val sliders = listOf(
        Slider("height", "Height (m)", 0.5f, 2.2f, 1.2f, "Body"),
        Slider("head", "Head size", 0.6f, 1.8f, 1.0f, "Body"),
        Slider("body-width", "Body width", 0.6f, 1.6f, 1.0f, "Body"),
        Slider("belly", "Belly", 0.7f, 1.8f, 1.0f, "Body"),
        Slider("torso", "Torso length", 0.7f, 1.4f, 1.0f, "Body"),
        Slider("neck", "Neck length", 0.3f, 2.5f, 1.0f, "Body"),
        Slider("arm-length", "Arm length", 0.6f, 1.5f, 1.0f, "Limbs"),
        Slider("arm-thickness", "Arm thickness", 0.5f, 2.0f, 1.0f, "Limbs"),
        Slider("leg-length", "Leg length", 0.6f, 1.5f, 1.0f, "Limbs"),
        Slider("leg-thickness", "Leg thickness", 0.5f, 2.0f, 1.0f, "Limbs"),
        Slider("hand-size", "Hand size", 0.5f, 2.0f, 1.0f, "Limbs"),
        Slider("foot-size", "Foot size", 0.5f, 2.0f, 1.0f, "Limbs"),
        Slider("eye-size", "Eye size", 0.5f, 2.0f, 1.0f, "Face"),
    )
    val choices = listOf(
        Choice("eyes", "Eyes", listOf("dot", "round", "anime")),
        Choice("mouth", "Mouth", listOf("smile", "flat", "open")),
        Choice("ears", "Ears", listOf("round", "pointy", "animal", "none")),
        Choice("hair", "Hair", listOf("none", "short", "spiky", "long", "bun", "mohawk")),
        Choice("top", "Top", listOf("none", "tshirt", "hoodie", "tank", "jacket")),
        Choice("bottom", "Bottom", listOf("shorts", "pants", "skirt")),
        Choice("shoes", "Shoes", listOf("sneakers", "boots", "none")),
        Choice("tail", "Tail", listOf("none", "cat", "lizard")),
        Choice("accessory", "Accessory", listOf("none", "glasses", "hat", "cape", "crown")),
    )
    val colors = listOf(
        "skin" to "Skin", "hair-color" to "Hair", "top-color" to "Top", "bottom-color" to "Bottom",
        "shoe-color" to "Shoes", "eye-color" to "Eyes", "accent-color" to "Accent",
    )
    val defaultColors = mapOf(
        "skin" to "#F2C9A0", "hair-color" to "#3B2417", "top-color" to "#E8553D", "bottom-color" to "#2F4A8A",
        "shoe-color" to "#F4F1EA", "eye-color" to "#3A6EA5", "accent-color" to "#F5C542",
    )
}

data class BuilderSpec(
    val name: String = "Custom Hero",
    val values: Map<String, Float> = Builder.sliders.associate { it.arg to it.default },
    val picks: Map<String, String> = Builder.choices.associate { it.arg to it.options.first() },
    val colors: Map<String, String> = Builder.defaultColors,
    val smooth: Int = 1,
) {
    fun args(rig: Boolean, blend: Boolean): List<String> = buildList {
        add("--name"); add(name.ifBlank { "Custom Hero" }.trim())
        values.forEach { (k, v) -> add("--$k"); add("%.3f".format(java.util.Locale.US, v)) }
        picks.forEach { (k, v) -> add("--$k"); add(v) }
        colors.forEach { (k, v) -> add("--$k"); add(v) }
        add("--smooth"); add(smooth.toString())
        if (!rig) add("--no-rig")
        if (blend) add("--blend")
    }

    fun randomized(rnd: kotlin.random.Random = kotlin.random.Random): BuilderSpec = copy(
        values = Builder.sliders.associate { s -> s.arg to (s.default + (rnd.nextFloat() - 0.5f) * (s.max - s.min) * 0.5f).coerceIn(s.min, s.max) },
        picks = Builder.choices.associate { it.arg to it.options.random(rnd) },
        colors = Builder.colors.associate { (k, _) -> k to "#%06X".format(rnd.nextInt(0x1000000)) },
    )
}

/** bx_export presets (scenario-labs, MIT) offered in the Result screen's export menu. */
enum class ExportPreset(val id: String, val label: String, val note: String) {
    UNITY("unity", "Unity", "FBX · static mesh"),
    UNITY_RIG("unity_skeletal", "Unity (rigged)", "FBX · armature + skin"),
    UNREAL("unreal", "Unreal", "FBX · static mesh"),
    UNREAL_RIG("unreal_skeletal", "Unreal (rigged)", "FBX · armature + skin"),
    ROBLOX("roblox", "Roblox", "FBX · Creator Hub settings"),
    GODOT("godot", "Godot", "GLB"),
    WEB("web", "Web / three.js", "GLB · WebP textures"),
    PRINT("print", "3D print", "STL · millimetres"),
}
