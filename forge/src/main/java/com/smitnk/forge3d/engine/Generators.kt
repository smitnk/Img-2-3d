package com.smitnk.forge3d.engine

/** A colour argument of scripts/generate_model.py. */
enum class ColorParam(val arg: String, val label: String) {
    SKIN("--skin", "Skin / clay"),
    HAIR("--hair", "Hair / fur"),
    SHIRT("--shirt", "Shirt"),
    SHORTS("--shorts", "Shorts"),
    SHOES("--shoes", "Shoes"),
}

/**
 * The Model Forge generators (smitnk/Android-3D-Model-Forge scripts/), exactly as registered in
 * generate_model.py. [colors] lists only the arguments each generator's own script reads; [defaults] are the
 * script's argparse defaults. A colour absent from both is left to the generator (skin/hair "type default").
 */
enum class Generator(
    val modelType: String,
    val title: String,
    val tagline: String,
    val colors: List<ColorParam>,
    val defaults: Map<ColorParam, String>,
    val referenceDriven: Boolean,
) {
    FORGE_KID("chibi_humanoid", "Forge Kid", "Dressed chibi · 17-bone rig",
        listOf(ColorParam.SKIN, ColorParam.HAIR, ColorParam.SHIRT, ColorParam.SHORTS, ColorParam.SHOES),
        mapOf(ColorParam.SHIRT to "#E8553D", ColorParam.SHORTS to "#2F4A8A", ColorParam.SHOES to "#F4F1EA"), false),
    CLAY_CHIBI("clay_chibi", "Clay Chibi", "Metaball sculpt base mesh",
        listOf(ColorParam.SKIN), emptyMap(), false),
    SNOW_BUDDY("snow_buddy", "Snow Buddy", "Round cartoon, dot eyes",
        listOf(ColorParam.SKIN), emptyMap(), false),
    ROUND_BUDDY("round_buddy", "Round Buddy", "From a 24-view turnaround",
        listOf(ColorParam.SKIN), emptyMap(), true),
    GORILLA("gorilla", "Gorilla", "Visual hull · projected texture", emptyList(), emptyMap(), true),
    LIZARD_WARRIOR("lizard_warrior", "Lizard Warrior", "12-view carve · bone daggers", emptyList(), emptyMap(), true),
    LIZARD_REFINED("lizard_refined", "Lizard Refined", "Armour + loincloth parts", emptyList(), emptyMap(), true),
}

data class GenRequest(
    val generator: Generator,
    val name: String,
    val colors: Map<ColorParam, String>,
    val rig: Boolean,
    val saveBlend: Boolean,
) {
    fun scriptArgs(outDir: String): List<String> = buildList {
        add("--model-type"); add(generator.modelType)
        if (name.isNotBlank()) { add("--name"); add(name.trim()) }
        for (p in generator.colors) colors[p]?.let { add(p.arg); add(it) }
        if (!rig) add("--no-rig")
        if (saveBlend) add("--blend")
        add("--out"); add(outDir)
    }
}
