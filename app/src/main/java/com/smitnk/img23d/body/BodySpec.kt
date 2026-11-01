package com.smitnk.img23d.body

import org.json.JSONObject

/** A Body Creator character: MakeHuman macro values (0..1), slider values (-1..1) and looks. Serialized as spec.json. */
data class BodySpec(
    val name: String = "My character",
    val macros: Map<String, Float> = emptyMap(),
    val sliders: Map<String, Float> = emptyMap(),
    val skin: String = "#C9A08A",
    val subdiv: Int = 1,
    val symmetric: Boolean = true,
) {
    fun macro(k: String) = macros[k] ?: BodyMorph.DEFAULT_MACROS.getValue(k)

    /** Sets a slider; with symmetry on, the matching l-/r- slider moves too. */
    fun withSlider(id: String, v: Float): BodySpec {
        val twin = if (symmetric) mirror(id) else null
        return copy(sliders = sliders + (id to v) + listOfNotNull(twin?.let { it to v }))
    }

    fun toJson(): String = JSONObject()
        .put("name", name.ifBlank { "Character" }).put("macros", JSONObject(macros as Map<*, *>))
        .put("sliders", JSONObject(sliders.filterValues { it != 0f } as Map<*, *>)).put("skin", skin).put("subdiv", subdiv).put("rig", true)
        .toString(2)

    companion object {
        /** "eyes/l-eye-scale-decr|incr" <-> "eyes/r-eye-scale-decr|incr" */
        fun mirror(id: String): String? {
            val (g, t) = id.split("/", limit = 2).let { it[0] to it.getOrElse(1) { "" } }
            return when {
                t.startsWith("l-") -> "$g/r-${t.drop(2)}"
                t.startsWith("r-") -> "$g/l-${t.drop(2)}"
                else -> null
            }
        }

        val MACROS = listOf(
            "gender" to "Feminine ↔ Masculine", "age" to "Age", "muscle" to "Muscle", "weight" to "Weight",
            "height" to "Height", "proportions" to "Proportions", "cup" to "Breast size", "firmness" to "Breast firmness",
        )
        val ETHNIC = listOf("african" to "African", "asian" to "Asian", "caucasian" to "European")

        val SKINS = listOf("#F6D7C3", "#EBC2A4", "#E0AC8A", "#C9A08A", "#C68B59", "#B07A5A", "#9C6644", "#8D5524",
            "#7A4A2E", "#5C3A21", "#4A2C1A", "#3B2417")

        /** Age 0..1 maps 1 -> 90 years in MakeHuman (0.1875 = 10 y, 0.5 = 25 y). */
        val PRESETS: List<Pair<String, BodySpec>> = listOf(
            "Kid" to BodySpec(name = "Kid", macros = mapOf("age" to 0.15f, "gender" to 0.6f)),
            "Teen" to BodySpec(name = "Teen", macros = mapOf("age" to 0.35f, "weight" to 0.45f)),
            "Man" to BodySpec(name = "Man", macros = mapOf("gender" to 1f, "age" to 0.5f)),
            "Woman" to BodySpec(name = "Woman", macros = mapOf("gender" to 0f, "age" to 0.5f)),
            "Athlete" to BodySpec(name = "Athlete", macros = mapOf("gender" to 1f, "muscle" to 0.85f, "weight" to 0.45f, "height" to 0.65f)),
            "Hero" to BodySpec(name = "Hero", macros = mapOf("gender" to 1f, "muscle" to 1f, "weight" to 0.7f, "height" to 0.8f, "proportions" to 1f),
                sliders = mapOf("torso/torso-vshape-decr|incr" to 0.7f, "chin/chin-width-decr|incr" to 0.4f)),
            "Heavy" to BodySpec(name = "Heavy", macros = mapOf("gender" to 1f, "weight" to 1f, "muscle" to 0.3f, "age" to 0.6f)),
            "Elder" to BodySpec(name = "Elder", macros = mapOf("age" to 0.95f, "muscle" to 0.3f)),
        )
    }
}
