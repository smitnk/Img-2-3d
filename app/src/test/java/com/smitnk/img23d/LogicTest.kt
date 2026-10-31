package com.smitnk.img23d

import com.smitnk.img23d.ai.ScriptWriter
import com.smitnk.img23d.blender.Builder
import com.smitnk.img23d.blender.BuilderSpec
import com.smitnk.img23d.blender.Preset
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LogicTest {
    @Test fun parsesFencedScriptAndNotes() {
        val d = ScriptWriter.parse("<notes>A fox.</notes>\n```python\nimport bpy\nprint(1)\n```\nbye")
        assertEquals("import bpy\nprint(1)\n", d.script)
        assertEquals("A fox.", d.notes)
    }

    @Test fun parsesBareScript() {
        assertTrue(ScriptWriter.parse("import bpy\nx = 1").script.startsWith("import bpy"))
    }

    @Test fun builderArgsCoverEverySliderChoiceAndColour() {
        val a = BuilderSpec(name = " Hero ").args(rig = false, blend = true)
        assertEquals("Hero", a[a.indexOf("--name") + 1])
        Builder.sliders.forEach { assertTrue(it.arg, "--${it.arg}" in a) }
        Builder.choices.forEach { assertTrue(it.arg, "--${it.arg}" in a) }
        Builder.colors.forEach { assertTrue(it.first, "--${it.first}" in a) }
        assertTrue("--no-rig" in a && "--blend" in a)
        assertEquals("1.200", a[a.indexOf("--height") + 1]) // locale-independent decimal point
    }

    @Test fun randomizedStaysInRange() {
        repeat(50) {
            val r = BuilderSpec().randomized()
            Builder.sliders.forEach { s -> assertTrue(r.values[s.arg]!! in s.min..s.max) }
            Builder.choices.forEach { c -> assertTrue(r.picks[c.arg] in c.options) }
        }
    }

    @Test fun presetPassesOnlyItsOwnColours() {
        assertEquals(listOf("--model-type", "gorilla"), Preset.GORILLA.args("", mapOf("skin" to "#000000"), rig = true, blend = false))
        assertEquals(listOf("--model-type", "snow_buddy", "--name", "Ice", "--skin", "#FFFFFF", "--no-rig"),
            Preset.SNOW_BUDDY.args("Ice", mapOf("skin" to "#FFFFFF", "shirt" to "#000000"), rig = false, blend = false))
    }
}
