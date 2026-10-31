package com.smitnk.forge3d.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class GenRequestTest {
    @Test fun forgeKidPassesOnlyItsColoursAndFlags() {
        val r = GenRequest(Generator.FORGE_KID, " Hero ", mapOf(ColorParam.SKIN to "#AABBCC", ColorParam.SHIRT to "#112233"), rig = false, saveBlend = true)
        assertEquals(
            listOf("--model-type", "chibi_humanoid", "--name", "Hero", "--skin", "#AABBCC", "--shirt", "#112233", "--no-rig", "--blend", "--out", "/tmp/x"),
            r.scriptArgs("/tmp/x"),
        )
    }

    @Test fun referenceGeneratorIgnoresColours() {
        val r = GenRequest(Generator.GORILLA, "", mapOf(ColorParam.SKIN to "#000000"), rig = true, saveBlend = false)
        assertEquals(listOf("--model-type", "gorilla", "--out", "/o"), r.scriptArgs("/o"))
    }
}
