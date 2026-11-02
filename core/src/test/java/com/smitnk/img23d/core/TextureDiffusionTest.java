package com.smitnk.img23d.core;

import org.junit.Test;

import static org.junit.Assert.*;

public class TextureDiffusionTest {
    @Test public void lcmTimestepsMatchDiffusers() {
        // diffusers LCMScheduler.set_timesteps(4, strength=0.5) with 50 original steps
        assertArrayEquals(new int[]{499, 379, 259, 139}, TextureDiffusion.timesteps(4, 0.5f));
        assertArrayEquals(new int[]{999, 759, 519, 279}, TextureDiffusion.timesteps(4, 1f));
    }

    @Test public void alphasCumprodScaledLinear() {
        double[] ac = TextureDiffusion.alphasCumprod();
        assertEquals(1 - 0.00085, ac[0], 1e-9);
        assertEquals(0.00466, ac[999], 1e-4); // SD1.5 final alpha_cumprod
    }
}
