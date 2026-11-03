package com.smitnk.img23d.core;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class HunyuanShapeTest {
    @Test public void nearSurfaceMarksBothSidesOfASignChange() {
        int g = 4;
        float[] v = new float[g * g * g];
        for (int a = 0; a < g; a++) for (int b = 0; b < g; b++) for (int c = 0; c < g; c++) v[(a * g + b) * g + c] = a < 2 ? 5f : -5f;
        byte[] m = HunyuanShape.nearSurface(v, g);
        for (int a = 0; a < g; a++) assertEquals(a == 1 || a == 2 ? 1 : 0, m[(a * g) * g]);
    }

    @Test public void dilateGrowsByOneCell() {
        int g = 5;
        byte[] m = new byte[g * g * g];
        m[(2 * g + 2) * g + 2] = 1;
        byte[] d = HunyuanShape.dilate(m, g);
        int n = 0;
        for (byte b : d) n += b;
        assertEquals(27, n);
    }

    @Test public void preprocessCentresSubjectOnWhite() {
        int w = 10, h = 20;
        int[] rgba = new int[w * h * 4];
        for (int y = 5; y < 15; y++) for (int x = 2; x < 8; x++) { int i = (y * w + x) * 4; rgba[i + 3] = 255; }
        float[] chw = HunyuanShape.preprocess(rgba, w, h, 40);
        assertEquals(1f, chw[0], 1e-6);            // corner: white
        assertTrue(chw[20 * 40 + 20] < -0.9f);       // centre: the black subject
    }

    @Test public void readsInfo() { // no HunyuanShape instance: core tests run without ONNX Runtime on the classpath
        String j = "{\"L\": 3072, \"C\": 64, \"scale_factor\": 1.01, \"sigmas\": [0.0, 0.5, 1.0]}";
        org.junit.Assert.assertArrayEquals(new float[]{0f, 0.5f, 1f}, HunyuanShape.sigmasOf(j), 0f);
        assertEquals(3072, (int) HunyuanShape.num(j, "L"));
        assertEquals(64, (int) HunyuanShape.num(j, "C"));
    }
}
