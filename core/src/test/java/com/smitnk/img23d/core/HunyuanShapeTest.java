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

    private static void cube(java.util.List<Float> v, java.util.List<Integer> f, float c, float h, boolean inward) {
        int b = v.size() / 3;
        for (int i = 0; i < 8; i++) { v.add(c + ((i & 1) == 0 ? -h : h)); v.add(c + ((i & 2) == 0 ? -h : h)); v.add(c + ((i & 4) == 0 ? -h : h)); }
        int[][] q = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
        for (int[] r : q) {
            int[][] tris = {{r[0], r[1], r[2]}, {r[0], r[2], r[3]}};
            for (int[] t : tris) { f.add(b + t[0]); if (inward) { f.add(b + t[2]); f.add(b + t[1]); } else { f.add(b + t[1]); f.add(b + t[2]); } }
        }
    }

    private static Mesh mesh(java.util.List<Float> v, java.util.List<Integer> f) {
        float[] p = new float[v.size()];
        for (int i = 0; i < p.length; i++) p[i] = v.get(i);
        return new Mesh(p, f.stream().mapToInt(Integer::intValue).toArray(), null);
    }

    @Test public void cleanDropsInwardBubblesAndDegenerateFaces() {
        java.util.List<Float> v = new java.util.ArrayList<>();
        java.util.List<Integer> f = new java.util.ArrayList<>();
        cube(v, f, 0f, 1f, false);  // body
        cube(v, f, 0f, 0.5f, true); // sealed inward bubble
        f.add(0); f.add(0); f.add(1); // zero-area triangle
        Mesh m = HunyuanShape.clean(mesh(v, f), 0);
        assertEquals(12, m.faceCount());
        assertEquals(8, m.vertexCount());
        assertTrue(HunyuanShape.signedVolume(m) > 0);
    }

    @Test public void cleanDropsSmallFloaters() {
        java.util.List<Float> v = new java.util.ArrayList<>();
        java.util.List<Integer> f = new java.util.ArrayList<>();
        cube(v, f, 0f, 1f, false);
        cube(v, f, 5f, 0.1f, false); // a second part with the same face count: 50% share each
        assertEquals(24, HunyuanShape.clean(mesh(v, f), 0.4).faceCount());
        assertEquals(0, HunyuanShape.clean(mesh(v, f), 0.6).faceCount());
    }

    @org.junit.Test
    public void fewerStepsKeepsEnds() {
        float[] o = HunyuanShape.fewerSteps(new float[]{0f, .2f, .4f, .6f, .8f, 1f});
        org.junit.Assert.assertArrayEquals(new float[]{0f, .4f, .8f, 1f}, o, 0f);
    }
}
