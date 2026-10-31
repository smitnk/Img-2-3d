package com.smitnk.img23d.core;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Reads GLBs written by {@link Mesh#writeGlb} back into render-ready arrays (Y-up). Not a general glTF loader. */
public final class GlbReader {
    public final float[] positions, normals, colors;
    public final int[] indices;

    public GlbReader(byte[] glb) {
        ByteBuffer b = ByteBuffer.wrap(glb).order(ByteOrder.LITTLE_ENDIAN);
        if (b.getInt(0) != 0x46546C67) throw new IllegalArgumentException("Not a GLB file");
        int jl = b.getInt(12);
        String json = new String(glb, 20, jl, StandardCharsets.UTF_8);
        int bin = 28 + jl;
        Matcher m = Pattern.compile("\"byteOffset\":(\\d+),\"byteLength\":(\\d+)").matcher(json);
        int[][] views = new int[4][];
        int n = 0;
        while (m.find() && n < 4) views[n++] = new int[]{Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2))};
        boolean hasColor = json.contains("\"COLOR_0\"");
        positions = floats(b, bin, views[0]);
        normals = floats(b, bin, views[1]);
        colors = hasColor ? floats(b, bin, views[2]) : null;
        int[] iv = views[hasColor ? 3 : 2];
        indices = new int[iv[1] / 4];
        for (int i = 0; i < indices.length; i++) indices[i] = b.getInt(bin + iv[0] + i * 4);
    }

    private static float[] floats(ByteBuffer b, int bin, int[] v) {
        float[] f = new float[v[1] / 4];
        for (int i = 0; i < f.length; i++) f[i] = b.getFloat(bin + v[0] + i * 4);
        return f;
    }
}
