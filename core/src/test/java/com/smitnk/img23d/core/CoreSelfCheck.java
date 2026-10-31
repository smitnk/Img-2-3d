package com.smitnk.img23d.core;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;

/** Assert-based checks for the meshing/export/preprocess code (run by CoreTest and by `java` directly). */
public final class CoreSelfCheck {
    public static void main(String[] args) {
        sphereIsWatertightAndRound();
        glbIsWellFormed();
        resizeMatchesTorchSemantics();
        resizeForegroundCentersObject();
        System.out.println("core self-check OK");
    }

    static Mesh sphere(int r, float radius) {
        float[] v = new float[r * r * r];
        for (int i = 0; i < r; i++) for (int j = 0; j < r; j++) for (int k = 0; k < r; k++) {
            float x = (float) i / (r - 1) * 2 * 0.87f - 0.87f, y = (float) j / (r - 1) * 2 * 0.87f - 0.87f, z = (float) k / (r - 1) * 2 * 0.87f - 0.87f;
            v[i * r * r + j * r + k] = radius - (float) Math.sqrt(x * x + y * y + z * z); // > 0 inside, like density - threshold
        }
        Mesh m = MarchingCubes.run(v, r);
        m.transform(2 * 0.87f / (r - 1), -0.87f);
        return m;
    }

    static void sphereIsWatertightAndRound() {
        Mesh m = sphere(48, 0.5f);
        check(m.faceCount() > 1000, "too few faces: " + m.faceCount());
        for (int i = 0; i < m.positions.length; i += 3) {
            double d = Math.sqrt(m.positions[i] * m.positions[i] + m.positions[i + 1] * m.positions[i + 1] + m.positions[i + 2] * m.positions[i + 2]);
            check(Math.abs(d - 0.5) < 0.02, "vertex off sphere: " + d);
        }
        HashMap<Long, Integer> edges = new HashMap<>();
        for (int f = 0; f < m.indices.length; f += 3)
            for (int e = 0; e < 3; e++) {
                long a = m.indices[f + e], b = m.indices[f + (e + 1) % 3];
                edges.merge(Math.min(a, b) << 32 | Math.max(a, b), 1, Integer::sum);
            }
        for (int c : edges.values()) check(c == 2, "non-manifold edge, count " + c);
        int euler = m.vertexCount() - edges.size() + m.faceCount();
        check(euler == 2, "euler characteristic " + euler);
        // consistent winding: signed volume must be +-(4/3)pi r^3
        double vol = 0;
        for (int f = 0; f < m.indices.length; f += 3) {
            int a = m.indices[f] * 3, b = m.indices[f + 1] * 3, c = m.indices[f + 2] * 3;
            float[] p = m.positions;
            vol += (p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
                    + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])) / 6.0;
        }
        check(Math.abs(Math.abs(vol) - 4.0 / 3 * Math.PI * 0.125) < 0.01, "volume " + vol);
    }

    static void glbIsWellFormed() {
        Mesh m = sphere(16, 0.5f);
        m.colors = new float[m.positions.length];
        byte[] glb = m.toGlb();
        ByteBuffer b = ByteBuffer.wrap(glb).order(ByteOrder.LITTLE_ENDIAN);
        check(b.getInt(0) == 0x46546C67 && b.getInt(4) == 2 && b.getInt(8) == glb.length, "glb header");
        int jl = b.getInt(12);
        check(jl % 4 == 0 && b.getInt(16) == 0x4E4F534A, "json chunk");
        int bl = b.getInt(20 + jl);
        check(b.getInt(24 + jl) == 0x004E4942 && 28 + jl + bl == glb.length, "bin chunk");
        String json = new String(glb, 20, jl, java.nio.charset.StandardCharsets.UTF_8);
        check(json.contains("\"COLOR_0\":2") && json.contains("\"count\":" + m.indices.length), "json content");
        GlbReader r = new GlbReader(glb);
        check(r.indices.length == m.indices.length && r.positions.length == m.positions.length && r.colors != null, "glb round trip");
        check(r.positions[1] == m.positions[2] && r.positions[0] == -m.positions[1], "y-up transform");
    }

    static void resizeMatchesTorchSemantics() {
        float[] c = new float[10 * 6];
        java.util.Arrays.fill(c, 0.3f);
        for (float v : ImageOps.resize(c, 10, 6, 1, 3, 7)) check(Math.abs(v - 0.3f) < 1e-6, "constant not preserved");
        // 4 -> 2 antialiased bilinear: triangle filter of support 2 (torch/PIL): [.75,.75,.25]/1.75
        float[] r = ImageOps.resize(new float[]{0, 1, 2, 3}, 4, 1, 1, 2, 1);
        check(Math.abs(r[0] - 1.25f / 1.75f) < 1e-6 && Math.abs(r[1] - 4f / 1.75f) < 1e-6, "downscale " + r[0] + "," + r[1]);
        // 2 -> 4 upsample == align_corners=False bilinear: [0, .25, .75, 1]
        float[] u = ImageOps.resize(new float[]{0, 1}, 2, 1, 1, 4, 1);
        check(Math.abs(u[0]) < 1e-6 && Math.abs(u[1] - .25f) < 1e-6 && Math.abs(u[2] - .75f) < 1e-6 && Math.abs(u[3] - 1) < 1e-6, "upscale");
    }

    static void resizeForegroundCentersObject() {
        int w = 40, h = 20;
        int[] img = new int[w * h * 4];
        for (int y = 2; y < 12; y++) for (int x = 5; x < 15; x++) img[(y * w + x) * 4 + 3] = 255;
        int[] size = new int[1];
        int[] out = ImageOps.resizeForeground(img, w, h, 0.85f, size);
        // fg = [2:11, 5:14] -> 9x9, padded to int(9/0.85)=10
        check(size[0] == 10, "size " + size[0]);
        check(out[(0 * 10 + 0) * 4 + 3] == 255 && out[(9 * 10 + 9) * 4 + 3] == 0, "placement");
    }

    static void check(boolean ok, String msg) {
        if (!ok) throw new AssertionError(msg);
    }
}
