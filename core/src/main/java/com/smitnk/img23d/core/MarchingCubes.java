package com.smitnk.img23d.core;

import java.util.HashMap;

/**
 * Port of torchmcubes mcubes_cpu.cpp + TripoSR MarchingCubeHelper.
 * Input: values[i*R*R + j*R + k] = density - threshold on TripoSR's ij-meshgrid (i=x, j=y, k=z).
 * Output vertices are in grid units, already reordered like TripoSR's v_pos[..., [2,1,0]].
 */
public final class MarchingCubes {
    private MarchingCubes() {}

    private static final int[] INDEX_TABLE = {0, 1, 4, 5, 3, 2, 7, 6};

    public static Mesh run(float[] values, int r) {
        if (values.length != r * r * r) throw new IllegalArgumentException("volume size mismatch");
        FloatList verts = new FloatList(1 << 16);
        IntList faces = new IntList(1 << 17);
        HashMap<Key, Integer> unique = new HashMap<>(1 << 16);
        float[] cx = new float[8], cy = new float[8], cz = new float[8], cv = new float[8];
        float[] vx = new float[12], vy = new float[12], vz = new float[12];
        Key probe = new Key();
        // torchmcubes indexes vol[z][y][x]; TripoSR's tensor axis 0 is "z" there.
        for (int z = 0; z < r - 1; z++) {
            for (int y = 0; y < r - 1; y++) {
                for (int x = 0; x < r - 1; x++) {
                    for (int i = 0; i < 8; i++) {
                        int dx = i % 2, dy = (i / 2) % 2, dz = i / 4, t = INDEX_TABLE[i];
                        cx[t] = x + dx; cy[t] = y + dy; cz[t] = z + dz;
                        cv[t] = values[(z + dz) * r * r + (y + dy) * r + (x + dx)];
                    }
                    int ci = 0;
                    for (int i = 0; i < 8; i++) if (cv[i] < 0f) ci |= 1 << i;
                    int edges = McTables.EDGE[ci];
                    if (edges == 0) continue;
                    for (int e = 0; e < 12; e++) {
                        if ((edges & (1 << e)) == 0) continue;
                        int a = EA[e], b = EB[e];
                        interp(cx[a], cy[a], cz[a], cv[a], cx[b], cy[b], cz[b], cv[b], vx, vy, vz, e);
                    }
                    for (int i = 0; McTables.TRI[ci * 16 + i] != -1; i++) {
                        int e = McTables.TRI[ci * 16 + i];
                        // TripoSR swaps xyz -> zyx after meshing.
                        probe.set(vz[e], vy[e], vx[e]);
                        Integer idx = unique.get(probe);
                        if (idx == null) {
                            idx = verts.size() / 3;
                            unique.put(new Key(probe), idx);
                            verts.add(vz[e]); verts.add(vy[e]); verts.add(vx[e]);
                        }
                        faces.add(idx);
                    }
                }
            }
        }
        return new Mesh(verts.toArray(), faces.toArray(), null);
    }

    private static final int[] EA = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3};
    private static final int[] EB = {1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7};

    private static void interp(float x1, float y1, float z1, float v1, float x2, float y2, float z2, float v2,
                               float[] ox, float[] oy, float[] oz, int o) {
        // Neighbouring cells see an edge in opposite order; canonical order makes the float result identical,
        // which is what trimesh's vertex merge achieves after torchmcubes in the original pipeline.
        if (x1 > x2 || (x1 == x2 && (y1 > y2 || (y1 == y2 && z1 > z2)))) {
            float t;
            t = x1; x1 = x2; x2 = t; t = y1; y1 = y2; y2 = t; t = z1; z1 = z2; z2 = t; t = v1; v1 = v2; v2 = t;
        }
        if (Math.abs(v1) < 1e-4f) { ox[o] = x1; oy[o] = y1; oz[o] = z1; return; }
        if (Math.abs(v2) < 1e-4f) { ox[o] = x2; oy[o] = y2; oz[o] = z2; return; }
        if (Math.abs(v1 - v2) < 1e-4f) { ox[o] = x1; oy[o] = y1; oz[o] = z1; return; }
        float mu = -v1 / (v2 - v1);
        ox[o] = x1 + mu * (x2 - x1); oy[o] = y1 + mu * (y2 - y1); oz[o] = z1 + mu * (z2 - z1);
    }

    private static final class Key {
        float a, b, c;
        Key() {}
        Key(Key k) { a = k.a; b = k.b; c = k.c; }
        void set(float a, float b, float c) { this.a = a; this.b = b; this.c = c; }
        @Override public boolean equals(Object o) {
            Key k = (Key) o;
            return a == k.a && b == k.b && c == k.c;
        }
        @Override public int hashCode() {
            return (Float.floatToIntBits(a) * 31 + Float.floatToIntBits(b)) * 31 + Float.floatToIntBits(c);
        }
    }

    static final class FloatList {
        float[] d; int n;
        FloatList(int cap) { d = new float[cap]; }
        void add(float v) { if (n == d.length) d = java.util.Arrays.copyOf(d, n * 2); d[n++] = v; }
        int size() { return n; }
        float[] toArray() { return java.util.Arrays.copyOf(d, n); }
    }

    static final class IntList {
        int[] d; int n;
        IntList(int cap) { d = new int[cap]; }
        void add(int v) { if (n == d.length) d = java.util.Arrays.copyOf(d, n * 2); d[n++] = v; }
        int[] toArray() { return java.util.Arrays.copyOf(d, n); }
    }
}
