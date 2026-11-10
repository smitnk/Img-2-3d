package com.smitnk.img23d.core;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/** Reads the mesh trellis-cli (TRELLIS.2 on ggml) writes next to its GLB: a binary PLY, Z-up, facing +Y. */
public final class TrellisMesh {
    private TrellisMesh() {}

    /**
     * PLY -> app frame (Mesh.yUpPositions then writes Y-up glTF facing -Z, like Hunyuan models). No cleanup here:
     * the patched trellis-cli already remeshed it watertight and dropped floaters.
     */
    public static Mesh read(File ply) throws IOException {
        byte[] b = Files.readAllBytes(ply.toPath());
        int end = indexOf(b, "end_header\n");
        if (end < 0) throw new IOException("Not a PLY file: " + ply.getName());
        String[] header = new String(b, 0, end, StandardCharsets.US_ASCII).split("\n");
        int nv = 0, nf = 0, floats = 0;
        boolean inVertex = false;
        for (String l : header) {
            String[] t = l.trim().split("\\s+");
            if (t[0].equals("format") && !t[1].equals("binary_little_endian")) throw new IOException("Unsupported PLY format " + t[1]);
            if (t[0].equals("element")) {
                inVertex = t[1].equals("vertex");
                if (inVertex) nv = Integer.parseInt(t[2]);
                if (t[1].equals("face")) nf = Integer.parseInt(t[2]);
            }
            if (t[0].equals("property") && inVertex) {
                if (!t[1].equals("float")) throw new IOException("Unsupported PLY vertex property " + l);
                floats++;
            }
        }
        if (nv <= 0 || nf <= 0 || floats < 3) throw new IOException("Empty PLY mesh");
        ByteBuffer d = ByteBuffer.wrap(b, end + 11, b.length - end - 11).order(ByteOrder.LITTLE_ENDIAN);
        float[] p = new float[nv * 3];
        for (int i = 0; i < nv; i++) {
            float x = d.getFloat(), y = d.getFloat(), z = d.getFloat();
            d.position(d.position() + (floats - 3) * 4);
            p[i * 3] = y; // app frame (a, b, c) with glTF (X, Y, Z) = (-b, c, -a) = (x, z, -y)
            p[i * 3 + 1] = -x;
            p[i * 3 + 2] = z;
        }
        int[] idx = new int[nf * 3];
        int n3 = 0;
        for (int f = 0; f < nf; f++) {
            int n = d.get() & 255, a = d.getInt(), prev = d.getInt();
            for (int k = 2; k < n; k++) { // fan: quads and n-gons become triangles
                int c = d.getInt();
                if (n3 + 3 > idx.length) idx = java.util.Arrays.copyOf(idx, idx.length * 2);
                idx[n3++] = a; idx[n3++] = prev; idx[n3++] = c;
                prev = c;
            }
        }
        return new Mesh(p, java.util.Arrays.copyOf(idx, n3), null);
    }

    private static int indexOf(byte[] b, String s) {
        byte[] n = s.getBytes(StandardCharsets.US_ASCII);
        outer:
        for (int i = 0; i <= Math.min(b.length, 4096) - n.length; i++) {
            for (int j = 0; j < n.length; j++) if (b[i + j] != n[j]) continue outer;
            return i;
        }
        return -1;
    }
}
