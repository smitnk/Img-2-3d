package com.smitnk.img23d.core;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;

public class TrellisMeshTest {
    @Test public void readsTrellisPlyIntoAppFrame() throws Exception {
        // a tetrahedron, outward faces; PLY is Z-up, the model faces +Y
        float[] v = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
        int[] f = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
        String head = "ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            + "element face 4\nproperty list uchar int vertex_indices\nend_header\n";
        ByteBuffer b = ByteBuffer.allocate(head.length() + 48 + 4 * 13).order(ByteOrder.LITTLE_ENDIAN);
        b.put(head.getBytes("US-ASCII"));
        for (float x : v) b.putFloat(x);
        for (int i = 0; i < 4; i++) { b.put((byte) 3); for (int k = 0; k < 3; k++) b.putInt(f[i * 3 + k]); }
        File ply = File.createTempFile("trellis", ".ply");
        try (FileOutputStream o = new FileOutputStream(ply)) { o.write(b.array()); }
        Mesh m = TrellisMesh.read(ply);
        ply.delete();
        assertEquals(4, m.faceCount());
        // PLY (x, y, z) is stored as (y, -x, z): the +Y (front) vertex and the +Z (up) vertex
        assertArrayEquals(new float[]{1, 0, 0}, java.util.Arrays.copyOfRange(m.positions, 6, 9), 0f);
        assertArrayEquals(new float[]{0, 0, 1}, java.util.Arrays.copyOfRange(m.positions, 9, 12), 0f);
    }
}
