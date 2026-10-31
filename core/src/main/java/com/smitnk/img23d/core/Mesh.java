package com.smitnk.img23d.core;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Triangle mesh: positions xyz, triangle indices, optional per-vertex rgb in 0..1. */
public final class Mesh {
    public final float[] positions;
    public final int[] indices;
    public float[] colors;

    public Mesh(float[] positions, int[] indices, float[] colors) {
        this.positions = positions;
        this.indices = indices;
        this.colors = colors;
    }

    public int vertexCount() { return positions.length / 3; }
    public int faceCount() { return indices.length / 3; }

    public void transform(float scale, float offset) {
        for (int i = 0; i < positions.length; i++) positions[i] = positions[i] * scale + offset;
    }

    /** TripoSR gradio_app.to_gradio_3d_orientation: Rx(-90) then Ry(90), i.e. (x,y,z) -> (-y, z, -x). Y-up for glTF. */
    public float[] yUpPositions() {
        float[] p = new float[positions.length];
        for (int i = 0; i < p.length; i += 3) {
            p[i] = -positions[i + 1];
            p[i + 1] = positions[i + 2];
            p[i + 2] = -positions[i];
        }
        return p;
    }

    public static float[] vertexNormals(float[] p, int[] idx) {
        float[] n = new float[p.length];
        for (int f = 0; f < idx.length; f += 3) {
            int a = idx[f] * 3, b = idx[f + 1] * 3, c = idx[f + 2] * 3;
            float ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
            float vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
            float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            for (int v : new int[]{a, b, c}) { n[v] += nx; n[v + 1] += ny; n[v + 2] += nz; }
        }
        for (int i = 0; i < n.length; i += 3) {
            float l = (float) Math.sqrt(n[i] * n[i] + n[i + 1] * n[i + 1] + n[i + 2] * n[i + 2]);
            if (l > 0) { n[i] /= l; n[i + 1] /= l; n[i + 2] /= l; } else n[i + 1] = 1f;
        }
        return n;
    }

    /** Writes a binary glTF 2.0 file (Y-up, vertex colors, double-sided). */
    public void writeGlb(OutputStream out) throws IOException {
        float[] p = yUpPositions();
        float[] n = vertexNormals(p, indices);
        int vc = vertexCount();
        boolean hasColor = colors != null;
        int posBytes = vc * 12, colBytes = hasColor ? vc * 12 : 0, idxBytes = indices.length * 4;
        int binLen = posBytes * 2 + colBytes + idxBytes;
        ByteBuffer bin = ByteBuffer.allocate(binLen).order(ByteOrder.LITTLE_ENDIAN);
        float[] min = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE};
        float[] max = {-Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
        for (int i = 0; i < p.length; i++) {
            bin.putFloat(p[i]);
            min[i % 3] = Math.min(min[i % 3], p[i]);
            max[i % 3] = Math.max(max[i % 3], p[i]);
        }
        for (float v : n) bin.putFloat(v);
        if (hasColor) for (float v : colors) bin.putFloat(Math.max(0f, Math.min(1f, v)));
        for (int v : indices) bin.putInt(v);

        StringBuilder j = new StringBuilder();
        j.append("{\"asset\":{\"version\":\"2.0\",\"generator\":\"Img-2-3D (TripoSR)\"},\"scene\":0,")
         .append("\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],")
         .append("\"materials\":[{\"doubleSided\":true,\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],\"metallicFactor\":0,\"roughnessFactor\":0.9}}],")
         .append("\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1");
        if (hasColor) j.append(",\"COLOR_0\":2");
        int idxAcc = hasColor ? 3 : 2;
        j.append("},\"indices\":").append(idxAcc).append(",\"material\":0}]}],");
        j.append("\"buffers\":[{\"byteLength\":").append(binLen).append("}],\"bufferViews\":[");
        int off = 0, view = 0;
        j.append(view(off, posBytes, 34962)); off += posBytes; view++;
        j.append(',').append(view(off, posBytes, 34962)); off += posBytes; view++;
        if (hasColor) { j.append(',').append(view(off, colBytes, 34962)); off += colBytes; view++; }
        j.append(',').append(view(off, idxBytes, 34963));
        j.append("],\"accessors\":[");
        j.append("{\"bufferView\":0,\"componentType\":5126,\"count\":").append(vc).append(",\"type\":\"VEC3\",\"min\":[")
         .append(min[0]).append(',').append(min[1]).append(',').append(min[2]).append("],\"max\":[")
         .append(max[0]).append(',').append(max[1]).append(',').append(max[2]).append("]},");
        j.append("{\"bufferView\":1,\"componentType\":5126,\"count\":").append(vc).append(",\"type\":\"VEC3\"}");
        if (hasColor) j.append(",{\"bufferView\":2,\"componentType\":5126,\"count\":").append(vc).append(",\"type\":\"VEC3\"}");
        j.append(",{\"bufferView\":").append(idxAcc).append(",\"componentType\":5125,\"count\":").append(indices.length).append(",\"type\":\"SCALAR\"}]}");

        byte[] json = j.toString().getBytes(StandardCharsets.UTF_8);
        int jsonPad = (4 - json.length % 4) % 4;
        int total = 12 + 8 + json.length + jsonPad + 8 + binLen;
        ByteBuffer h = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
        h.putInt(0x46546C67).putInt(2).putInt(total).putInt(json.length + jsonPad).putInt(0x4E4F534A);
        out.write(h.array());
        out.write(json);
        for (int i = 0; i < jsonPad; i++) out.write(' ');
        ByteBuffer bh = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        bh.putInt(binLen).putInt(0x004E4942);
        out.write(bh.array());
        out.write(bin.array());
    }

    private static String view(int off, int len, int target) {
        return "{\"buffer\":0,\"byteOffset\":" + off + ",\"byteLength\":" + len + ",\"target\":" + target + "}";
    }

    public byte[] toGlb() {
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        try { writeGlb(bo); } catch (IOException e) { throw new IllegalStateException(e); }
        return bo.toByteArray();
    }
}
