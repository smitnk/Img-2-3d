package com.smitnk.img23d.cli;

import com.smitnk.img23d.core.Mesh;
import com.smitnk.img23d.core.TripoSR;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * Desktop runner for the exact Java pipeline the Android app uses. CI runs it against the
 * PyTorch reference produced by tools/export/export_onnx.py.
 *
 * usage: cli <modelDir> <image.png|image_512.f32> <out.glb> [resolution] [reference.obj] [maxChamfer]
 */
public final class Main {
    public static void main(String[] a) throws Exception {
        if (a.length > 0 && a[0].equals("texgen")) { TexGen.main(a); return; }
        File models = new File(a[0]);
        File input = new File(a[1]);
        File out = new File(a[2]);
        TripoSR.Options opt = new TripoSR.Options();
        opt.resolution = a.length > 3 ? Integer.parseInt(a[3]) : 128;
        opt.threads = Runtime.getRuntime().availableProcessors();
        TripoSR pipe = new TripoSR(models);
        TripoSR.Listener log = (s, f) -> { if (f == 0f || f == 1f) System.out.println("stage " + s + " " + f); };
        long t0 = System.nanoTime();
        TripoSR.Result res;
        if (input.getName().endsWith(".f32")) {
            byte[] raw = Files.readAllBytes(input.toPath());
            float[] img = new float[raw.length / 4];
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(img);
            res = new TripoSR.Result();
            float[] scene = pipe.encode(img, opt.threads, log, res);
            res.mesh = pipe.extractMesh(scene, opt, log, res);
        } else {
            BufferedImage bi = ImageIO.read(input);
            int w = bi.getWidth(), h = bi.getHeight();
            int[] argb = bi.getRGB(0, 0, w, h, null, 0, w);
            int[] rgba = new int[w * h * 4];
            boolean alpha = bi.getColorModel().hasAlpha();
            for (int i = 0; i < w * h; i++) {
                int p = argb[i];
                rgba[i * 4] = (p >> 16) & 255; rgba[i * 4 + 1] = (p >> 8) & 255; rgba[i * 4 + 2] = p & 255;
                rgba[i * 4 + 3] = alpha ? (p >>> 24) : 255;
            }
            res = pipe.run(rgba, w, h, opt, log);
        }
        Mesh m = res.mesh;
        try (FileOutputStream fo = new FileOutputStream(out)) { m.writeGlb(fo); }
        Runtime rt = Runtime.getRuntime();
        System.out.printf("vertices=%d faces=%d total=%dms timings=%s heapUsedMB=%d glbBytes=%d%n", m.vertexCount(), m.faceCount(),
                (System.nanoTime() - t0) / 1_000_000, res.timingsMs, (rt.totalMemory() - rt.freeMemory()) >> 20, out.length());
        float[] lo = {9, 9, 9}, hi = {-9, -9, -9};
        for (int i = 0; i < m.positions.length; i++) { lo[i % 3] = Math.min(lo[i % 3], m.positions[i]); hi[i % 3] = Math.max(hi[i % 3], m.positions[i]); }
        System.out.printf("bbox min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f)%n", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);

        if (a.length > 4) {
            float[] ref = readObjVertices(new File(a[4]));
            double max = a.length > 5 ? Double.parseDouble(a[5]) : 0.01;
            double c = chamfer(m.positions, ref);
            System.out.printf("chamfer vs PyTorch reference = %.5f (limit %.5f), ref vertices=%d%n", c, max, ref.length / 3);
            if (!(c <= max)) { System.err.println("FAIL: mesh deviates from original TripoSR output"); System.exit(2); }
        }
    }

    static float[] readObjVertices(File f) throws Exception {
        List<String> lines = Files.readAllLines(f.toPath());
        ArrayList<Float> v = new ArrayList<>();
        for (String l : lines) if (l.startsWith("v ")) {
            String[] p = l.trim().split("\\s+");
            for (int i = 1; i <= 3; i++) v.add(Float.parseFloat(p[i]));
        }
        float[] r = new float[v.size()];
        for (int i = 0; i < r.length; i++) r[i] = v.get(i);
        return r;
    }

    /** Symmetric mean nearest-neighbour distance on up to 3000 sampled points per side. */
    static double chamfer(float[] a, float[] b) {
        return (oneWay(a, b) + oneWay(b, a)) / 2;
    }

    static double oneWay(float[] a, float[] b) {
        int na = a.length / 3, nb = b.length / 3, step = Math.max(1, na / 3000);
        double sum = 0; int cnt = 0;
        for (int i = 0; i < na; i += step) {
            double best = Double.MAX_VALUE;
            for (int j = 0; j < nb; j++) {
                double dx = a[i * 3] - b[j * 3], dy = a[i * 3 + 1] - b[j * 3 + 1], dz = a[i * 3 + 2] - b[j * 3 + 2];
                best = Math.min(best, dx * dx + dy * dy + dz * dz);
            }
            sum += Math.sqrt(best); cnt++;
        }
        return sum / cnt;
    }
}
