package com.smitnk.img23d.cli;

import com.smitnk.img23d.core.HunyuanShape;
import com.smitnk.img23d.core.Mesh;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;

/** cli hunyuan <modelDir> <info.json> <cutout.png> <out.glb> [octree] : the app's Hunyuan3D pipeline on the desktop. */
final class Hunyuan {
    static void main(String[] a) throws Exception {
        int threads = Runtime.getRuntime().availableProcessors();
        HunyuanShape hs = HunyuanShape.fromInfo(new File(a[1]), new String(Files.readAllBytes(new File(a[2]).toPath()), "UTF-8"), threads);
        BufferedImage bi = ImageIO.read(new File(a[3]));
        int w = bi.getWidth(), h = bi.getHeight();
        int[] argb = bi.getRGB(0, 0, w, h, null, 0, w);
        int[] rgba = new int[w * h * 4];
        for (int i = 0; i < w * h; i++) {
            int p = argb[i];
            rgba[i * 4] = (p >> 16) & 255; rgba[i * 4 + 1] = (p >> 8) & 255; rgba[i * 4 + 2] = p & 255;
            rgba[i * 4 + 3] = bi.getColorModel().hasAlpha() ? (p >>> 24) : 255;
        }
        hs.fast = Boolean.getBoolean("hunyuan.fast");
        int octree = a.length > 5 ? Integer.parseInt(a[5]) : 256;
        long t0 = System.nanoTime();
        final String[] last = {""};
        Mesh m = hs.run(rgba, w, h, octree, 7, (s, f) -> { if (!s.equals(last[0])) { System.out.println("stage " + s); last[0] = s; } });
        try (FileOutputStream fo = new FileOutputStream(a[4])) { m.writeGlb(fo); }
        float[] y = m.yUpPositions();
        float[] lo = {9, 9, 9}, hi = {-9, -9, -9};
        for (int i = 0; i < y.length; i++) { lo[i % 3] = Math.min(lo[i % 3], y[i]); hi[i % 3] = Math.max(hi[i % 3], y[i]); }
        Runtime rt = Runtime.getRuntime();
        System.out.printf("HUNYUAN_JAVA faces=%d vertices=%d total=%dms timings=%s bbox=[%.3f %.3f %.3f]..[%.3f %.3f %.3f] heapMB=%d%n",
                m.faceCount(), m.vertexCount(), (System.nanoTime() - t0) / 1_000_000, hs.timingsMs,
                lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], (rt.totalMemory() - rt.freeMemory()) >> 20);
    }
}
