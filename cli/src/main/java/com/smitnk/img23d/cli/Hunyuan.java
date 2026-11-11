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
        // one cut-out, or a folder of ref_cutout_{front,left,back,right}.png for the multi-view pack
        File src = new File(a[3]);
        String[] names = {"front", "left", "back", "right"};
        java.util.List<int[]> px = new java.util.ArrayList<>(), dims = new java.util.ArrayList<>();
        for (int v = 0; v < 4; v++) {
            File f = src.isDirectory() ? new File(src, "ref_cutout_" + names[v] + ".png") : src;
            if (!f.isFile() || (!src.isDirectory() && v > 0)) continue;
            BufferedImage bi = ImageIO.read(f);
            int w = bi.getWidth(), h = bi.getHeight();
            int[] argb = bi.getRGB(0, 0, w, h, null, 0, w);
            int[] rgba = new int[w * h * 4];
            for (int i = 0; i < w * h; i++) {
                int p = argb[i];
                rgba[i * 4] = (p >> 16) & 255; rgba[i * 4 + 1] = (p >> 8) & 255; rgba[i * 4 + 2] = p & 255;
                rgba[i * 4 + 3] = bi.getColorModel().hasAlpha() ? (p >>> 24) : 255;
            }
            px.add(rgba); dims.add(new int[]{w, h, v});
        }
        int n = px.size();
        int[][] rgba = px.toArray(new int[0][]);
        int[] w = new int[n], h = new int[n], view = new int[n];
        for (int i = 0; i < n; i++) { w[i] = dims.get(i)[0]; h[i] = dims.get(i)[1]; view[i] = dims.get(i)[2]; }
        hs.fast = Boolean.getBoolean("hunyuan.fast");
        int octree = a.length > 5 ? Integer.parseInt(a[5]) : 256;
        long t0 = System.nanoTime();
        final String[] last = {""};
        Mesh m = hs.run(rgba, w, h, view, octree, 7, (s, f) -> { if (!s.equals(last[0])) { System.out.println("stage " + s); last[0] = s; } });
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
