package com.smitnk.img23d.cli;

import com.smitnk.img23d.core.TextureDiffusion;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

/**
 * texgen check <dir> <unet> <init.f32> <noise.f32> <out.f32> [maxMae]   Java pipeline vs tools/texgen reference
 * texgen paint <dir> <unet> <in.png> <view> <out.png> [strength]        repaint one view image
 */
final class TexGen {
    static float[] f32(String path) throws Exception {
        byte[] raw = Files.readAllBytes(new File(path).toPath());
        float[] f = new float[raw.length / 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(f);
        return f;
    }

    static void main(String[] a) throws Exception {
        int threads = Runtime.getRuntime().availableProcessors();
        TextureDiffusion td = new TextureDiffusion(new File(a[2]), a[3], threads);
        long t0 = System.nanoTime();
        if (a[1].equals("check")) {
            float[] init = f32(a[4]), noise = f32(a[5]), ref = f32(a[6]);
            int n = 4 * 64 * 64;
            float[] out = td.paint(init, 1, 4, 0.5f, (dst, step) -> System.arraycopy(noise, step * n, dst, 0, n), null);
            double mae = 0;
            for (int i = 0; i < out.length; i++) mae += Math.abs(out[i] - ref[i]);
            mae /= out.length;
            double max = a.length > 7 ? Double.parseDouble(a[7]) : 0.02;
            System.out.printf("TEXGEN check mae=%.5f (max %.3f) ms=%d%n", mae, max, (System.nanoTime() - t0) / 1_000_000);
            if (mae > max) throw new AssertionError("Java texture pipeline differs from reference");
        } else {
            BufferedImage bi = ImageIO.read(new File(a[4]));
            int w = bi.getWidth(), h = bi.getHeight();
            int[] argb = bi.getRGB(0, 0, w, h, null, 0, w), rgba = new int[w * h * 4];
            for (int i = 0; i < w * h; i++) {
                int p = argb[i];
                rgba[i * 4] = (p >> 16) & 255; rgba[i * 4 + 1] = (p >> 8) & 255; rgba[i * 4 + 2] = p & 255; rgba[i * 4 + 3] = p >>> 24;
            }
            float[] alpha = new float[512 * 512];
            int view = java.util.Arrays.asList(TextureDiffusion.VIEWS).indexOf(a[5]);
            float strength = a.length > 7 ? Float.parseFloat(a[7]) : 0.5f;
            int[] o = TextureDiffusion.toRgba(td.paint(TextureDiffusion.toChw(rgba, w, h, alpha), view, 4, strength,
                    TextureDiffusion.seeded(1), null), alpha);
            BufferedImage out = new BufferedImage(512, 512, BufferedImage.TYPE_INT_ARGB);
            for (int i = 0; i < 512 * 512; i++) out.setRGB(i % 512, i / 512, (o[i * 4 + 3] << 24) | (o[i * 4] << 16) | (o[i * 4 + 1] << 8) | o[i * 4 + 2]);
            ImageIO.write(out, "png", new File(a[6]));
            System.out.printf("TEXGEN paint %s ms=%d%n", a[5], (System.nanoTime() - t0) / 1_000_000);
        }
    }
}
