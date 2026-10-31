package com.smitnk.img23d.core;

/**
 * Pixel ops replicating TripoSR run.py preprocessing (resize_foreground, gray composite)
 * and torch F.interpolate(bilinear, align_corners=False, antialias=True).
 * Images are float arrays, HWC, channel-interleaved.
 */
public final class ImageOps {
    private ImageOps() {}

    /** torch/PIL antialiased bilinear resize (separable triangle filter). */
    public static float[] resize(float[] src, int sw, int sh, int ch, int dw, int dh) {
        float[] tmp = new float[dw * sh * ch];
        Kernel kx = kernel(sw, dw);
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < dw; x++)
                for (int c = 0; c < ch; c++) {
                    double acc = 0;
                    for (int t = 0; t < kx.len[x]; t++)
                        acc += kx.w[x][t] * src[(y * sw + kx.start[x] + t) * ch + c];
                    tmp[(y * dw + x) * ch + c] = (float) acc;
                }
        float[] dst = new float[dw * dh * ch];
        Kernel ky = kernel(sh, dh);
        for (int y = 0; y < dh; y++)
            for (int x = 0; x < dw; x++)
                for (int c = 0; c < ch; c++) {
                    double acc = 0;
                    for (int t = 0; t < ky.len[y]; t++)
                        acc += ky.w[y][t] * tmp[((ky.start[y] + t) * dw + x) * ch + c];
                    dst[(y * dw + x) * ch + c] = (float) acc;
                }
        return dst;
    }

    private static final class Kernel { int[] start, len; double[][] w; }

    private static Kernel kernel(int in, int out) {
        double scale = (double) in / out;
        double support = scale >= 1.0 ? scale : 1.0;
        double inv = scale >= 1.0 ? 1.0 / scale : 1.0;
        Kernel k = new Kernel();
        k.start = new int[out]; k.len = new int[out]; k.w = new double[out][];
        for (int i = 0; i < out; i++) {
            double center = scale * (i + 0.5);
            int xmin = Math.max((int) (center - support + 0.5), 0);
            int xmax = Math.min((int) (center + support + 0.5), in);
            int n = xmax - xmin;
            double[] w = new double[n];
            double sum = 0;
            for (int j = 0; j < n; j++) {
                double v = 1.0 - Math.abs((j + xmin - center + 0.5) * inv);
                w[j] = v > 0 ? v : 0;
                sum += w[j];
            }
            if (sum > 0) for (int j = 0; j < n; j++) w[j] /= sum;
            k.start[i] = xmin; k.len[i] = n; k.w[i] = w;
        }
        return k;
    }

    /** tsr.utils.resize_foreground on RGBA uint8 (as int[] of 0..255, 4 channels). Returns {image, size}. */
    public static int[] resizeForeground(int[] rgba, int w, int h, float ratio, int[] outSize) {
        int y1 = Integer.MAX_VALUE, y2 = -1, x1 = Integer.MAX_VALUE, x2 = -1;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (rgba[(y * w + x) * 4 + 3] > 0) {
                    y1 = Math.min(y1, y); y2 = Math.max(y2, y); x1 = Math.min(x1, x); x2 = Math.max(x2, x);
                }
        if (y2 < 0) throw new IllegalArgumentException("No foreground found in image");
        // numpy slicing fg = image[y1:y2, x1:x2] excludes the max row/col, kept for fidelity.
        int fh = y2 - y1, fw = x2 - x1;
        if (fh <= 0 || fw <= 0) throw new IllegalArgumentException("Foreground is too small");
        int size = Math.max(fh, fw);
        int ph0 = (size - fh) / 2, pw0 = (size - fw) / 2;
        int newSize = (int) (size / ratio);
        int qh0 = (newSize - size) / 2, qw0 = (newSize - size) / 2;
        int[] out = new int[newSize * newSize * 4];
        for (int y = 0; y < fh; y++)
            for (int x = 0; x < fw; x++) {
                int dy = y + ph0 + qh0, dx = x + pw0 + qw0;
                System.arraycopy(rgba, ((y + y1) * w + x + x1) * 4, out, (dy * newSize + dx) * 4, 4);
            }
        outSize[0] = newSize;
        return out;
    }

    /** run.py: rgb*a + (1-a)*0.5, then uint8 truncation, returned as 0..1 floats (HWC, 3ch). */
    public static float[] compositeGray(int[] rgba, int n) {
        float[] out = new float[n * 3];
        for (int i = 0; i < n; i++) {
            float a = rgba[i * 4 + 3] / 255f;
            for (int c = 0; c < 3; c++) {
                float v = rgba[i * 4 + c] / 255f * a + (1 - a) * 0.5f;
                out[i * 3 + c] = ((int) (v * 255f)) / 255f;
            }
        }
        return out;
    }
}
