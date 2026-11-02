package com.smitnk.img23d.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * On-device texture painter: img2img with LCM Dreamshaper v7 (SD1.5 latent consistency) on ONNX Runtime.
 * A coarse render of one side of the model (TripoSR colours) goes in; a detailed repaint of that side comes out.
 * Text prompts are pre-encoded per view in prompts.bin (no tokenizer/text encoder on the phone).
 * Scheduler = diffusers LCMScheduler (scaled-linear betas, 50 original steps), mirrored by tools/texgen/export_texgen.py.
 */
public final class TextureDiffusion implements AutoCloseable {
    public static final int SIZE = 512, LAT = 64;
    public static final float SCALE = 0.18215f;
    public static final String[] VIEWS = {"front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"};

    public interface Noise { void fill(float[] dst, int step); }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final File dir;
    private final String unetName;
    private final int threads;
    private final float[] prompts; // VIEWS.length * 77 * 768, then 256
    private final double[] alphasCumprod = alphasCumprod();

    public TextureDiffusion(File dir, String unetName, int threads) throws IOException {
        this.dir = dir;
        this.unetName = unetName;
        this.threads = threads;
        byte[] raw = Files.readAllBytes(new File(dir, "prompts.bin").toPath());
        prompts = new float[raw.length / 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(prompts);
        if (prompts.length != VIEWS.length * 77 * 768 + 256) throw new IOException("prompts.bin has unexpected size");
    }

    static double[] alphasCumprod() {
        double[] ac = new double[1000];
        double lo = Math.sqrt(0.00085), hi = Math.sqrt(0.012), p = 1;
        for (int i = 0; i < 1000; i++) {
            double b = lo + (hi - lo) * i / 999.0;
            p *= 1 - b * b;
            ac[i] = p;
        }
        return ac;
    }

    static int[] timesteps(int steps, float strength) {
        int n = (int) (50 * strength), skip = Math.max(1, n / steps);
        int[] t = new int[Math.min(steps, (n + skip - 1) / skip)];
        for (int i = 0; i < t.length; i++) t[i] = (n - i * skip) * 20 - 1; // origin[::-1][::skip]
        return t;
    }

    private OrtSession open(String name) throws OrtException {
        File f = new File(dir, name);
        if (!f.isFile()) throw new IllegalStateException("Missing texture model file: " + f);
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(threads);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        return env.createSession(f.getAbsolutePath(), o);
    }

    private float[] run1(OrtSession s, Map<String, OnnxTensor> in) throws OrtException {
        try (OrtSession.Result r = s.run(in)) {
            FloatBuffer b = ((OnnxTensor) r.get(0)).getFloatBuffer();
            float[] out = new float[b.remaining()];
            b.get(out);
            return out;
        } finally {
            for (OnnxTensor t : in.values()) t.close();
        }
    }

    /**
     * @param initChw image [3,512,512] in -1..1
     * @param view    index into VIEWS (picks the prompt)
     * @return repainted image [3,512,512] in -1..1
     */
    public float[] paint(float[] initChw, int view, int steps, float strength, Noise noise, TripoSR.Listener l) throws OrtException {
        // Sessions open one at a time (VAE encoder, UNet, VAE decoder) to keep peak RAM near the UNet's size.
        float[] lat;
        try (OrtSession enc = open("vae_encoder.onnx")) {
            Map<String, OnnxTensor> in = new HashMap<>();
            in.put("image", OnnxTensor.createTensor(env, FloatBuffer.wrap(initChw), new long[]{1, 3, SIZE, SIZE}));
            lat = run1(enc, in);
        }
        for (int i = 0; i < lat.length; i++) lat[i] *= SCALE;
        int[] ts = timesteps(steps, strength);
        float[] nz = new float[lat.length];
        noise.fill(nz, 0);
        double a = alphasCumprod[ts[0]];
        float[] x = new float[lat.length];
        for (int i = 0; i < x.length; i++) x[i] = (float) (Math.sqrt(a) * lat[i] + Math.sqrt(1 - a) * nz[i]);
        float[] emb = new float[77 * 768], tcond = new float[256];
        System.arraycopy(prompts, view * 77 * 768, emb, 0, emb.length);
        System.arraycopy(prompts, VIEWS.length * 77 * 768, tcond, 0, 256);
        try (OrtSession unet = open(unetName)) {
            for (int s = 0; s < ts.length; s++) {
                if (l != null) l.onProgress(TripoSR.Stage.ENCODER, (float) s / ts.length);
                Map<String, OnnxTensor> in = new HashMap<>();
                in.put("sample", OnnxTensor.createTensor(env, FloatBuffer.wrap(x), new long[]{1, 4, LAT, LAT}));
                in.put("timestep", OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{ts[s]}), new long[]{1}));
                in.put("encoder_hidden_states", OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), new long[]{1, 77, 768}));
                in.put("timestep_cond", OnnxTensor.createTensor(env, FloatBuffer.wrap(tcond), new long[]{1, 256}));
                float[] eps = run1(unet, in);
                double at = alphasCumprod[ts[s]], sc = ts[s] * 10.0;
                double cSkip = 0.25 / (sc * sc + 0.25), cOut = sc / Math.sqrt(sc * sc + 0.25);
                boolean last = s + 1 == ts.length;
                double ap = last ? 0 : alphasCumprod[ts[s + 1]];
                if (!last) noise.fill(nz, s + 1);
                for (int i = 0; i < x.length; i++) {
                    double x0 = (x[i] - Math.sqrt(1 - at) * eps[i]) / Math.sqrt(at);
                    double den = cOut * x0 + cSkip * x[i];
                    x[i] = (float) (last ? den : Math.sqrt(ap) * den + Math.sqrt(1 - ap) * nz[i]);
                }
            }
        }
        for (int i = 0; i < x.length; i++) x[i] /= SCALE;
        float[] out;
        try (OrtSession dec = open("vae_decoder.onnx")) {
            Map<String, OnnxTensor> in = new HashMap<>();
            in.put("latent", OnnxTensor.createTensor(env, FloatBuffer.wrap(x), new long[]{1, 4, LAT, LAT}));
            out = run1(dec, in);
        }
        for (int i = 0; i < out.length; i++) out[i] = Math.max(-1f, Math.min(1f, out[i]));
        if (l != null) l.onProgress(TripoSR.Stage.DONE, 1f);
        return out;
    }

    /** Gaussian noise from a seed (the app's default). */
    public static Noise seeded(long seed) {
        Random r = new Random(seed);
        return (dst, step) -> { for (int i = 0; i < dst.length; i++) dst[i] = (float) r.nextGaussian(); };
    }

    /** RGBA 0..255 (any size) composited on mid gray -> [3,512,512] in -1..1, plus the 512 alpha mask. */
    public static float[] toChw(int[] rgba, int w, int h, float[] alphaOut) {
        float[] hwc = new float[w * h * 4];
        for (int i = 0; i < hwc.length; i++) hwc[i] = rgba[i] / 255f;
        float[] r = ImageOps.resize(hwc, w, h, 4, SIZE, SIZE);
        float[] chw = new float[3 * SIZE * SIZE];
        for (int p = 0; p < SIZE * SIZE; p++) {
            float al = r[p * 4 + 3];
            if (alphaOut != null) alphaOut[p] = al;
            for (int c = 0; c < 3; c++) chw[c * SIZE * SIZE + p] = (r[p * 4 + c] * al + 0.5f * (1 - al)) * 2 - 1;
        }
        return chw;
    }

    /** [3,512,512] -1..1 + alpha -> RGBA 0..255. */
    public static int[] toRgba(float[] chw, float[] alpha) {
        int n = SIZE * SIZE;
        int[] o = new int[n * 4];
        for (int p = 0; p < n; p++) {
            for (int c = 0; c < 3; c++) o[p * 4 + c] = Math.round(Math.max(0, Math.min(255, (chw[c * n + p] + 1) * 127.5f)));
            o[p * 4 + 3] = alpha == null ? 255 : Math.round(alpha[p] * 255);
        }
        return o;
    }

    @Override public void close() { }
}
