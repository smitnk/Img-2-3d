package com.smitnk.img23d.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * TripoSR image-to-3D pipeline on ONNX Runtime (same API on Android and desktop JVM).
 * Stages mirror run.py: remove_background -> resize_foreground -> gray composite ->
 * TSR.forward (encoder graph) -> TSR.extract_mesh (field graph + marching cubes) -> vertex colors.
 * One job at a time; each session is released as soon as its stage is done to cap peak RAM.
 */
public final class TripoSR {
    public static final int IMAGE_SIZE = 512;
    public static final float RADIUS = 0.87f;
    public static final float THRESHOLD = 25.0f;
    public static final float FOREGROUND_RATIO = 0.85f;
    private static final int FIELD_CHUNK = 65536;

    public enum Stage { BACKGROUND, PREPROCESS, ENCODER_LOAD, ENCODER, FIELD, MARCHING_CUBES, COLORS, DONE }

    public interface Listener {
        void onProgress(Stage stage, float fraction);
    }

    public static final class Options {
        public int resolution = 256;
        public boolean removeBackground = true;
        public float foregroundRatio = FOREGROUND_RATIO;
        public boolean vertexColors = true;
        public int threads = 4;
    }

    public static final class Result {
        public Mesh mesh;
        public float[] preprocessed; // 512x512x3, what the network saw
        public final Map<String, Long> timingsMs = new HashMap<>();
    }

    private final File modelDir;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final AtomicBoolean cancelled = new AtomicBoolean();

    public TripoSR(File modelDir) { this.modelDir = modelDir; }

    public void cancel() { cancelled.set(true); }

    private void check() {
        if (cancelled.get()) throw new java.util.concurrent.CancellationException("Cancelled");
    }

    private OrtSession open(String name, int threads) throws OrtException {
        File f = new File(modelDir, name);
        if (!f.isFile()) throw new IllegalStateException("Missing model file: " + f.getAbsolutePath());
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(threads);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        o.setMemoryPatternOptimization(true);
        return env.createSession(f.getAbsolutePath(), o);
    }

    /** @param rgba interleaved 0..255 RGBA, w*h*4 */
    public Result run(int[] rgba, int w, int h, Options opt, Listener l) throws OrtException {
        cancelled.set(false);
        Result res = new Result();
        long t = System.nanoTime();

        boolean hasAlpha = false;
        for (int i = 3; i < rgba.length; i += 4) if (rgba[i] < 255) { hasAlpha = true; break; }
        if (opt.removeBackground && !hasAlpha) {
            l.onProgress(Stage.BACKGROUND, 0f);
            rgba = removeBackground(rgba, w, h, opt.threads);
            res.timingsMs.put("background", ms(t)); t = System.nanoTime();
        }
        check();
        l.onProgress(Stage.PREPROCESS, 0f);
        int[] size = new int[1];
        int[] fg = ImageOps.resizeForeground(rgba, w, h, opt.foregroundRatio, size);
        float[] comp = ImageOps.compositeGray(fg, size[0] * size[0]);
        float[] img = ImageOps.resize(comp, size[0], size[0], 3, IMAGE_SIZE, IMAGE_SIZE);
        res.preprocessed = img;
        res.timingsMs.put("preprocess", ms(t)); t = System.nanoTime();
        check();

        float[] scene = encode(img, opt.threads, l, res);
        t = System.nanoTime();
        check();

        res.mesh = extractMesh(scene, opt, l, res);
        res.timingsMs.put("mesh_total", ms(t));
        l.onProgress(Stage.DONE, 1f);
        return res;
    }

    public float[] encode(float[] image512, int threads, Listener l, Result res) throws OrtException {
        long t = System.nanoTime();
        l.onProgress(Stage.ENCODER_LOAD, 0f);
        try (OrtSession s = open("triposr_encoder.onnx", threads)) {
            res.timingsMs.put("encoder_load", ms(t)); t = System.nanoTime();
            check();
            l.onProgress(Stage.ENCODER, 0f);
            try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(image512), new long[]{1, IMAGE_SIZE, IMAGE_SIZE, 3});
                 OrtSession.Result r = s.run(java.util.Collections.singletonMap("image", in))) {
                FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
                float[] scene = new float[fb.remaining()];
                fb.get(scene);
                res.timingsMs.put("encoder", ms(t));
                return scene;
            }
        }
    }

    public Mesh extractMesh(float[] scene, Options opt, Listener l, Result res) throws OrtException {
        int r = opt.resolution;
        long t = System.nanoTime();
        try (OrtSession field = open("triposr_field.onnx", opt.threads);
             OnnxTensor sceneT = OnnxTensor.createTensor(env, FloatBuffer.wrap(scene), new long[]{3, scene.length / 3 / 4096, 64, 64})) {
            int total = r * r * r;
            float[] values = new float[total];
            float[] lin = new float[r];
            for (int i = 0; i < r; i++) lin[i] = (r == 1 ? 0f : (float) i / (r - 1)) * 2 * RADIUS - RADIUS;
            float[] pts = new float[Math.min(FIELD_CHUNK, total) * 3];
            for (int start = 0; start < total; start += FIELD_CHUNK) {
                check();
                int n = Math.min(FIELD_CHUNK, total - start);
                for (int q = 0; q < n; q++) {
                    int idx = start + q;
                    pts[q * 3] = lin[idx / (r * r)];
                    pts[q * 3 + 1] = lin[(idx / r) % r];
                    pts[q * 3 + 2] = lin[idx % r];
                }
                float[] d = query(field, sceneT, pts, n, "density_act");
                for (int q = 0; q < n; q++) values[start + q] = d[q] - THRESHOLD;
                l.onProgress(Stage.FIELD, (start + n) / (float) total);
            }
            res.timingsMs.put("field", ms(t)); t = System.nanoTime();

            l.onProgress(Stage.MARCHING_CUBES, 0f);
            Mesh mesh = MarchingCubes.run(values, r);
            values = null;
            // v_pos / (R-1) then scale (0,1) -> (-radius, radius)
            mesh.transform(2 * RADIUS / (r - 1), -RADIUS);
            res.timingsMs.put("marching_cubes", ms(t)); t = System.nanoTime();
            if (mesh.faceCount() == 0) throw new IllegalStateException("TripoSR produced an empty surface");

            if (opt.vertexColors) {
                int vc = mesh.vertexCount();
                float[] colors = new float[vc * 3];
                for (int start = 0; start < vc; start += FIELD_CHUNK) {
                    check();
                    int n = Math.min(FIELD_CHUNK, vc - start);
                    float[] p = new float[n * 3];
                    System.arraycopy(mesh.positions, start * 3, p, 0, n * 3);
                    float[] c = query(field, sceneT, p, n, "color");
                    System.arraycopy(c, 0, colors, start * 3, n * 3);
                    l.onProgress(Stage.COLORS, (start + n) / (float) vc);
                }
                mesh.colors = colors;
                res.timingsMs.put("colors", ms(t));
            }
            return mesh;
        }
    }

    private float[] query(OrtSession field, OnnxTensor scene, float[] pts, int n, String output) throws OrtException {
        Map<String, OnnxTensor> in = new HashMap<>();
        try (OnnxTensor p = OnnxTensor.createTensor(env, FloatBuffer.wrap(pts, 0, n * 3), new long[]{n, 3})) {
            in.put("scene_code", scene);
            in.put("positions", p);
            try (OrtSession.Result r = field.run(in, java.util.Collections.singleton(output))) {
                FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
                float[] out = new float[fb.remaining()];
                fb.get(out);
                return out;
            }
        }
    }

    /** rembg u2net session: predict + naive_cutout. */
    public int[] removeBackground(int[] rgba, int w, int h, int threads) throws OrtException {
        final int s = 320;
        float[] rgb = new float[w * h * 3];
        for (int i = 0; i < w * h; i++) for (int c = 0; c < 3; c++) rgb[i * 3 + c] = rgba[i * 4 + c];
        // shortcut: rembg uses PIL LANCZOS here; antialiased bilinear differs by a few gray levels in the mask.
        float[] small = ImageOps.resize(rgb, w, h, 3, s, s);
        float max = 1e-6f;
        for (float v : small) max = Math.max(max, v);
        float[] mean = {0.485f, 0.456f, 0.406f}, std = {0.229f, 0.224f, 0.225f};
        float[] chw = new float[3 * s * s];
        for (int i = 0; i < s * s; i++)
            for (int c = 0; c < 3; c++) chw[c * s * s + i] = (small[i * 3 + c] / max - mean[c]) / std[c];
        float[] pred;
        try (OrtSession sess = open("u2net.onnx", threads);
             OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, 3, s, s})) {
            String name = sess.getInputNames().iterator().next();
            try (OrtSession.Result r = sess.run(java.util.Collections.singletonMap(name, in))) {
                FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
                pred = new float[s * s];
                fb.get(pred, 0, s * s); // [1,1,320,320] -> channel 0
            }
        }
        float mi = Float.MAX_VALUE, ma = -Float.MAX_VALUE;
        for (float v : pred) { mi = Math.min(mi, v); ma = Math.max(ma, v); }
        float[] m8 = new float[s * s];
        for (int i = 0; i < m8.length; i++) m8[i] = (int) (Math.max(0f, Math.min(1f, (pred[i] - mi) / (ma - mi))) * 255);
        float[] mask = ImageOps.resize(m8, s, s, 1, w, h);
        int[] out = new int[w * h * 4];
        for (int i = 0; i < w * h; i++) {
            int a = Math.max(0, Math.min(255, Math.round(mask[i])));
            // PIL Image.composite(img, transparent, mask) blends every channel, so rgb is premultiplied.
            for (int c = 0; c < 3; c++) out[i * 4 + c] = (rgba[i * 4 + c] * a + 127) / 255;
            out[i * 4 + 3] = a;
        }
        return out;
    }

    private static long ms(long t0) { return (System.nanoTime() - t0) / 1_000_000; }
}
