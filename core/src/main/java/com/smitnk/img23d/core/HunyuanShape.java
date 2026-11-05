package com.smitnk.img23d.core;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Photo -> shape with Hunyuan3D-2mini Turbo (Tencent Hunyuan 3D 2.0 Community License) on ONNX Runtime.
 * Mirrors hy3dgen's Hunyuan3DDiTFlowMatchingPipeline as checked by tools/hunyuan/export_hunyuan.py:
 * recentre on white -> cond.onnx (DINOv2) -> 5 consistency-flow Euler steps with dit.onnx -> vae.onnx ->
 * hierarchical octree decode with geo.onnx (HierarchicalVolumeDecoding) -> marching cubes at level 0.
 * Parts are opened one at a time to keep peak RAM near the largest part.
 */
public final class HunyuanShape {
    public static final float BOUNDS = 1.01f, BORDER = 0.15f;
    public static final int CHUNK = 8000;
    /** Fast mode: 3 shape steps instead of 5, and one dilation (not two) around the surface on the finest level. */
    public boolean fast;

    public interface Listener { void onProgress(String stage, float fraction); }

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final File dir;
    private final float[] sigmas;
    private final float scaleFactor;
    private final int latents, channels, threads;
    private final AtomicBoolean cancelled = new AtomicBoolean();
    public final Map<String, Long> timingsMs = new HashMap<>();

    public HunyuanShape(File dir, float[] sigmas, float scaleFactor, int latents, int channels, int threads) {
        this.dir = dir;
        this.sigmas = sigmas;
        this.scaleFactor = scaleFactor;
        this.latents = latents;
        this.channels = channels;
        this.threads = threads;
    }

    /** From hunyuan_info.json (or the CI's dit_info.json): every part carries L, C and scale_factor; dit adds sigmas. */
    public static HunyuanShape fromInfo(File dir, String json, int threads) {
        return new HunyuanShape(dir, sigmasOf(json), (float) num(json, "scale_factor"), (int) num(json, "L"), (int) num(json, "C"), threads);
    }

    /** Fast mode: every other sigma, keeping the last (5 steps -> 3), so the shape model runs ~40% less. */
    static float[] fewerSteps(float[] s) {
        int n = (s.length + 1) / 2 + (s.length % 2 == 0 ? 1 : 0);
        float[] o = new float[n];
        for (int i = 0; i < n - 1; i++) o[i] = s[2 * i];
        o[n - 1] = s[s.length - 1];
        return o;
    }

    static float[] sigmasOf(String json) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("\"sigmas\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(json);
        if (!m.find()) throw new IllegalArgumentException("hunyuan info has no sigmas");
        String[] parts = m.group(1).split(",");
        float[] sig = new float[parts.length];
        for (int i = 0; i < parts.length; i++) sig[i] = Float.parseFloat(parts[i].trim());
        return sig;
    }

    static double num(String json, String key) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("\"" + key + "\"\\s*:\\s*([-0-9.eE]+)").matcher(json);
        if (!m.find()) throw new IllegalArgumentException("hunyuan info has no " + key);
        return Double.parseDouble(m.group(1));
    }

    public void cancel() { cancelled.set(true); }

    private void check() { if (cancelled.get()) throw new java.util.concurrent.CancellationException(); }

    private OrtSession open(String name) throws OrtException {
        File f = new File(dir, name);
        if (!f.isFile()) throw new IllegalStateException("Missing Hunyuan3D file: " + f);
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(threads);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        // The arena + memory-pattern planner held 15.5 GB during the DiT steps on the desktop: free buffers as they go.
        // Arena off cost speed; with attention now traced in small head groups the pool may fit (-Dhunyuan.arena=true to test).
        // hunyuan.arena: "false" (lowest RAM), "dit" (pool only for the shape steps, where it doubles speed), "true" (all parts)
        String mode = System.getProperty("hunyuan.arena", "false");
        boolean arena = mode.equals("true") || (mode.equals("dit") && name.equals("dit.onnx"));
        o.setCPUArenaAllocator(arena);
        o.setMemoryPatternOptimization(arena);
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

    private OnnxTensor t(float[] a, long... shape) throws OrtException {
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(a), shape);
    }

    /**
     * @param rgba cut-out subject (alpha = mask), any size
     * @param octree final grid resolution (256 in the reference; 128 is ~8x fewer surface queries)
     * @return mesh in the app's TripoSR frame (Mesh.writeGlb maps it to Y-up)
     */
    public Mesh run(int[] rgba, int w, int h, int octree, long seed, Listener l) throws OrtException {
        long t0 = System.nanoTime();
        float[] cond;
        int[] condShape = new int[2];
        try (OrtSession s = open("cond.onnx")) {
            long[] shape = ((TensorInfo) s.getInputInfo().get("image").getInfo()).getShape();
            int size = (int) shape[2];
            l.onProgress("Reading the photo", 0f);
            Map<String, OnnxTensor> in = new HashMap<>();
            in.put("image", t(preprocess(rgba, w, h, size), 1, 3, size, size));
            cond = run1(s, in);
            long[] os = ((TensorInfo) s.getOutputInfo().get("cond").getInfo()).getShape();
            condShape[0] = (int) os[1];
            condShape[1] = (int) os[2];
            if (condShape[0] <= 0) { condShape[1] = 1536; condShape[0] = cond.length / condShape[1]; }
        }
        timingsMs.put("encoder", ms(t0)); t0 = System.nanoTime();
        check();

        int n = latents * channels;
        float[] x = new float[n];
        Random r = new Random(seed);
        for (int i = 0; i < n; i++) x[i] = (float) r.nextGaussian();
        float[] sigmas = fast ? fewerSteps(this.sigmas) : this.sigmas;
        int steps = sigmas.length - 1;
        try (OrtSession s = open("dit.onnx")) {
            for (int i = 0; i < steps; i++) {
                l.onProgress("Shaping", (float) i / steps);
                Map<String, OnnxTensor> in = new HashMap<>();
                in.put("x", t(x.clone(), 1, latents, channels));
                in.put("t", t(new float[]{sigmas[i]}, 1));
                in.put("cond", t(cond, 1, condShape[0], condShape[1]));
                float[] v = run1(s, in);
                float d = sigmas[i + 1] - sigmas[i];
                for (int k = 0; k < n; k++) x[k] += d * v[k];
                check();
            }
        }
        cond = null;
        timingsMs.put("shape_steps", ms(t0)); t0 = System.nanoTime();

        for (int k = 0; k < n; k++) x[k] /= scaleFactor;
        float[] kv;
        int kvWidth;
        try (OrtSession s = open("vae.onnx")) {
            Map<String, OnnxTensor> in = new HashMap<>();
            in.put("latents", t(x, 1, latents, channels));
            kv = run1(s, in);
            kvWidth = kv.length / latents;
        }
        timingsMs.put("decoder", ms(t0)); t0 = System.nanoTime();

        float[] grid;
        int g;
        try (OrtSession s = open("geo.onnx");
             OnnxTensor kvT = t(kv, 1, latents, kvWidth)) {
            Volume v = decode(s, kvT, octree, l);
            grid = v.values;
            g = v.size;
        }
        timingsMs.put("surface", ms(t0)); t0 = System.nanoTime();
        l.onProgress("Building the mesh", 0f);
        Mesh m = MarchingCubes.run(grid, g);
        grid = null;
        if (m.faceCount() == 0) throw new IllegalStateException("Hunyuan3D produced an empty surface");
        m.transform(2 * BOUNDS / (g - 1), -BOUNDS);
        if (signedVolume(m) < 0) flip(m);
        m = clean(m);
        Mesh out = toAppFrame(m);
        timingsMs.put("marching_cubes", ms(t0));
        l.onProgress("Done", 1f);
        return out;
    }

    static final class Volume { float[] values; int size; }

    /** HierarchicalVolumeDecoding: dense coarse grid, then only cells near the surface at each finer level. */
    Volume decode(OrtSession geo, OnnxTensor kv, int octree, Listener l) throws OrtException {
        List<Integer> res = new ArrayList<>();
        int o = octree;
        if (o < 63) res.add(o);
        while (o >= 63) { res.add(0, o); o /= 2; }
        float size = 2 * BOUNDS;
        int g = res.get(0) + 1;
        int[] idx = new int[g * g * g];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        float[] vals = new float[idx.length];
        query(geo, kv, idx, g, size / res.get(0), vals, l, 0, res.size());
        for (int level = 1; level < res.size(); level++) {
            int ng = res.get(level) + 1;
            boolean last = level == res.size() - 1;
            byte[] curr = nearSurface(vals, g);
            int expand = last ? 0 : 1;
            for (int e = 0; e < expand; e++) curr = dilate(curr, g);
            byte[] next = new byte[ng * ng * ng];
            for (int a = 0; a < g; a++)
                for (int b = 0; b < g; b++)
                    for (int c = 0; c < g; c++)
                        if (curr[(a * g + b) * g + c] != 0 && 2 * a < ng && 2 * b < ng && 2 * c < ng)
                            next[((2 * a) * ng + 2 * b) * ng + 2 * c] = 1;
            for (int e = 0; e < (fast && last ? 1 : 2 - expand); e++) next = dilate(next, ng);
            int count = 0;
            for (byte bt : next) if (bt != 0) count++;
            int[] sel = new int[count];
            for (int i = 0, k = 0; i < next.length; i++) if (next[i] != 0) sel[k++] = i;
            next = null;
            float[] nv = new float[ng * ng * ng];
            java.util.Arrays.fill(nv, -10000f);
            query(geo, kv, sel, ng, size / res.get(level), nv, l, level, res.size());
            vals = nv;
            g = ng;
        }
        Volume v = new Volume();
        v.values = vals;
        v.size = g;
        return v;
    }

    private void query(OrtSession geo, OnnxTensor kv, int[] sel, int g, float step, float[] dst, Listener l, int level, int levels)
            throws OrtException {
        float[] q = new float[Math.min(CHUNK, sel.length) * 3];
        for (int start = 0; start < sel.length; start += CHUNK) {
            check();
            int m = Math.min(CHUNK, sel.length - start);
            for (int i = 0; i < m; i++) {
                int id = sel[start + i];
                q[i * 3] = (id / (g * g)) * step - BOUNDS;
                q[i * 3 + 1] = ((id / g) % g) * step - BOUNDS;
                q[i * 3 + 2] = (id % g) * step - BOUNDS;
            }
            Map<String, OnnxTensor> in = new HashMap<>();
            in.put("queries", OnnxTensor.createTensor(env, FloatBuffer.wrap(q, 0, m * 3), new long[]{1, m, 3}));
            in.put("kv", kv);
            float[] out;
            try (OrtSession.Result r = geo.run(in)) {
                FloatBuffer b = ((OnnxTensor) r.get(0)).getFloatBuffer();
                out = new float[b.remaining()];
                b.get(out);
            } finally {
                in.get("queries").close();
            }
            for (int i = 0; i < m; i++) dst[sel[start + i]] = out[i];
            l.onProgress("Carving the surface", (level + (start + m) / (float) sel.length) / levels);
        }
    }

    /** extract_near_surface_volume_fn (+ |logit| < 0.95): a 6-neighbour with the other sign, edges replicated. */
    static byte[] nearSurface(float[] v, int g) {
        byte[] out = new byte[v.length];
        int[][] d = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (int a = 0; a < g; a++)
            for (int b = 0; b < g; b++)
                for (int c = 0; c < g; c++) {
                    int i = (a * g + b) * g + c;
                    float x = v[i];
                    if (x <= -9000f) continue;
                    boolean s = x > 0, hit = Math.abs(x) < 0.95f;
                    for (int k = 0; k < 6 && !hit; k++) {
                        int na = Math.max(0, Math.min(g - 1, a + d[k][0]));
                        int nb = Math.max(0, Math.min(g - 1, b + d[k][1]));
                        int nc = Math.max(0, Math.min(g - 1, c + d[k][2]));
                        float y = v[(na * g + nb) * g + nc];
                        if (y > -9000f && (y > 0) != s) hit = true;
                    }
                    if (hit) out[i] = 1;
                }
        return out;
    }

    /** 3x3x3 box dilation (the all-ones Conv3d, padding 1, then > 0). */
    static byte[] dilate(byte[] m, int g) {
        byte[] out = new byte[m.length];
        for (int a = 0; a < g; a++)
            for (int b = 0; b < g; b++)
                for (int c = 0; c < g; c++) {
                    if (m[(a * g + b) * g + c] == 0) continue;
                    for (int da = -1; da <= 1; da++)
                        for (int db = -1; db <= 1; db++)
                            for (int dc = -1; dc <= 1; dc++) {
                                int x = a + da, y = b + db, z = c + dc;
                                if (x >= 0 && y >= 0 && z >= 0 && x < g && y < g && z < g) out[(x * g + y) * g + z] = 1;
                            }
                }
        return out;
    }

    /** ImageProcessorV2: subject box scaled to (1 - border) of the square, centred on white, -> [3,size,size] in -1..1. */
    static float[] preprocess(int[] rgba, int w, int h, int size) {
        int x0 = w, y0 = h, x1 = -1, y1 = -1;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (rgba[(y * w + x) * 4 + 3] > 0) {
                    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
                }
        if (x1 < x0) throw new IllegalArgumentException("The photo has no subject after background removal");
        int cw = x1 - x0 + 1, ch = y1 - y0 + 1;
        float[] crop = new float[cw * ch * 4];
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++)
                for (int c = 0; c < 4; c++) crop[(y * cw + x) * 4 + c] = rgba[((y + y0) * w + x + x0) * 4 + c] / 255f;
        int desired = (int) (size * (1 - BORDER));
        float scale = desired / (float) Math.max(cw, ch);
        int nw = Math.max(1, (int) (cw * scale)), nh = Math.max(1, (int) (ch * scale));
        // shortcut: hy3dgen resizes INTER_AREA to the max(H,W) square then INTER_CUBIC to 512+; one antialiased resize here.
        float[] small = ImageOps.resize(crop, cw, ch, 4, nw, nh);
        int ox = (size - nw) / 2, oy = (size - nh) / 2, plane = size * size;
        float[] chw = new float[3 * plane];
        java.util.Arrays.fill(chw, 1f);
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++) {
                int s = (y * nw + x) * 4, p = (y + oy) * size + x + ox;
                float a = Math.max(0f, Math.min(1f, small[s + 3]));
                for (int c = 0; c < 3; c++) chw[c * plane + p] = (Math.min(1f, small[s + c]) * a + (1 - a)) * 2 - 1;
            }
        return chw;
    }

    /**
     * Keep the real outer surface: drop zero-area triangles, inward-facing closed parts (sealed bubbles inside the body,
     * which the app's game-ready check flagged as "normals pointing inward" / "enclosed parts") and loose fragments
     * under 1% of the faces.
     */
    static Mesh clean(Mesh m) { return clean(m, 0.01); }

    static Mesh clean(Mesh m, double minFraction) {
        float[] p = m.positions;
        int[] f = m.indices;
        int nf = f.length / 3, nv = p.length / 3;
        int[] parent = new int[nv];
        for (int i = 0; i < nv; i++) parent[i] = i;
        boolean[] keepFace = new boolean[nf];
        for (int t = 0; t < nf; t++) {
            int a = f[t * 3], b = f[t * 3 + 1], c = f[t * 3 + 2];
            if (a == b || b == c || a == c || area2(p, a, b, c) < 1e-14) continue;
            keepFace[t] = true;
            union(parent, a, b);
            union(parent, a, c);
        }
        java.util.Map<Integer, double[]> parts = new HashMap<>(); // root -> {faces, signed volume}
        for (int t = 0; t < nf; t++) {
            if (!keepFace[t]) continue;
            double[] st = parts.computeIfAbsent(find(parent, f[t * 3]), k -> new double[2]);
            st[0]++;
            st[1] += tetVolume(p, f[t * 3], f[t * 3 + 1], f[t * 3 + 2]);
        }
        int total = 0;
        for (double[] st : parts.values()) total += (int) st[0];
        IntListLocal out = new IntListLocal();
        for (int t = 0; t < nf; t++) {
            if (!keepFace[t]) continue;
            double[] st = parts.get(find(parent, f[t * 3]));
            if (st[1] <= 0 || st[0] < minFraction * total) continue;
            out.add(f[t * 3]); out.add(f[t * 3 + 1]); out.add(f[t * 3 + 2]);
        }
        // compact vertices
        int[] idx = out.toArray(), remap = new int[nv];
        java.util.Arrays.fill(remap, -1);
        int n = 0;
        for (int v : idx) if (remap[v] < 0) remap[v] = n++;
        float[] np = new float[n * 3];
        for (int v = 0; v < nv; v++) if (remap[v] >= 0) System.arraycopy(p, v * 3, np, remap[v] * 3, 3);
        for (int i = 0; i < idx.length; i++) idx[i] = remap[idx[i]];
        return new Mesh(np, idx, null);
    }

    private static int find(int[] par, int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    }

    private static void union(int[] par, int a, int b) {
        int ra = find(par, a), rb = find(par, b);
        if (ra != rb) par[ra] = rb;
    }

    private static double area2(float[] p, int a, int b, int c) {
        double ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
        double vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
        double x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
        return x * x + y * y + z * z;
    }

    private static double tetVolume(float[] p, int a3, int b3, int c3) {
        int a = a3 * 3, b = b3 * 3, c = c3 * 3;
        return (p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
                + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c])) / 6.0;
    }

    private static final class IntListLocal {
        int[] a = new int[1 << 16];
        int n;
        void add(int v) { if (n == a.length) a = java.util.Arrays.copyOf(a, n * 2); a[n++] = v; }
        int[] toArray() { return java.util.Arrays.copyOf(a, n); }
    }

    static double signedVolume(Mesh m) {
        double v = 0;
        float[] p = m.positions;
        int[] f = m.indices;
        for (int i = 0; i < f.length; i += 3) {
            int a = f[i] * 3, b = f[i + 1] * 3, c = f[i + 2] * 3;
            v += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
                    + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
        }
        return v / 6;
    }

    static void flip(Mesh m) {
        for (int i = 0; i < m.indices.length; i += 3) {
            int t = m.indices[i + 1];
            m.indices[i + 1] = m.indices[i + 2];
            m.indices[i + 2] = t;
        }
    }

    /**
     * Hunyuan meshes are Y-up; the app stores meshes in TripoSR's frame, which Mesh.yUpPositions maps
     * (x,y,z) -> (-y, z, -x). Store (Z, X, Y) so the written glTF is (X, Y, Z) turned 180° about Y, i.e. facing
     * the same way as TripoSR models (Blender +Y), so painting / Body Creator treat both alike.
     */
    static Mesh toAppFrame(Mesh m) {
        float[] p = m.positions, o = new float[p.length];
        for (int i = 0; i < p.length; i += 3) {
            float X = -p[i], Y = p[i + 1], Z = -p[i + 2]; // 180° about Y
            o[i] = -Z;
            o[i + 1] = -X;
            o[i + 2] = Y;
        }
        return new Mesh(o, m.indices, null);
    }

    private static long ms(long t0) { return (System.nanoTime() - t0) / 1_000_000; }
}
