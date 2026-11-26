// Host test: runs the app's Real-ESRGAN core (ncnn, CPU) on a small synthetic line-art image.
// usage: esrgan_test <model.param> <model.bin>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "esrgan_core.h"

typedef std::vector<unsigned char> Img;  // RGB

static Img make_ground_truth(int w, int h) {  // flat colour regions, a gradient, thin dark lines, a ring
    Img im((size_t)w * h * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char* p = &im[((size_t)y * w + x) * 3];
            int r = 230, g = 200, b = 170;
            if (x > w / 2) { r = 60; g = 120; b = 220; }
            if (y > h * 2 / 3) { r = x * 255 / w; g = 90; b = 90 + y * 100 / h; }
            double d = std::hypot(x - w * 0.3, y - h * 0.35);
            if (d > 28 && d < 31) r = g = b = 20;
            if (std::abs(x - y * 2 / 3 - 10) <= 1) r = g = b = 25;
            p[0] = r; p[1] = g; p[2] = b;
        }
    return im;
}

static Img box_down(const Img& s, int w, int h, int k) {
    int ow = w / k, oh = h / k;
    Img o((size_t)ow * oh * 3);
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                for (int j = 0; j < k; j++) for (int i = 0; i < k; i++) sum += s[((size_t)(y * k + j) * w + x * k + i) * 3 + c];
                o[((size_t)y * ow + x) * 3 + c] = (unsigned char)((sum + k * k / 2) / (k * k));
            }
    return o;
}

static double cubic(double x) {  // Catmull-Rom
    x = std::fabs(x);
    if (x < 1) return 1.5 * x * x * x - 2.5 * x * x + 1;
    if (x < 2) return -0.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
    return 0;
}

static Img bicubic_up(const Img& s, int w, int h, int k) {
    int ow = w * k, oh = h * k;
    Img o((size_t)ow * oh * 3);
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            double fx = (x + 0.5) / k - 0.5, fy = (y + 0.5) / k - 0.5;
            int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
            for (int c = 0; c < 3; c++) {
                double acc = 0;
                for (int j = -1; j <= 2; j++)
                    for (int i = -1; i <= 2; i++) {
                        int sx = std::min(std::max(ix + i, 0), w - 1), sy = std::min(std::max(iy + j, 0), h - 1);
                        acc += s[((size_t)sy * w + sx) * 3 + c] * cubic(fx - (ix + i)) * cubic(fy - (iy + j));
                    }
                o[((size_t)y * ow + x) * 3 + c] = (unsigned char)std::min(255.0, std::max(0.0, acc + 0.5));
            }
        }
    return o;
}

static double psnr(const Img& a, const Img& b) {
    double mse = 0;
    for (size_t i = 0; i < a.size(); i++) { double d = (double)a[i] - b[i]; mse += d * d; }
    mse /= a.size();
    return mse <= 1e-9 ? 99.0 : 10 * std::log10(255.0 * 255.0 / mse);
}

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) g_fail++;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: esrgan_test model.param model.bin\n"); return 2; }
    const int GW = 192, GH = 160, W = GW / 4, H = GH / 4;
    Img gt = make_ground_truth(GW, GH);
    Img lr = box_down(gt, GW, GH, 4);

    esrgan::Options opt;
    opt.try_vulkan = false;  // CI has no GPU: the CPU path is what the app falls back to
    opt.threads = 2;
    opt.tile = opt.cpu_tile = 32;  // several tiles in both directions, like a big photo would have
    Img out((size_t)W * 4 * H * 4 * 3);
    std::string backend;
    int code = esrgan::upscale(argv[1], argv[2], lr.data(), W, H, out.data(), opt, nullptr, nullptr, &backend);
    std::printf("upscale code=%d backend=%s size=%dx%d -> %dx%d\n", code, backend.c_str(), W, H, W * 4, H * 4);
    check(code == 0, "upscale returns 0");
    if (code != 0) return 1;
    check(out.size() == (size_t)GW * GH * 3, "output is exactly 4x in both directions");

    Img bic = bicubic_up(lr, W, H, 4);
    double pe = psnr(out, gt), pb = psnr(bic, gt), pd = psnr(out, bic);
    std::printf("PSNR vs ground truth: esrgan %.2f dB, bicubic %.2f dB; esrgan vs bicubic %.2f dB\n", pe, pb, pd);
    check(pd < 45.0, "differs from bicubic (PSNR esrgan-vs-bicubic < 45 dB)");
    check(pd > 14.0, "still the same picture as bicubic (> 14 dB)");
    check(pe > 14.0, "close to the ground truth (> 14 dB)");
    check(pe > pb, "closer to the ground truth than bicubic");

    // tiling must not leave seams: one big tile vs many small tiles
    esrgan::Options one = opt; one.tile = one.cpu_tile = 128;
    Img out2(out.size());
    code = esrgan::upscale(argv[1], argv[2], lr.data(), W, H, out2.data(), one, nullptr, nullptr, nullptr);
    check(code == 0, "single-tile run returns 0");
    double ptile = psnr(out, out2);
    std::printf("PSNR small tiles vs single tile: %.2f dB\n", ptile);
    check(ptile > 35.0, "tiled result matches the single-tile result (> 35 dB)");

    // Vulkan requested on a machine without a GPU: must fall back to CPU and still succeed
    esrgan::Options vk = one; vk.try_vulkan = true;
    std::string b3;
    code = esrgan::upscale(argv[1], argv[2], lr.data(), W, H, out2.data(), vk, nullptr, nullptr, &b3);
    std::printf("try_vulkan run: code=%d backend=%s\n", code, b3.c_str());
    check(code == 0 && (b3.rfind("cpu", 0) == 0 || b3 == "vulkan"), "try_vulkan=true works (GPU or CPU fallback)");

    std::printf(g_fail ? "RESULT: FAILED (%d)\n" : "RESULT: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
