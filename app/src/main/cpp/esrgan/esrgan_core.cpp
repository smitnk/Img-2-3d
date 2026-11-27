#include "esrgan_core.h"

#include <algorithm>
#include <vector>

#include "gpu.h"
#include "mat.h"
#include "net.h"

namespace esrgan {

namespace {

// One pass over every tile with the given net. Returns 0 or a negative ncnn error.
int run_tiles(ncnn::Net& net, const unsigned char* rgb, int w, int h, unsigned char* out, int tile, int pad,
              ProgressFn progress, void* user) {
    const int tiles_x = (w + tile - 1) / tile, tiles_y = (h + tile - 1) / tile, total = tiles_x * tiles_y;
    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    const float denorm[3] = {255.f, 255.f, 255.f};
    const int ow = w * kScale;
    std::vector<unsigned char> buf;
    int done = 0;
    for (int ty = 0; ty < tiles_y; ty++) {
        for (int tx = 0; tx < tiles_x; tx++) {
            const int x0 = tx * tile, y0 = ty * tile;
            const int x1 = std::min(x0 + tile, w), y1 = std::min(y0 + tile, h);
            const int pw = (x1 - x0) + 2 * pad, ph = (y1 - y0) + 2 * pad;
            buf.resize((size_t)pw * ph * 3);
            for (int y = 0; y < ph; y++) {
                const int sy = std::min(std::max(y0 - pad + y, 0), h - 1);  // replicate the border
                for (int x = 0; x < pw; x++) {
                    const int sx = std::min(std::max(x0 - pad + x, 0), w - 1);
                    const unsigned char* s = rgb + ((size_t)sy * w + sx) * 3;
                    unsigned char* d = &buf[((size_t)y * pw + x) * 3];
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
                }
            }
            ncnn::Mat in = ncnn::Mat::from_pixels(buf.data(), ncnn::Mat::PIXEL_RGB, pw, ph);
            in.substract_mean_normalize(0, norm);
            ncnn::Mat res;
            {
                ncnn::Extractor ex = net.create_extractor();
                int r = ex.input("data", in);
                if (r != 0) return -10;
                r = ex.extract("output", res);
                if (r != 0 || res.empty()) return -11;
            }
            if (res.w != pw * kScale || res.h != ph * kScale || res.c != 3) return -12;
            res.substract_mean_normalize(0, denorm);
            std::vector<unsigned char> px((size_t)res.w * res.h * 3);
            res.to_pixels(px.data(), ncnn::Mat::PIXEL_RGB);
            // copy the tile without its padding
            const int cw = (x1 - x0) * kScale, ch = (y1 - y0) * kScale;
            for (int y = 0; y < ch; y++) {
                const unsigned char* s = &px[((size_t)(y + pad * kScale) * res.w + pad * kScale) * 3];
                unsigned char* d = out + ((size_t)(y0 * kScale + y) * ow + x0 * kScale) * 3;
                std::copy(s, s + (size_t)cw * 3, d);
            }
            done++;
            if (progress) progress(user, (float)done / total);
        }
    }
    return 0;
}

// Loads the model into [net] for GPU or CPU. Options must be set before loading.
int load(ncnn::Net& net, const char* param, const char* bin, bool vulkan, int threads) {
    net.opt.use_vulkan_compute = vulkan;
    net.opt.num_threads = threads;
    if (vulkan) {
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_int8_storage = true;
        net.opt.use_int8_arithmetic = false;
    }
    if (net.load_param(param) != 0) return -1;
    if (net.load_model(bin) != 0) return -2;
    return 0;
}

}  // namespace

int upscale(const char* param_path, const char* bin_path, const unsigned char* rgb, int w, int h,
            unsigned char* out, const Options& opt, ProgressFn progress, void* user, std::string* backend) {
    if (!param_path || !bin_path || !rgb || !out || w < 1 || h < 1) return -100;
    const int pad = opt.prepadding;
    std::string note;
#if NCNN_VULKAN
    bool instance = false;
    if (opt.try_vulkan) {
        if (ncnn::create_gpu_instance() == 0) instance = true;
        if (instance && ncnn::get_gpu_count() > 0) {
            int r;
            {
                ncnn::Net net;
                r = load(net, param_path, bin_path, true, opt.threads);
                if (r == 0) r = run_tiles(net, rgb, w, h, out, std::max(opt.tile, 32), pad, progress, user);
            }
            if (r == 0) {
                if (backend) *backend = "vulkan";
                ncnn::destroy_gpu_instance();
                return 0;
            }
            note = " (Vulkan failed with " + std::to_string(r) + ", fell back)";
        } else {
            note = " (no Vulkan GPU)";
        }
    }
    if (instance) ncnn::destroy_gpu_instance();
#else
    if (opt.try_vulkan) note = " (built without Vulkan)";
#endif
    ncnn::Net net;
    int r = load(net, param_path, bin_path, false, opt.threads);
    if (r == 0) r = run_tiles(net, rgb, w, h, out, std::max(std::min(opt.tile, opt.cpu_tile), 32), pad, progress, user);
    if (backend) *backend = "cpu" + note;
    return r;
}

}  // namespace esrgan
