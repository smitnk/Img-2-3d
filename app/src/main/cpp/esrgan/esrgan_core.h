// Real-ESRGAN x4 upscaler on ncnn (BSD-3). Tiled processing as in Real-ESRGAN-ncnn-vulkan (tile + prepadding),
// Vulkan GPU first, CPU with smaller tiles when Vulkan is missing or fails. Plain C++ (no JNI) so a host test can run it.
#pragma once
#include <string>

namespace esrgan {

constexpr int kScale = 4;

struct Options {
    int tile = 200;          // Vulkan tile (input pixels, before padding); CPU uses min(tile, cpu_tile)
    int cpu_tile = 100;      // smaller tiles on CPU keep RAM low
    int prepadding = 10;     // x4plus-anime uses 10, like Real-ESRGAN-ncnn-vulkan
    int threads = 4;         // CPU threads
    bool try_vulkan = true;
};

// progress(user, 0..1); return false from nothing - the run cannot be cancelled mid-tile.
typedef void (*ProgressFn)(void* user, float fraction);

// rgb: w*h*3 bytes. out: (w*4)*(h*4)*3 bytes, caller-allocated. Returns 0 on success, else a negative error code.
// [backend] receives "vulkan" or "cpu" (+ a note when a fallback happened).
int upscale(const char* param_path, const char* bin_path, const unsigned char* rgb, int w, int h,
            unsigned char* out, const Options& opt, ProgressFn progress, void* user, std::string* backend);

}  // namespace esrgan
