package com.smitnk.img23d.ai

/** JNI to libimg23d_esrgan (ncnn Real-ESRGAN x4, Vulkan first, CPU fallback). See app/src/main/cpp/esrgan. */
object EsrganNative {
    fun interface Progress { fun onProgress(fraction: Float) }

    private val loaded: Boolean = runCatching { System.loadLibrary("img23d_esrgan") }.isSuccess
    val available get() = loaded

    /**
     * [rgb] is w*h*3 bytes; returns (4w)*(4h)*3 bytes or null on failure. [backendOut][0] gets "vulkan", "cpu (note)" or the error.
     * Vulkan uses [tile]; the CPU path (no GPU, or Vulkan failed) uses at most [cpuTile] to keep RAM low.
     */
    @JvmStatic
    external fun upscale(param: String, bin: String, rgb: ByteArray, w: Int, h: Int, tile: Int, cpuTile: Int, threads: Int,
                         tryVulkan: Boolean, progress: Progress?, backendOut: Array<String?>): ByteArray?
}
