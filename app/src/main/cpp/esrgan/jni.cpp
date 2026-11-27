// JNI wrapper: com.smitnk.img23d.ai.EsrganNative
#include <jni.h>

#include <string>

#include "esrgan_core.h"

namespace {
struct Cb {
    JNIEnv* env;
    jobject obj;
    jmethodID mid;
};
void progress_cb(void* user, float f) {
    Cb* c = static_cast<Cb*>(user);
    if (c->obj && c->mid) c->env->CallVoidMethod(c->obj, c->mid, f);
}
}  // namespace

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_smitnk_img23d_ai_EsrganNative_upscale(JNIEnv* env, jobject, jstring param, jstring bin, jbyteArray rgb, jint w, jint h,
                                               jint tile, jint cpuTile, jint threads, jboolean tryVulkan, jobject progress,
                                               jobjectArray backendOut) {
    if (w < 1 || h < 1 || env->GetArrayLength(rgb) != w * h * 3) return nullptr;
    const char* p = env->GetStringUTFChars(param, nullptr);
    const char* b = env->GetStringUTFChars(bin, nullptr);
    jbyte* in = env->GetByteArrayElements(rgb, nullptr);
    const size_t outBytes = (size_t)w * h * esrgan::kScale * esrgan::kScale * 3;
    jbyteArray result = env->NewByteArray((jsize)outBytes);
    jbyte* out = result ? env->GetByteArrayElements(result, nullptr) : nullptr;
    int code = -200;
    std::string backend;
    if (out) {
        esrgan::Options opt;
        opt.tile = tile; opt.cpu_tile = cpuTile; opt.threads = threads; opt.try_vulkan = tryVulkan;
        Cb cb{env, progress, nullptr};
        if (progress) cb.mid = env->GetMethodID(env->GetObjectClass(progress), "onProgress", "(F)V");
        code = esrgan::upscale(p, b, reinterpret_cast<const unsigned char*>(in), w, h, reinterpret_cast<unsigned char*>(out), opt,
                               progress_cb, &cb, &backend);
        env->ReleaseByteArrayElements(result, out, 0);
    }
    env->ReleaseByteArrayElements(rgb, in, JNI_ABORT);
    env->ReleaseStringUTFChars(param, p);
    env->ReleaseStringUTFChars(bin, b);
    if (backendOut && env->GetArrayLength(backendOut) > 0) {
        std::string s = (code == 0 ? backend : "error " + std::to_string(code) + " " + backend);
        env->SetObjectArrayElement(backendOut, 0, env->NewStringUTF(s.c_str()));
    }
    return code == 0 ? result : nullptr;
}
