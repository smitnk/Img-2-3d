// The prebuilt ncnn/glslang archives were compiled with a newer NDK libc++ whose std::unordered_map calls
// std::__hash_memory (added in LLVM 20). Older NDK libc++ builds lack it, so provide it here (FNV-1a; any stable hash works).
#include <cstddef>
#include <cstdint>

#if defined(__ANDROID__) && defined(_LIBCPP_VERSION)
namespace std {
inline namespace __ndk1 {
__attribute__((weak, visibility("default"))) size_t __hash_memory(const void* ptr, size_t size) noexcept;
size_t __hash_memory(const void* ptr, size_t size) noexcept {
    const unsigned char* p = static_cast<const unsigned char*>(ptr);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return static_cast<size_t>(h);
}
}  // namespace __ndk1
}  // namespace std
#endif
