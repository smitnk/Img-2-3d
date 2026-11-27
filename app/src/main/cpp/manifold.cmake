# Pulls the manifold library (Apache-2.0, https://github.com/elalish/manifold) at an exact tag and builds it static,
# with every optional piece off: no TBB (MANIFOLD_PAR), no CrossSection/Clipper2, no python/C/JS bindings, no assimp, no tests.
# Shared by the Android build (app/src/main/cpp/CMakeLists.txt) and the host test (tests/native/CMakeLists.txt).
include(FetchContent)
set(MANIFOLD_PAR OFF CACHE BOOL "" FORCE)
set(MANIFOLD_CROSS_SECTION OFF CACHE BOOL "" FORCE)
set(MANIFOLD_PYBIND OFF CACHE BOOL "" FORCE)
set(MANIFOLD_CBIND OFF CACHE BOOL "" FORCE)
set(MANIFOLD_JSBIND OFF CACHE BOOL "" FORCE)
set(MANIFOLD_TEST OFF CACHE BOOL "" FORCE)
set(MANIFOLD_EXPORT OFF CACHE BOOL "" FORCE)
set(ASSIMP_ENABLE OFF CACHE BOOL "" FORCE)
set(MANIFOLD_DEBUG OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(manifold
  GIT_REPOSITORY https://github.com/elalish/manifold.git
  GIT_TAG v3.5.5          # commit 32271a1a999ce721ffaf5ab10e9746ceeb696547
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(manifold)
