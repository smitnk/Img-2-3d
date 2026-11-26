cmake_minimum_required(VERSION 3.22)
project(meshrepair CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(${CMAKE_CURRENT_LIST_DIR}/manifold.cmake)
add_library(meshrepair SHARED jni_repair.cpp mesh_repair.cpp)
target_link_libraries(meshrepair PRIVATE manifold log)
# 16 KB page alignment (Android 15+) and no unused symbols.
target_link_options(meshrepair PRIVATE -Wl,-z,max-page-size=16384 -Wl,--gc-sections)

# Photo enhance (Real-ESRGAN on ncnn): libimg23d_esrgan.so, see esrgan/CMakeLists.txt
add_subdirectory(esrgan)
