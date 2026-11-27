# Host (linux x86_64) build of the mesh-repair code, run by .github/workflows/native-test.yml:
#   cmake -S tests/native -B build/native -DCMAKE_BUILD_TYPE=Release && cmake --build build/native -j && ctest --test-dir build/native --output-on-failure
cmake_minimum_required(VERSION 3.22)
project(meshrepair_test CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(${CMAKE_CURRENT_LIST_DIR}/../../app/src/main/cpp/manifold.cmake)
add_executable(repair_test repair_test.cpp ../../app/src/main/cpp/mesh_repair.cpp)
target_include_directories(repair_test PRIVATE ../../app/src/main/cpp)
target_link_libraries(repair_test PRIVATE manifold)
enable_testing()
add_test(NAME repair_test COMMAND repair_test)
