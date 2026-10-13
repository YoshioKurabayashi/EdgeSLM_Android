cmake_minimum_required(VERSION 3.22.1)
project(edgeslm_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(edgeslm_runner SHARED
    edgeslm_jni.cpp
    runner/session.cpp
    runner/job.cpp
)

target_compile_options(edgeslm_runner PRIVATE -O3 -Wall -Wextra)
target_include_directories(edgeslm_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# ORT は dlopen("libonnxruntime.so") で動的に読む（ZipDepth と同じ方式）。
# AAR が jni/arm64-v8a/libonnxruntime.so を APK に入れるため、
# 実行時は nativeLibraryDir に展開されている。リンクはしない。
target_link_libraries(edgeslm_runner log android dl)
