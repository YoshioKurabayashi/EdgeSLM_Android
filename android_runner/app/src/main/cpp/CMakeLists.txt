# EdgeSLM Runner - 端末で動く C++ 部分（libedgeslm_runner.so）の組み立て方（2026-10-18 に説明を追記）
#
# 何のためか: 画面・サービス（Kotlin）から JNI で呼ぶ共有ライブラリ libedgeslm_runner.so を作る。
#   推論の中身（ORT＋QNN のセッション・task の実行）はすべてこの .so の中にあり、Kotlin には置かない
#   （EdgeSLM_Android/android_runner/README.md「設計上の約束」。最終目標が C++ の推論サービスのため）。
# どこで使うか: Gradle が app/build.gradle の externalNativeBuild（path 'src/main/cpp/CMakeLists.txt'・
#   cmake の版 3.22.1）からこのファイルを読み、arm64-v8a（abiFilters）向けに NDK でビルドする。
#   STL は build.gradle の '-DANDROID_STL=c++_static'（.so を 1 つだけにするため静的にリンク）。
# 入力: 下の add_library に並べた .cpp と、ort/（ORT の C API のヘッダ）・third_party/json.hpp（nlohmann/json）。
# 出力: libedgeslm_runner.so（APK の lib/arm64-v8a/ に入る）。Kotlin の NativeBridge が
#   System.loadLibrary("edgeslm_runner") で読む。
# 動かし方: このファイルを直接 cmake に渡すことはない。アプリのビルドで一緒に作られる:
#   $env:JAVA_HOME="D:\Android\jdk17"; cd <アプリのフォルダ>; .\gradlew.bat assembleDebug
cmake_minimum_required(VERSION 3.22.1)
# プロジェクト名。使う言語は C++ だけ（CXX）
project(edgeslm_runner CXX)

# C++17 でコンパイルする。REQUIRED: 使えないときは古い規格へ黙って下げず、止める
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SHARED = 共有ライブラリ（.so）。名前 edgeslm_runner → libedgeslm_runner.so
#   edgeslm_jni.cpp      Kotlin ↔ C++ のつなぎ（JSON 文字列の受け渡しだけ。ロジックを置かない）
#   runner/session.cpp   ORT＋QNN のセッション（開く・入出力を調べる・推論する）
#   runner/job.cpp       task ごとの処理（probe・bench・op_check・infer）
add_library(edgeslm_runner SHARED
    edgeslm_jni.cpp
    runner/session.cpp
    runner/job.cpp
)

# -O3: 速さを優先した最適化。-Wall -Wextra: 警告を多めに出す（見落としを減らすため）。
# PRIVATE: このライブラリを作るときだけ効く（使う側には伝えない）
target_compile_options(edgeslm_runner PRIVATE -O3 -Wall -Wextra)
# #include の起点。"runner/session.h"・"ort/onnxruntime_c_api.h"・"third_party/json.hpp" はこのフォルダから引く。
# third_party を足しているのは json.hpp を名前だけでも引けるようにするため（いまのソースは "third_party/json.hpp" と書くので、無くても通ると推定）
target_include_directories(edgeslm_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# ORT は dlopen("libonnxruntime.so") で動的に読む（ZipDepth と同じ方式）。
# AAR が jni/arm64-v8a/libonnxruntime.so を APK に入れるため、
# 実行時は nativeLibraryDir に展開されている。リンクはしない。
# log: __android_log_print（logcat へ出す。タグ "EdgeSLMRunner"）、android: NDK の Android の基本ライブラリ、
# dl: dlopen / dlsym（上の注記のとおり libonnxruntime.so を実行時に読むため）
target_link_libraries(edgeslm_runner log android dl)
