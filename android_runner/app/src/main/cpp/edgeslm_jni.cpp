// EdgeSLM Runner - JNI シム
//
// ここには**ロジックを置かない**。JSON 文字列と C++ 呼び出しの変換だけを行う。
// 引数・戻り値を JSON に統一しているのは、パラメータ追加のたびに
// シグネチャが変わって APK の再ビルドが必要になるのを避けるため（基本設計③ 6.2節）。

#include <jni.h>

#include <android/log.h>

#include <memory>
#include <string>

#include "runner/job.h"
#include "runner/session.h"
#include "third_party/json.hpp"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "EdgeSLMRunner", __VA_ARGS__)

using nlohmann::json;

namespace {

// ハンドルは Session と中断フラグを束ねたもの
struct Handle {
    edgeslm::Session session;
    bool cancel = false;
};

std::string to_std(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* p = env->GetStringUTFChars(s, nullptr);
    std::string out(p ? p : "");
    if (p) env->ReleaseStringUTFChars(s, p);
    return out;
}

jstring to_j(JNIEnv* env, const json& j) {
    return env->NewStringUTF(j.dump().c_str());
}

jstring error_json(JNIEnv* env, const std::string& where, const std::string& what) {
    json j;
    j["ok"] = false;
    j["error"] = where + ": " + what;
    return to_j(env, j);
}

Handle* as_handle(jlong h) { return reinterpret_cast<Handle*>(h); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeCreate(JNIEnv* env, jobject, jstring configJson) {
    try {
        const json config = json::parse(to_std(env, configJson));
        auto* h = new Handle();
        if (!h->session.open(config)) {
            // 失敗しても Handle は保持しない。理由は nativeDescribe が
            // 「開けなかった理由」を返せるよう、ログに出したうえで 0 を返す設計。
            LOGE("session.open failed: %s", h->session.last_error().c_str());
            delete h;
            return 0;
        }
        return reinterpret_cast<jlong>(h);
    } catch (const std::exception& e) {
        LOGE("nativeCreate exception: %s", e.what());
        return 0;
    }
}

JNIEXPORT jstring JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeDescribe(JNIEnv* env, jobject, jlong handle) {
    if (handle == 0) {
        return error_json(env, "describe", "session handle is null (open failed; see logcat)");
    }
    try {
        return to_j(env, as_handle(handle)->session.describe());
    } catch (const std::exception& e) {
        return error_json(env, "describe", e.what());
    }
}

JNIEXPORT jstring JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeRunJob(JNIEnv* env, jobject, jlong handle,
                                                  jstring jobJson) {
    if (handle == 0) {
        return error_json(env, "runJob", "session handle is null");
    }
    try {
        Handle* h = as_handle(handle);
        h->cancel = false;
        const json job = json::parse(to_std(env, jobJson));
        const json result = edgeslm::run_job(h->session, job, &h->cancel);
        return to_j(env, result);
    } catch (const std::exception& e) {
        return error_json(env, "runJob", e.what());
    }
}

JNIEXPORT jboolean JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeCancel(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return JNI_FALSE;
    as_handle(handle)->cancel = true;
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return;
    delete as_handle(handle);
}

JNIEXPORT jstring JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeVersions(JNIEnv* env, jobject) {
    try {
        return to_j(env, edgeslm::Session::versions());
    } catch (const std::exception& e) {
        return error_json(env, "versions", e.what());
    }
}

}  // extern "C"
