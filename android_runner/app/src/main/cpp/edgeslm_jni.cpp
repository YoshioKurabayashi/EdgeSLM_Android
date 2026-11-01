// EdgeSLM Runner - JNI シム
//
// ここには**ロジックを置かない**。JSON 文字列と C++ 呼び出しの変換だけを行う。
// 引数・戻り値を JSON に統一しているのは、パラメータ追加のたびに
// シグネチャが変わって APK の再ビルドが必要になるのを避けるため（基本設計③ 6.2節）。
//
// （2026-10-18 追記）
// 何のためか: Kotlin の NativeBridge（external fun の宣言）と C++ の runner をつなぐ入口。
//   C++ の関数名 Java_<パッケージ>_NativeBridge_<名前> が、Kotlin の宣言と 1 対 1 に結び付く（JNI の名前の決まり）。
//   パッケージ名（ここでは org.edgeslm.runner）を変えるときは、ここの関数名もすべて変える必要がある。
// どこで使うか: 画面（Activity）や検証用の RunnerService が NativeBridge 経由で呼ぶ。
// 入力と出力: 引数も戻り値も JSON 文字列（失敗は {"ok": false, "error": "…"}）。開いた Session は
//   ハンドル（jlong ＝ C++ の Handle の番地）で Kotlin 側が持ち回す。
// 動かし方: 単独では動かない。アプリのビルド（gradlew assembleDebug）で libedgeslm_runner.so に入り、
//   NativeBridge が System.loadLibrary("edgeslm_runner") で読む

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
// nativeCreate が new し、その番地を jlong で Kotlin に返す。nativeDestroy が delete する。
// cancel は nativeCancel が立て、job.cpp のループ（cancel_flag）が見て止まる。
// ★2026-10-18 気づいた点: cancel は普通の bool（std::atomic ではない）で、別のスレッドから書く。
//   C++ の決まりではデータ競合にあたり、最適化によっては立てたことが見えない恐れがある（推定。コードは直していない）
struct Handle {
    edgeslm::Session session;
    bool cancel = false;
};

// jstring → std::string。GetStringUTFChars は Java の「修正 UTF-8」を返す（日本語は普通の UTF-8 と同じ形。
// 絵文字など 4 バイトの文字と NUL だけ形が違う）。null は空の文字列。使い終わったら必ず Release する
std::string to_std(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* p = env->GetStringUTFChars(s, nullptr);
    std::string out(p ? p : "");
    if (p) env->ReleaseStringUTFChars(s, p);
    return out;
}

// json → jstring。dump() は普通の UTF-8。NewStringUTF は修正 UTF-8 を期待するので、
// 絵文字など 4 バイトの文字が入ると化けるか失敗する恐れがある（推定。日本語の文言は 3 バイトの文字なので、そのままで問題ない）
jstring to_j(JNIEnv* env, const json& j) {
    return env->NewStringUTF(j.dump().c_str());
}

// 失敗を {"ok": false, "error": "<where>: <what>"} の JSON で返す。
// C++ の例外を Kotlin へ投げない（JNI の境目で例外が抜けるとアプリが落ちる）ので、各関数は try で包んでここへ来る
jstring error_json(JNIEnv* env, const std::string& where, const std::string& what) {
    json j;
    j["ok"] = false;
    j["error"] = where + ": " + what;
    return to_j(env, j);
}

// jlong → Handle*（nativeCreate が返した値を戻すだけ。0 かどうかは呼ぶ側で確かめる）
Handle* as_handle(jlong h) { return reinterpret_cast<Handle*>(h); }

}  // namespace

extern "C" {

// セッションを開く。configJson: job.json 全体（runtime.native_lib_dir・runtime.base_dir は Kotlin 側の JsonInject が足す）。
// 戻り値: 開けたら Handle の番地（0 以外）、開けなければ 0（理由は logcat。この版は nativeDescribe(0) では理由を返さない）
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

// 開いたセッションの説明（Session::describe: 配置のラベル・入出力の名前と形など）を JSON で返す。
// handle が 0 なら「開けなかった（logcat を見よ）」の error
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

// job を 1 つ実行して結果の JSON を返す（job.cpp の run_job。task で振る舞いが変わる）。
// 終わるまで戻らない（同期）。始める前に cancel を下ろす（前の job の「止めて」を持ち越さない）
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

// 実行中の job に止めるよう頼む（cancel を立てるだけ。実際に止まるのは job.cpp のループがフラグを見たとき）。
// handle が 0 なら何もせず false
JNIEXPORT jboolean JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeCancel(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return JNI_FALSE;
    as_handle(handle)->cancel = true;
    return JNI_TRUE;
}

// セッションを閉じて Handle を消す（Session のデストラクタが ORT の部品を解放する）。
// 実行中の nativeRunJob があるかはここでは確かめない。呼ぶ側が、終わってから呼ぶこと
JNIEXPORT void JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    if (handle == 0) return;
    delete as_handle(handle);
}

// ORT の版などを返す（Session::versions。セッションを開かずに呼べる）
JNIEXPORT jstring JNICALL
Java_org_edgeslm_runner_NativeBridge_nativeVersions(JNIEnv* env, jobject) {
    try {
        return to_j(env, edgeslm::Session::versions());
    } catch (const std::exception& e) {
        return error_json(env, "versions", e.what());
    }
}

}  // extern "C"
