// EdgeSLM Runner - ORT/QNN セッション管理
//
// 設計上の要点（基本設計③ / 参考実装_ZipDepth解説 5章）:
//  - ORT は dlopen("libonnxruntime.so") + dlsym("OrtGetApiBase") で動的に読む
//  - QNN EP の backend_path には **実ファイルパス**（nativeLibraryDir/libQnnHtp.so）を渡す。
//    FastRPC が V68 skeleton に通常のファイルシステムパスを要求するため（C-6）
//  - グラフ配置のラベルは推測しない。**disable_cpu_ep_fallback=1 で成功したか**で決める。
//    strict 成功 → "all nodes" / strict 失敗かつ通常成功 → "CPU I/O boundary"
//  - 電力プロファイル（htp_performance_mode / rpc_control_latency）は **設定しない**。
//    unsigned PD では setPowerConfig が許可されず Run が失敗する（要件定義書 3.4節 C-3）
//
// （2026-10-18 追記）このファイルの役目:
//  - 何のためか: 端末の NPU（Snapdragon の HTP。ORT では QNN の実行役＝EP として使う）でモデルを動かす「セッション」の部品。
//    ORT（ONNX Runtime）の C API を直接呼び、モデルを開く・入出力の形を調べる・1 回推論する、を受け持つ。
//  - どこで使うか: job.cpp（task ごとの処理）が使う。JNI（edgeslm_jni.cpp）の Handle が 1 つ持ち、Kotlin の画面・サービスが
//    ハンドルで持ち回す。
//  - 入力: job.json（config。ep・runtime・model など）。出力: 推論の出力のバイト列と所要時間、describe() の JSON。
//  - 動かし方: 単独では動かない。アプリをビルドし、job を流す（各プロジェクトの README）。

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ort/onnxruntime_c_api.h"
#include "third_party/json.hpp"

namespace edgeslm {

// モデルの入力・出力 1 本の名前・形・型（collect_io が開いたときに ORT から読む）
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;   // 動的次元は -1
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t element_count() const;                 // 動的次元がある場合は 0
    size_t element_size() const;                  // 型1要素のバイト数
};

// ORT＋QNN のセッション 1 つ。モデル 1 本を開いて持ち、何度でも run できる。
// 使い方: open(config) → inputs()/outputs() で形を見て入力のバッファを用意 → run(...) を繰り返す → 破棄。
// コピーは禁止（ORT のハンドルを 2 重に解放しないため）
class Session {
public:
    // 作るだけでは何も開かない（開くのは open）。破棄で ORT のセッションと Env を解放する（libonnxruntime.so は閉じない）
    Session();
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // config は job.json 全体。runtime.native_lib_dir / runtime.base_dir を含む前提。
    // 戻り値: 開けたら true。失敗の理由は last_error()。
    // 流れ: ORT を dlopen → Env を作る → reopen(config["model"])（まず CPU へ回すのを禁じて開き、だめなら許して開く）
    bool open(const nlohmann::json& config);

    // 同じ Session で**別のモデルに張り替える**（v0.2）。
    // op_check のように多数のモデルを次々試す用途のために足した。
    // OrtEnv を作り直さないのは、QNN バックエンドがプロセス全体の状態を持つためで、
    // モデルごとに Env を作って壊すと不安定になる（Session のデストラクタの注記と同じ理由）。
    bool reopen(const std::string& model_rel);

    // 1回の推論。入力バッファは呼び出し側が用意する。所要時間(ms)を out_ms に返す。
    // 多入力版が本体で、単入力版はその薄い包み（KVキャッシュは多入力になる）。
    // outputs を渡すと出力テンソルの中身を取り出す（v0.3）。
    // **実機での精度を測るために足した**——ここまでの精度はすべてホスト上の
    // シミュレーションで、HTP の演算順序・丸めの違いが含まれていなかった。
    //   inputs : 入力ごとのバイト列（inputs() の順。大きさ＝要素数 × 型のバイト数。形はモデルの形をそのまま使う）
    //   out_ms : OrtRun だけの時間（ms。入力を包む・出力を写す時間は入らない）
    //   outputs: null でなければ、出力ごとの中身を写して返す（outputs() の順）
    //   戻り値 : 成功で true。失敗の理由は last_error()
    bool run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms,
             std::vector<std::vector<uint8_t>>* outputs);
    bool run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms);
    bool run(const std::vector<uint8_t>& input_bytes, double* out_ms);

    // job.json の相対パスを base_dir から解決する（infer task のファイル入出力用）
    std::string resolve_path(const std::string& p) const;

    // 入力・出力の一覧（開いたときに作る。並びはモデルの定義の順＝ run に渡す順）
    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }

    // 開いた結果の説明（ok・配置のラベル backend・strict_no_cpu_fallback・model・入出力の name/type/shape/static など）。
    // job の結果に毎回 "describe" として入る
    nlohmann::json describe() const;

    // ok: セッションが開いているか。backend_label: 配置のラベル（"…all nodes" か "…CPU I/O boundary"）。
    // last_error: 直近の失敗の理由（「どこで: 何が」の形）
    bool ok() const { return session_ != nullptr; }
    const std::string& backend_label() const { return backend_label_; }
    const std::string& last_error() const { return last_error_; }

    // ORT の版など（libonnxruntime.so から読む）。セッションを開かずに呼べる
    static nlohmann::json versions();

private:
    // load_ort: libonnxruntime.so を dlopen し、ORT の C API の表（OrtApi）を得る（1 回だけ）
    // create_session: オプションを組んで ORT のセッションを作る。strict=true なら CPU へ回すのを禁じる
    // collect_io: 入力・出力の名前・形・型を ORT から読んで inputs_ / outputs_ に入れる
    // set_error / check: 失敗の理由を last_error_ と logcat に残す。check は OrtStatus が null（成功）なら true
    bool load_ort();
    bool create_session(const std::string& model_path,
                        const nlohmann::json& config,
                        bool strict);
    bool collect_io();
    void set_error(const std::string& where, const std::string& what);
    bool check(OrtStatus* status, const std::string& where);

    // ORT の部品。api_ は関数の表、env_ はプロセスで使い回す環境、session_ が開いたモデル、
    // run_options_ は Run ごとの設定（既定のまま）、mem_info_ は CPU のメモリの説明（入力のバッファを包むときに使う）
    void* ort_lib_ = nullptr;
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtRunOptions* run_options_ = nullptr;
    OrtMemoryInfo* mem_info_ = nullptr;

    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    // 開いた結果。strict_succeeded_: CPU へ回すのを禁じて開けたか（＝全ノードが NPU）
    std::string backend_label_;
    std::string last_error_;
    std::string model_path_;
    bool strict_succeeded_ = false;
    nlohmann::json config_;   // reopen で EP オプションを再利用するため保持する
};

}  // namespace edgeslm
