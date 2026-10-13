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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ort/onnxruntime_c_api.h"
#include "third_party/json.hpp"

namespace edgeslm {

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;   // 動的次元は -1
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    size_t element_count() const;                 // 動的次元がある場合は 0
    size_t element_size() const;                  // 型1要素のバイト数
};

class Session {
public:
    Session();
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // config は job.json 全体。runtime.native_lib_dir / runtime.base_dir を含む前提。
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
    bool run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms,
             std::vector<std::vector<uint8_t>>* outputs);
    bool run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms);
    bool run(const std::vector<uint8_t>& input_bytes, double* out_ms);

    // job.json の相対パスを base_dir から解決する（infer task のファイル入出力用）
    std::string resolve_path(const std::string& p) const;

    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }

    nlohmann::json describe() const;

    bool ok() const { return session_ != nullptr; }
    const std::string& backend_label() const { return backend_label_; }
    const std::string& last_error() const { return last_error_; }

    static nlohmann::json versions();

private:
    bool load_ort();
    bool create_session(const std::string& model_path,
                        const nlohmann::json& config,
                        bool strict);
    bool collect_io();
    void set_error(const std::string& where, const std::string& what);
    bool check(OrtStatus* status, const std::string& where);

    void* ort_lib_ = nullptr;
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtRunOptions* run_options_ = nullptr;
    OrtMemoryInfo* mem_info_ = nullptr;

    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    std::string backend_label_;
    std::string last_error_;
    std::string model_path_;
    bool strict_succeeded_ = false;
    nlohmann::json config_;   // reopen で EP オプションを再利用するため保持する
};

}  // namespace edgeslm
