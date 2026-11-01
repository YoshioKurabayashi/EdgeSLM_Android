// EdgeSLM Runner - ORT/QNN セッション管理の本体（宣言と設計の要点は session.h）（2026-10-18 に説明を追記）
//
// 何のためか: ORT の C API を dlopen で読み、QNN の実行役（HTP ＝ NPU）でモデルを開いて推論する。
// どこで使うか: job.cpp（task ごとの処理）・edgeslm_jni.cpp（Handle が 1 つ持つ）から。
// 入力: job.json の ep（QNN のオプション・ログの細かさ・スレッド数）・runtime（置き場）・model（モデルの場所）。
// 出力: 推論の出力（バイト列）と時間（ms）、describe() の JSON（配置のラベル・入出力の形）。
// 動かし方: 単独では動かない。アプリをビルドし、job を流す（各プロジェクトの README）。
// ログ: logcat のタグ "EdgeSLMRunner"（adb logcat -s EdgeSLMRunner で見られる）
#include "runner/session.h"

#include <android/log.h>
#include <dlfcn.h>

#include <chrono>
#include <cstdlib>   // setenv / getenv（FastRPC のパス設定に使う）
#include <cstring>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "EdgeSLMRunner", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

namespace {

// 今の時刻（ms）。steady_clock（時計合わせで戻らない時計）なので、差を取る用途だけに使う
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// ORT の型の番号 → describe() に出す名前
const char* type_name(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:return "float16";
        default: return "other";
    }
}

// job.json の相対パスを base_dir から解決する。絶対パスならそのまま。
std::string resolve(const std::string& base, const std::string& p) {
    if (!p.empty() && p[0] == '/') return p;
    if (base.empty()) return p;
    return base + "/" + p;
}

// FastRPC のライブラリ探索パスを**自分で**設定する（既定では呼ばない）。
//
// ★重要な教訓（2026-09-12 に実際に踏んだ）:
//   ORT の QNN EP は `ADSP_LIBRARY_PATH` を**自分で設定する**。こちらが先に設定すると
//   ORT は自分の設定をスキップし、次の警告を出したうえで HTP バックエンドの
//   デバイス生成に失敗する:
//     W qnn_backend_manager.cc:308] Using existing ADSP_LIBRARY_PATH setting of <path>,
//       which may cause the HTP backend to fail.
//     E qnn_execution_provider.cc:1046 GetCapability] QNN SetupBackend failed
//       Failed to create device. Error: QNN_DEVICE_ERROR_INVALID_CONFIG
//   さらにその後の症状は「EPContext node is not compatible with any execution provider」
//   という**真因を示さないエラー**になる。
//
//   → したがって既定では**何もしない**。ORT に任せる。
//      必要な場合のみ job.json の runtime.set_fastrpc_paths=true で有効化する
//      （ZipDepth は自前で設定しており、比較実験のために残してある）。
//
// 区切り文字が2つで異なる点に注意:
//   LD_LIBRARY_PATH   → ':'（POSIX 流儀）
//   ADSP_LIBRARY_PATH → ';'（Qualcomm 流儀）
void configure_fastrpc_paths(const std::string& dir) {
    if (dir.empty()) return;
    const char* old_ld = getenv("LD_LIBRARY_PATH");
    const char* old_adsp = getenv("ADSP_LIBRARY_PATH");
    if (old_ld && strstr(old_ld, dir.c_str()) &&
        old_adsp && strstr(old_adsp, dir.c_str())) {
        return;  // 既に設定済み
    }
    std::string ld = dir;
    if (old_ld && old_ld[0]) { ld += ":"; ld += old_ld; }
    setenv("LD_LIBRARY_PATH", ld.c_str(), 1);

    std::string adsp = dir;
    if (old_adsp && old_adsp[0]) { adsp += ";"; adsp += old_adsp; }
    setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);

    LOGI("fastrpc paths configured: %s", dir.c_str());
}

// OrtStatus* を返すが失敗しても続行してよい呼び出し用。
// **黙って捨てない**——落とすと -Wunused-result が出るうえ、
// 後から「設定できていたのか」を確認できなくなる。警告として残す。
void discard(const OrtApi* api, OrtStatus* status, const char* where) {
    if (!status) return;
    const char* msg = api->GetErrorMessage(status);
    LOGI("(non-fatal) %s: %s", where, msg ? msg : "(no message)");
    api->ReleaseStatus(status);
}

}  // namespace

// 型 1 要素のバイト数。知らない型は 0（呼ぶ側は 0 を「扱えない」とみなす）
size_t TensorInfo::element_size() const {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return 8;
        default: return 0;
    }
}

// 要素の数（形の各次元の積）。動的な次元（0 以下）があれば 0
size_t TensorInfo::element_count() const {
    size_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) return 0;  // 動的次元。QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）
        n *= static_cast<size_t>(d);
    }
    return n;
}

Session::Session() = default;

// ORT の部品を解放する（api_ が無い＝ ORT を読めなかったときは、解放するものも無い）
Session::~Session() {
    if (api_) {
        if (run_options_) api_->ReleaseRunOptions(run_options_);
        if (mem_info_)    api_->ReleaseMemoryInfo(mem_info_);
        if (session_)     api_->ReleaseSession(session_);
        if (env_)         api_->ReleaseEnv(env_);
    }
    // ort_lib_ は dlclose しない。QNN バックエンドはプロセス全体の状態を持つため、
    // 閉じて開き直すと不安定になる（ZipDepth も同じ判断をしている）。
}

// 失敗の理由を「どこで: 何が」の形で last_error_ に残し、logcat にも出す
void Session::set_error(const std::string& where, const std::string& what) {
    last_error_ = where + ": " + what;
    LOGE("%s", last_error_.c_str());
}

// ORT の呼び出しの結果を確かめる。成功（status が null）なら true。
// 失敗なら理由を残し、status を解放して false（ORT の決まりで、status は受け取った側が解放する）
bool Session::check(OrtStatus* status, const std::string& where) {
    if (status == nullptr) return true;
    const char* msg = api_->GetErrorMessage(status);
    set_error(where, msg ? msg : "(no message)");
    api_->ReleaseStatus(status);
    return false;
}

// libonnxruntime.so を読み、ORT の C API の表（OrtApi）を得る。2 回目以降は何もしない。
// .so は AAR が APK に入れたもの（nativeLibraryDir に展開される）。リンクせずに実行時に読む（ZipDepth と同じ方式）
bool Session::load_ort() {
    if (api_) return true;
    // ① .so を開く。RTLD_NOW: 足りない関数があれば今すぐ失敗させる。RTLD_LOCAL: 中の名前を他の .so に見せない
    ort_lib_ = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!ort_lib_) {
        // ★2026-10-18 気づいた点: dlerror() は呼ぶたびに中身を消す。1 回目で「理由あり」と分かっても、2 回目の dlerror() は
        //   null を返し、null から std::string を作ることになる（未定義の動き。落ちる恐れ）。コードは直していない（下の dlsym の所も同じ）
        set_error("dlopen libonnxruntime.so", dlerror() ? dlerror() : "(null)");
        return false;
    }
    // ② 入口の関数 OrtGetApiBase を名前で探す（ORT の C API はこの 1 つの関数から全部たどれる）
    using GetApiBaseFn = const OrtApiBase* (*)(void);
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(dlsym(ort_lib_, "OrtGetApiBase"));
    if (!get_api_base) {
        set_error("dlsym OrtGetApiBase", dlerror() ? dlerror() : "(null)");
        return false;
    }
    // ③ ヘッダ（ort/onnxruntime_c_api.h）の版の API を求める。.so の方が古ければ null が返る
    //    （ヘッダは AAR 1.27.0 と対応。EdgeSLM_Android/android_runner/README.md）
    api_ = get_api_base()->GetApi(ORT_API_VERSION);
    if (!api_) {
        set_error("GetApi", "requested ORT C API version is unavailable");
        return false;
    }
    return true;
}

// ORT のセッションを 1 つ作る（session_ に入れる）。
//   model_path: モデルの絶対パス。config: job.json 全体（ep.session・ep.options・ep.backend・runtime.native_lib_dir など）
//   strict    : true なら CPU へ回すのを禁じて開く（開けたら全ノードが NPU）
//   戻り値    : 開けたら true。strict で開けないのは「異常」ではなく判定の材料なので、ログは INFO で残す
bool Session::create_session(const std::string& model_path,
                             const json& config,
                             bool strict) {
    OrtSessionOptions* options = nullptr;
    if (!check(api_->CreateSessionOptions(&options), "CreateSessionOptions")) return false;

    // ① job の ep の部分を取り出す（無ければ空）
    const json ep      = config.value("ep", json::object());
    const json sess    = ep.value("session", json::object());
    const json ep_opts = ep.value("options", json::object());

    // ② 共通のオプション。ログの細かさ（既定 2 ＝警告以上）、CPU 側のスレッド数（既定 1）、グラフの最適化（全部）。
    //    失敗しても続ける（discard が logcat に残す）
    discard(api_, api_->SetSessionLogSeverityLevel(options, sess.value("log_severity_level", 2)),
            "SetSessionLogSeverityLevel");
    discard(api_, api_->SetIntraOpNumThreads(options, sess.value("intra_op_num_threads", 1)),
            "SetIntraOpNumThreads");
    discard(api_, api_->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL),
            "SetSessionGraphOptimizationLevel");

    // --- QNN EP のプロバイダオプション ---
    // backend_path は実ファイルパスでなければならない（C-6）。
    const std::string native_lib_dir = config.value("runtime", json::object())
                                             .value("native_lib_dir", std::string());
    const std::string backend_so = ep.value("backend", std::string("libQnnHtp.so"));
    const std::string backend_path = native_lib_dir.empty()
                                   ? backend_so
                                   : native_lib_dir + "/" + backend_so;

    // ep.options の中身を、名前と値の文字列の組にしてそのまま QNN の実行役へ渡す（文字列でない値は JSON の文字にする。
    // 例: 数値 1 → "1"）。オプションを足しても APK を作り直さなくてよい
    std::vector<std::string> key_store, val_store;
    key_store.push_back("backend_path");
    val_store.push_back(backend_path);
    for (auto it = ep_opts.begin(); it != ep_opts.end(); ++it) {
        key_store.push_back(it.key());
        val_store.push_back(it.value().is_string() ? it.value().get<std::string>()
                                                   : it.value().dump());
    }
    // ORT は const char* の並びを求める。文字列の中身は key_store / val_store が持ち、ここは番地を並べるだけ
    std::vector<const char*> keys, vals;
    keys.reserve(key_store.size());
    vals.reserve(val_store.size());
    for (size_t i = 0; i < key_store.size(); ++i) {
        keys.push_back(key_store[i].c_str());
        vals.push_back(val_store[i].c_str());
    }

    // ③ QNN の実行役をオプションに付ける（失敗＝ .so が無い・オプション名の誤りなど。ここで止める）
    OrtStatus* append = api_->SessionOptionsAppendExecutionProvider(
        options, "QNN", keys.data(), vals.data(), keys.size());
    if (!check(append, "SessionOptionsAppendExecutionProvider(QNN)")) {
        api_->ReleaseSessionOptions(options);
        return false;
    }

    // strict の場合のみ CPU フォールバックを禁止する。
    // これが成功したかどうかが、そのままグラフ配置のラベルになる。
    if (strict) {
        discard(api_, api_->AddSessionConfigEntry(options, "session.disable_cpu_ep_fallback", "1"),
                "disable_cpu_ep_fallback");
    }

    // 電力プロファイル（htp_performance_mode / rpc_control_latency）は設定しない。
    // unsigned PD では setPowerConfig が許可されず Run が失敗するため（C-3）。

    // ④ セッションを作る。EPContext のモデル（NPU 用に前もって変換した context binary を包んだ ONNX）なら、
    //    ここで NPU へ読み込む（数秒。ヒープに読んでから NPU 側の dma-buf へ写すので、一時的に binary の約 2 倍のメモリが要る。
    //    EdgeSLM_AndroidScale P0A）。オプションは作り終えたら要らないので、成否にかかわらず解放する
    OrtStatus* st = api_->CreateSession(env_, model_path.c_str(), options, &session_);
    api_->ReleaseSessionOptions(options);
    if (st != nullptr) {
        const char* msg = api_->GetErrorMessage(st);
        // strict 失敗は「異常」ではなく判定材料なので、ログレベルを落として記録する
        LOGI("CreateSession(strict=%d) failed: %s", strict ? 1 : 0, msg ? msg : "(no message)");
        last_error_ = std::string("CreateSession(strict=") + (strict ? "1" : "0") + "): " +
                      (msg ? msg : "(no message)");
        api_->ReleaseStatus(st);
        session_ = nullptr;
        return false;
    }
    return true;
}

// 開いたセッションの入力・出力の名前・型・形を読み、inputs_ / outputs_ に入れる。
// 名前は ORT が確保した文字列なので、写したらすぐ解放する。型・形が取れなくても致命的にはしない（discard）
bool Session::collect_io() {
    OrtAllocator* alloc = nullptr;
    if (!check(api_->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions"))
        return false;

    size_t n_in = 0, n_out = 0;
    if (!check(api_->SessionGetInputCount(session_, &n_in), "SessionGetInputCount")) return false;
    if (!check(api_->SessionGetOutputCount(session_, &n_out), "SessionGetOutputCount")) return false;

    // collect: is_input で入力・出力を切り替え、count 本を順に読んで out に足す
    auto collect = [&](bool is_input, size_t count, std::vector<TensorInfo>* out) -> bool {
        for (size_t i = 0; i < count; ++i) {
            char* name = nullptr;
            OrtStatus* st = is_input ? api_->SessionGetInputName(session_, i, alloc, &name)
                                     : api_->SessionGetOutputName(session_, i, alloc, &name);
            if (!check(st, "SessionGetName")) return false;

            TensorInfo info;
            info.name = name ? name : "";
            alloc->Free(alloc, name);

            OrtTypeInfo* type_info = nullptr;
            st = is_input ? api_->SessionGetInputTypeInfo(session_, i, &type_info)
                          : api_->SessionGetOutputTypeInfo(session_, i, &type_info);
            if (!check(st, "SessionGetTypeInfo")) return false;

            const OrtTensorTypeAndShapeInfo* tinfo = nullptr;
            if (check(api_->CastTypeInfoToTensorInfo(type_info, &tinfo), "CastTypeInfoToTensorInfo")
                && tinfo != nullptr) {
                discard(api_, api_->GetTensorElementType(tinfo, &info.type),
                        "GetTensorElementType");
                size_t dims = 0;
                discard(api_, api_->GetDimensionsCount(tinfo, &dims), "GetDimensionsCount");
                info.shape.resize(dims);
                if (dims > 0) {
                    discard(api_, api_->GetDimensions(tinfo, info.shape.data(), dims),
                            "GetDimensions");
                }
            }
            api_->ReleaseTypeInfo(type_info);
            out->push_back(std::move(info));
        }
        return true;
    };

    inputs_.clear();
    outputs_.clear();
    if (!collect(true, n_in, &inputs_))   return false;
    if (!collect(false, n_out, &outputs_)) return false;
    return true;
}

// モデルを開く（最初の 1 回）。config は job.json 全体。戻り値: 開けたら true（理由は last_error()）。
// 流れ: FastRPC のパス（既定は ORT に任せる）→ ORT を読む → Env → RunOptions・CPU の MemoryInfo
//       → reopen でモデルを開く（strict → だめなら通常）
bool Session::open(const json& config) {
    const json runtime = config.value("runtime", json::object());
    const std::string native_lib_dir = runtime.value("native_lib_dir", std::string());

    // ★既定では FastRPC のパスを自分で設定しない。ORT の QNN EP に任せる。
    // 先に設定すると ORT が自分の設定をスキップしてデバイス生成に失敗する
    // （configure_fastrpc_paths の説明を参照）。
    if (runtime.value("set_fastrpc_paths", false)) {
        LOGI("runtime.set_fastrpc_paths=true: setting paths ourselves (ORT will skip its own)");
        configure_fastrpc_paths(native_lib_dir);
    } else {
        LOGI("runtime.set_fastrpc_paths=false: leaving ADSP_LIBRARY_PATH to ORT");
    }

    // ORT を読む（失敗は .so が無い・版が合わない など）
    if (!load_ort()) return false;

    // Env はプロセスで使い回す（reopen では作り直さない。QNN の実行役がプロセス全体の状態を持つため）
    if (!env_) {
        if (!check(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "edgeslm", &env_), "CreateEnv"))
            return false;
    }

    // reopen が EP のオプションを使い回せるよう、config を覚えておく
    config_ = config;

    // RunOptions（Run ごとの設定。既定のまま）と、CPU のメモリの説明（入力のバッファを包むときに使う）を作る
    if (!check(api_->CreateRunOptions(&run_options_), "CreateRunOptions")) return false;
    if (!check(api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_),
               "CreateCpuMemoryInfo")) return false;

    return reopen(config.value("model", std::string()));
}

// 同じ Session でモデルを開き直す（張り替え。open の最後もこれを呼ぶ）。
//   model_rel: base_dir からの相対パス（絶対パスも可）。戻り値: 開けたら true（入出力の一覧も作り直す）
// 前のセッションを閉じてから開く（同時に 2 本持たない）
bool Session::reopen(const std::string& model_rel) {
    if (!api_) {
        set_error("reopen", "session was never opened");
        return false;
    }
    const std::string base_dir = config_.value("runtime", json::object())
                                        .value("base_dir", std::string());
    const std::string path = resolve(base_dir, model_rel);
    if (path.empty()) {
        set_error("reopen", "model path is empty");
        return false;
    }

    // 前のセッションを閉じる。Env / RunOptions / MemoryInfo は使い回す。
    if (session_) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
    }
    inputs_.clear();
    outputs_.clear();
    backend_label_.clear();
    strict_succeeded_ = false;
    model_path_ = path;
    LOGI("model=%s", model_path_.c_str());

    // ① まず strict（CPU フォールバック禁止）で開く。
    // ② 失敗したら通常で開き直す。どちらで開けたかがグラフ配置のラベルになる。
    strict_succeeded_ = create_session(model_path_, config_, /*strict=*/true);
    if (!strict_succeeded_) {
        if (!create_session(model_path_, config_, /*strict=*/false)) {
            return false;  // どちらでも開けなかった
        }
    }
    backend_label_ = strict_succeeded_ ? "QNN HTP - EPContext - all nodes"
                                       : "QNN HTP - EPContext - CPU I/O boundary";
    LOGI("session opened: %s", backend_label_.c_str());

    // 入出力の名前・形を読み直す（モデルが変わったので）
    return collect_io();
}

// 単入力版（入力が 1 本のモデル用）。多入力版の薄い包み
bool Session::run(const std::vector<uint8_t>& input_bytes, double* out_ms) {
    return run(std::vector<std::vector<uint8_t>>{input_bytes}, out_ms);
}

// job の相対パスを runtime.base_dir から解決する（絶対パスはそのまま。中身は上の resolve）
std::string Session::resolve_path(const std::string& p) const {
    const std::string base_dir = config_.value("runtime", json::object())
                                        .value("base_dir", std::string());
    return resolve(base_dir, p);
}

// 出力を取り出さない版（時間だけ測るとき）
bool Session::run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms) {
    return run(inputs, out_ms, nullptr);
}

// 1 回の推論の本体（引数の説明は session.h）。
// 入力は写さずに包み、出力は ORT に確保させてから写す。out_ms は OrtRun だけの時間
bool Session::run(const std::vector<std::vector<uint8_t>>& inputs, double* out_ms,
                  std::vector<std::vector<uint8_t>>* outputs) {
    if (!session_ || inputs_.empty() || outputs_.empty()) {
        set_error("run", "session is not ready");
        return false;
    }
    // 数が合わないまま黙って先頭だけ渡すと、原因の分かりにくい失敗になる。
    if (inputs.size() != inputs_.size()) {
        set_error("run", "input count mismatch: model expects " +
                             std::to_string(inputs_.size()) + ", got " +
                             std::to_string(inputs.size()));
        return false;
    }

    // ① 入力・出力の名前を ORT に渡す形（const char* の並び）にする
    std::vector<const char*> in_names;
    std::vector<const char*> out_names;
    in_names.reserve(inputs_.size());
    out_names.reserve(outputs_.size());
    for (const auto& i : inputs_)  in_names.push_back(i.name.c_str());
    for (const auto& o : outputs_) out_names.push_back(o.name.c_str());

    // ② 入力のバイト列を、写さずに OrtValue で包む（中身は呼ぶ側の vector のまま）。形はモデルの形（inputs_ の shape）。
    //    バイト数が形と合わなければ ORT が失敗を返す。1 本でも失敗したら、包んだ分を解放して終わる
    std::vector<OrtValue*> in_values(inputs_.size(), nullptr);
    bool built = true;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        const TensorInfo& t = inputs_[i];
        OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
            mem_info_,
            const_cast<uint8_t*>(inputs[i].data()), inputs[i].size(),
            t.shape.data(), t.shape.size(), t.type, &in_values[i]);
        if (!check(st, "CreateTensorWithDataAsOrtValue(" + t.name + ")")) {
            built = false;
            break;
        }
    }
    if (!built) {
        for (OrtValue* v : in_values) if (v) api_->ReleaseValue(v);
        return false;
    }

    // ③ 出力は ORT に確保させる（null の並びを渡す）。Run の前後で時刻を取る
    //    （out_ms ＝ 推論そのもの（NPU の計算）＋ ORT の中の受け渡し。入力を包む・出力を写す時間は入らない）
    std::vector<OrtValue*> out_values(outputs_.size(), nullptr);

    const double t0 = now_ms();
    OrtStatus* st = api_->Run(session_, run_options_,
                              in_names.data(),
                              (const OrtValue* const*)in_values.data(), in_values.size(),
                              out_names.data(), out_names.size(), out_values.data());
    const double t1 = now_ms();

    const bool ok = check(st, "Run");

    // ④ 出力の中身を vector へ写す（要素数 × 型のバイト数）
    if (ok && outputs != nullptr) {
        outputs->clear();
        outputs->reserve(out_values.size());
        for (size_t i = 0; i < out_values.size(); ++i) {
            std::vector<uint8_t> buf;
            OrtTensorTypeAndShapeInfo* info = nullptr;
            if (out_values[i] != nullptr &&
                check(api_->GetTensorTypeAndShape(out_values[i], &info), "GetTensorTypeAndShape")) {
                size_t count = 0;
                ONNXTensorElementDataType t = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                discard(api_, api_->GetTensorShapeElementCount(info, &count),
                        "GetTensorShapeElementCount");
                discard(api_, api_->GetTensorElementType(info, &t), "GetTensorElementType(out)");
                TensorInfo ti;
                ti.type = t;
                const size_t esz = ti.element_size();
                void* data = nullptr;
                if (esz > 0 && count > 0 &&
                    check(api_->GetTensorMutableData(out_values[i], &data), "GetTensorMutableData")
                    && data != nullptr) {
                    buf.assign(static_cast<uint8_t*>(data),
                               static_cast<uint8_t*>(data) + count * esz);
                }
                api_->ReleaseTensorTypeAndShapeInfo(info);
            }
            outputs->push_back(std::move(buf));
        }
    }

    // ⑤ 包み（OrtValue）を解放する。出力は ORT が確保したものなのでここで消える。入力の中身は呼ぶ側の vector なので残る
    for (OrtValue* v : out_values) if (v) api_->ReleaseValue(v);
    for (OrtValue* v : in_values)  if (v) api_->ReleaseValue(v);

    if (ok && out_ms) *out_ms = t1 - t0;
    return ok;
}

// セッションの説明を JSON にする（job の結果の "describe" に毎回入る）。
//   ok・backend（配置のラベル）・strict_no_cpu_fallback（CPU へ回すのを禁じて開けたか）・model・last_error（あれば）・
//   inputs / outputs（name・type・shape・static）。static=false（動的な形）の入出力は QNN では扱えない
json Session::describe() const {
    json j;
    j["ok"] = (session_ != nullptr);
    j["backend"] = backend_label_;
    j["strict_no_cpu_fallback"] = strict_succeeded_;
    j["model"] = model_path_;
    if (!last_error_.empty()) j["last_error"] = last_error_;

    auto dump_io = [](const std::vector<TensorInfo>& v) {
        json arr = json::array();
        for (const auto& t : v) {
            json e;
            e["name"] = t.name;
            e["type"] = type_name(t.type);
            e["shape"] = t.shape;
            e["static"] = (t.element_count() != 0);
            arr.push_back(e);
        }
        return arr;
    };
    j["inputs"] = dump_io(inputs_);
    j["outputs"] = dump_io(outputs_);
    return j;
}

// 版の記録（job の結果の "versions" に入る）。ORT の版は .so から読む。
// dlopen は同じ .so なら開いているものを返す（数を 1 つ増やすだけ。ここでは閉じない）
json Session::versions() {
    json j;
    void* lib = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        j["onnxruntime"] = "unavailable";
        // ★2026-10-18 気づいた点: dlerror() は呼ぶたびに中身を消す。1 回目で「理由あり」と分かっても、2 回目の dlerror() は
        //   null を返し、null から std::string を作ることになる（未定義の動き。落ちる恐れ）。コードは直していない
        j["error"] = dlerror() ? dlerror() : "(null)";
        return j;
    }
    using GetApiBaseFn = const OrtApiBase* (*)(void);
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(dlsym(lib, "OrtGetApiBase"));
    if (get_api_base) {
        const OrtApiBase* base = get_api_base();
        j["onnxruntime"] = base->GetVersionString() ? base->GetVersionString() : "unknown";
        j["ort_api_version"] = ORT_API_VERSION;
    } else {
        j["onnxruntime"] = "unknown";
    }
    // QNN の版はランタイム API から直接は取れないため、
    // 同梱した QAIRT のセット版をホスト側 manifest と突き合わせる（FR-12）。
    j["qnn_runtime"] = "see manifest (bundled QAIRT set)";
    return j;
}

}  // namespace edgeslm
