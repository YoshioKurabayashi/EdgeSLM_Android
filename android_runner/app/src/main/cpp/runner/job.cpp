// EdgeSLM Runner - ジョブ実行の本体（job.h の宣言の実装）
//
// （2026-10-18 追記）学習用の説明:
// 何のための部品か: PC から送った job（JSON）の task の値を見て、端末（SOG05・HTP v68）の NPU で
//   「開くだけ（probe）」「速さを測る（bench）」「単体 op を次々に試す（op_check）」「実データで推論して出力を書く（infer）」
//   のどれかを行い、結果を JSON で返す。task の一覧と理由は job.h の冒頭と android_runner/README.md。
// どの検証の段で使うか: 基本設計③ 5.2節（task 分岐）・基本設計① 5章（計測の手順）。
//   op_check は P2b（results/P2b_20260912_op_matrix_device.md）、infer は P3 以降の実機の精度
//   （PC 側は app/pipeline/device_accuracy.py。results/P3_20260913_device_accuracy.md）。
// 入力と出力: 開いた Session と job の JSON → 結果の JSON（RunnerService が out/ に書く）。
//   infer はさらに、出力のテンソルを生のバイト列で out/<job_id>_<id>_<出力名>.bin に書く。
// 動かし方: bash ../scripts/run_runner.sh run jobs/<job>.json（例 jobs/p2b_op_check.json・jobs/p3_qwen24.json）
//
// 読む順番（StudyMemo 13.4）: run_job の task 分岐 → infer の段 → op_check の段 → bench の段。
#include "runner/job.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

// logcat に "EdgeSLMRunner" の札で 1 行出す（adb logcat -s EdgeSLMRunner で見える）
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

// ここから下の無名名前空間は、このファイルの中だけで使う下請けの関数。
namespace {

// 入力バッファを用意する（v0.2 で多入力対応）。
//
// 型ごとに作り方を変える。**整数入力を乱数バイトで埋めてはいけない**:
// int64 を乱数バイトで埋めると天文学的な値になり、Gather の索引なら範囲外参照、
// position_ids なら意味のない位置になる。実際の入力分布に寄せる。
//
// （2026-10-18 追記）
//   s      : 開いた Session。describe() の inputs（name・type・shape・static）から、作るべき入力の数と大きさを決める
//   spec   : job の "input"。kind（"zeros" か "random"。既定 "random"）・seed（乱数の種。既定 0）・
//            int_max（{入力名: 上限}。整数入力の値をこの上限までに収める）
//   out    : 入力ごとのバイト列（モデルの入力順）。大きさは 要素数 × 型の大きさ（float32=4・int64=8・uint16=2 など）
//   err    : 失敗の理由。戻り値 false のときに入る
// 使う所: op_check（多入力のまま）と bench（build_input 経由で単入力に限る）。infer は使わない（実データを読む）。
bool build_inputs(const Session& s, const json& spec,
                  std::vector<std::vector<uint8_t>>* out, std::string* err) {
    const json d = s.describe();
    if (!d.contains("inputs") || d["inputs"].empty()) {
        *err = "model has no inputs";
        return false;
    }

    // ① 入力の作り方（kind）を確かめる。知らない値は黙って既定に倒さず、エラーで返す
    const std::string kind = spec.value("kind", std::string("random"));
    if (kind != "zeros" && kind != "random") {
        *err = "unsupported input.kind: " + kind + " (supports zeros/random)";
        return false;
    }
    // 名前ごとの上限（Gather の索引など）。外すと範囲外参照になる。
    const json int_max = spec.value("int_max", json::object());

    out->clear();
    std::mt19937 rng(spec.value("seed", 0u));

    // ② モデルの入力ごとに、形から要素数を出し、型から 1 要素のバイト数を決めてバッファを作る
    for (const auto& in : d["inputs"]) {
        if (!in.value("static", false)) {
            // QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）。
            *err = "input '" + in.value("name", std::string("?")) +
                   "' shape is not static; QNN EP requires fixed shapes";
            return false;
        }
        size_t count = 1;
        for (const auto& dim : in["shape"]) count *= static_cast<size_t>(dim.get<int64_t>());
        const std::string type = in.value("type", std::string("float32"));
        const std::string name = in.value("name", std::string());

        // 型の大きさ。ここに無い型（float32・int32 など）は 4 バイトとして扱う
        // （推定: bool・double などの入力は想定していない。来たら大きさが合わない）
        size_t elem = 4;
        if (type == "uint8" || type == "int8") elem = 1;
        else if (type == "uint16" || type == "int16" || type == "float16") elem = 2;
        else if (type == "int64") elem = 8;

        std::vector<uint8_t> buf(count * elem, 0);
        // ③ 中身を埋める。"zeros" ならバッファは 0 のまま（上の初期値）
        if (kind == "random") {
            if (type == "float32") {
                auto* p = reinterpret_cast<float*>(buf.data());
                // 活性化は 0..1 に収まらない。負側も踏ませる
                // （ホスト側のキャリブレーションと同じ値域にしないと分布がずれる）。
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (size_t i = 0; i < count; ++i) p[i] = dist(rng);
            } else if (type == "int64") {
                // 整数は乱数ではなく 0, 1, …, hi, 0, 1, … と順に並べる（hi は int_max の値か、なければ 要素数−1）。
                // こうすると Gather の索引が範囲内に収まり、position_ids なら 0 から並んだ位置らしい値になる
                auto* p = reinterpret_cast<int64_t*>(buf.data());
                const int64_t hi = int_max.contains(name)
                                 ? int_max[name].get<int64_t>()
                                 : static_cast<int64_t>(count) - 1;
                for (size_t i = 0; i < count; ++i) {
                    p[i] = (hi <= 0) ? 0 : static_cast<int64_t>(i % (hi + 1));
                }
            } else if (type == "int32") {
                auto* p = reinterpret_cast<int32_t*>(buf.data());
                const int32_t hi = int_max.contains(name)
                                 ? int_max[name].get<int32_t>()
                                 : static_cast<int32_t>(count) - 1;
                for (size_t i = 0; i < count; ++i) {
                    p[i] = (hi <= 0) ? 0 : static_cast<int32_t>(i % (hi + 1));
                }
            } else {
                // それ以外の型（uint8・uint16・float16 など）は 1 バイトずつ 0〜255 の乱数で埋める。
                // （2026-10-18 注）float16 をこの方法で埋めると、ビットの並びによっては NaN や無限大が混ざる
                std::uniform_int_distribution<int> dist(0, 255);
                for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(dist(rng));
            }
        }
        out->push_back(std::move(buf));
    }
    return true;
}

// （2026-10-18 追記）build_inputs で作った入力が 1 本だけのときに、その 1 本を out に移す。
// 2 本以上なら false（bench は単入力のモデルだけを測る。多入力は op_check を使う、とエラーに書く）。
// 単入力版（bench 用）。多入力モデルはここで明示的に弾く。
bool build_input(const Session& s, const json& spec, std::vector<uint8_t>* out,
                 std::string* err) {
    std::vector<std::vector<uint8_t>> all;
    if (!build_inputs(s, spec, &all, err)) return false;
    if (all.size() != 1) {
        *err = "bench supports single-input models only; this model has " +
               std::to_string(all.size()) + " inputs (use task=op_check)";
        return false;
    }
    *out = std::move(all[0]);
    return true;
}

// 生のバイト列としてファイルを読む。**サイズが期待と違えば失敗させる**——
// 黙って足りないまま推論すると、結果が静かに壊れる。
// （2026-10-18 追記）このファイル自体はファイルの大きさを読むだけで、期待との照合は呼び出し元（infer の段）が行う。
//   path: 読むファイルの絶対パス（resolve_path 済み）。out: 中身全部。err: 開けない・読めないときの理由
bool read_all(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { *err = "cannot open: " + path; return false; }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out->resize(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(out->data()), n)) {
        *err = "read failed: " + path;
        return false;
    }
    return true;
}

// （2026-10-18 追記）data をそのまま path に書く（上書き）。開けない・書けないときは false と理由を err に入れる。
// infer の出力（.bin）を書くのに使う。PC 側がこのバイト列を numpy で読み戻して fp32 と比べる
bool write_all(const std::string& path, const std::vector<uint8_t>& data, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { *err = "cannot write: " + path; return false; }
    if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()),
                               static_cast<std::streamsize>(data.size()));
    if (!f) { *err = "write failed: " + path; return false; }
    return true;
}

// （2026-10-18 追記）時間の標本（ミリ秒の並び）から統計量を作る。
//   v      : 標本。値渡しで受けて、この中で並べ替える（呼び出し元の並びは変えない）
//   戻り値 : median・p90・min・max・mean・stddev（標本標準偏差。n−1 で割る）・n。空なら空の JSON
// 中央値と p90 を主に見るのは、初回の外れ値に平均が引きずられるため（基本設計① 5章。下の行のコメントも同じ）
json stats_of(std::vector<double> v) {
    json j;
    if (v.empty()) return j;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    // 百分位点 p（0〜1）を、並べた標本の間を直線で補って求める（例: n=10 の p90 は 8.1 番目の値）
    auto pct = [&](double p) {
        const double idx = p * (static_cast<double>(n) - 1.0);
        const size_t lo = static_cast<size_t>(std::floor(idx));
        const size_t hi = static_cast<size_t>(std::ceil(idx));
        if (lo == hi) return v[lo];
        return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
    };
    double sum = 0.0;
    for (double x : v) sum += x;
    const double mean = sum / static_cast<double>(n);
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);

    // 平均は主指標にしない（基本設計① 5章）。初回の外れ値に引きずられるため。
    j["median"] = pct(0.5);
    j["p90"]    = pct(0.9);
    j["min"]    = v.front();
    j["max"]    = v.back();
    j["mean"]   = mean;
    j["stddev"] = (n > 1) ? std::sqrt(var / static_cast<double>(n - 1)) : 0.0;
    j["n"]      = n;
    return j;
}

}  // namespace

// （2026-10-18 追記）task の名前が、この版で実装済みの 4 つ（probe・bench・op_check・infer）のどれかなら true
bool is_supported_task(const std::string& task) {
    return task == "probe" || task == "bench" || task == "op_check" || task == "infer";
}

// （2026-10-18 追記）ジョブ 1 件を実行する入口。宣言と引数の説明は job.h。
// 段の並び: ① 共通の項目（job_id・task・describe・versions・power_profile）を結果に入れる
//           ② 未実装の task を弾く → ③ probe → ④ infer → ⑤ op_check → ⑥ 残りは bench
json run_job(Session& session, const json& job, const bool* cancel_flag) {
    // ① どの task でも結果に必ず入る項目
    json out;
    const std::string task = job.value("task", std::string("probe"));
    out["job_id"] = job.value("job_id", std::string(""));
    out["task"] = task;
    out["describe"] = session.describe();
    out["versions"] = Session::versions();
    // 電力プロファイルは制御できない。全レコードに固定で刻み、
    // 他者の burst 測定値や公称値と混ぜられないようにする（基本設計① 2.2節 / 6章）。
    out["power_profile"] = "default(unsigned_pd)";

    // ② 未実装の task（soak・generate など）は、ok=false と理由を返して何もしない
    if (!is_supported_task(task)) {
        out["ok"] = false;
        out["error"] = "task '" + task + "' is not implemented in v0.3 (probe/bench/op_check/infer)";
        return out;
    }

    // ③ probe: 開けたかどうか（session.ok）だけを返す。配置（全ノード NPU か）は ① の describe に入っている
    if (task == "probe") {
        out["ok"] = session.ok();
        return out;
    }

    // ④ infer: job の batches（[{id, inputs:{入力名: 端末上のパス}}]）を 1 件ずつ実行する。
    //    PC 側は app/pipeline/device_accuracy.py がこの job と入力の .bin を作って push する
    if (task == "infer") {
        // 実データを読んで推論し、出力を生バイト列で書き出す。
        // ホスト側が fp32 の参照と突き合わせて**実機での精度**を出す。
        const json batches = job.value("batches", json::array());
        const std::string out_dir = job.value("output", json::object())
                                       .value("dir", std::string("out"));
        const json desc = session.describe();
        json records = json::array();
        int n_ok = 0;

        for (const auto& b : batches) {
            json rec;
            const std::string id = b.value("id", std::string("0"));
            rec["id"] = id;
            if (cancel_flag && *cancel_flag) { rec["error"] = "cancelled"; records.push_back(rec); break; }

            // ④-1 入力の .bin を読む。読む順はモデルの入力順（describe の inputs の並び）で、
            //      ファイルは入力の名前で job から引く（下の行の「名前順」はこの意味。2026-10-18 補足）
            // 入力を名前順（モデルの入力順）に読む
            std::vector<std::vector<uint8_t>> inputs;
            std::string err;
            bool loaded = true;
            const json files = b.value("inputs", json::object());
            for (const auto& in : desc["inputs"]) {
                const std::string name = in.value("name", std::string());
                if (!files.contains(name)) {
                    err = "no input file for '" + name + "'";
                    loaded = false;
                    break;
                }
                std::vector<uint8_t> buf;
                if (!read_all(session.resolve_path(files[name].get<std::string>()), &buf, &err)) {
                    loaded = false;
                    break;
                }
                // （2026-10-18 追記）期待 = 形の要素数 × 型の大きさ（build_inputs と同じ表）。
                //   PC 側で型を取り違える（例 float32 を float16 で書く）と、ここで size mismatch になって気付ける
                // 期待バイト数と一致しなければ止める
                size_t count = 1;
                for (const auto& d : in["shape"]) count *= static_cast<size_t>(d.get<int64_t>());
                const std::string type = in.value("type", std::string("float32"));
                size_t esz = 4;
                if (type == "uint8" || type == "int8") esz = 1;
                else if (type == "uint16" || type == "int16" || type == "float16") esz = 2;
                else if (type == "int64") esz = 8;
                if (buf.size() != count * esz) {
                    err = "size mismatch for '" + name + "': expected " +
                          std::to_string(count * esz) + " bytes, got " + std::to_string(buf.size());
                    loaded = false;
                    break;
                }
                inputs.push_back(std::move(buf));
            }
            if (!loaded) {
                rec["error"] = err;
                records.push_back(rec);
                continue;
            }

            // ④-2 実行する。ms は OrtRun の時間（ミリ秒）。出力はモデルの出力順にバイト列で返る
            std::vector<std::vector<uint8_t>> outs;
            double ms = 0.0;
            if (!session.run(inputs, &ms, &outs)) {
                rec["error"] = session.last_error();
                records.push_back(rec);
                continue;
            }
            rec["ort_run_ms"] = ms;

            // ④-3 出力を .bin に書き、名前・パス・バイト数・形・型を記録する（PC 側はこれを見て読み戻す）
            json written = json::array();
            bool wrote = true;
            for (size_t i = 0; i < outs.size() && i < desc["outputs"].size(); ++i) {
                const std::string oname = desc["outputs"][i].value("name", std::string("out"));
                const std::string rel = out_dir + "/" + job.value("job_id", std::string("infer")) +
                                        "_" + id + "_" + oname + ".bin";
                if (!write_all(session.resolve_path(rel), outs[i], &err)) {
                    rec["error"] = err;
                    wrote = false;
                    break;
                }
                written.push_back({{"name", oname}, {"path", rel},
                                   {"bytes", outs[i].size()},
                                   {"shape", desc["outputs"][i]["shape"]},
                                   {"type", desc["outputs"][i]["type"]}});
            }
            if (!wrote) { records.push_back(rec); continue; }
            rec["outputs"] = written;
            records.push_back(rec);
            ++n_ok;
            LOGI("infer[%s] %.2f ms, %zu outputs", id.c_str(), ms, outs.size());
        }

        // ④-4 全件が成功し、かつ 1 件以上あったときだけ ok。途中で失敗した件は records の error に理由が残る
        out["records"] = records;
        out["ok"] = (n_ok == static_cast<int>(batches.size())) && n_ok > 0;
        out["completed"] = n_ok;
        return out;
    }

    if (task == "op_check") {
        // P2b: 単体 op モデルを次々に開いて「どこまで通るか」を判定する。
        //
        // **APK を作り直さずに対象を足せる**ことが肝で、models は job.json 側にある。
        // 判定は「通る / 通らない」の2値にせず、次の手が決まる粒度まで割る:
        //   pass_strict        … 全ノードが HTP。そのまま使える
        //   pass_cpu_boundary  … 開けたが CPU フォールバックあり。delegate しきれていない
        //   open_failed        … ロードで落ちた（C-1〜C-7 / op 非対応）
        //   run_failed         … 開けたが実行で落ちた
        // ⑤ op_check: models の各項目（パスの文字列、または {model, name, input}）を順に開き直して試す
        json results = json::array();
        const json models = job.value("models", json::array());
        const json proto = job.value("protocol", json::object());
        const int warmup = proto.value("warmup", 2);
        const int iters  = proto.value("iterations", 5);
        int n_strict = 0, n_boundary = 0, n_failed = 0;

        // ⑤-1 1 つ目のモデルから順に、開く → 入力を作る → ウォームアップ → 反復 → 判定
        for (const auto& entry : models) {
            const bool is_str = entry.is_string();
            const std::string path = is_str ? entry.get<std::string>()
                                            : entry.value("model", std::string());
            const std::string label = is_str ? path : entry.value("name", path);

            json r;
            r["name"] = label;
            r["model"] = path;

            if (cancel_flag && *cancel_flag) {
                r["verdict"] = "cancelled";
                results.push_back(r);
                break;
            }

            // 同じ Session のまま、モデルだけを張り替える（OrtEnv は作り直さない。README の「Session::reopen」）
            if (!session.reopen(path)) {
                r["verdict"] = "open_failed";
                r["error"] = session.last_error();
                results.push_back(r);
                ++n_failed;
                LOGI("op_check %s: open_failed", label.c_str());
                continue;
            }
            const json desc = session.describe();
            r["describe"] = desc;
            // CPU へ回すのを禁じて開けたなら strict=true（全ノード NPU）。推測ではなく開けたかどうかで決める
            const bool strict = desc.value("strict_no_cpu_fallback", false);
            r["strict_no_cpu_fallback"] = strict;

            const json in_spec = is_str ? job.value("input", json::object())
                                        : entry.value("input", job.value("input", json::object()));
            std::vector<std::vector<uint8_t>> inputs;
            std::string err;
            if (!build_inputs(session, in_spec, &inputs, &err)) {
                r["verdict"] = "run_failed";
                r["error"] = "input: " + err;
                results.push_back(r);
                ++n_failed;
                continue;
            }

            // ⑤-2 ウォームアップ（既定 2 回）と反復（既定 5 回）。1 回でも失敗したら run_ok=false で打ち切る。
            //      （2026-10-18 注）反復の途中で取り消しが来ると、run_ok は true のまま抜けるので、判定は pass_* になる
            bool run_ok = true;
            std::vector<double> warm, samples;
            for (int i = 0; i < warmup && run_ok; ++i) {
                double ms = 0.0;
                run_ok = session.run(inputs, &ms);
                if (run_ok) warm.push_back(ms);
            }
            for (int i = 0; i < iters && run_ok; ++i) {
                if (cancel_flag && *cancel_flag) break;
                double ms = 0.0;
                run_ok = session.run(inputs, &ms);
                if (run_ok) samples.push_back(ms);
            }
            r["warmup_ms"] = warm;
            r["samples_ms"] = samples;
            if (!samples.empty()) r["ort_run_ms"] = stats_of(samples);

            // ⑤-3 4 つの判定のどれかにする（意味は上の一覧）
            if (!run_ok) {
                r["verdict"] = "run_failed";
                r["error"] = session.last_error();
                ++n_failed;
            } else if (strict) {
                r["verdict"] = "pass_strict";
                ++n_strict;
            } else {
                r["verdict"] = "pass_cpu_boundary";
                ++n_boundary;
            }
            LOGI("op_check %s: %s", label.c_str(),
                 r["verdict"].get<std::string>().c_str());
            results.push_back(r);
        }

        // ⑤-4 判定ごとの件数。全件が失敗なし（pass_strict か pass_cpu_boundary）のときだけ ok。
        //      （2026-10-18 注）終わった後の Session は最後に開いたモデルのまま
        out["results"] = results;
        out["summary"] = {{"pass_strict", n_strict},
                          {"pass_cpu_boundary", n_boundary},
                          {"failed", n_failed},
                          {"total", results.size()}};
        // describe は「最後に開いたモデル」のものになってしまい誤解を招くので外す。
        out.erase("describe");
        out["ok"] = (n_failed == 0) && !results.empty();
        return out;
    }

    // ⑥ bench: 単入力のモデルの速さを測る（基本設計① 5章: ウォームアップ 3 回を捨て、10 回の標本を残す。既定値）
    // ---- bench ----
    const json proto = job.value("protocol", json::object());
    const int warmup = proto.value("warmup", 3);
    const int iters  = proto.value("iterations", 10);

    // ⑥-1 入力を作る（build_input。多入力のモデルはここでエラー）
    std::vector<uint8_t> input;
    std::string err;
    if (!build_input(session, job.value("input", json::object()), &input, &err)) {
        out["ok"] = false;
        out["error"] = err;
        return out;
    }
    out["input_bytes"] = input.size();

    // ウォームアップ分は破棄する。ただし「破棄した」ことと値は記録する。
    // 初回が遅いのは既知（EdgeSLM の知見）で、その事実自体が情報だからである。
    std::vector<double> warm_ms;
    for (int i = 0; i < warmup; ++i) {
        if (cancel_flag && *cancel_flag) { out["cancelled"] = true; break; }
        double ms = 0.0;
        if (!session.run(input, &ms)) {
            out["ok"] = false;
            out["error"] = "warmup run failed: " + session.last_error();
            out["warmup_ms"] = warm_ms;
            return out;
        }
        warm_ms.push_back(ms);
        LOGI("warmup[%d] OrtRun %.2f ms", i, ms);
    }
    out["warmup_ms"] = warm_ms;

    std::vector<double> samples;
    for (int i = 0; i < iters; ++i) {
        if (cancel_flag && *cancel_flag) { out["cancelled"] = true; break; }
        double ms = 0.0;
        if (!session.run(input, &ms)) {
            out["ok"] = false;
            out["error"] = "run failed at iteration " + std::to_string(i) + ": " +
                           session.last_error();
            out["samples_ms"] = samples;
            return out;
        }
        samples.push_back(ms);
        LOGI("bench[%d] OrtRun %.2f ms", i, ms);
    }

    // 生サンプルを必ず残す（基本設計① 5章）。統計量だけでは
    // 「外れ値1個に引きずられていた」ことを後から検証できない。
    out["samples_ms"] = samples;
    out["ort_run_ms"] = stats_of(samples);
    out["ok"] = !samples.empty();
    return out;
}

}  // namespace edgeslm
