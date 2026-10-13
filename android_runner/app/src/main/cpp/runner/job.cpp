#include "runner/job.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

namespace {

// 入力バッファを用意する（v0.2 で多入力対応）。
//
// 型ごとに作り方を変える。**整数入力を乱数バイトで埋めてはいけない**:
// int64 を乱数バイトで埋めると天文学的な値になり、Gather の索引なら範囲外参照、
// position_ids なら意味のない位置になる。実際の入力分布に寄せる。
bool build_inputs(const Session& s, const json& spec,
                  std::vector<std::vector<uint8_t>>* out, std::string* err) {
    const json d = s.describe();
    if (!d.contains("inputs") || d["inputs"].empty()) {
        *err = "model has no inputs";
        return false;
    }

    const std::string kind = spec.value("kind", std::string("random"));
    if (kind != "zeros" && kind != "random") {
        *err = "unsupported input.kind: " + kind + " (supports zeros/random)";
        return false;
    }
    // 名前ごとの上限（Gather の索引など）。外すと範囲外参照になる。
    const json int_max = spec.value("int_max", json::object());

    out->clear();
    std::mt19937 rng(spec.value("seed", 0u));

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

        size_t elem = 4;
        if (type == "uint8" || type == "int8") elem = 1;
        else if (type == "uint16" || type == "int16" || type == "float16") elem = 2;
        else if (type == "int64") elem = 8;

        std::vector<uint8_t> buf(count * elem, 0);
        if (kind == "random") {
            if (type == "float32") {
                auto* p = reinterpret_cast<float*>(buf.data());
                // 活性化は 0..1 に収まらない。負側も踏ませる
                // （ホスト側のキャリブレーションと同じ値域にしないと分布がずれる）。
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                for (size_t i = 0; i < count; ++i) p[i] = dist(rng);
            } else if (type == "int64") {
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
                std::uniform_int_distribution<int> dist(0, 255);
                for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>(dist(rng));
            }
        }
        out->push_back(std::move(buf));
    }
    return true;
}

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

bool write_all(const std::string& path, const std::vector<uint8_t>& data, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { *err = "cannot write: " + path; return false; }
    if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()),
                               static_cast<std::streamsize>(data.size()));
    if (!f) { *err = "write failed: " + path; return false; }
    return true;
}

json stats_of(std::vector<double> v) {
    json j;
    if (v.empty()) return j;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
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

bool is_supported_task(const std::string& task) {
    return task == "probe" || task == "bench" || task == "op_check" || task == "infer";
}

json run_job(Session& session, const json& job, const bool* cancel_flag) {
    json out;
    const std::string task = job.value("task", std::string("probe"));
    out["job_id"] = job.value("job_id", std::string(""));
    out["task"] = task;
    out["describe"] = session.describe();
    out["versions"] = Session::versions();
    // 電力プロファイルは制御できない。全レコードに固定で刻み、
    // 他者の burst 測定値や公称値と混ぜられないようにする（基本設計① 2.2節 / 6章）。
    out["power_profile"] = "default(unsigned_pd)";

    if (!is_supported_task(task)) {
        out["ok"] = false;
        out["error"] = "task '" + task + "' is not implemented in v0.3 (probe/bench/op_check/infer)";
        return out;
    }

    if (task == "probe") {
        out["ok"] = session.ok();
        return out;
    }

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

            std::vector<std::vector<uint8_t>> outs;
            double ms = 0.0;
            if (!session.run(inputs, &ms, &outs)) {
                rec["error"] = session.last_error();
                records.push_back(rec);
                continue;
            }
            rec["ort_run_ms"] = ms;

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
        json results = json::array();
        const json models = job.value("models", json::array());
        const json proto = job.value("protocol", json::object());
        const int warmup = proto.value("warmup", 2);
        const int iters  = proto.value("iterations", 5);
        int n_strict = 0, n_boundary = 0, n_failed = 0;

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

    // ---- bench ----
    const json proto = job.value("protocol", json::object());
    const int warmup = proto.value("warmup", 3);
    const int iters  = proto.value("iterations", 10);

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
