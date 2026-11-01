// EdgeSLM Runner - ジョブ実行
//
// task の値で振る舞いを切り替える（基本設計③ 5.2節）。
// 新しい検証を追加するたびに APK を作り直さないための仕掛けである。
//   probe     : セッションを開いて配置情報だけ返す
//   bench     : ウォームアップ + 反復 + 統計（基本設計① 5章のプロトコル）
//   op_check  : 単体 op モデルを次々に開いて可否を判定する（v0.2 / P2b）
//   infer     : 実データを読んで推論し、**出力をファイルに書く**（v0.3 / P3）
//               実機での精度を測るため。ホスト上のシミュレーションでは
//               HTP の演算順序・丸めの違いが分からない
//   soak      : 指定秒数の連続実行（計画のみ。★2026-10-18 訂正: 旧「v0.3 で実装」。この android_runner の
//               is_supported_task に soak は無く、渡すと「未実装」のエラーを返す）
//   generate  : prefill + decode の生成ループ（計画。★2026-10-18 訂正: 旧「v0.4 で実装」。この android_runner には
//               入っていない（is_supported_task に無い）。生成ループは写し先の EdgeSLM_AndroidDemo/demo_app で
//               generate（窓ずらし）・generate_kv（KV キャッシュ）として実装された）
//
// v0.3 では probe / bench / op_check / infer を実装し、未実装の task は「未実装」と明示して返す。
// （2026-10-18 確認: いまもこの 4 つだけ。job.cpp の is_supported_task）
// 黙って別の動作をしないこと。
//
// （2026-10-18 追記）どこで使うか: edgeslm_jni.cpp の nativeRunJob が、開いた Session と job の JSON を渡して run_job を呼ぶ。
//   job は PC から adb push した jobs/*.json（RunnerService が読む。このアプリに画面は無い）。
// 入力と出力: job（task・job_id と task ごとの項目）→ 結果の JSON。RunnerService は out/<種類>_<時刻>.json と
//   out/last_<種類>.json に書き、PC が adb pull で持ち帰る。
// 動かし方: bash ../scripts/run_runner.sh run jobs/<job>.json（android_runner/README.md。例 jobs/probe_zipdepth.json）

#pragma once

#include <string>

#include "runner/session.h"
#include "third_party/json.hpp"

namespace edgeslm {

// 実装済みの task か（task の名前の一覧と照らすだけ。一覧はこのファイルの冒頭と同じ）。
// run_job はこれが false の task を実行せず、ok=false・「… is not implemented …」の error を返す
bool is_supported_task(const std::string& task);

// ジョブを実行して結果 JSON を返す。session は open 済みであること。
//   job        : job.json の中身。task（既定 "probe"）・job_id と、task ごとの項目（例: bench の protocol、infer の batches）
//   cancel_flag: nativeCancel が立てるフラグ（null 可）。反復のループ（op_check・infer・bench）が毎回見て、立っていれば途中で止める
//   戻り値     : job_id・task・describe（Session::describe）・versions・power_profile（いつも "default(unsigned_pd)"）と、
//                task ごとの結果（ok・records など）
// 時間がかかる task（bench など）でも、終わるまで戻らない（同期）
nlohmann::json run_job(Session& session, const nlohmann::json& job, const bool* cancel_flag);

}  // namespace edgeslm
