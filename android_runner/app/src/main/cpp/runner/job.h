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
//   soak      : 指定秒数の連続実行（v0.3 で実装）
//   generate  : prefill + decode の生成ループ（v0.4 で実装）
//
// v0.3 では probe / bench / op_check / infer を実装し、未実装の task は「未実装」と明示して返す。
// 黙って別の動作をしないこと。

#pragma once

#include <string>

#include "runner/session.h"
#include "third_party/json.hpp"

namespace edgeslm {

// 実装済みの task か
bool is_supported_task(const std::string& task);

// ジョブを実行して結果 JSON を返す。session は open 済みであること。
nlohmann::json run_job(Session& session, const nlohmann::json& job, const bool* cancel_flag);

}  // namespace edgeslm
