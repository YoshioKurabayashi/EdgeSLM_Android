# -*- coding: utf-8 -*-
"""P12u: D0' — 位置0だけFP32事前計算 + KV入力range修正(16bitのみ) + 位置1以降はbaseline(narrowなし)。

P12k: D0(FP32位置0 + baseline)はbaselineより悪化した(1位保持85.7→80.4%, 点B+3.35→-0.22,
自由生成turn1「エアコンの音楽をかけています」)。しかしP12kと同時に、past_k/past_v入力の
KV量子化範囲が狭すぎて対話のほぼ全位置でclipする別問題が見つかっていた
（P12e〜P12kの全変種はこのclipを抱えたまま評価されていた）。
P12l: KV系252グループのうち16bitの224グループだけ広げると(baseline_kvwide16)、
1位保持87.5%・margin保持51.8%まで改善する(8bitグループは広げるとC-8制約で悪化するため除外)。

D0'はこの2つを組み合わせる: FP32位置0のKV + KV入力range修正(16bitのみ広げる) +
位置1以降はbaseline QDQ(narrowは一切入れない、mul_41/linear_20もbaselineのまま)。
旧D0の悪化が「FP32 KV→狭いKV入力でclip」由来だったなら、D0'で解消するはず。

新規ビルドは不要: P12l保存済みのwiden_plan.json(16bitサブセット)を、
D:/AI/artifacts/.../baseline.onnx にメモリ上で適用してORTセッションを作る
（kv_range_recal16.pyのapply()をそのまま再利用）。
（★2026-10-18 訂正: 旧「kv_range_recal16.pyのapply()」。apply は kv_range_recal.py に定義があり、
  kv_range_recal16.py もこのファイルも、そこから import している）

判定は自由生成を最優先(teacher forcingの集計値は診断用)。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）
- 結果（results/P12u_20260916_d0prime_fp32pos0_kvwide.md）: 仮説は否定された。KV 入力の範囲を直しても、
  自由生成の turn1（「音楽をかけて」）は旧 D0 とほぼ同じ「エアコンの音楽をかけています。」という失敗文になった
- 入力: P12l の widen_plan.json、P12e の baseline.onnx（ART）、work/ops/qwen3_dec28c256/ の prepped.onnx（FP32）
- 出力: results/P12u_20260916_d0prime_fp32pos0_kvwide/report.json と画面の 1 行の要約・自由生成の文
- 動かし方: python -m app.pipeline.d0_prime_fp32pos0_kvwide（引数なし）
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, OPS  # noqa: E402
from block_revert_ab import eval_points, score, ART  # noqa: E402
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402
from kv_range_recal import OUTDIR, apply  # noqa: E402

OUT = ROOT / "results/P12u_20260916_d0prime_fp32pos0_kvwide"


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def summarize(sc):
    """score の結果を 1 行の要約（1位保持・margin 保持・最小 margin・KL・点B・点A・BANANA・バナナ）にする。"""
    a = sc["ALL"]
    return (f"1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
            f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}")


def main():
    """① 評価点 → ② 16bit の計画 → ③ FP32 の基準 → ④ D0'（位置0 は FP32、以降は kvwide16 の baseline）を評価して保存。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "start", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    plan_all = json.loads((OUTDIR / "widen_plan.json").read_text(encoding="utf-8"))["plan"]
    plan16 = [p for p in plan_all if "new_scale" in p and p["dtype"] != "uint8"]
    skipped_u8 = [p for p in plan_all if "new_scale" in p and p["dtype"] == "uint8"]
    emit({"phase": "plan", "kv_16bit_groups_widened": len(plan16), "kv_8bit_groups_kept_baseline": len(skipped_u8)})

    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    emit({"phase": "fp32_reference_done"})

    base16 = apply(ART / "baseline.onnx", plan16)
    pipe = Pipeline(fp32, base16, base16, "pos0")
    sc = score(run_points(pipe, pts), ref)
    gen = free_generation(pipe, tok)
    print(f"D0'(fp32pos0+kvwide16+baseline)  {summarize(sc)}", flush=True)
    for t in gen:
        print(f"      turn{t['turn']}: {t['text']!r}", flush=True)
    (OUT / "report.json").write_text(
        json.dumps({"scores": {"D0_prime": sc}, "free_generation": {"D0_prime": gen}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "done", "wrote": str(OUT / "report.json")})


if __name__ == "__main__":
    main()
