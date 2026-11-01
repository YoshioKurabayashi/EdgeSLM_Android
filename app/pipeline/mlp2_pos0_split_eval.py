# -*- coding: utf-8 -*-
"""P12o: 位置0はbaseline(既存範囲)、位置1以降だけP12nのmlp2later(後続範囲・56本)に切り替えて評価する。

P12n: 56本(28層×SiLU(gate)×up出力/down_proj出力)を後続範囲にして全位置(先頭トークン込み)に
適用すると、Hostで完全に崩壊した(1位保持0%, KL16.8, 自由生成が文字列として破綻)。
「後続範囲そのものがダメ」なのか「位置0にまで後続範囲を適用したのがダメ」なのかを切り分ける。

  条件         位置0    位置1以降
  baseline     wide     wide       (P12nで実測済み。ここでは再計算せずreport.jsonの値を使う)
  P12n         narrow   narrow     (同上)
  P12o(本実験) wide     narrow

変数を増やしすぎないため、KV range再較正(P12l)は混ぜない。pos0/pos1+の切り替えだけを見る。
仮にこれで崩壊が消えれば、位置0のmassive activationにwide rangeが必要という仮説が強く支持される。
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】引数は無い（argparse を使わない）。
# どの段か: StudyMemo 11.5 節の表「位置0 だけ元の範囲に戻す → 崩壊は消える（80.4%）。だが『音楽をかけて』だけ壊れる」。
# 結果（results/P12o_20260916_pos0_wide_pos1_narrow.md）: 1位保持 80.4%・KL 0.304 で全面崩壊は消えた。
#   ただし点B（音楽をかけて）の margin が +3.35 → −0.43 になり、自由生成でも turn1 が英語の自問自答を始めて壊れた。
#   → mul と linear のどちらが主犯かを分ける mlp2_mul_linear_split.py（P12p）へ続く。
# 入力:  baseline（SOURCE）、P12n で作った mlp2later_qdq.onnx（mlp2_all_layers_later.MODEL）、
#        P12n の report.json（baseline と mlp2later の数値は再計算せずに再掲する）。
# 出力:  results/P12o_20260916_pos0_wide_pos1_narrow/report.json
# 動かし方: & $py -B app/pipeline/mlp2_pos0_split_eval.py
# ---------------------------------------------------------------------------
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, SOURCE  # noqa: E402
from block_revert_ab import eval_points, score  # noqa: E402
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402
from mlp2_all_layers_later import MODEL, OUT as P12N_OUT  # noqa: E402

import onnxruntime as ort  # noqa: E402,F401 (imported for session() dependency check)

OUT = ROOT / "results/P12o_20260916_pos0_wide_pos1_narrow"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"


def emit(obj):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def fmt(name, sc, gen):
    """score の結果と自由生成の文を、画面に出す数行の文字列にまとめる（点B = 音楽、点A = 最初の分岐、BANANA/バナナ = System Prompt）。"""
    a = sc["ALL"]
    lines = [f"{name:12s} 1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
             f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
             f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
             f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}"]
    for t in gen:
        lines.append(f"      turn{t['turn']}: {t['text']!r}")
    return "\n".join(lines)


def main():
    """P12n の数値を再掲し、新しく「位置0 = baseline、位置1以降 = mlp2later」の 2 モデル構成だけを採点・自由生成する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "evaluate", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    prior = json.loads((P12N_OUT / "report.json").read_text(encoding="utf-8"))
    for name in ("baseline", "mlp2later"):
        print(fmt(name + "(P12n再掲)", prior["scores"][name], prior["free_generation"][name]))

    fp32 = session(OPS_DIR / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    emit({"phase": "evaluate", "step": "fp32_reference_done"})

    # Pipeline(fp32, first, later, mode): first（位置0）は既存範囲の baseline、later（位置1以降）は 56 本を後続範囲にしたモデル。
    # FP32 の事前計算（fp32=None）や KV 範囲の較正し直しは混ぜない（変える変数を 1 つにする）。
    base = session(SOURCE)
    later = session(MODEL)
    pipe = Pipeline(None, base, later, "none")
    sc = score(run_points(pipe, pts), ref)
    gen = free_generation(pipe, tok)
    del base, later, pipe
    print(fmt("split(wide0/narrow1+)", sc, gen))

    report = {
        "baseline_from_P12n": prior["scores"]["baseline"],
        "mlp2later_from_P12n": prior["scores"]["mlp2later"],
        "split_wide0_narrow1plus": sc,
        "free_generation": {
            "baseline_from_P12n": prior["free_generation"]["baseline"],
            "mlp2later_from_P12n": prior["free_generation"]["mlp2later"],
            "split_wide0_narrow1plus": gen,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "evaluate", "wrote": str(OUT / "report.json")})


if __name__ == "__main__":
    main()
