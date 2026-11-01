# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# P13b「KV の組み合わせ swap」（results/P13b_20260917_kv_combo_swap.md）の 1 段。中身は kv_combo_swap.py の
# run_variants() に「どの層の K・V を baseline の値に差し替えるか」の表（VARIANTS）を渡すだけ。
# 枠組み（P13a・P12z と同じ）: 最後の 1 歩は oracle モデル（linear20_oracle.py の ORACLE_FULL）で動かし、
# 判定位置（79）の直前の KV のうち指定した層だけを baseline モデルの KV に替えて、
# margin（FP32 の正解トークンへの logit の差）がどれだけ変わるか（Δmargin。oracle 対照は −1.0210）を測る。
# どの段か: StudyMemo 11.5 節「直そうとして起きたこと」（KV の差し替え）・11.6 節（層11-17 の V が悪いと分かった実験）。
# 入力: work/p13a_layer_block_kv_swap/caches.npz・meta.json（P13a で保存したキャッシュ）。
# 出力: results/P13b_20260917_kv_combo_swap/step2_v_split4.json
# 動かし方: & $py -B app/pipeline/p13b_step2_v_split4.py
# この段の結果: ② 層7-20 を 4 つに分けると +0.983 / +1.707 / +1.634 / +0.332。合計（+4.655）は一括（+2.513）の約 1.85 倍で、組み合わせると頭打ちする。
# ---------------------------------------------------------------------------
from pathlib import Path
import sys
# このファイルと同じフォルダ（app/pipeline）を import の探し先に足す（kv_combo_swap を読むため）。
sys.path.insert(0, str(Path(__file__).parent))
from kv_combo_swap import run_variants  # noqa: E402

# 名前 → {"k": K を差し替える層の番号, "v": V を差し替える層の番号}（層は 0 始まり、range の終わりは含まない）。
VARIANTS = {
    "V_L7-10": {"k": [], "v": list(range(7, 11))},
    "V_L11-13": {"k": [], "v": list(range(11, 14))},
    "V_L14-17": {"k": [], "v": list(range(14, 18))},
    "V_L18-20": {"k": [], "v": list(range(18, 21))},
}

if __name__ == "__main__":
    run_variants(VARIANTS, "step2_v_split4")
