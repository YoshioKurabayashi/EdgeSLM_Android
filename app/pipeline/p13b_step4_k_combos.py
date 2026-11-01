# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# P13b「KV の組み合わせ swap」（results/P13b_20260917_kv_combo_swap.md）の 1 段。中身は kv_combo_swap.py の
# run_variants() に「どの層の K・V を baseline の値に差し替えるか」の表（VARIANTS）を渡すだけ。
# 枠組み（P13a・P12z と同じ）: 最後の 1 歩は oracle モデル（linear20_oracle.py の ORACLE_FULL）で動かし、
# 判定位置（79）の直前の KV のうち指定した層だけを baseline モデルの KV に替えて、
# margin（FP32 の正解トークンへの logit の差）がどれだけ変わるか（Δmargin。oracle 対照は −1.0210）を測る。
# どの段か: StudyMemo 11.5 節「直そうとして起きたこと」（KV の差し替え）・11.6 節（層11-17 の V が悪いと分かった実験）。
# 入力: work/p13a_layer_block_kv_swap/caches.npz・meta.json（P13a で保存したキャッシュ）。
# 出力: results/P13b_20260917_kv_combo_swap/step4_k_combos.json
# 動かし方: & $py -B app/pipeline/p13b_step4_k_combos.py
# この段の結果: ④ K の層0-6 は単独では悪化（−1.544）だが、層14-20 と組み合わせると +1.064 に符号が反転する（層7-13・21-27 との組では反転しない）。
# ---------------------------------------------------------------------------
from pathlib import Path
import sys
# このファイルと同じフォルダ（app/pipeline）を import の探し先に足す（kv_combo_swap を読むため）。
sys.path.insert(0, str(Path(__file__).parent))
from kv_combo_swap import run_variants  # noqa: E402

# 名前 → {"k": K を差し替える層の番号, "v": V を差し替える層の番号}（層は 0 始まり、range の終わりは含まない）。
VARIANTS = {
    "K_L0-6+L7-13": {"k": list(range(0, 7)) + list(range(7, 14)), "v": []},
    "K_L0-6+L14-20": {"k": list(range(0, 7)) + list(range(14, 21)), "v": []},
    "K_L0-6+L21-27": {"k": list(range(0, 7)) + list(range(21, 28)), "v": []},
}

if __name__ == "__main__":
    run_variants(VARIANTS, "step4_k_combos")
