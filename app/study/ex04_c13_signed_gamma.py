# -*- coding: utf-8 -*-
"""演習 4: C-13 ― ONNX では正しいのに実機だけ壊れる（RMSNorm の gamma の符号付き 16bit）（StudyMemo 9 章）。

    python -B -m app.study.ex04_c13_signed_gamma      # numpy だけ。1 秒

## 何が起きたか（9.4 節）

ORT は RMSNorm の gamma（0〜1.99）を **uint16**（zero_point 0、scale ＝ 最大 / 65535）で量子化する。
ところが変換器（qairt-converter）は RMSNorm を融合 op `RmsNorm` にし、gamma を **`sFxp_16`（符号付き）**として渡していた。
符号付き 16bit は 32767 までしか表せないので、**量子化値が 32767 を超える要素は黙って頭打ち**になる（警告もエラーも出ない）。

## 何を確かめるか

1. 頭打ちの再現: gamma のうち大きい要素だけが 32767 × scale ≒ 0.996 に張り付く
2. 直し方（`app/pipeline/fix_norm_gamma.py` の `run`）: 符号付き対称（scale ＝ |gamma|最大 / 32767）で入れ直す
3. 見つけ方（`app/pipeline/dlc_audit.py` の `audit`）: DLC のエンコーディングから「必要な量子化値」を計算し、型の範囲と比べる
4. 実際の検査の記録（`results/P9_20260914_qwen_pm64pc_c13_audit.json`）

`SIGNED_MAX`・`LIMITS` は本物のソースの値を読む（`app/study/_src.py`）。
"""
from __future__ import annotations

import json

import numpy as np

from app.study._src import ROOT, load


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def main():
    SIGNED_MAX = load("app/pipeline/fix_norm_gamma.py", ["SIGNED_MAX"])["SIGNED_MAX"]
    LIMITS = load("app/pipeline/dlc_audit.py", ["LIMITS"])["LIMITS"]
    rng = np.random.default_rng(0)
    # 本物に似せた gamma: ほとんどは 1 未満、33 個だけ 1 を超え、最大は 1.9921875（dlc_audit.py の説明にある DLC の max）
    gamma = np.clip(rng.normal(0.6, 0.12, 896), 0.05, 0.95)
    idx = rng.choice(896, 33, replace=False)
    gamma[idx] = rng.uniform(1.0, 1.9, 33)
    gamma[idx[0]] = 1.9921875

    print("== 1. ORT が作る uint16 の量子化（zero_point 0）")
    sc_u = gamma.max() / 65535
    q = np.rint(gamma / sc_u)                       # 0〜65535
    print(f"  scale = {sc_u:.4e}、量子化値の最大 {q.max():.0f}")
    print("\n== 2. 変換器が符号付き（sFxp_16）として読むと")
    q_signed = np.minimum(q, SIGNED_MAX)            # 32767 を超える分は頭打ち
    g_dev = q_signed * sc_u
    n_over = int((q > SIGNED_MAX).sum())
    print(f"  32767 を超える要素: {n_over} / {len(gamma)}（本物の記録は 896 中 33）")
    print(f"  頭打ちの値 = 32767 × scale = {SIGNED_MAX * sc_u:.4f}（これを超える gamma が全部この値に張り付く。本物の実測も 0.9961）")
    print(f"  gamma の相対L2 {rel(gamma, g_dev):.2%}（本物は RMSNorm 単体の出力で 13.54%。測っているものは違うが、桁は同じ）")

    print("\n== 3. 直し方（fix_norm_gamma.run と同じ式: 符号付き対称で入れ直す）")
    new_sc = float(np.abs(gamma).max()) / SIGNED_MAX
    new_q = np.clip(np.rint(gamma / new_sc), -32768, SIGNED_MAX).astype(np.int16)
    print(f"  新しい scale = {new_sc:.4e}、量子化値の最大 {int(new_q.max())}（≦ {SIGNED_MAX}）")
    print(f"  符号付きとして読んでも同じ値。入れ直しの誤差 {rel(gamma, new_q * new_sc):.4%}（分解能が半分になった分だけ）")

    print("\n== 4. 見つけ方（dlc_audit.audit の式）")
    e = {"min": 0.0, "max": 1.9921875, "scale": 3.0399e-05, "offset": 0.0}   # dlc_audit.py の説明にある実際の DLC の行
    lo, hi = LIMITS["sFxp_16"]
    q_max_needed = e["max"] / e["scale"] - e["offset"]     # ★QNN の約束は 実数 = (量子化値 + offset) × scale
    print(f"  DLC: data type sFxp_16 / encoding min {e['min']} max {e['max']} scale {e['scale']} offset {e['offset']:.0f}")
    print(f"  必要な量子化値の最大 = max/scale - offset = {q_max_needed:.0f}  ↔ sFxp_16 の範囲 [{lo}, {hi}]")
    print(f"  → {'★不整合（端末へ送る前に止める）' if q_max_needed > hi + 1.5 else '問題なし'}")

    print("\n== 5. 実際の検査の記録（results/P9_20260914_qwen_pm64pc_c13_audit.json）")
    d = json.loads((ROOT / "results/P9_20260914_qwen_pm64pc_c13_audit.json").read_text(encoding="utf-8"))
    print(f"  直す前: gamma {d['before']['count']} 本のうち {d['before']['n_unsafe']} 本がはみ出し、"
          f"はみ出した要素は合計 {d['before']['total_over']:,} 個")
    print(f"  直した後: はみ出し {d['after']['n_unsafe']} 本。入れ直しの誤差は最大 {d['worst_requant_rel_l2']:.1e}")
    print("  → ONNX の検査（fix_norm_gamma --audit）と DLC の検査（dlc_audit）の両方が要る。融合されると ONNX のノードは消える（9.6 節 ①）")


if __name__ == "__main__":
    main()
