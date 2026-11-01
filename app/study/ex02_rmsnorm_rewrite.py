# -*- coding: utf-8 -*-
"""演習 2: RMSNorm を「トークンごとの max で割ってから」計算する書き換え（C-11）と、マスク定数（C-9）。

    python -B -m app.study.ex02_rmsnorm_rewrite      # numpy だけ。1 秒

## 何を確かめるか（StudyMemo 3.3〜3.5・5.1・5.2・5.5 節）

1. **数式として同じ**: `x / sqrt(mean(x²))` と `(x/m) / sqrt(mean((x/m)²))` は m によらず同じ（スケール不変）
2. **量子化すると違いが出る**: `x²` の範囲を 1 つの scale（per-tensor・uint16）で表すと、巨大チャンネルを持つ
   トークンに範囲を取られ、ふつうのトークンの RMSNorm が壊れる。max で割ってから二乗すると範囲が [0,1] になり壊れない
3. eps の扱い（5.2 節）: 割った後の `mean(xs²)` は元の `1/m²` 倍なので、元の eps 1e-6 のままだと効きすぎる
4. マスク定数（C-9、5.5 節）: `-3.4e38` を量子化の範囲に入れると、本物のスコアが全部同じ値に潰れる

本物の書き換えは `app/pipeline/preprocess.py` の `normalize_rmsnorm_input`（ONNX のノードを足す）。
ここでは同じ計算を numpy で行う。マスクの置き換え値 `MASK_REPLACE` は本物のソースから読む。
"""
from __future__ import annotations

import numpy as np

from app.study._src import load


def q_dq(x, lo, hi, bits=16):
    """非対称の QDQ（活性化）。lo〜hi を 2^bits-1 段に割って丸め、戻す"""
    qmax = 2 ** bits - 1
    s = (hi - lo) / qmax
    zp = np.rint(-lo / s)
    return (np.clip(np.rint(x / s) + zp, 0, qmax) - zp) * s


def rms_orig(x, eps=1e-6, quant=None):
    x2 = x ** 2
    if quant:
        x2 = q_dq(x2, *quant)
    return x / np.sqrt(x2.mean(-1, keepdims=True) + eps)


def rms_maxnorm(x, eps=1e-12, quant=None):
    m = np.abs(x).max(-1, keepdims=True) + 1e-6            # 本物も 1e-6 の下限を足す（全ゼロで 1/0 にしないため）
    xs = x / m
    x2 = xs ** 2
    if quant:
        x2 = q_dq(x2, *quant)
    return xs / np.sqrt(x2.mean(-1, keepdims=True) + eps)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def main():
    rng = np.random.default_rng(0)
    H = 896
    x = rng.normal(0, 0.2, (4, H))                          # 4 トークン
    x[0, 35] = 1677.0                                       # トークン 0 だけ巨大チャンネルを持つ（3.2 節の実測に合わせた値）

    print("== 1. 数式として同じか（量子化なし、float64）")
    a, b = rms_orig(x, eps=0.0), rms_maxnorm(x, eps=0.0)
    for t in range(4):
        print(f"  トークン {t}: 元の式 と max で割った式 の差 {np.abs(a[t] - b[t] * 1).max() / np.abs(a[t]).max():.1e}"
              f"（max|x| = {np.abs(x[t]).max():8.2f}）")

    print("\n== 2. x² を uint16 の per-tensor で量子化したとき（較正の範囲は 4 トークン全体の最小〜最大）")
    ref = rms_orig(x, eps=1e-6)
    x2_range = (0.0, float((x ** 2).max()))
    xs2_range = (0.0, 1.0)                                  # max で割ると必ず [0, 1]
    o = rms_orig(x, eps=1e-6, quant=x2_range)
    n = rms_maxnorm(x, eps=1e-12, quant=xs2_range)
    print(f"  x² の範囲 {x2_range[1]:,.0f} → 1 段 {x2_range[1] / 65535:.1f}      xs² の範囲 1 → 1 段 {1 / 65535:.1e}")
    print("  トークン |  元の式（量子化）の誤差 | max で割った式（量子化）の誤差")
    for t in range(4):
        # 書き換え後の出力は xs/rms(xs) = x/rms(x)（スケール不変）なので、そのまま元の参照と比べられる
        print(f"     {t}     |  {rel(ref[t], o[t]):>18.1%}    | {rel(ref[t], n[t]):>18.4%}")
    print("  → 巨大チャンネルの無いトークン（1〜3）は、元の式では x² が全部 0 に丸まり、出力が eps で割った巨大な値になる")

    print("\n== 3. eps の扱い（5.2 節）")
    print("  割った後の mean(xs²) は元の 1/m² 倍。だから同じ eps を足すと、効き目は m² 倍になる")
    for t in (0, 1):
        m = np.abs(x[t]).max()
        base = rms_orig(x[t], eps=1e-6)
        mm = m + 1e-6
        row = "、".join(f"{lab} → 差 {rel(base, rms_maxnorm(x[t], eps=eps)):.1e}"
                       for lab, eps in (("eps 1e-6", 1e-6), ("eps 1e-12（実装）", 1e-12), ("eps 1e-6/m²（厳密）", 1e-6 / mm ** 2)))
        print(f"  トークン {t}（max|x| = {m:8.2f}）: {row}")
    print("  → 巨大値を持つトークン（m が大きい）では、元の 1e-6 のままだと効きすぎてずれる。実装の 1e-12 はそれを消すが、")
    print("    m が 1 より小さいトークンでは元の eps より弱くなり、少しずれる（近似）。eps/m² なら厳密に同じ。")
    print("    5.2 節「ここは厳密解を取り逃した」: グラフの中で eps * rmx * rmx と書けば厳密にできた")

    print("\n== 4. マスク定数（C-9）")
    MASK_REPLACE = load("app/pipeline/preprocess.py", ["MASK_REPLACE"])["MASK_REPLACE"]
    scores = rng.normal(0, 5, 1000).clip(-20, 20)           # 本物のスコアは ±20 程度（preprocess.py の説明）
    for lo, name in ((-3.4028235e38, "HF のまま -3.4e38"), (MASK_REPLACE, f"置き換え {MASK_REPLACE:g}")):
        step = (20 - lo) / 65535
        kinds = len(np.unique(q_dq(scores, lo, 20.0)))
        print(f"  {name:>18s}: 1 段 {step:.3g}、1000 個のスコアが量子化後に {kinds} 種類の値になる")
    print(f"  → exp({MASK_REPLACE:g}) = {np.exp(MASK_REPLACE):.1e} なので、softmax の後は -inf と同じく 0 になる")


if __name__ == "__main__":
    main()
