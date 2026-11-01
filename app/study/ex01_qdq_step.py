# -*- coding: utf-8 -*-
"""演習 1: QDQ の式と「1 段階の幅（step）」— bit 数より Dynamic Range（StudyMemo 2 章・3 章・7 章）。

    python -B -m app.study.ex01_qdq_step          # numpy だけ。1 秒

## 何を確かめるか

1. QuantizeLinear / DequantizeLinear の式（2.1 節）を、そのまま numpy で書いて往復させる
2. `step = 42.7` の正体（2.2 節）: RMSNorm の中の `x²` が 0〜約 280 万の幅を持つと、uint16（65,536 段）でも
   1 段が 42.7 になる。ふつうのチャンネルの `x²`（0.04 前後）は**全部 0 に丸まる**
3. 重みの per-tensor と per-channel（7 章）: 列ごとに大きさの違う重みを 1 つの scale で量子化すると、
   小さい列の誤差が大きくなる。列ごとの scale（per-channel、axis=1）にすると誤差が下がる

本物の道具（`app/pipeline/quantize_qdq.py` など）は ORT の量子化器を呼ぶので onnxruntime が要る。
ここでは**式だけ**を取り出して、数字の意味を手で確かめる。
"""
from __future__ import annotations

import numpy as np


def quantize(x, scale, zp, qmin, qmax):
    """QuantizeLinear: q = clip(round(x / scale) + zero_point, qmin, qmax)"""
    return np.clip(np.rint(x / scale) + zp, qmin, qmax)


def dequantize(q, scale, zp):
    """DequantizeLinear: x' = (q - zero_point) * scale"""
    return (q - zp) * scale


def asym_params(lo, hi, bits):
    """非対称（活性化）: 最小〜最大を 2^bits - 1 段に割る。0 がちょうど表せるよう zero_point を整数に丸める"""
    qmax = 2 ** bits - 1
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / qmax
    zp = np.rint(-lo / scale)
    return scale, zp, 0, qmax


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def main():
    rng = np.random.default_rng(0)

    print("== 1. QDQ の往復（uint16、範囲 [-10, 6]）")
    x = rng.uniform(-10, 6, 8).astype(np.float32)
    s, zp, qmin, qmax = asym_params(-10, 6, 16)
    q = quantize(x, s, zp, qmin, qmax)
    print(f"  scale = {s:.3e}（1 段の幅）  zero_point = {zp:.0f}")
    print(f"  x    = {np.round(x, 4)}")
    print(f"  q    = {q.astype(int)}")
    print(f"  x'   = {np.round(dequantize(q, s, zp), 4)}   最大のずれ {np.abs(dequantize(q, s, zp) - x).max():.2e} ≦ scale/2")

    print("\n== 2. step = 42.7 の正体（RMSNorm の中の x²）")
    hi = 2801222.2                                     # ログの実測: `pow_13  16bit range=2801222.2  1段階=42.74326`（2.2 節）
    s2, zp2, _, qm2 = asym_params(0.0, hi, 16)
    print(f"  x² の範囲 [0, {hi:,.1f}] を uint16 で → 1 段 = {s2:.5f}")
    print(f"  （ログの 42.74326 は範囲を 65,536 で割った値 {hi / 65536:.5f}。式（65,535 で割る）との差は 0.002% で、意味は同じ）")
    print(f"  （残差の巨大チャンネル ±1677 の二乗は {1677.0 ** 2:,.0f}。この 1 本が範囲を決めている）")
    normal = rng.normal(0, 0.2, 895)                   # 巨大値を持たないトークンのふつうのチャンネル
    xx = normal ** 2
    qx = dequantize(quantize(xx, s2, zp2, 0, qm2), s2, zp2)
    print(f"  ふつうのチャンネルの x² は平均 {xx.mean():.3f}。量子化すると 0 でない値が {int((qx != 0).sum())} / {len(xx)} 個")
    print(f"  → mean(x²) が 0 に潰れ、RMSNorm の分母 sqrt(mean(x²)+eps) が壊れる（5.1 節の書き換えが必要になった理由）")

    print("\n== 3. 重みの per-tensor と per-channel（7 章）")
    K, N = 896, 64
    col_scale = np.exp(rng.normal(0, 1.0, N))          # 列（出力チャンネル）ごとに大きさが違う
    W = (rng.normal(0, 0.02, (K, N)) * col_scale).astype(np.float32)
    def sym_q8(w, absmax):                              # 重みは対称 int8（zero_point 0）
        sc = absmax / 127.0
        return np.clip(np.rint(w / sc), -127, 127) * sc
    pt = sym_q8(W, np.abs(W).max())                                  # 全体で 1 つの scale
    pc = sym_q8(W, np.abs(W).max(axis=0, keepdims=True))             # 列ごとの scale（axis=1 の N 本）
    worst = int(np.argmin(col_scale))
    print(f"  重み [K={K}, N={N}]、列の大きさの比 最大/最小 = {col_scale.max() / col_scale.min():.0f} 倍")
    print(f"  per-tensor : 相対L2 {rel_l2(W, pt):.2%}   一番小さい列 {rel_l2(W[:, worst], pt[:, worst]):.1%}")
    print(f"  per-channel: 相対L2 {rel_l2(W, pc):.2%}   一番小さい列 {rel_l2(W[:, worst], pc[:, worst]):.1%}")
    print("  → 本物では「per_channel=True を渡したのに生成物は 390 本すべて per-tensor」だった（7.2 節）。")
    print("    生成物を数える道具が app/pipeline/weight_percg.py の inspect_weights（StudyMemo 14.4 節）")


if __name__ == "__main__":
    main()
