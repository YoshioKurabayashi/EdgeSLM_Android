# -*- coding: utf-8 -*-
"""演習 5: 固定長の KV キャッシュ（左シフト＋マスク）と RoPE の表（StudyMemo 1.6 節・10 章）。

    python -B -m app.study.ex05_decode_cache      # numpy だけ。1 秒

## 何を確かめるか

NPU（v68）は動的形状を扱えないので、decode のグラフは**長さ L の固定のキャッシュ**を入出力に持つ
（`app/pipeline/export_qwen_decode.py`）。毎回 1 つ左にずらして末尾に新しい K・V を書き、空きはマスクで塞ぐ。

1. `cat(past[:, 1:], k)` と `cat(past, k)[:, 1:]` は同じ（1.6 節: 後者にしたのは、量子化器が Slice に入る float の入力を量子化しないため）
2. 小さな 1 層の注意で、固定長キャッシュ（L 個）＋マスクで 1 語ずつ回した結果が、全部をまとめて計算した結果と一致する（窓に収まる間）
3. 窓が埋まると、古い語から見えなくなる（10.4 節）
4. RoPE の表（`app/pipeline/make_rope_table.py` の `build`）: 行数は「位置を何番まで表せるか」で、キャッシュの容量とは別（10.3 節）

`build`（RoPE の表）と `MASK_NEG` は本物のソースから取り出して使う（`app/study/_src.py`）。
"""
from __future__ import annotations

import numpy as np

from app.study._src import load


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def rope(x, cos, sin):
    """export_qwen_decode.QwenDecodeStep._rope と同じ rotate_half"""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


def main():
    build = load("app/pipeline/make_rope_table.py", ["build"])["build"]
    MASK_NEG = load("app/pipeline/export_qwen_decode.py", ["MASK_NEG"])["MASK_NEG"]
    rng = np.random.default_rng(0)
    D, L, T = 16, 8, 6                                       # head_dim 16、キャッシュ 8 個、6 語
    cos, sin = build(64, head_dim=D)
    Wq, Wk, Wv = (rng.normal(0, 0.3, (D, D)) for _ in range(3))
    xs = rng.normal(0, 1, (T, D))

    print("== 1. 「ずらしてから足す」と「足してからずらす」は同じ")
    past = rng.normal(0, 1, (1, L, D))
    k = rng.normal(0, 1, (1, 1, D))
    a = np.concatenate([past[:, 1:], k], 1)
    b = np.concatenate([past, k], 1)[:, 1:]
    print(f"  最大差 {np.abs(a - b).max():.1e}  形 {a.shape}（長さ L のまま）")

    print(f"\n== 2. 固定長キャッシュ（L = {L}）で 1 語ずつ ↔ まとめて計算（{T} 語。窓に収まる）")
    q_all = np.stack([rope(xs[t] @ Wq, cos[t], sin[t]) for t in range(T)])
    k_all = np.stack([rope(xs[t] @ Wk, cos[t], sin[t]) for t in range(T)])
    v_all = xs @ Wv
    full = np.stack([softmax(q_all[t] @ k_all[:t + 1].T / np.sqrt(D)) @ v_all[:t + 1] for t in range(T)])
    pk, pv = np.zeros((L, D)), np.zeros((L, D))
    step = []
    for t in range(T):
        kt = rope(xs[t] @ Wk, cos[t], sin[t])               # 位置 t の cos/sin は表の t 行目（ホストが渡す）
        pk = np.concatenate([pk, kt[None]], 0)[1:]           # 足してからずらす（export_qwen_decode と同じ順）
        pv = np.concatenate([pv, (xs[t] @ Wv)[None]], 0)[1:]
        mask = np.full(L, MASK_NEG)
        mask[L - min(t + 1, L):] = 0.0                       # 有効なのは後ろ t+1 個
        s = q_all[t] @ pk.T / np.sqrt(D) + mask
        step.append(softmax(s) @ pv)
    print(f"  最大差 {np.abs(full - np.stack(step)).max():.1e}（マスクの {MASK_NEG:g} は exp で {np.exp(MASK_NEG):.0e} ＝ 実質 0）")

    print(f"\n== 3. 窓が埋まった後（{L + 4} 語目）: 最初の 4 語はキャッシュから押し出されて見えない")
    T2 = L + 4
    xs2 = rng.normal(0, 1, (T2, D))
    pk = np.zeros((L, D))
    for t in range(T2):
        pk = np.concatenate([pk, rope(xs2[t] @ Wk, cos[t], sin[t])[None]], 0)[1:]
    first = rope(xs2[0] @ Wk, cos[0], sin[0])
    print(f"  キャッシュの中に 0 語目の K があるか: {bool(np.any(np.all(np.isclose(pk, first), axis=1)))}")
    print("  → 固定長の窓では古い語を忘れる。品質が落ちる原因は移植ではなく文脈の持ち方（10.4 節）。固定 prefix・sink の比較は 10.5・10.6 節")

    print("\n== 4. RoPE の表の行数とキャッシュの容量は別")
    c2048, _ = build(2048, head_dim=64)
    print(f"  Qwen2.5（head_dim 64）の表: {c2048.shape}。位置 2047 まで表せる。キャッシュの容量（何語覚えるか）はモデルの L で別に決まる")
    print("  窓が一杯になっても位置は増え続けるので、表の行数は「生成したい語数」で決める（make_rope_table.py の説明）")


if __name__ == "__main__":
    main()
