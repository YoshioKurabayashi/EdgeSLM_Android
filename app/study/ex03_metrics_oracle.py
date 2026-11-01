# -*- coding: utf-8 -*-
"""演習 3: 指標の読み方と Oracle 実験の結果（StudyMemo 4 章・8.4 節）。

    python -B -m app.study.ex03_metrics_oracle      # numpy だけ。1 秒

## 何を確かめるか

1. 本物の指標の関数（`app/pipeline/accuracy_logits.py` の `metrics`）を、**ソースから取り出して**そのまま使う
   - 相対 L2 が大きくても 1 位が保たれれば top-1 は 100%（4.4 節「見る場所を間違えない」）
   - L2 が小さくても 1 位と 2 位が入れ替われば top-1 は落ちる
2. 誤差の分解（`error_decompose.py` の `_rel`）: **分母をそろえないと比べられない**（4.5 節の罠 1）
3. 実際の Oracle 実験の結果（`results/P3_oracle6.json`）と、RMSNorm の書き換えの結果（`results/P3_rms6.json`）を読む

`metrics`・`_softmax`・`_rel` は本物のソースの定義をそのまま実行している（`app/study/_src.py`。onnxruntime は要らない）。
"""
from __future__ import annotations

import json

import numpy as np

from app.study._src import ROOT, load


def main():
    f = load("app/pipeline/accuracy_logits.py", ["_softmax", "metrics"])
    metrics = f["metrics"]
    metrics.__globals__["_softmax"] = f["_softmax"]
    rel = load("app/pipeline/error_decompose.py", ["_rel"])["_rel"]
    rng = np.random.default_rng(0)

    print("== 1. 相対 L2 と top-1 は別物")
    V = 1000
    ref = rng.normal(0, 3, (16, V))
    ref[np.arange(16), rng.integers(0, V, 16)] += 15          # 1 位がはっきりしている logits
    noisy = ref + rng.normal(0, 2.0, ref.shape)               # 全体に大きなノイズ（1 位は保たれる）
    swapped = ref.copy()                                      # ほとんど同じだが、1 位と 2 位だけ入れ替える
    for i in range(16):
        o = np.argsort(-swapped[i])
        swapped[i, o[0]], swapped[i, o[1]] = swapped[i, o[1]], swapped[i, o[0]]
    for name, t in (("全体にノイズ", noisy), ("1 位と 2 位を入れ替え", swapped)):
        m = metrics(ref, t)
        print(f"  {name:12s}: 相対L2 {m['logits_rel_l2']:6.1%}  top-1 一致 {m['top1_agreement']:6.1%}  "
              f"KL {m['kl_fp32_to_quant']:.3f}  fp32 の 1 位の順位（平均）{m['rank_of_fp32_top1_mean']:.2f}")
    print("  → 「L2 が大きい＝使えない」「L2 が小さい＝大丈夫」とは言えない。生成に直結する top-1・KL・順位を併記する（8.7 節）")

    print("\n== 2. 誤差を分解するとき、分母をそろえる（error_decompose._rel）")
    fp32 = rng.normal(0, 1, 1000)
    host = fp32 + rng.normal(0, 0.8, 1000)                    # 量子化の誤差が大きく、ホストの出力のノルムが fp32 より大きい
    dev = host + rng.normal(0, 0.5, 1000)                     # 実機で足された誤差
    print(f"  量子化        fp32 → ホスト（分母 ‖fp32‖） : {rel(fp32, host, fp32):.1%}")
    print(f"  実機の追加分  ホスト → 実機（分母 ‖fp32‖） : {rel(host, dev, fp32):.1%}")
    print(f"  実機の追加分  ホスト → 実機（分母 ‖ホスト‖）: {rel(host, dev, host):.1%}  ← 同じずれなのに小さく見える")
    print(f"  （‖ホスト‖ / ‖fp32‖ = {np.linalg.norm(host) / np.linalg.norm(fp32):.2f}。量子化で値が膨らむと、分母が大きくなって誤差が薄まる）")

    print("\n== 3. Oracle 実験の実際の結果（6 層、176 位置。results/P3_oracle6.json）")
    rows = json.loads((ROOT / "results/P3_oracle6.json").read_text(encoding="utf-8"))
    print("  量子化を外した所       | 外したノード | top-1 一致 |   KL")
    for r in rows:
        print(f"  {r['pattern']:22s} | {r['excluded_nodes']:>10d} | {r['top1_agreement']:9.1%} | {r['kl_fp32_to_quant']:7.3f}")
    print("  → 残差（Add）を外しても 6.2% のまま、RMSNorm を外すと 74.4%。推定（残差が犯人）は外れで、RMSNorm が主因（4.2 節）")

    rows = json.loads((ROOT / "results/P3_rms6.json").read_text(encoding="utf-8"))
    print("\n  RMSNorm を書き換えた結果（results/P3_rms6.json。量子化は全部かけたまま）:")
    for r in rows:
        print(f"  {r['label']:22s} top-1 一致 {r['top1_agreement']:6.1%}  KL {r['kl_fp32_to_quant']:.3f}")
    print("  → 書き換え（実機で動く形）だけで、Oracle の上限（RMSNorm を外した 74.4%）と同じ所まで戻った（6.1 節 例 3）")


if __name__ == "__main__":
    main()
