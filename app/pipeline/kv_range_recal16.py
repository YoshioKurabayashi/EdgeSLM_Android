# -*- coding: utf-8 -*-
"""KV 範囲の拡張を **16bit のグループだけ**に限る切り分け（P12l-2）。

P12l で KV 系252グループ全部を広げたら、baseline は悪化した（1位保持 85.7→78.6%、自由生成がおうむ返し）。
容疑は uint8 の28グループ。C-8 のため attention の act-act MatMul の右辺（GQA で複製した KV）は 8bit で、
255段しかない。範囲を 2.55倍に広げると刻みが値の大きさに対して粗くなりすぎる可能性がある。

そこで uint8 グループは**元の範囲のまま**にし、16bit のグループだけ広げて比べる。
これで「飽和を消す利得」と「8bit の分解能の損失」を分けられる。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）。kv_range_recal.py（P12l）の続き。記録は results/P12l_20260916_kv_range_recal.md
- 結果（同じ md）: baseline_kvwide16 は 1位保持 87.5%・margin 保持 51.8% と、全部広げた版（78.6%）より良い。
  ただし点B（+3.35→−0.67）・日本語の System Prompt（+4.45→+0.88）は悪化し、baseline を全面的には上回らない
- 入力: kv_range_recal.py が書いた widen_plan.json（OUTDIR）、P12e の成果物（ART の baseline.onnx・focused_later.onnx）など
- 出力: OUTDIR の report16.json と progress.log（追記）
- 動かし方: python -m app.pipeline.kv_range_recal16（引数なし。kv_range_recal を先に流しておく）
- 関数は無く、下の if __name__ == "__main__" の中が本体。2 変種: baseline_kvwide16（全位置 baseline）と
  D1_fp32pos0_later_kvwide16（位置0 は FP32、位置1 以降は focused_later）
"""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, ROOT, OPS, tokenizer
from block_revert_ab import eval_points, score, ART
from fp32_prefix_eval import Pipeline, run_points, free_generation
from kv_range_recal import OUTDIR, apply

import numpy as np

if __name__ == "__main__":
    # ① 広げる計画から、16bit のグループ（dtype が uint8 でないもの）だけを取り出す。uint8 は据え置き
    plan_all = json.loads((OUTDIR / "widen_plan.json").read_text(encoding="utf-8"))["plan"]
    plan16 = [p for p in plan_all if "new_scale" in p and p["dtype"] != "uint8"]
    skipped = [p for p in plan_all if "new_scale" in p and p["dtype"] == "uint8"]
    log = open(OUTDIR / "progress.log", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    # 下の「倍率 最大」は、据え置いた uint8 のグループの倍率の最大（計画では何倍に広げるはずだったか）
    say(f"16bit だけ広げる: {len(plan16)} グループ（uint8 {len(skipped)} は据え置き。倍率 最大 "
        f"{max(p['widen_factor'] for p in skipped):.2f}）")
    # ② 評価点と FP32 の基準を作り、2 変種を同じ枠で評価する（自由生成も）
    tok = tokenizer()
    pts = eval_points(tok)
    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    report = {"scores": {}, "free_generation": {}}
    base16 = apply(ART / "baseline.onnx", plan16)
    later16 = apply(ART / "focused_later.onnx", plan16)
    for name, first, later, mode in (("baseline_kvwide16", base16, base16, "none"),
                                     ("D1_fp32pos0_later_kvwide16", later16, later16, "pos0")):
        t0 = time.time()
        pipe = Pipeline(fp32, first, later, mode)
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        report["scores"][name] = sc
        report["free_generation"][name] = gen
        a = sc["ALL"]
        say(f"{name:28s} 1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  最小margin {a['margin_min']:+6.2f}  "
            f"KL {a['kl_mean']:.3f} | 点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}  ({time.time() - t0:.0f}s)")
        for t in gen:
            say(f"      turn{t['turn']}: {t['text']!r}")
        (OUTDIR / "report16.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"-> {OUTDIR / 'report16.json'}")
