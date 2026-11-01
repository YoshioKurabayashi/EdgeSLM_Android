# -*- coding: utf-8 -*-
"""P13b: 任意のlayer範囲・K/V組み合わせでswap実験を行う汎用ランナー。

P13aで保存したキャッシュ(work/p13a_layer_block_kv_swap/caches.npz)を再利用し、
final-stepはoracleモデルに固定したまま、指定した層集合のK・Vをbaseline化して
margin(FP32基準トークンへのlogit差)を測る。VARIANTSを書き換えて使う。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P13 系（StudyMemo 11.5 節）。P13a（layer_block_kv_swap.py）の続き
- 結果（results/P13b_20260917_kv_combo_swap.md）: V 側は L11〜17 の 7 層だけで、全 28 層の V を替えた P12z（+2.202）を上回る
  回復（+2.349）。K 側は L0〜6 単独では悪化（−1.544）だが、L14〜20 と組むと改善に転じる（+1.064）
- 使い方: このファイルには入口が無い。p13b_step1_v_l7_20.py〜p13b_step4_k_combos.py が、表（VARIANTS）を作って
  run_variants を呼ぶ（「VARIANTSを書き換えて使う」は、そうした小さなファイルを段ごとに作る、という意味）
- 入力: P13a が保存したキャッシュ（WORK の caches.npz と meta.json）と oracle のモデル。
  出力: results/P13b_20260917_kv_combo_swap/<tag>.json（例 step1_v_l7_20.json）
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import tables, ROOT  # noqa: E402
from add26_causal_probe import sess  # noqa: E402
from linear20_oracle import ORACLE_FULL  # noqa: E402
from layer_block_kv_swap import WORK, N_LAYERS, T1, feed_final, margin_of  # noqa: E402

import numpy as np  # noqa: E402

OUT = ROOT / "results/P13b_20260917_kv_combo_swap"


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def run_variants(variants, tag):
    """variants: {name: {"k": [layers], "v": [layers]}}"""
    # （2026-10-18 追記）variants: 条件の名前 → baseline の値に替える K の層・V の層のリスト。tag: 出力の名前（<tag>.json）。
    # 戻り値は無し（ファイルと画面に出す）。最後の 1 歩はいつも oracle のモデル（P12z と同じ枠）。
    # 何も替えない「oracle 対照」を先に測り、各条件の差（Δ）をそこから出す
    OUT.mkdir(parents=True, exist_ok=True)
    npz = dict(np.load(WORK / "caches.npz"))
    meta = json.loads((WORK / "meta.json").read_text(encoding="utf-8"))
    last_tid, pos = meta["last_tid"], meta["pos"]
    cos, sin = tables()
    oracle_sess = sess(ORACLE_FULL)

    def mixed_cache(k_layers, v_layers):
        # 層ごとに、指定した層だけ baseline のキャッシュ、それ以外は oracle のキャッシュを使う（npz の鍵は baseline_k_<層> など）
        c = {}
        for L in range(N_LAYERS):
            c[f"past_k_{L}"] = npz[f"baseline_k_{L}"] if L in k_layers else npz[f"oracle_k_{L}"]
            c[f"past_v_{L}"] = npz[f"baseline_v_{L}"] if L in v_layers else npz[f"oracle_v_{L}"]
        return c

    m_oracle = margin_of(feed_final(oracle_sess, cos, sin, mixed_cache([], []), last_tid, pos))
    emit({"phase": "run", "tag": tag, "condition": "oracle_control", "margin": m_oracle})

    results = {"oracle_control": m_oracle}
    for name, spec in variants.items():
        m = margin_of(feed_final(oracle_sess, cos, sin, mixed_cache(spec["k"], spec["v"]), last_tid, pos))
        results[name] = m
        emit({"phase": "run", "tag": tag, "condition": name, "margin": m, "delta_vs_oracle": m - m_oracle})

    out = {"decision_top1_token": T1, "margin": results,
           "delta_vs_oracle": {k: v - m_oracle for k, v in results.items()}}
    out_path = OUT / f"{tag}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "done", "tag": tag, "wrote": str(out_path)})
    print(f"\n  [{tag}] oracle対照 margin = {m_oracle:+.4f}")
    for name in variants:
        print(f"  {name:20s} margin={results[name]:+8.4f}  Δ={results[name]-m_oracle:+8.4f}")
