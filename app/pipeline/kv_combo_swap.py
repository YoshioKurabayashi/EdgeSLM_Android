# -*- coding: utf-8 -*-
"""P13b: 任意のlayer範囲・K/V組み合わせでswap実験を行う汎用ランナー。

P13aで保存したキャッシュ(work/p13a_layer_block_kv_swap/caches.npz)を再利用し、
final-stepはoracleモデルに固定したまま、指定した層集合のK・Vをbaseline化して
margin(FP32基準トークンへのlogit差)を測る。VARIANTSを書き換えて使う。
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
    print(json.dumps(o, ensure_ascii=False), flush=True)


def run_variants(variants, tag):
    """variants: {name: {"k": [layers], "v": [layers]}}"""
    OUT.mkdir(parents=True, exist_ok=True)
    npz = dict(np.load(WORK / "caches.npz"))
    meta = json.loads((WORK / "meta.json").read_text(encoding="utf-8"))
    last_tid, pos = meta["last_tid"], meta["pos"]
    cos, sin = tables()
    oracle_sess = sess(ORACLE_FULL)

    def mixed_cache(k_layers, v_layers):
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
