# -*- coding: utf-8 -*-
"""P13a: layer block × K/V分解 — どのブロックがKV回復効果の大半を説明するか。

P12z: KV状態を丸ごとoracle→baselineに交換すると、KだけでもVだけでも判定が反転した
(oracle基準margin -1.0210 → K単独+0.8388 → V単独+1.1808 → 両方+3.1374)。
K単独の回復(+1.8598)とV単独の回復(+2.2018)の合計(+4.0616)は、両方交換の実測回復(+4.1584)と
ほぼ一致(差 +0.097)しており、KとVの悪影響は全体としてはかなり加法的に分解できる。

「多数のposition×layerに分散したKV差がAttentionでまとめて参照され、その集合効果が判定を
動かしている」という理解のもと、28層を4ブロック(L0-6/L7-13/L14-20/L21-27)に分け、
各ブロック・各K/Vだけをbaseline化した場合の回復度(Δmargin)を測る。

final-stepはP12zと同じくoracleモデルに固定。先入観を排し、4ブロックを同じ条件で測る。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P13 系（StudyMemo 11.5 節）。P12z（global_kv_swap_gate.py）の続き、後は P13b（kv_combo_swap.py）
- 結果（results/P13a_20260917_layer_block_kv_swap.md）: V は L7〜13・L14〜20 が大きく効く（+1.27・+1.88）分散型。
  K は L0〜6 単独では悪化（−1.54）するのに全 K 一括では改善（+1.86）する、強い相互作用型
- 入力: oracle のモデル（linear20_oracle.ORACLE_FULL）・baseline の QDQ（SOURCE）・点B の入力（block_revert_ab.eval_points）
- 出力: work/p13a_layer_block_kv_swap/ の caches.npz（oracle・baseline の全層の K・V。鍵は oracle_k_<層> など）と meta.json、
  results/P13a_20260917_layer_block_kv_swap/result.json。caches.npz は P13b 以降（kv_combo_swap.py など）も使う
- プロジェクトの直下からなら python -m app.pipeline.layer_block_kv_swap --phase cache のようにも動く
## 使い方

  python layer_block_kv_swap.py --phase cache   # oracle/baselineのKVキャッシュを保存(初回のみ)
  python layer_block_kv_swap.py --phase run      # 8変種(4ブロック×K/V)を測定
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, ROOT  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from add26_causal_probe import sess  # noqa: E402
from linear20_oracle import ORACLE_FULL  # noqa: E402
from temporal_qdq_experiment import SOURCE  # noqa: E402

import numpy as np  # noqa: E402

# 結果の置き場・キャッシュの置き場・層の数・基準の語（FP32 の 1 位）
OUT = ROOT / "results/P13a_20260917_layer_block_kv_swap"
WORK = ROOT / "work/p13a_layer_block_kv_swap"
N_LAYERS = 28
T1 = 78685  # FP32自身の1位トークン(P12w/P12x/P12y/P12zと同じ基準)

# 28 層を 7 層ずつ 4 つに分けたブロック
BLOCKS = {
    "L0-6": list(range(0, 7)),
    "L7-13": list(range(7, 14)),
    "L14-20": list(range(14, 21)),
    "L21-27": list(range(21, 28)),
}


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def build_cache(model_sess, cos, sin, ids):
    """ids を先頭から 1 つずつ model_sess に流し（teacher forcing）、最後のキャッシュ {past_k_i/past_v_i} を返す。"""
    r = Runner(model_sess, cos, sin)
    for tid in ids:
        r.feed(tid, model_sess)
    return r.cache


def feed_final(model_sess, cos, sin, cache, last_tid, pos):
    """キャッシュ cache を与えて最後の 1 トークン last_tid（位置 pos）だけを流し、logits（float64 の 1 次元）を返す。

    マスクは右詰めで直近 pos+1 個だけ 0、ほかは −60。cos/sin は Qwen3 の head_dim 128 の行。global_kv_swap_gate.py と同じ
    """
    mask = np.full((1, 1, 1, 256), -60, np.float32)
    mask[..., 256 - min(pos + 1, 256):] = 0
    inputs = {"input_ids": np.array([[last_tid]], np.int64),
              "cos": cos[pos].reshape(1, 1, 1, 128), "sin": sin[pos].reshape(1, 1, 1, 128),
              "attn_mask": mask, **cache}
    names = [o.name for o in model_sess.get_outputs()]
    result = dict(zip(names, model_sess.run(None, inputs)))
    return result["logits"].ravel().astype(np.float64)


def margin_of(lg):
    """基準の語 T1 の logit − それ以外の最大の logit（正なら FP32 と同じ語を選ぶ）。"""
    mask = np.ones(lg.size, dtype=bool)
    mask[T1] = False
    return float(lg[T1] - lg[mask].max())


def phase_cache():
    """点B の位置 0〜78 を oracle と baseline でそれぞれ流し、全層の K・V と最後の語・位置を保存する（初回だけ）。"""
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "point_B_music")
    ids = p["ids"]
    prefix = ids[:-1]
    emit({"phase": "cache", "tokens": len(ids)})

    cos, sin = tables()
    oracle_sess = sess(ORACLE_FULL)
    oracle_cache = build_cache(oracle_sess, cos, sin, prefix)
    emit({"phase": "cache", "variant": "oracle", "done": True})
    del oracle_sess

    baseline_sess = sess(SOURCE)
    baseline_cache = build_cache(baseline_sess, cos, sin, prefix)
    emit({"phase": "cache", "variant": "baseline", "done": True})
    del baseline_sess

    npz = {}
    for L in range(N_LAYERS):
        npz[f"oracle_k_{L}"] = oracle_cache[f"past_k_{L}"]
        npz[f"oracle_v_{L}"] = oracle_cache[f"past_v_{L}"]
        npz[f"baseline_k_{L}"] = baseline_cache[f"past_k_{L}"]
        npz[f"baseline_v_{L}"] = baseline_cache[f"past_v_{L}"]
    np.savez(WORK / "caches.npz", **npz)
    (WORK / "meta.json").write_text(json.dumps({"ids": ids, "last_tid": ids[-1], "pos": len(ids) - 1}),
                                    encoding="utf-8")
    emit({"phase": "cache", "wrote": str(WORK / "caches.npz")})


def phase_run():
    """保存したキャッシュから、ブロック × K/V の 8 変種と対照を作り、最後の 1 歩を oracle で流して margin を比べる。"""
    OUT.mkdir(parents=True, exist_ok=True)
    npz = dict(np.load(WORK / "caches.npz"))
    meta = json.loads((WORK / "meta.json").read_text(encoding="utf-8"))
    last_tid, pos = meta["last_tid"], meta["pos"]
    cos, sin = tables()
    oracle_sess = sess(ORACLE_FULL)

    def mixed_cache(k_swap_layers, v_swap_layers):
        # 指定した層だけ baseline、ほかは oracle のキャッシュを使う
        c = {}
        for L in range(N_LAYERS):
            c[f"past_k_{L}"] = npz[f"baseline_k_{L}"] if L in k_swap_layers else npz[f"oracle_k_{L}"]
            c[f"past_v_{L}"] = npz[f"baseline_v_{L}"] if L in v_swap_layers else npz[f"oracle_v_{L}"]
        return c

    results = {}
    # 対照: 純oracle(全部oracle)
    m_oracle = margin_of(feed_final(oracle_sess, cos, sin, mixed_cache([], []), last_tid, pos))
    results["oracle_oracle(control)"] = m_oracle
    emit({"phase": "run", "condition": "oracle_oracle(control)", "margin": m_oracle})

    for block_name, layers in BLOCKS.items():
        for kv in ("K", "V"):
            k_layers = layers if kv == "K" else []
            v_layers = layers if kv == "V" else []
            m = margin_of(feed_final(oracle_sess, cos, sin, mixed_cache(k_layers, v_layers), last_tid, pos))
            name = f"{block_name}_{kv}"
            results[name] = m
            emit({"phase": "run", "condition": name, "margin": m, "delta_vs_oracle": m - m_oracle})

    out = {"decision_top1_token": T1, "margin": results,
           "delta_vs_oracle": {k: v - m_oracle for k, v in results.items()}}
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "done", "wrote": str(OUT / "result.json")})

    print(f"\n  oracle対照 margin = {m_oracle:+.4f}\n")
    print(f"  {'block':10s} {'K Δmargin':>12s} {'V Δmargin':>12s}")
    for block_name in BLOCKS:
        dk = results[f"{block_name}_K"] - m_oracle
        dv = results[f"{block_name}_V"] - m_oracle
        print(f"  {block_name:10s} {dk:+12.4f} {dv:+12.4f}")


def main():
    """コマンドの入口。--phase cache か run のどちらか 1 段を流す。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["cache", "run"], required=True)
    a = ap.parse_args()
    {"cache": phase_cache, "run": phase_run}[a.phase]()


if __name__ == "__main__":
    main()
