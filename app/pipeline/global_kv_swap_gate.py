# -*- coding: utf-8 -*-
"""P12z: 大域K-only/V-only swap — KV仮説そのものを検証するゲート試験。

3回連続の局所swap失敗（P12w: add_26単体、P12y: KV1点）は「KVが無関係」ではなく
「単一点では説明できない」ことを示しているだけ。ここでは判定位置に至るまでの
K/V**全体**（全position・全28層）を丸ごと入れ替えて、KV state仮説そのものにゲートをかける。

同じteacher-forced履歴（点Bの80トークン）を使い、**最終step（判定位置）のモデルはoracleに
固定**する。position 0-78で構築したKVキャッシュだけを、oracle自身のものとbaselineのもので
以下の4通りに組み合わせ、最終step(position 79のトークンをfeedして得るlogits)を比較する。

  K        V         意味
  oracle   oracle    失敗側対照(oracle自身)
  baseline oracle    Kをbaseline化すると救えるか
  oracle   baseline  Vをbaseline化すると救えるか
  baseline baseline  成功側対照(KV両方baseline化。ただし最終stepの計算自体はoracle固定)

## 判定基準(事前に決めたゲート)

  A. KまたはVの片方の交換だけで大きく回復 → layer block×K/Vへ分解する探索へ
  B. 両方baselineでしか回復しない(相互作用) → layer block単位で両方同時にswapする実験へ
  C. baseline K+Vの組み合わせでも回復しない → KV探索はここで終了。
     hidden trajectory/residual/MLP/final-step処理側、あるいはKVと現在hiddenの
     組み合わせへ方向転換する

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節「KV の差し替え」）。前は P12y（local_swap_pos45_L14V.py）、後は P13a（layer_block_kv_swap.py）
- 結果（results/P12z_20260916_global_kv_swap_gate.md）: K だけでも V だけでも判定が負→正に反転した（ゲート A）。
  margin: oracle/oracle −1.0210、K を baseline +0.8388、V を baseline +1.1808、両方 +3.1374
- 入力: P12v の oracle モデル（linear20_oracle.ORACLE_FULL）・baseline の QDQ（SOURCE）・トークナイザ・RoPE 表
- 出力: results/P12z_20260916_global_kv_swap_gate/result.json と画面の表
- 「oracle」= linear_20 の Q/DQ を外したモデル（P12v）。自由生成で点B に失敗する側。「baseline」= 既存の QDQ（点B に成功する側）
## 使い方

  python global_kv_swap_gate.py
  （2026-10-18 補足）app/pipeline の中で上のように動かす。プロジェクトの直下からなら python -m app.pipeline.global_kv_swap_gate
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, ROOT  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from add26_causal_probe import sess  # noqa: E402
from linear20_oracle import ORACLE_FULL  # noqa: E402
from temporal_qdq_experiment import SOURCE  # noqa: E402

import numpy as np  # noqa: E402

# 結果の置き場と、Qwen3-0.6B の層の数
OUT = ROOT / "results/P12z_20260916_global_kv_swap_gate"
N_LAYERS = 28


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def build_cache(model_sess, cos, sin, ids):
    """ids(先頭からlen(ids)個)をteacher forcingし、そのあとのpast_k/v(全28層)を返す。"""
    # （2026-10-18 追記）戻り値: {past_k_0〜27・past_v_0〜27: [1, KV ヘッド数, 256, 128]}（最後の feed の後のキャッシュ。
    #   形は推定: Qwen3-0.6B の KV ヘッドは 8、head_dim は 128）
    r = Runner(model_sess, cos, sin)
    for tid in ids:
        r.feed(tid, model_sess)
    return r.cache


def feed_final(model_sess, cos, sin, cache, last_tid, pos):
    """cache(過去分)を与えて、最後の1トークンだけをmodel_sessでfeedしてlogitsを返す。"""
    # （2026-10-18 追記）pos: このトークンの位置（RoPE の行とマスクの長さを決める）。マスクは右詰めで直近 pos+1 個だけ 0、
    #   ほかは −60（Runner.feed と同じ作り方）。戻り値: logits（語彙数の 1 次元、float64）
    mask = np.full((1, 1, 1, 256), -60, np.float32)
    mask[..., 256 - min(pos + 1, 256):] = 0
    inputs = {"input_ids": np.array([[last_tid]], np.int64),
              "cos": cos[pos].reshape(1, 1, 1, 128), "sin": sin[pos].reshape(1, 1, 1, 128),
              "attn_mask": mask, **cache}
    names = [o.name for o in model_sess.get_outputs()]
    result = dict(zip(names, model_sess.run(None, inputs)))
    return result["logits"].ravel().astype(np.float64)


def main():
    """① 点B の入力を取る → ② oracle と baseline でそれぞれ位置 0〜78 のキャッシュを作る
    → ③ K・V の出どころを 4 通りに組み合わせ、最後の 1 歩だけ oracle のモデルで流す → ④ FP32 の 1 位の語の margin。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "point_B_music")
    ids = p["ids"]
    prefix, last_tid, pos = ids[:-1], ids[-1], len(ids) - 1
    emit({"phase": "start", "tokens": len(ids), "decision_position": pos})

    cos, sin = tables()
    oracle_sess = sess(ORACLE_FULL)
    baseline_sess = sess(SOURCE)

    oracle_cache = build_cache(oracle_sess, cos, sin, prefix)
    emit({"phase": "cache", "variant": "oracle", "done": True})
    baseline_cache = build_cache(baseline_sess, cos, sin, prefix)
    emit({"phase": "cache", "variant": "baseline", "done": True})

    # FP32の1位トークン(これまでの実験と同じ基準)を得るため、既存のP12x/P12yと同じ
    # decision_top1_tokenを使う(78685)。念のためoracle自身のfeedでも整合を取る。
    # ★2026-10-18 訂正: 旧「念のためoracle自身のfeedでも整合を取る」。このコードには整合を確かめる処理は無い
    #   （78685 を下で決め打ちするだけ）。数は P12w の compare.json の decision_top1_token と同じ、と記録にある
    conditions = {
        "oracle_oracle": {"K": oracle_cache, "V": oracle_cache},
        "baseline_oracle": {"K": baseline_cache, "V": oracle_cache},
        "oracle_baseline": {"K": oracle_cache, "V": baseline_cache},
        "baseline_baseline": {"K": baseline_cache, "V": baseline_cache},
    }
    logits_by_cond = {}
    # ③ 層ごとに、K は条件の K 側のキャッシュから、V は V 側のキャッシュから取って混ぜる
    for name, kv in conditions.items():
        mixed = {}
        for L in range(N_LAYERS):
            mixed[f"past_k_{L}"] = kv["K"][f"past_k_{L}"]
            mixed[f"past_v_{L}"] = kv["V"][f"past_v_{L}"]
        logits_by_cond[name] = feed_final(oracle_sess, cos, sin, mixed, last_tid, pos)
        emit({"phase": "run", "condition": name, "done": True})

    t1 = 78685  # P12w/P12x/P12yと同じ、FP32自身の1位トークン
    margins = {}
    for name, lg in logits_by_cond.items():
        mask = np.ones(lg.size, dtype=bool)
        mask[t1] = False
        margins[name] = float(lg[t1] - lg[mask].max())

    result = {"decision_top1_token": t1, "margin": margins}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "done", **result})
    print(f"\n  最終step固定=oracleモデル / K・Vの出どころを4通りに組み合わせ:")
    print(f"  {'K':>10s} {'V':>10s} {'margin':>10s}")
    for name in ("oracle_oracle", "baseline_oracle", "oracle_baseline", "baseline_baseline"):
        k, v = name.split("_")
        print(f"  {k:>10s} {v:>10s} {margins[name]:+10.4f}")
    print(f"\n-> {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
