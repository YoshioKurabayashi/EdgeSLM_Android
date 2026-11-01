# -*- coding: utf-8 -*-
"""P12y: position45・layer14のVだけをbaseline/4x/oracleの値に差し替えたFP32軌跡で、
判定位置(79)のmarginがどう動くかを見る（誤差の「向き」＝decision projectionの直接測定）。

P12xで、baseline/4x/oracleのKV誤差は全体としては似た大きさなのに、position45・layer14の
Vだけoracleが突出して悪化していた（baseline/4x ~5% に対し oracle 71.7%）ことが分かった。
これが実際に判定を悪い方向へ動かしているかを、線形近似のgradientではなく、実際の順伝播で
確かめる: FP32の正しい軌跡（position0-44は完全にFP32）に対し、position45のlayer14 Vだけを
P12xで採取済みのbaseline/4x/oracleの値に差し替え、そこから先(position46-79)もFP32モデルで
継続する。差し替えの効果だけが、判定位置79のmarginにどう表れるかを見る。

仮説の更新: 「小さな誤差→KVへ徐々に蓄積→最後に壊れる」ではなく、
「KV誤差は早期から大きい→大部分は無害→特定のtoken×layer×Vだけdecision方向へ有害にずれる
→生成軌道が分岐」

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節「KV の差し替え」）。P12x（kv_trajectory_probe.py）の続き、後は P12z（global_kv_swap_gate.py）
- 結果（results/P12y_20260916_local_swap_pos45_L14V.md）: 3 条件とも判定への効果はほぼ無視できる大きさで、oracle はむしろ良くなる向き
  （+0.0188）。P12w の add_26 単体の差し替えと同じく「1 点では説明できない」→ KV 全体を入れ替える P12z へ
- 入力: P12x の work/p12x_kv_trajectory/kv_trajectory.npz（各モデルが位置ごとに新しく書いた K・V）と FP32 のモデル
- 出力: results/P12y_20260916_local_swap_pos45_L14V/result.json と画面の表
- プロジェクトの直下からなら python -m app.pipeline.local_swap_pos45_L14V でも動く
## 使い方

  python local_swap_pos45_L14V.py
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, ROOT, OPS  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from add26_causal_probe import sess  # noqa: E402

import numpy as np  # noqa: E402

# 結果の置き場、P12x の採取結果の置き場、差し替える位置と層
OUT = ROOT / "results/P12y_20260916_local_swap_pos45_L14V"
WORK = ROOT / "work/p12x_kv_trajectory"
SWAP_POSITION = 45
SWAP_LAYER = 14


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def run_with_swap(fp32_sess, cos, sin, ids, swap_v):
    """positions 0..len(ids)-1 をFP32モデルで実行。swap_v is not None なら、
    position==SWAP_POSITION の直後に past_v_{SWAP_LAYER} の新規スロットを差し替える。"""
    # （2026-10-18 追記）swap_v: 差し替える V の新しいスロット（1024 = 8 ヘッド × 128）。None なら差し替えない（対照）。
    # 戻り値: 最後の位置（判定位置）の logits（float64 の 1 次元）。
    # 位置 45 を流した直後に、キャッシュの最後のスロット（いま書いたばかりの位置 45 の V）を書き換え、以降は FP32 で続ける
    r = Runner(fp32_sess, cos, sin)
    logits = None
    for pos, tid in enumerate(ids):
        got, _ = r.feed(tid, fp32_sess)
        if pos == SWAP_POSITION and swap_v is not None:
            r.cache[f"past_v_{SWAP_LAYER}"][:, :, -1, :] = swap_v.reshape(1, 8, 128)
        if pos == len(ids) - 1:
            logits = got["logits"].ravel().astype(np.float64)
    return logits


def main():
    """① 点B の入力と、P12x で採取した位置45・層14 の V（baseline・4x・oracle）を読む → ② 4 条件で FP32 の軌跡を流す → ③ margin。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "point_B_music")
    ids = p["ids"]
    emit({"phase": "start", "tokens": len(ids), "swap_position": SWAP_POSITION, "swap_layer": SWAP_LAYER})

    kv = dict(np.load(WORK / "kv_trajectory.npz"))
    swap_values = {
        "fp32": None,  # 対照(差し替えなし)
        "baseline": kv["baseline_v"][SWAP_POSITION, SWAP_LAYER],
        "4x": kv["4x_v"][SWAP_POSITION, SWAP_LAYER],
        "oracle": kv["oracle_v"][SWAP_POSITION, SWAP_LAYER],
    }

    cos, sin = tables()
    fp32_sess = sess(OPS / "qwen3_dec28c256_prepped.onnx")

    logits_by_cond = {}
    for name, v in swap_values.items():
        logits_by_cond[name] = run_with_swap(fp32_sess, cos, sin, ids, v)
        emit({"phase": "run", "condition": name, "done": True})

    t1 = int(logits_by_cond["fp32"].argmax())
    margins = {}
    for name, lg in logits_by_cond.items():
        mask = np.ones(lg.size, dtype=bool)
        mask[t1] = False
        margins[name] = float(lg[t1] - lg[mask].max())

    result = {"decision_top1_token": t1, "margin": margins,
              "margin_delta_vs_fp32": {k: margins[k] - margins["fp32"] for k in margins}}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "done", **result})
    print(f"\n  position{SWAP_POSITION}・layer{SWAP_LAYER}のVだけを差し替えたFP32軌跡:")
    print(f"  {'条件':10s} {'margin':>10s} {'Δmargin(FP32比)':>16s}")
    for name in ("fp32", "baseline", "4x", "oracle"):
        print(f"  {name:10s} {margins[name]:+10.4f} {margins[name]-margins['fp32']:+16.4f}")
    print(f"\n-> {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
