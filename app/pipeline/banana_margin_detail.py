# -*- coding: utf-8 -*-
"""BANANA(sys_en_BANANA)評価点のlogit marginを高精度で見る(P12o/P12p-M/P12p-Lで margin=+0.00 だった件)。

score()の表示は小数2桁に丸めているため、+0.00 が厳密なゼロなのか±0.005程度のばらつきなのかを
区別できない。ここではtop1/top2のtoken IDと、logit差を小数6桁以上で出す。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）。P12o（results/P12o_20260916_pos0_wide_pos1_narrow.md）と
  P12p（results/P12p_20260916_mul_linear_split.md）の表で、System Prompt「BANANA」の評価点が margin +0.00 と出た件の確かめ
- 比べる 4 つ: baseline（既存 QDQ）・P12o（位置0 は baseline、位置1 以降は mlp2later）・P12p-M・P12p-L
  （位置0 は baseline、位置1 以降は mlp2_mul_linear_split.py の M／L のモデル）。基準の 1 位の語は FP32 で決める
- 入力: work/ops/qwen3_dec28c256/ の prepped.onnx（FP32）と final_qdq.onnx、P12n・P12p で作ったモデル、トークナイザ・RoPE 表
- 出力: 画面だけ（ファイルは書かない）。この道具の結果を書いた results のファイルは見当たらない
- 動かし方: python -m app.pipeline.banana_margin_detail（引数なし）
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS, SOURCE  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from mlp2_all_layers_later import MODEL as MLP2LATER  # noqa: E402
from mlp2_mul_linear_split import MODELS as ML_MODELS  # noqa: E402

import numpy as np  # noqa: E402


# （2026-10-18 追記）評価点の入力 ids を、位置0 は sess_first・位置1 以降は sess_later で 1 つずつ流し、
# 位置 pos_want の logits（語彙数の 1 次元、float64）を返す。KV キャッシュは 2 つのモデルで持ち回す（Runner の中で共通）。
# 位置0 と後続でモデルを替えるのは、P12o・P12p が「位置0 だけ別の範囲」の変種だから
def logits_at(sess_first, sess_later, ids, pos_want):
    cos, sin = tables()
    r = Runner(sess_first, cos, sin)
    got = None
    for pos, tid in enumerate(ids):
        s = sess_first if pos == 0 else sess_later
        got, _ = r.feed(tid, s)
        if pos == pos_want:
            return got["logits"].ravel().astype(np.float64)
    raise RuntimeError("position not reached")


# （2026-10-18 追記）logits lg について、FP32 の 1 位の語 t1 の logit・t1 を除いた最大の語（次点）・margin
# （t1 の logit − 次点の logit）と、lg 自身の 1 位・2 位を小数 8 桁で 1 行に出す。margin が負なら t1 を失っている
def detail(name, lg, t1):
    order = np.argsort(lg)[::-1]
    top1_idx, top2_idx = order[0], order[1]
    # index of the runner-up relative to t1 (excluding t1 itself)
    mask = np.ones(lg.size, dtype=bool)
    mask[t1] = False
    runner_idx = np.arange(lg.size)[mask][np.argmax(lg[mask])]
    margin = lg[t1] - lg[mask].max()
    print(f"{name:12s} FP32top1(id={t1}) logit={lg[t1]:.8f}  runner-up(id={runner_idx}) logit={lg[runner_idx]:.8f}  "
          f"margin={margin:.8f}  argmax_overall(id={top1_idx}) logit={lg[top1_idx]:.8f}  2nd(id={top2_idx}) logit={lg[top2_idx]:.8f}")


# （2026-10-18 追記）① 評価点 sys_en_BANANA の入力と判定位置を取る → ② FP32 で基準の 1 位の語を決める
# → ③ baseline・P12o・P12p-M・P12p-L の順に、同じ入力で margin を出す。メモリを空けるため、使い終えたセッションは del する
def main():
    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "sys_en_BANANA")
    ids, pos_want = p["ids"], p["eval_positions"][0]
    print(f"sys_en_BANANA: {len(ids)} tokens, eval position {pos_want}")

    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    ref_lg = logits_at(fp32, fp32, ids, pos_want)
    t1 = int(ref_lg.argmax())
    del fp32
    print(f"FP32 top1 token id = {t1}  logit = {ref_lg[t1]:.8f}")

    baseline = session(SOURCE)
    detail("baseline", logits_at(baseline, baseline, ids, pos_want), t1)

    mlp2later = session(MLP2LATER)
    detail("P12o(split)", logits_at(baseline, mlp2later, ids, pos_want), t1)
    del mlp2later

    for tag in ("M", "L"):
        variant = session(ML_MODELS[tag])
        detail(f"P12p-{tag}", logits_at(baseline, variant, ids, pos_want), t1)
        del variant


if __name__ == "__main__":
    main()
