# -*- coding: utf-8 -*-
"""BANANA(sys_en_BANANA)評価点のlogit marginを高精度で見る(P12o/P12p-M/P12p-Lで margin=+0.00 だった件)。

score()の表示は小数2桁に丸めているため、+0.00 が厳密なゼロなのか±0.005程度のばらつきなのかを
区別できない。ここではtop1/top2のtoken IDと、logit差を小数6桁以上で出す。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS, SOURCE  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from mlp2_all_layers_later import MODEL as MLP2LATER  # noqa: E402
from mlp2_mul_linear_split import MODELS as ML_MODELS  # noqa: E402

import numpy as np  # noqa: E402


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
