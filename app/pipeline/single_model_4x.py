# -*- coding: utf-8 -*-
"""P16: 「実機がよければよい」を確かめるための前段。**1モデルで 4x range** が Host で成立するか（P16-1）。

## なぜこれが要るのか

P12r の 4x（Host の自由生成が唯一きれいだった設定）は、実際には
**位置0 は baseline モデル／位置1以降は 4x モデル**という 2モデル構成だった。
実機では会話の途中でモデルを切り替えられない（context binary は1本、セッションも1つ）。
したがって実機で試すには **位置0 も 4x で通す1モデル**である必要があり、その条件は未検証。

位置0 には巨大活性（層2 MLP 出力で 8142）が出るので、4x（後続レンジ×6.0 ＝ おおむね ±20〜40）では
**位置0 が確実に飽和する**。それが致命傷になるかどうかを Host で先に見る。

## 変種（いずれも既存 QDQ から scale/zero だけを差し替え。重み・ノードは不変）

  A1  層2 の `linear_20` だけ 4x（margin 6.0）。**位置0 も同じモデル**
  B1  全28層の down_proj（`linear_{7L+6}`）を 4x。**位置0 も同じモデル**
  A2  参考: A1 と同じ範囲だが、位置0 だけ baseline モデルを使う2モデル構成（= P12r の 4x 相当）

判定は P12t で確立した順序: ①自由生成 → ②回答開始 token の margin → ③集計指標（診断用）。
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, SOURCE, digest, fit
from block_revert_ab import eval_points, score, REP
from fp32_prefix_eval import Pipeline, run_points, free_generation

import numpy as np
import onnx
from onnx import numpy_helper

OUT = ROOT / "results/P16_20260917_single_model_4x"
WORK = ROOT / "work/p16_single_4x"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"
MARGIN = 6.0                                   # P12r の 4x
DOWN_PROJ = {L: f"linear_{7 * L + 6}" for L in range(28)}   # 層2 → linear_20
VARIANTS = {"A1_layer2_only": [DOWN_PROJ[2]], "B1_all_layers": [DOWN_PROJ[L] for L in range(28)]}


def emit(o):
    print(json.dumps(o, ensure_ascii=False), flush=True)


def build(tag, names):
    calib = json.loads((REP / "calibration_ranges.json").read_text(encoding="utf-8"))["ranges"]["later"]
    qmap = json.loads((REP / "qparam_map.json").read_text(encoding="utf-8"))["groups"]
    m = onnx.load(str(SOURCE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    original = {t.name: hashlib.sha256(t.SerializeToString()).hexdigest() for t in m.graph.initializer}
    node_hash = hashlib.sha256(b"".join(n.SerializeToString() for n in m.graph.node)).hexdigest()
    allowed, rows = set(), []
    for name in names:
        lo, hi = calib[name]
        g = next(g for g in qmap if name in g["sources"])
        dt = np.dtype(g["dtype"])
        qmax = np.iinfo(dt).max
        s, z = fit(lo, hi, qmax, MARGIN)
        old_s = float(numpy_helper.to_array(init[g["scale_name"]]).ravel()[0])
        for nm, val, cast in ((g["scale_name"], s, np.float32), (g["zero_name"], z, dt)):
            shape = numpy_helper.to_array(init[nm]).shape
            init[nm].CopyFrom(numpy_helper.from_array(np.asarray(val, dtype=cast).reshape(shape), nm))
            allowed.add(nm)
        rows.append({"tensor": name, "observed": [lo, hi], "old_scale": old_s, "new_scale": float(s),
                     "zero": int(z), "represented": [-z * float(s), (qmax - z) * float(s)],
                     "narrower_by": old_s / float(s)})
    # 重み・ノードが変わっていないことを必ず確かめる（scale/zero 以外は1バイトも動かさない）
    unexpected = [t.name for t in m.graph.initializer if t.name not in allowed and
                  hashlib.sha256(t.SerializeToString()).hexdigest() != original[t.name]]
    assert not unexpected, unexpected
    assert hashlib.sha256(b"".join(n.SerializeToString() for n in m.graph.node)).hexdigest() == node_hash
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{tag}_qdq.onnx"
    onnx.save(m, str(path))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"build_{tag}.json").write_text(json.dumps(
        {"tag": tag, "margin": MARGIN, "tensors": rows, "source_sha256": digest(SOURCE),
         "model_sha256": digest(path), "node_sha256": node_hash}, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", "tag": tag, "tensors": len(rows),
          "narrower_by_median": float(np.median([r["narrower_by"] for r in rows])),
          "represented_layer2": rows[0]["represented"] if names[0] == DOWN_PROJ[2] else None})
    del m, init
    gc.collect()
    return path


def evaluate():
    tok = tokenizer()
    pts = eval_points(tok)
    fp32 = session(OPS_DIR / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    base = session(ROOT / "work/ops/qwen3_dec28c256/qwen3_dec28c256_final_qdq.onnx")
    report = {}
    plan = [("baseline", base, base)]
    for tag in VARIANTS:
        s = session(WORK / f"{tag}_qdq.onnx")
        plan.append((tag, s, s))                       # 1モデル: 位置0 も同じモデル
        if tag == "A1_layer2_only":
            plan.append(("A2_layer2_pos0_baseline", base, s))   # 参考: 2モデル（P12r の 4x 相当）
    for tag, first, later in plan:
        t0 = time.time()
        pipe = Pipeline(None, first, later, "none")
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        a = sc["ALL"]
        report[tag] = {"score": sc, "free_generation": gen}
        emit({"phase": "evaluate", "tag": tag, "top1_keep": a["top1_keep"], "retention": a["retention_mean"],
              "margin_min": a["margin_min"], "kl": a["kl_mean"],
              "point_B": sc["point_B_music"]["margins"][0], "banana_en": sc["sys_en_BANANA"]["margins"][0],
              "banana_ja": sc["sys_ja_banana"]["margins"][0], "sec": round(time.time() - t0)})
        for t in gen:
            emit({"phase": "gen", "tag": tag, "turn": t["turn"], "text": t["text"]})
        (OUT / "evaluate.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        del pipe
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["build", "evaluate"], required=True)
    a = ap.parse_args()
    if a.phase == "build":
        for tag, names in VARIANTS.items():
            build(tag, names)
    else:
        evaluate()
