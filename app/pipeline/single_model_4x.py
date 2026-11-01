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
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 11.7b 節「『実機に載せられる形か』を先に確かめる」（P14 でクローズした後の追試）。
# 結果（results/P16_20260917_single_model_4x.md）: 1 モデルにまとめると完全に崩れた。
#   A1（層2 のみ 4x・1 モデル）1位保持 0.0%・KL 18.24、B1（全 28 層）0.0%・KL 16.25、どちらも文字化け。
#   A2（同じ scale で位置0 だけ baseline・2 モデル）は 83.9% で正常。位置0 の 8142 が 4x の範囲（−15.7〜+31.4）で
#   飽和し、KV の先頭スロットとして最後まで参照され続けるため。→ 4x は実機に載せられる形が無い。
# 入力:  SOURCE（work/ops/qwen3_dec28c256/qwen3_dec28c256_final_qdq.onnx）、
#        REP = results/P12e_20260915_temporal_qdq/ の calibration_ranges.json（後続の観測範囲）と qparam_map.json。
# 出力:  work/p16_single_4x/<変種>_qdq.onnx、results/P16_20260917_single_model_4x/build_<変種>.json・evaluate.json。
# 動かし方: & $py -B app/pipeline/single_model_4x.py --phase build  →  --phase evaluate
# ---------------------------------------------------------------------------
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
# MARGIN: 後続トークンで観測した範囲を何倍に広げるか（6.0 が P12r の「4x」。1x が 1.5 に当たる）。
# DOWN_PROJ: 層 L の down_proj の出力の名前。export のノード番号は 1 層に 7 つずつ進むので linear_{7L+6}。
# VARIANTS: 変種の名前 → scale/zero を差し替えるテンソルの一覧。
MARGIN = 6.0                                   # P12r の 4x
DOWN_PROJ = {L: f"linear_{7 * L + 6}" for L in range(28)}   # 層2 → linear_20
VARIANTS = {"A1_layer2_only": [DOWN_PROJ[2]], "B1_all_layers": [DOWN_PROJ[L] for L in range(28)]}


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def build(tag, names):
    """既存の QDQ から、names のテンソルの scale・zero point だけを「後続の観測範囲 × MARGIN」に差し替えたモデルを作る。

    引数: tag = 変種の名前、names = 差し替えるテンソル名のリスト。戻り値: 作ったモデルのパス。
    記録（build_<tag>.json）: テンソルごとの観測範囲・新旧の scale・表せる範囲・何倍細かくなったか（narrower_by）。
    最後に、重みとノードが 1 バイトも変わっていないことを sha256 で確かめる（変える変数を scale/zero だけにする）。
    """
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
    """FP32 の基準と比べて、baseline・A1・A2・B1 を同じ評価点で採点し、元の日本語 3 ターンで自由生成する。

    Pipeline(fp32, first, later, mode) の first は位置0 を通すモデル、later は位置1以降のモデル。
    A1・B1 は first も later も同じ（1 モデル）、A2 は first だけ baseline（2 モデル）。
    判定の順序は ① 自由生成 → ② 回答の最初のトークンの margin → ③ 集計指標（診断用）（P12t・StudyMemo 11.8 (4)）。
    """
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


# 引数: --phase build（2 つの変種を作る）/ evaluate（採点と自由生成）。
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["build", "evaluate"], required=True)
    a = ap.parse_args()
    if a.phase == "build":
        for tag, names in VARIANTS.items():
            build(tag, names)
    else:
        evaluate()
