# -*- coding: utf-8 -*-
"""P12q: 層2だけ narrow にした3変種(L2-M/L2-L/L2-ML)をHostで評価する。

P12p: mul_X 28本/linear_X 28本を分離すると、linearの方が全体的に有害だが、
「音楽をかけて」「BANANA」の失敗はどちらの部分集合にも共通して残った。
mul_X/linear_Xは層番号で1対1対応しているため、この共通の失敗が「層2という特定層が
両方の部分集合に必ず含まれているから」という可能性を検証する。

P12mでは、層2の mul_41/linear_20 だけを後続範囲にすると、実機でadd_26のHost↔HTP誤差が
21.95%→8.61%に改善した(層2部分モデルでの実機検証)。これがHost全体(28層モデル)でも
自由生成を壊さずに使える最小パッチになるかを確認する。

  変種    pos0   層2 mul_41   層2 linear_20   他27層
  baseline wide  wide         wide            wide
  L2-M    wide   narrow       wide            wide
  L2-L    wide   wide         narrow          wide
  L2-ML   wide   narrow       narrow          wide

判定は「① L2-ML単独で崩れるか」(十分性)と、後続でP12o-minus-Layer2を見る「② 全層narrowから
層2だけ戻すと回復するか」(必要性)の両方向で行う。本スクリプトは①のみ(L2-M/L2-L/L2-MLの3変種)。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節・11.8 節）
- 結果: results/P12t_20260916_session_summary_layer2_htp_tradeoff.md の「P12q」の節（P12t が挙げる P12q の md は results に見当たらず、
  数字は results/P12q_20260916_layer2_only_variants/report.json）。1位保持は L2-M 85.7%・L2-L 85.7%・L2-ML 87.5%（baseline 超え）
  なのに、自由生成は 3 つとも崩れた（L2-ML は turn1 が turn0 を繰り返す）。「集計指標は生成の健全性を予測しない」の例
- 入力: baseline の QDQ（SOURCE。重みを中に持つ形であること）、P12e の build_audit.json（REP。後続の範囲の値）、
  work/ops/qwen3_dec28c256/ の prepped.onnx（FP32）
- 出力: work/p12q_layer2_only/l2only_<M|L|ML>_qdq.onnx、results/P12q_20260916_layer2_only_variants/ の build_audit_<tag>.json・report.json
- 「narrow」= P12e の較正の「後続トークンだけ」の範囲（margin 1.5）。「wide」= 既存の範囲（位置0 の巨大値を含む）。
  評価では位置0 をいつも baseline で流す（Pipeline の first = baseline、later = 変種）
- プロジェクトの直下からなら python -m app.pipeline.layer2_only_variants --phase build のようにも動く
## 使い方

  python layer2_only_variants.py --phase build
  python layer2_only_variants.py --phase evaluate   # teacher forcing + 自由生成
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, SOURCE, digest  # noqa: E402
from block_revert_ab import eval_points, score, REP  # noqa: E402
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

# 結果・作ったモデルの置き場と、FP32 のモデルの置き場
OUT = ROOT / "results/P12q_20260916_layer2_only_variants"
WORK = ROOT / "work/p12q_layer2_only"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"

LAYER2_MUL, LAYER2_LINEAR = "mul_41", "linear_20"  # P12j/P12mと同じ層2の対象

# 変種の名前 → narrow にするテンソル。MODELS はその保存先
VARIANTS = {
    "M": [LAYER2_MUL],
    "L": [LAYER2_LINEAR],
    "ML": [LAYER2_MUL, LAYER2_LINEAR],
}
MODELS = {tag: WORK / f"l2only_{tag}_qdq.onnx" for tag in VARIANTS}


def emit(obj):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def build_one(tag, tensors):
    """baseline の QDQ のうち、tensors の Q/DQ の scale・zero point だけを P12e の「後続」の値にしたモデルを作る。

    置き換えを許した初期化子以外が変わっていないこと・ノードが同じことを hash で確かめてから保存し、監査の記録を残す
    （focused_qdq_experiment.build_focused と同じやり方）
    """
    audit = json.loads((REP / "build_audit.json").read_text(encoding="utf-8"))
    later = {c["scale_name"]: c for c in audit["later"]["changes"] if len(c["sources"]) == 1}
    selected = [t + "_scale" for t in tensors]
    missing = [s for s in selected if s not in later]
    assert not missing, missing

    m = onnx.load(str(SOURCE), load_external_data=False)
    assert not any(t.external_data for t in m.graph.initializer), "Expected self-contained QDQ"
    init = {t.name: t for t in m.graph.initializer}
    original = {t.name: hashlib.sha256(t.SerializeToString()).hexdigest() for t in m.graph.initializer}
    node_hash = hashlib.sha256(b"".join(n.SerializeToString() for n in m.graph.node)).hexdigest()

    allowed = set()
    changes = []
    for name in selected:
        c = later[name]
        for nm, value in [(c["scale_name"], c["scale"]), (c["zero_name"], c["zero"])]:
            before = numpy_helper.to_array(init[nm])
            init[nm].CopyFrom(numpy_helper.from_array(
                np.asarray(value, dtype=before.dtype).reshape(before.shape), nm))
            allowed.add(nm)
        changes.append(c)

    unexpected = [t.name for t in m.graph.initializer if t.name not in allowed and
                  hashlib.sha256(t.SerializeToString()).hexdigest() != original[t.name]]
    assert not unexpected, unexpected
    assert hashlib.sha256(b"".join(n.SerializeToString() for n in m.graph.node)).hexdigest() == node_hash

    path = MODELS[tag]
    onnx.save(m, str(path))
    onnx.checker.check_model(str(path))
    result = {"tag": tag, "tensors": tensors, "changed_groups": len(changes), "changes": changes,
              "source_sha256": digest(SOURCE), "model_sha256": digest(path), "node_sha256": node_hash}
    (OUT / f"build_audit_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", "tag": tag, "tensors": tensors, "wrote": str(path)})
    del m, init
    gc.collect()


def build():
    """3 変種（M・L・ML）をすべて作る。"""
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, tensors in VARIANTS.items():
        build_one(tag, tensors)


def summarize(sc):
    """score の結果を 1 行の要約にする。"""
    a = sc["ALL"]
    return (f"1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
            f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}")


def evaluate():
    """FP32 の基準を作り、3 変種を P12i と同じ評価点・指標（teacher forcing）と自由生成で評価して report.json に書く。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "evaluate", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    fp32 = session(OPS_DIR / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    gc.collect()
    emit({"phase": "evaluate", "step": "fp32_reference_done"})

    base = session(SOURCE)
    results, gens = {}, {}
    for tag in VARIANTS:
        variant = session(MODELS[tag])
        pipe = Pipeline(None, base, variant, "none")
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        results[tag] = sc
        gens[tag] = gen
        print(f"L2-{tag:3s}  {summarize(sc)}", flush=True)
        for t in gen:
            print(f"      turn{t['turn']}: {t['text']!r}", flush=True)
        del variant, pipe
        gc.collect()
        (OUT / "report.json").write_text(json.dumps({"scores": results, "free_generation": gens},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    del base
    gc.collect()
    emit({"phase": "evaluate", "wrote": str(OUT / "report.json")})


# コマンドの入口。--phase build か evaluate のどちらか 1 段を流す
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "evaluate"], required=True)
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate}[a.phase]()
