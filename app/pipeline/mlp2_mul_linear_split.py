# -*- coding: utf-8 -*-
"""P12p: mul(SiLU(gate)×up出力)とlinear(down_proj出力)を分離して、どちらがP12oの崩れの主犯か切り分ける。

P12o: 位置0はbaseline(既存範囲)、位置1以降は56本(28層×mul/linear)すべてを後続範囲にすると、
全面崩壊(P12n)は消えたが「音楽をかけて」とBANANAだけ再現性高く崩れたままだった。
56本の中身は mul_X(SiLU(gate)×up出力)28本と linear_X(down_proj出力)28本の2系統。
どちらが「音楽をかけて」の崩れの主犯かを切り分ける。

  変種      pos0   mul pos1+   linear pos1+
  baseline  wide   wide        wide
  P12o      wide   narrow      narrow   (既存)
  P12p-M    wide   narrow      wide     (mulだけ後続範囲)
  P12p-L    wide   wide        narrow   (linearだけ後続範囲)

最初のscreeningは軽量化: 対話評価点のteacher forcing比較だけ行い、自由生成は省略する
(音楽=点B・BANANA・バナナ・点A・対話全体のmargin/KL/top1保持で十分に判別できる、既に感度の
高いcanary)。どちらかが明確に悪ければ、そちらだけ後で自由生成も確認する。

## 使い方

  python mlp2_mul_linear_split.py --phase build   # P12p-M / P12p-L のQDQを作る
  python mlp2_mul_linear_split.py --phase screen  # teacher forcingだけで高速比較
  python mlp2_mul_linear_split.py --phase generate --variant M   # 選んだ方だけ自由生成も見る
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 11.5 節（P12o の「音楽をかけて」の崩れの主犯探し）。
# 結果（results/P12p_20260916_mul_linear_split.md、teacher forcing のみ）:
#   M（mul だけ後続範囲）1位保持 83.9%・点B −0.01、L（linear だけ）76.8%・点B −0.50。
#   どちらでも「音楽をかけて」と BANANA の失敗は消えない。全体の悪化は linear（down_proj の出力）側が大きい。
#   → linear_20 1 本に絞る範囲掃引（linear20_range_sweep.py、P12r）へ続く。
# 入力:  baseline（SOURCE）、REP（results/P12e_20260915_temporal_qdq/）の build_audit.json の later の値、
#        mlp2_all_layers_later.TARGET_PAIRS（28 層の (mul, linear) の名前の組）。
# 出力:  work/p12p_mul_linear_split/{mulonly,linearonly}_later_qdq.onnx、
#        results/P12p_20260916_mul_linear_split/build_audit_{M,L}.json・screen_report.json・generate_{M,L}.json。
# ---------------------------------------------------------------------------
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
from mlp2_all_layers_later import TARGET_PAIRS  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

OUT = ROOT / "results/P12p_20260916_mul_linear_split"
WORK = ROOT / "work/p12p_mul_linear_split"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"

MODELS = {"M": WORK / "mulonly_later_qdq.onnx", "L": WORK / "linearonly_later_qdq.onnx"}


def emit(obj):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def build_one(tag, tensor_index):
    """tensor_index: 0=mul (TARGET_PAIRSの1個目), 1=linear (2個目) だけ後続範囲にする。"""
    # 【説明】28 本の scale・zero point を P12e の後続用の値に差し替えたモデルを MODELS[tag] に保存し、
    #   build_audit_<tag>.json に変えた組を残す。重み・ノードが変わっていないことを sha256 で確かめる
    #   （mlp2_all_layers_later.build と同じ手順で、対象だけを半分にしたもの）。
    audit = json.loads((REP / "build_audit.json").read_text(encoding="utf-8"))
    later = {c["scale_name"]: c for c in audit["later"]["changes"] if len(c["sources"]) == 1}
    selected = [pair[tensor_index] + "_scale" for pair in TARGET_PAIRS]
    missing = [s for s in selected if s not in later]
    assert not missing, missing
    assert len(selected) == 28

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
    result = {"tag": tag, "changed_groups": len(changes), "changes": changes,
              "source_sha256": digest(SOURCE), "model_sha256": digest(path), "node_sha256": node_hash}
    (OUT / f"build_audit_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", "tag": tag, "wrote": str(path), "changed_groups": len(changes)})
    del m, init
    gc.collect()


def build():
    """M（mul だけ）と L（linear だけ）の 2 つのモデルを作る。"""
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    build_one("M", 0)  # mul_X だけ後続範囲
    build_one("L", 1)  # linear_X だけ後続範囲


def summarize(sc):
    """score の結果を 1 行の文字列にする（1位保持・margin 保持・最小 margin・KL・点ごとの margin）。"""
    a = sc["ALL"]
    return (f"1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
            f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}")


def screen():
    """自由生成をせず、teacher forcing の採点だけで M と L を比べる（速く判別するため）。

    どちらも位置0 は baseline、位置1以降が変種（P12o と同じ 2 モデル構成）。出力: screen_report.json。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "screen", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    fp32 = session(OPS_DIR / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    gc.collect()
    emit({"phase": "screen", "step": "fp32_reference_done"})

    base = session(SOURCE)
    results = {}
    for tag, path in MODELS.items():
        variant = session(path)
        sc = score(run_points(Pipeline(None, base, variant, "none"), pts), ref)
        results[tag] = sc
        print(f"P12p-{tag}      {summarize(sc)}")
        del variant
        gc.collect()
    del base
    gc.collect()

    (OUT / "screen_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "screen", "wrote": str(OUT / "screen_report.json")})


def generate(variant):
    """選んだ変種（M か L）だけ、元の日本語 3 ターンで自由生成する。出力: generate_<variant>.json。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    base = session(SOURCE)
    other = session(MODELS[variant])
    pipe = Pipeline(None, base, other, "none")
    gen = free_generation(pipe, tok)
    for t in gen:
        print(f"  turn{t['turn']}: {t['text']!r}")
    out = OUT / f"generate_{variant}.json"
    out.write_text(json.dumps(gen, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "generate", "variant": variant, "wrote": str(out)})


# 引数: --phase build / screen / generate（generate の時は --variant M か L が要る）。
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "screen", "generate"], required=True)
    ap.add_argument("--variant", choices=["M", "L"])
    a = ap.parse_args()
    if a.phase == "build":
        build()
    elif a.phase == "screen":
        screen()
    elif a.phase == "generate":
        assert a.variant, "--phase generate には --variant M|L が必要"
        generate(a.variant)
