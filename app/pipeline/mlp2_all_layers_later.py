# -*- coding: utf-8 -*-
"""P12n: 層2 MLP の mul_41/linear_20 と同じパターンを28層全体に適用し、Hostで崩れないか確認する。

P12m: 層2部分モデルで `mul_41`/`linear_20`（SiLU(gate)×up 出力・down_proj 出力）だけを
後続トークン用の範囲に差し替えると、add_26 の Host↔HTP 誤差が 22%→9%に下がった。
28層本番ビルド（136分）に進む前に、同じ調整を **28層全体の該当2本×28層＝56テンソル** に適用した
単一QDQモデルを作り、P12i/P12k/P12l と同じ対話評価点・自由生成でHost上の品質が崩れないか確認する。

- scale/zero point は新規較正しない。P12e で計算済みの「後続」範囲（margin 1.5 でフィット済み、
  `results/P12e_20260915_temporal_qdq/build_audit.json` の later フェーズ）をそのまま56本だけ適用する。
- 全位置（先頭トークン含む）に同じモデルを使う。層2narrowの実機実験と同条件。
  先頭トークンの巨大値をこの範囲でクリップして品質が保たれるかどうかが、この実験の核心の問い
  （P12eの全体later/sharedは崩れたが、あれは2,226組すべてを変えた場合。今回は56本だけ）。

## 使い方

  python mlp2_all_layers_later.py --phase build     # 56テンソルだけ後続範囲にしたQDQを作る
  python mlp2_all_layers_later.py --phase evaluate  # 対話評価点・自由生成をHostで比較する
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, OPS, SOURCE, digest  # noqa: E402
from block_revert_ab import eval_points, score, ART, REP, P12D  # noqa: E402,F401
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

OUT = ROOT / "results/P12n_20260916_mlp2_all_layers_later"
WORK = ROOT / "work/p12n_mlp2later"
MODEL = WORK / "mlp2later_qdq.onnx"

# 層2 (mul_41, linear_20) と同じ役割の (SiLU(gate)×up 出力, down_proj 出力) を28層分。
# layer2_probe.py / P12j の対象と同じ、グラフ上の位置で機械的に特定した組（0始まり、層0が mul_13/linear_6）。
TARGET_PAIRS = [
    ("mul_13", "linear_6"), ("mul_27", "linear_13"), ("mul_41", "linear_20"), ("mul_55", "linear_27"),
    ("mul_69", "linear_34"), ("mul_83", "linear_41"), ("mul_97", "linear_48"), ("mul_111", "linear_55"),
    ("mul_125", "linear_62"), ("mul_139", "linear_69"), ("mul_153", "linear_76"), ("mul_167", "linear_83"),
    ("mul_181", "linear_90"), ("mul_195", "linear_97"), ("mul_209", "linear_104"), ("mul_223", "linear_111"),
    ("mul_237", "linear_118"), ("mul_251", "linear_125"), ("mul_265", "linear_132"), ("mul_279", "linear_139"),
    ("mul_293", "linear_146"), ("mul_307", "linear_153"), ("mul_321", "linear_160"), ("mul_335", "linear_167"),
    ("mul_349", "linear_174"), ("mul_363", "linear_181"), ("mul_377", "linear_188"), ("mul_391", "linear_195"),
]


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((REP / "build_audit.json").read_text(encoding="utf-8"))
    later = {c["scale_name"]: c for c in audit["later"]["changes"] if len(c["sources"]) == 1}
    selected = []
    for mu, li in TARGET_PAIRS:
        selected += [mu + "_scale", li + "_scale"]
    missing = [s for s in selected if s not in later]
    assert not missing, missing
    assert len(selected) == 56

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
    assert len(m.graph.node) == len(onnx.load(str(SOURCE), load_external_data=False).graph.node)

    onnx.save(m, str(MODEL))
    onnx.checker.check_model(str(MODEL))
    result = {"changed_groups": len(changes), "changes": changes, "source_sha256": digest(SOURCE),
              "model_sha256": digest(MODEL), "node_sha256": node_hash}
    (OUT / "build_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", "wrote": str(MODEL), "changed_groups": len(changes),
          "model_bytes": MODEL.stat().st_size})
    del m, init
    gc.collect()


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "evaluate", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    gc.collect()
    emit({"phase": "evaluate", "step": "fp32_reference_done"})

    results, gens = {}, {}
    variants = [("baseline", SOURCE), ("mlp2later", MODEL)]
    for name, path in variants:
        s = session(path)
        results[name] = score(run_points(Pipeline(None, s, s, "none"), pts), ref)
        gens[name] = free_generation(Pipeline(None, s, s, "none"), tok)
        del s
        gc.collect()
        a = results[name]["ALL"]
        print(f"{name:12s} 1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
              f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
              f"点B {results[name]['point_B_music']['margins'][0]:+.2f} "
              f"点A {results[name]['point_A_first_div']['margins'][0]:+.2f} "
              f"BANANA {results[name]['sys_en_BANANA']['margins'][0]:+.2f} "
              f"バナナ {results[name]['sys_ja_banana']['margins'][0]:+.2f}")
        for t in gens[name]:
            print(f"      turn{t['turn']}: {t['text']!r}")
        (OUT / "report.json").write_text(
            json.dumps({"scores": results, "free_generation": gens}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    emit({"phase": "evaluate", "wrote": str(OUT / "report.json")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "evaluate"], required=True)
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate}[a.phase]()
