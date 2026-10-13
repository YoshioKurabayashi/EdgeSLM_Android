# -*- coding: utf-8 -*-
"""P12v: linear_20 oracle — QDQを迂回してdown_proj出力を高精度(float32)のままresidual addへ渡す。

P12r/P12sで、linear_20のper-tensor rangeを段階的に広げても(1x〜8x)、Host自由生成と
HTP誤差改善を両立できなかった。これは「range幅の選び方が悪い」のではなく、そもそも
「linear_20の出力を1本のper-tensor scaleへ押し込めること自体」が問題である可能性を検証する。

## 迂回の設計

linear_20（down_projのMatMul出力）は現状:
  linear_20 --QuantizeLinear--> linear_20_QuantizeLinear_Output --DequantizeLinear-->
  linear_20_DequantizeLinear_Output --(唯一の消費者)--> node_add_26(Add)

QuantizeLinear/DequantizeLinearの1組を取り除き、node_add_26の入力を linear_20（float32、
量子化前）に直結する。add_26自体の量子化（下流の層3へ渡す際のQ/DQ）はそのまま残すので、
「量子化境界を層3側まで後ろにずらした」構成になる。down_projの積和精度ではなく、
「1本のper-tensor scaleに押し込めること」だけを取り除いている点が重要。

## 使い方

  python linear20_oracle.py --phase build          # Host用: 28層モデルでoracle化
  python linear20_oracle.py --phase evaluate        # Host: teacher forcing + 自由生成
  python linear20_oracle.py --phase build-device     # 実機用: 層2部分モデルでoracle化
"""
from pathlib import Path
import argparse
import gc
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, SOURCE, digest  # noqa: E402
from block_revert_ab import eval_points, score  # noqa: E402
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402

import onnx  # noqa: E402

OUT = ROOT / "results/P12v_20260916_linear20_oracle"
WORK = ROOT / "work/p12v_linear20_oracle"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"
ORACLE_FULL = WORK / "oracle_full28_qdq.onnx"

WORK_L2 = ROOT / "work/q3_l2"
L2_BASELINE = WORK_L2 / "l2ctl_qdq.onnx"
ORACLE_L2 = WORK_L2 / "l2ctl_oracle_qdq.onnx"


def emit(o):
    print(json.dumps(o, ensure_ascii=False), flush=True)


def bypass_linear20_qdq(model_path, out_path):
    """linear_20のQuantizeLinear/DequantizeLinear 1組を取り除き、下流を生のfloatに直結する。"""
    m = onnx.load(str(model_path), load_external_data=False)
    g = m.graph
    q = next(n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == "linear_20")
    dq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == q.output[0])
    dq_out = dq.output[0]
    consumers = [n for n in g.node if dq_out in n.input]
    assert len(consumers) == 1 and consumers[0].op_type == "Add", consumers
    add_node = consumers[0]
    for i, name in enumerate(add_node.input):
        if name == dq_out:
            add_node.input[i] = "linear_20"
    g.node.remove(q)
    g.node.remove(dq)
    onnx.checker.check_model(m)
    onnx.save(m, str(out_path))
    return {"removed_nodes": [q.name, dq.name], "rewired": f"{add_node.name}.input <- linear_20 (bypass {dq_out})"}


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    info = bypass_linear20_qdq(SOURCE, ORACLE_FULL)
    result = {**info, "source_sha256": digest(SOURCE), "model_sha256": digest(ORACLE_FULL)}
    (OUT / "build_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", **result, "wrote": str(ORACLE_FULL)})


def build_device():
    OUT.mkdir(parents=True, exist_ok=True)
    assert L2_BASELINE.exists(), "先に P12m/P12j の層2部分モデルが必要 (work/q3_l2/l2ctl_qdq.onnx)"
    info = bypass_linear20_qdq(L2_BASELINE, ORACLE_L2)
    result = {**info, "source_sha256": digest(L2_BASELINE), "model_sha256": digest(ORACLE_L2)}
    (OUT / "build_audit_device.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build-device", **result, "wrote": str(ORACLE_L2)})


def summarize(sc):
    a = sc["ALL"]
    return (f"1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
            f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}")


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "evaluate", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    fp32 = session(OPS_DIR / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    gc.collect()
    emit({"phase": "evaluate", "step": "fp32_reference_done"})

    results, gens = {}, {}
    variants = [("baseline", SOURCE), ("oracle", ORACLE_FULL)]
    for name, path in variants:
        s = session(path)
        pipe = Pipeline(None, s, s, "none")
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        results[name] = sc
        gens[name] = gen
        print(f"{name:10s} {summarize(sc)}", flush=True)
        for t in gen:
            print(f"      turn{t['turn']}: {t['text']!r}", flush=True)
        del s, pipe
        gc.collect()
        (OUT / "report.json").write_text(json.dumps({"scores": results, "free_generation": gens},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "evaluate", "wrote": str(OUT / "report.json")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "evaluate", "build-device"], required=True)
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate, "build-device": build_device}[a.phase]()
