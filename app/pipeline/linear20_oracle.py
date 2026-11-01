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

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）。P12r/P12s（linear20_range_sweep.py・layer2_linear20_4x_device.py）の続き
- 結果（results/P12v_20260916_linear20_oracle.md）: 量子化を完全に外すと baseline より悪化した（1位保持 85.7%→76.8%、
  点B +3.35→−1.02、自由生成の turn1 は「エアコンを鳴らすか？」という新しい壊れ方）。Host で不合格のため実機（build-device の先）は測っていない
  → 「誤差の打ち消し合い」を疑って P12w（add26_causal_probe.py）へ
- 入力: baseline の QDQ（SOURCE）・FP32（OPS_DIR の prepped.onnx）・層2 の部分モデル（work/q3_l2/l2ctl_qdq.onnx。build-device のとき）
- 出力: work/p12v_linear20_oracle/oracle_full28_qdq.onnx（ORACLE_FULL。P12w〜P13 の道具が「oracle」として import する）、
  work/q3_l2/l2ctl_oracle_qdq.onnx、results/P12v_20260916_linear20_oracle/ の build_audit*.json・report.json
- プロジェクトの直下からなら python -m app.pipeline.linear20_oracle --phase build のようにも動く
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

# 結果・作業の置き場。ORACLE_FULL は 28 層の oracle モデル、ORACLE_L2 は層2 の部分モデルの oracle 版
OUT = ROOT / "results/P12v_20260916_linear20_oracle"
WORK = ROOT / "work/p12v_linear20_oracle"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"
ORACLE_FULL = WORK / "oracle_full28_qdq.onnx"

WORK_L2 = ROOT / "work/q3_l2"
L2_BASELINE = WORK_L2 / "l2ctl_qdq.onnx"
ORACLE_L2 = WORK_L2 / "l2ctl_oracle_qdq.onnx"


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def bypass_linear20_qdq(model_path, out_path):
    """linear_20のQuantizeLinear/DequantizeLinear 1組を取り除き、下流を生のfloatに直結する。"""
    # （2026-10-18 追記）model_path → out_path。重み（外部データ）は読まずにグラフだけ書き換える。
    # ① linear_20 の Q と、その後ろの DQ を探す → ② DQ の出力の使い手が Add 1 つだけであることを確かめる
    # → ③ その Add の入力を linear_20（量子化前の float）に付け替え、Q と DQ を消す → ④ 検査して保存
    # 戻り値: 消したノードの名前と、付け替えの説明
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
    """28 層の baseline から oracle モデルを作り、元と結果の sha256 を記録する（Host の評価用）。"""
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    info = bypass_linear20_qdq(SOURCE, ORACLE_FULL)
    result = {**info, "source_sha256": digest(SOURCE), "model_sha256": digest(ORACLE_FULL)}
    (OUT / "build_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", **result, "wrote": str(ORACLE_FULL)})


def build_device():
    """層2 の部分モデルから oracle 版を作る（実機用。P12v では Host で不合格だったので使っていない）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    assert L2_BASELINE.exists(), "先に P12m/P12j の層2部分モデルが必要 (work/q3_l2/l2ctl_qdq.onnx)"
    info = bypass_linear20_qdq(L2_BASELINE, ORACLE_L2)
    result = {**info, "source_sha256": digest(L2_BASELINE), "model_sha256": digest(ORACLE_L2)}
    (OUT / "build_audit_device.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build-device", **result, "wrote": str(ORACLE_L2)})


def summarize(sc):
    """score の結果を 1 行の要約にする。"""
    a = sc["ALL"]
    return (f"1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
            f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}")


def evaluate():
    """FP32 の基準を作り、baseline と oracle を P12i と同じ評価点（teacher forcing）と自由生成で評価する（全位置を同じモデルで流す）。"""
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


# コマンドの入口。--phase で build / evaluate / build-device のどれか 1 段を流す
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "evaluate", "build-device"], required=True)
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate, "build-device": build_device}[a.phase]()
