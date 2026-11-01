# -*- coding: utf-8 -*-
"""P12r: 層2 linear_20(down_proj出力)だけのrange幅を段階的に動かし、Host自由生成が
どこで正常化するかを見る(P12qで narrow・baseline は判定済み。sweetspotがあるかを1回で決着)。

mul_41はbaseline固定(P12qでmul_41単独は文字化け、linear_20単独は指示無視という別の壊れ方をした。
今回は主犯と見ているlinear_20だけを動かす。変数を1つにする)。

  変種       linear_20のrange                              備考
  1x(narrow) margin=1.5 (P12e後続較正の素の値)              = P12qのL2-Lと同一。再計算せず再掲。
  2x         margin=3.0 (観測lo/hiを2倍に)
  4x         margin=6.0
  8x         margin=12.0
  baseline   既存range(変更なし)                            再計算せず再掲。

margin(観測min/maxへの掛け率)を2/4/8倍することで、narrowの「観測値ベースの広さ」をNx倍に
広げる。既存(baseline)はnarrowの約632倍広いので、8xでもbaselineにはまだ遠い
(意図的:baselineは別枠のアンカーとして5点目に置く)。

判定順序(ユーザー指定): ①自由生成 ②回答開始token(点B等) ③KV/履歴 ④System Prompt
⑤top1/margin/KL(診断用、合否には使わない)

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節・11.8 節「合否は自由生成で決める」。11 章の枠は build_one）
- 結果（results/P12r_20260916_linear20_range_sweep.md）: 1x〜8x の 7 点で、teacher forcing と自由生成の両方で安定したのは 4x だけ。
  2x・3x・3.5x・3.75x は別々の形で自由生成が崩れ、8x はバナナ/BANANA が急落。4x の実機の効果は小さかった（P12s）ので、
  範囲の調整はここで打ち切り
- ★2026-10-18 注: 上の表は計画の段階のもの。いまのコードの SWEEP_MARGINS は最後に足した 3.75x（margin 5.625）の 1 点だけで、
  2x・3x・3.5x・4x・8x は流すたびに書き換えて回した（推定。build_audit_<tag>.json は 6 つ残っている）。
  report.json は流すたびに上書きされるので、いま残っているのは最後の回の分だけと思われる。全点の数字は md の表にある
- 入力: baseline の QDQ（SOURCE。重みを中に持つ形）、P12e の calibration_ranges.json・qparam_map.json（REP）、
  P12q の report.json（1x の結果を再掲するため）、FP32（OPS_DIR の prepped.onnx）
- 出力: work/p12r_linear20_sweep/l20_<tag>_qdq.onnx、results/P12r_20260916_linear20_range_sweep/ の build_audit_<tag>.json・report.json
- 評価は位置0 をいつも baseline で流す（Pipeline の first = baseline、later = 変種）
- プロジェクトの直下からなら python -m app.pipeline.linear20_range_sweep --phase build のようにも動く
## 使い方

  python linear20_range_sweep.py --phase build
  python linear20_range_sweep.py --phase evaluate
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, SOURCE, digest, fit  # noqa: E402
from block_revert_ab import eval_points, score, REP  # noqa: E402
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402
from layer2_only_variants import OUT as P12Q_OUT  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

# 結果・作ったモデル・FP32 の置き場
OUT = ROOT / "results/P12r_20260916_linear20_range_sweep"
WORK = ROOT / "work/p12r_linear20_sweep"
OPS_DIR = ROOT / "work/ops/qwen3_dec28c256"

LINEAR20 = "linear_20"
# 作る変種: 名前 → margin（観測した min/max に掛ける倍率。1.5 が「1x」）
SWEEP_MARGINS = {"3.75x": 5.625}   # 3.5x(margin5.25)も×判定。range sweep最後の1点(1x=1.5, 4x=6.0)
MODELS = {tag: WORK / f"l20_{tag}_qdq.onnx" for tag in SWEEP_MARGINS}


def emit(obj):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def build_one(tag, margin):
    """linear_20 の scale・zero point だけを「後続の観測範囲 × margin」から作り直したモデルを作る。

    ① P12e の較正の後続の範囲と、その Q/DQ の名前・型を引き、fit で scale・zero point を作る
    ② baseline の QDQ のその 2 つの初期化子だけを置き換え、ほかの初期化子とノードが変わっていないことを hash で確かめて保存する
    """
    calib = json.loads((REP / "calibration_ranges.json").read_text(encoding="utf-8"))
    lo, hi = calib["ranges"]["later"][LINEAR20]
    qmap = json.loads((REP / "qparam_map.json").read_text(encoding="utf-8"))
    g = next(g for g in qmap["groups"] if LINEAR20 in g["sources"])
    scale_name, zero_name, dtype = g["scale_name"], g["zero_name"], g["dtype"]
    qmax = np.iinfo(np.dtype(dtype)).max
    s, z = fit(lo, hi, qmax, margin)
    represented = [-z * float(s), (qmax - z) * float(s)]

    m = onnx.load(str(SOURCE), load_external_data=False)
    assert not any(t.external_data for t in m.graph.initializer)
    init = {t.name: t for t in m.graph.initializer}
    original = {t.name: hashlib.sha256(t.SerializeToString()).hexdigest() for t in m.graph.initializer}
    node_hash = hashlib.sha256(b"".join(n.SerializeToString() for n in m.graph.node)).hexdigest()

    dt = np.dtype(dtype)
    before_scale_shape = numpy_helper.to_array(init[scale_name]).shape
    before_zero_shape = numpy_helper.to_array(init[zero_name]).shape
    init[scale_name].CopyFrom(numpy_helper.from_array(np.asarray(s, np.float32).reshape(before_scale_shape), scale_name))
    init[zero_name].CopyFrom(numpy_helper.from_array(np.asarray(z, dtype=dt).reshape(before_zero_shape), zero_name))
    allowed = {scale_name, zero_name}

    unexpected = [t.name for t in m.graph.initializer if t.name not in allowed and
                  hashlib.sha256(t.SerializeToString()).hexdigest() != original[t.name]]
    assert not unexpected, unexpected
    assert hashlib.sha256(b"".join(n.SerializeToString() for n in m.graph.node)).hexdigest() == node_hash

    path = MODELS[tag]
    onnx.save(m, str(path))
    onnx.checker.check_model(str(path))
    result = {"tag": tag, "margin": margin, "observed": [lo, hi], "scale": float(s), "zero": int(z),
              "represented": represented, "source_sha256": digest(SOURCE), "model_sha256": digest(path),
              "node_sha256": node_hash}
    (OUT / f"build_audit_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", **result, "wrote": str(path)})
    del m, init
    gc.collect()


def build():
    """SWEEP_MARGINS のすべてを作る。"""
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, margin in SWEEP_MARGINS.items():
        build_one(tag, margin)


def summarize(sc):
    """score の結果を 1 行の要約にする。"""
    a = sc["ALL"]
    return (f"1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f} | "
            f"点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}")


def evaluate():
    """① 1x（= P12q の L2-L）の結果を再掲 → ② FP32 の基準 → ③ 各変種を teacher forcing と自由生成で評価して report.json に書く。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    emit({"phase": "evaluate", "points": [{"name": p["name"], "positions": len(p["eval_positions"])} for p in pts]})

    prior = json.loads((P12Q_OUT / "report.json").read_text(encoding="utf-8"))
    print("1x(=P12qのL2-L再掲)  " + summarize(prior["scores"]["L"]), flush=True)
    for t in prior["free_generation"]["L"]:
        print(f"      turn{t['turn']}: {t['text']!r}", flush=True)

    fp32 = session(OPS_DIR / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    gc.collect()
    emit({"phase": "evaluate", "step": "fp32_reference_done"})

    base = session(SOURCE)
    results, gens = {}, {}
    for tag in SWEEP_MARGINS:
        variant = session(MODELS[tag])
        pipe = Pipeline(None, base, variant, "none")
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        results[tag] = sc
        gens[tag] = gen
        print(f"{tag:20s} {summarize(sc)}", flush=True)
        for t in gen:
            print(f"      turn{t['turn']}: {t['text']!r}", flush=True)
        del variant, pipe
        gc.collect()
        (OUT / "report.json").write_text(json.dumps({
            "reused_1x_from_P12q": {"scores": prior["scores"]["L"], "free_generation": prior["free_generation"]["L"]},
            "scores": results, "free_generation": gens,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    del base
    gc.collect()
    emit({"phase": "evaluate", "wrote": str(OUT / "report.json")})


# コマンドの入口。--phase build か evaluate のどちらか 1 段を流す
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "evaluate"], required=True)
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate}[a.phase]()
