# -*- coding: utf-8 -*-
"""KV キャッシュ系の量子化範囲だけを、実際の対話データで広げ直して Host で評価する（P12l）。

P12k で、past_k/past_v 入力の量子化範囲が狭すぎ、対話のほぼ全位置で新しい KV が範囲外になると分かった
（日本語 107/107 位置、英語 107/109 位置。「22度」の「2」や「アシスタント」で past_v_14 の 8〜9倍）。
Host と HTP で同じように飽和するので Host↔HTP の差の原因ではないが、FP32 からの劣化としては大きい。

## 方針

- 対象: KV の値が流れる Q ノード（kv_family.json、448本）= past 入力 / 新しいスロット / Concat→present・GQA 複製・転置
- 範囲は**広げるだけ**。新しい範囲 = 既存範囲 ∪ （FP32 で観測した min/max × 1.2）。狭めない
- scale を共有する Q ノードは共有グループ単位で観測範囲を合わせる（qparam_map.json の groups）
- 較正データは**評価に使わない**対話: P12d の long_ja / long_code / chat_long（FP32 軌跡、計約1,900トークン）
- uint8 のテンソル（C-8 で落としたもの）は広げると分解能の損失が大きいので、倍率を記録して注意喚起する

## 評価（P12k と同じ）

  baseline / baseline_kvwide / D1_fp32pos0_later / D1_fp32pos0_later_kvwide
  評価点の margin・KL・自由生成 ＋ 評価対話で新しい KV が（広げた）範囲からはみ出すかの再計数
"""
from pathlib import Path
import argparse
import collections
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS
from block_revert_ab import eval_points, score, load_json, ART, REP, P12D, sess_from
from fp32_prefix_eval import Pipeline, run_points, free_generation

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OUTDIR = ROOT / "results/P12l_20260916_kv_range_recal"
CALIB_CASES = ("long_ja", "long_code", "chat_long")
MARGIN = 1.2


def kv_groups():
    fam = load_json(ROOT / "results/P12k_20260916_fp32_prefix/kv_family.json")
    groups = load_json(REP / "qparam_map.json")["groups"]
    by_q = {}
    for g in groups:
        for q in g["q_nodes"]:
            by_q[q] = g
    sel, mixed = {}, []
    for qname in fam:
        g = by_q[qname]
        sel[g["scale_name"]] = g
    for g in sel.values():
        non_kv = [q for q in g["q_nodes"] if q not in fam]
        if non_kv:
            mixed.append({"scale_name": g["scale_name"], "non_kv_q_nodes": non_kv})
    return list(sel.values()), mixed


def fp32_name(src):
    return src[: -len("_QuantizeLinear_Input")] if src.endswith("_QuantizeLinear_Input") else src


def observe(groups):
    """FP32 で較正対話を流し、各グループの入力テンソルの min/max を取る。"""
    names = sorted({fp32_name(s) for g in groups for s in g["sources"]})
    m = onnx.load(str(OPS / "qwen3_dec28c256_prepped.onnx"), load_external_data=False)
    produced = {o for n in m.graph.node for o in n.output} | {o.name for o in m.graph.output}
    have = {o.name for o in m.graph.output}
    for n in names:
        if n in produced and n not in have:
            m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    probe = OUTDIR / "fp32_kvprobe.onnx"
    link = OUTDIR / "qwen3_dec28c256_prepped.onnx.data"
    if not link.exists():
        import os
        os.link(OPS / "qwen3_dec28c256_prepped.onnx.data", link)
    onnx.save(m, str(probe))
    del m
    s = session(probe)
    cos, sin = tables()
    lo = collections.defaultdict(lambda: np.inf)
    hi = collections.defaultdict(lambda: -np.inf)
    n_tok = 0
    for case in CALIB_CASES:
        rows = load_json(P12D / f"{case}.json")["rows"]
        r = Runner(s, cos, sin)
        for row in rows:
            feeds_cache = r.cache                   # past_* は入力そのものを観測
            for k, v in feeds_cache.items():
                if r.position > 0:
                    n_valid = min(r.position, 256)
                    vv = v[:, :, -n_valid:, :]
                    lo[k] = min(lo[k], float(vv.min()))
                    hi[k] = max(hi[k], float(vv.max()))
            got, _ = r.feed(row["token_id"], s)
            for n in names:
                if n in got:
                    a = got[n]
                    lo[n] = min(lo[n], float(a.min()))
                    hi[n] = max(hi[n], float(a.max()))
            n_tok += 1
        print(f"  較正 {case}: 累計 {n_tok} トークン", flush=True)
    missing = [n for n in names if n not in lo]
    return dict(lo), dict(hi), n_tok, missing


def fit(lo, hi, qmax):
    lo, hi = min(0.0, lo), max(0.0, hi)
    scale = (hi - lo) / qmax
    zp = int(np.clip(round(-lo / scale), 0, qmax))
    scale = max(scale, (-lo / zp) if zp > 0 and lo < 0 else 0.0, (hi / (qmax - zp)) if zp < qmax and hi > 0 else 0.0)
    return np.float32(np.nextafter(np.float32(scale), np.float32(np.inf))), zp


def widen_plan(groups, lo, hi, base_model):
    init = {t.name: t for t in base_model.graph.initializer}
    plan = []
    for g in groups:
        s_old = float(numpy_helper.to_array(init[g["scale_name"]]).ravel()[0])
        z_arr = numpy_helper.to_array(init[g["zero_name"]])
        qmax = 65535 if z_arr.dtype == np.uint16 else 255
        z_old = float(z_arr.ravel()[0])
        old = ((0 - z_old) * s_old, (qmax - z_old) * s_old)
        srcs = [fp32_name(s) for s in g["sources"]]
        obs_lo = min(lo.get(s, np.inf) for s in srcs)
        obs_hi = max(hi.get(s, -np.inf) for s in srcs)
        if not np.isfinite(obs_lo):
            plan.append({"scale_name": g["scale_name"], "skipped": "no observation", "sources": srcs})
            continue
        new_lo, new_hi = min(old[0], obs_lo * MARGIN), max(old[1], obs_hi * MARGIN)
        if new_lo >= old[0] and new_hi <= old[1]:
            plan.append({"scale_name": g["scale_name"], "unchanged": True, "old": old, "observed": [obs_lo, obs_hi]})
            continue
        s_new, z_new = fit(new_lo, new_hi, qmax)
        plan.append({"scale_name": g["scale_name"], "zero_name": g["zero_name"], "dtype": str(z_arr.dtype),
                     "old": old, "observed": [obs_lo, obs_hi], "new_scale": float(s_new), "new_zero": int(z_new),
                     "new": [(0 - z_new) * float(s_new), (qmax - z_new) * float(s_new)],
                     "widen_factor": float(s_new) / s_old, "sources": srcs})
    return plan


def apply(model_path, plan):
    m = onnx.load(str(model_path))
    init = {t.name: t for t in m.graph.initializer}
    for p in plan:
        if "new_scale" not in p:
            continue
        for nm, val in ((p["scale_name"], p["new_scale"]), (p["zero_name"], p["new_zero"])):
            t = init[nm]
            a = numpy_helper.to_array(t)
            t.CopyFrom(numpy_helper.from_array(np.asarray(val, dtype=a.dtype).reshape(a.shape), nm))
    return sess_from(m.SerializeToString())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log = open(OUTDIR / "progress.log", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    groups, mixed = kv_groups()
    say(f"KV 系の量子化グループ {len(groups)}（KV 以外の Q と scale を共有しているもの {len(mixed)}）")
    t0 = time.time()
    lo, hi, n_tok, missing = observe(groups)
    say(f"FP32 観測 {n_tok} トークン {time.time() - t0:.0f}s / 観測できなかったテンソル {len(missing)}: {missing[:6]}")
    base = onnx.load(str(ART / "baseline.onnx"), load_external_data=False)
    plan = widen_plan(groups, lo, hi, base)
    del base
    changed = [p for p in plan if "new_scale" in p]
    u8 = [p for p in changed if p["dtype"] == "uint8"]
    wf = sorted((p["widen_factor"] for p in changed), reverse=True)
    say(f"広げたグループ {len(changed)} / 変更なし {sum(1 for p in plan if p.get('unchanged'))} / 観測なし {sum(1 for p in plan if 'skipped' in p)}"
        f"  倍率 最大 {wf[0]:.2f} 中央値 {np.median(wf):.2f}  うち uint8 {len(u8)}（最大倍率 {max([p['widen_factor'] for p in u8], default=0):.2f}）")
    (OUTDIR / "widen_plan.json").write_text(json.dumps({"mixed_scale_groups": mixed, "plan": plan}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")

    tok = tokenizer()
    pts = eval_points(tok)
    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    report = {"scores": {}, "free_generation": {}}
    base_w = apply(ART / "baseline.onnx", plan)
    later_w = apply(ART / "focused_later.onnx", plan)
    variants = [("baseline_kvwide", base_w, base_w, "none"), ("D1_fp32pos0_later_kvwide", later_w, later_w, "pos0")]
    for name, first, later, mode in variants:
        t0 = time.time()
        pipe = Pipeline(fp32, first, later, mode)
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        report["scores"][name] = sc
        report["free_generation"][name] = gen
        a = sc["ALL"]
        say(f"{name:26s} 1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  最小margin {a['margin_min']:+6.2f}  "
            f"KL {a['kl_mean']:.3f} | 点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}  ({time.time() - t0:.0f}s)")
        for t in gen:
            say(f"      turn{t['turn']}: {t['text']!r}")
        (OUTDIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"-> {OUTDIR / 'report.json'}")


if __name__ == "__main__":
    main()
