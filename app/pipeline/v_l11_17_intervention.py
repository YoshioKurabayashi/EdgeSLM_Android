# -*- coding: utf-8 -*-
"""P13c: L11-17のVだけに対する最小介入（K・他layer・MLPは一切触らない）。

P13bで、点Bの失敗を説明する主因が「L11-17のVキャッシュ状態」に集中していることが分かった
（oracle/baseline swap実験）。ただしこれはoracle（linear_20 QDQ除去）という特殊条件での
話であり、「channel35のoutlier問題」と直接つながっているわけではない。

本実験は、**標準のbaseline量子化モデル**（linear_20は一切変更しない）のまま、L11-17の
V量子化パラメータだけを2通りの方向で変更し、Host自由生成への影響を見る。

V cacheの量子化は2段構成:
  write側（present_v_L、cache格納）: uint16, per-layer scale
  read側（repeat_interleave_2X、GQA複製後のact-actMatMul右辺）: uint8（C-8制約）, per-layer scale

  条件A: 両段ともP12lで較正済みの「較正データ観測 × margin1.2」のrangeに広げる（狭すぎる
         range由来のclipを解消。narrowにはしない）
  条件B: read側（8bit, C-8制約）だけをuint16に格上げし、同じ観測rangeでfitし直す
         （bit数そのものを上げる。実機でC-8制約に反する可能性があるためHost限定の実験）

## 使い方

  python v_l11_17_intervention.py --phase build
  python v_l11_17_intervention.py --phase evaluate
"""
from pathlib import Path
import argparse
import gc
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import session, tokenizer, ROOT, OPS, SOURCE, fit  # noqa: E402
from block_revert_ab import eval_points, score  # noqa: E402
from fp32_prefix_eval import Pipeline, run_points, free_generation  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

OUT = ROOT / "results/P13c_20260917_v_l11_17_intervention"
WORK = ROOT / "work/p13c_v_l11_17"
COND_A = WORK / "condA_widen_qdq.onnx"
COND_B = WORK / "condB_precision_qdq.onnx"

WIDEN_PLAN = ROOT / "results/P12l_20260916_kv_range_recal/widen_plan.json"

# L11-17: (write側scale_name, read側scale_name)
LAYER_GROUPS = {
    11: ("cat_47_scale", "repeat_interleave_23_scale_u8"),
    12: ("cat_51_scale", "repeat_interleave_25_scale_u8"),
    13: ("cat_55_scale", "repeat_interleave_27_scale_u8"),
    14: ("cat_59_scale", "repeat_interleave_29_scale_u8"),
    15: ("cat_63_scale", "repeat_interleave_31_scale_u8"),
    16: ("cat_67_scale", "repeat_interleave_33_scale_u8"),
    17: ("cat_71_scale", "repeat_interleave_35_scale_u8"),
}


def emit(o):
    print(json.dumps(o, ensure_ascii=False), flush=True)


def zero_name_of(scale_name):
    return scale_name.replace("_scale", "_zero_point") if scale_name.endswith("_scale") and "_u8" not in scale_name \
        else scale_name.replace("_scale_u8", "_zp_u8")


def apply_scale_zero(init, scale_name, zero_name, scale, zero, dtype):
    dt = np.dtype(dtype)
    s_shape = numpy_helper.to_array(init[scale_name]).shape
    z_shape = numpy_helper.to_array(init[zero_name]).shape
    init[scale_name].CopyFrom(numpy_helper.from_array(np.asarray(scale, np.float32).reshape(s_shape), scale_name))
    init[zero_name].CopyFrom(numpy_helper.from_array(np.asarray(zero, dtype=dt).reshape(z_shape), zero_name))


def build_cond_a():
    """write側・read側とも、P12l較正済みのwiden_plan(margin1.2)へ広げる。L11-17のみ。"""
    plan = {p["scale_name"]: p for p in json.loads(WIDEN_PLAN.read_text(encoding="utf-8"))["plan"] if "new_scale" in p}
    m = onnx.load(str(SOURCE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    changes = []
    for L, (write_s, read_s) in LAYER_GROUPS.items():
        for scale_name in (write_s, read_s):
            p = plan[scale_name]
            apply_scale_zero(init, p["scale_name"], p["zero_name"], p["new_scale"], p["new_zero"], p["dtype"])
            changes.append({"layer": L, "scale_name": scale_name, "new_scale": p["new_scale"],
                             "new_zero": p["new_zero"], "dtype": p["dtype"], "represented": p["new"]})
    onnx.save(m, str(COND_A))
    onnx.checker.check_model(str(COND_A))
    return changes


def build_cond_b():
    """read側(8bit, C-8制約)だけをuint16に格上げ。同じ観測rangeでfitし直す。write側はbaselineのまま。"""
    plan = {p["scale_name"]: p for p in json.loads(WIDEN_PLAN.read_text(encoding="utf-8"))["plan"] if "new_scale" in p}
    m = onnx.load(str(SOURCE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    changes = []
    for L, (write_s, read_s) in LAYER_GROUPS.items():
        p = plan[read_s]
        lo, hi = p["observed"]
        s, z = fit(lo, hi, 65535, 1.2)  # 同じ観測range・margin、ただしuint16でfit
        zero_name = zero_name_of(read_s)
        apply_scale_zero(init, read_s, zero_name, float(s), int(z), "uint16")
        represented = [-int(z) * float(s), (65535 - int(z)) * float(s)]
        changes.append({"layer": L, "scale_name": read_s, "old_dtype": "uint8", "new_dtype": "uint16",
                         "new_scale": float(s), "new_zero": int(z), "represented": represented})
    onnx.save(m, str(COND_B))
    onnx.checker.check_model(str(COND_B))
    return changes


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    changes_a = build_cond_a()
    (OUT / "build_audit_A.json").write_text(json.dumps(changes_a, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", "condition": "A", "changed_groups": len(changes_a), "wrote": str(COND_A)})

    changes_b = build_cond_b()
    (OUT / "build_audit_B.json").write_text(json.dumps(changes_b, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", "condition": "B", "changed_groups": len(changes_b), "wrote": str(COND_B)})


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

    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    del fp32
    gc.collect()
    emit({"phase": "evaluate", "step": "fp32_reference_done"})

    results, gens = {}, {}
    variants = [("baseline(対照)", SOURCE), ("条件A(range拡張)", COND_A), ("条件B(8bit->16bit)", COND_B)]
    for name, path in variants:
        s = session(path)
        pipe = Pipeline(None, s, s, "none")
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        results[name] = sc
        gens[name] = gen
        print(f"{name:20s} {summarize(sc)}", flush=True)
        for t in gen:
            print(f"      turn{t['turn']}: {t['text']!r}", flush=True)
        del s, pipe
        gc.collect()
        (OUT / "report.json").write_text(json.dumps({"scores": results, "free_generation": gens},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "evaluate", "wrote": str(OUT / "report.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "evaluate"], required=True)
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate}[a.phase]()


if __name__ == "__main__":
    main()
