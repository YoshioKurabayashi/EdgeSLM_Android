# -*- coding: utf-8 -*-
"""P12m: 層2 MLP の `mul_41` / `linear_20` だけを後続トークン用の範囲にして、
add_26 の Host↔HTP 誤差（P12j: 22.09%）が下がるかを実機で直接測る。

## 背景

P12j: Host↔HTP の差が層3入口 add_26 で 22.09% に跳ねる発生源は、層2 MLP の
`mul_41`（SiLU(gate)×up）と `linear_20`（down_proj 出力）の2本。scale が先頭
トークンの巨大値（mul_41 最大4211 / linear_20 最大8141）に合わせて決まって
いるため、後続トークンでは実質3〜4値でしか表現されていない。

P12e で全体（先頭/後続 分離）を試すと自由生成が崩れたが、その較正
（`results/P12e_20260915_temporal_qdq/calibration_ranges.json`・`build_audit.json`）
にはこの2本の「後続」観測範囲・フィット済み scale/zero point が既にある。
今回はそれを流用し、**この2本だけ**を P12j の層2部分モデル（l2ctl, 3出力の
最小構成）に適用し、実機 HTP で add_26 の Host 比誤差が下がるかを見る。
Host の指標はいじらず、実機の差だけを見る一発勝負の実験（ビルド15分程度）。

## 使い方

  python layer2_kv_narrow.py --phase build     # 較正済みscale/zeroを2本だけ適用したQDQを作る
  bash scripts/build_context.sh work/q3_l2/l2ctl_narrow_qdq.onnx work/q3_l2 l2ctl_narrow
  python layer2_kv_narrow.py --phase wrap      # epctx ONNX を作る
  python layer2_kv_narrow.py --phase compare   # 実機で走らせ、baseline(l2ctl)と比較する
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path("D:/AI/source/n_chap5/EdgeSLM_Android")
WORK = ROOT / "work/q3_l2"
BASELINE = WORK / "l2ctl_qdq.onnx"
NARROW = WORK / "l2ctl_narrow_qdq.onnx"
NARROW_EPCTX = WORK / "l2ctl_narrow_epctx.onnx"
CTXDIR = WORK / "ctx_l2ctl_narrow"
P12E = ROOT / "results/P12e_20260915_temporal_qdq"
OUT = ROOT / "results/P12m_20260916_layer2_kv_narrow"
DEV = "/sdcard/Android/data/org.edgeslm.runner/files"
ADB = "D:/Android/Sdk/platform-tools/adb.exe"

# P12e later-phase build (margin=1.5) の該当2本。build_audit.json['later']['changes'] より。
NEW_QPARAMS = {
    "mul_41": {"scale_name": "mul_41_scale", "zero_name": "mul_41_zero_point",
               "scale": 0.00037498539313673973, "zero": 39798, "dtype": "uint16"},
    "linear_20": {"scale_name": "linear_20_scale", "zero_name": "linear_20_zero_point",
                  "scale": 0.00017944365390576422, "zero": 21833, "dtype": "uint16"},
}


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def sess(p):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((P12E / "build_audit.json").read_text(encoding="utf-8"))
    for src, expect in NEW_QPARAMS.items():
        c = next(c for c in audit["later"]["changes"] if c["sources"] == [src])
        assert c["scale_name"] == expect["scale_name"] and c["zero_name"] == expect["zero_name"]
        assert abs(c["scale"] - expect["scale"]) < 1e-12 and c["zero"] == expect["zero"], (src, c, expect)
        assert c["dtype"] == expect["dtype"]

    m = onnx.load(str(BASELINE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    before = {}
    changed = []
    for src, q in NEW_QPARAMS.items():
        dtype = np.dtype(q["dtype"])
        s_old = float(numpy_helper.to_array(init[q["scale_name"]]))
        z_old = int(numpy_helper.to_array(init[q["zero_name"]]))
        before[src] = {"scale": s_old, "zero": z_old,
                       "represented": [-z_old * s_old, (np.iinfo(dtype).max - z_old) * s_old]}
        s_shape = numpy_helper.to_array(init[q["scale_name"]]).shape
        z_shape = numpy_helper.to_array(init[q["zero_name"]]).shape
        init[q["scale_name"]].CopyFrom(numpy_helper.from_array(
            np.asarray(q["scale"], np.float32).reshape(s_shape), q["scale_name"]))
        init[q["zero_name"]].CopyFrom(numpy_helper.from_array(
            np.asarray(q["zero"], dtype=dtype).reshape(z_shape), q["zero_name"]))
        represented = [-q["zero"] * q["scale"], (np.iinfo(dtype).max - q["zero"]) * q["scale"]]
        changed.append({"tensor": src, "before": before[src],
                         "after": {"scale": q["scale"], "zero": q["zero"], "represented": represented}})
    onnx.save(m, str(NARROW))
    onnx.checker.check_model(str(NARROW))
    (OUT / "build_audit.json").write_text(json.dumps({"changed": changed}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    for c in changed:
        emit({"phase": "build", **c})
    emit({"phase": "build", "wrote": str(NARROW)})

    # Host sanity: baseline と narrow の add_26 差（narrow は較正で変えた範囲そのものなので、
    # 差が出ること自体は期待通り。「実機で誤差が縮むか」が本題）
    feeds = dict(np.load(WORK / "l2_inputs.npz"))
    rb = dict(zip([o.name for o in sess(BASELINE).get_outputs()], sess(BASELINE).run(None, feeds)))
    rn = dict(zip([o.name for o in sess(NARROW).get_outputs()], sess(NARROW).run(None, feeds)))
    k = "add_26_QuantizeLinear_Output"
    d = int(np.abs(rb[k].astype(np.int64) - rn[k].astype(np.int64)).max())
    emit({"phase": "build", "host_baseline_vs_narrow_add26_max_code_diff": d})


def wrap():
    OUT.mkdir(parents=True, exist_ok=True)
    assert NARROW.exists(), "先に --phase build"
    bins = list(CTXDIR.glob("*.bin"))
    assert bins, f"context binary が無い。先に bash scripts/build_context.sh {NARROW} {WORK} l2ctl_narrow"
    b = bins[0]
    from layer2_probe import ROOT as _  # noqa: F401 (import guard, not used)
    import importlib
    wrap_epctx = importlib.import_module("wrap_epctx")
    r = wrap_epctx.run(b, NARROW, NARROW_EPCTX, "l2ctl_narrow")
    (OUT / "epctx_meta.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "wrap", **{k: v for k, v in r.items() if k != "outputs"}})


def adb(*a):
    env = dict(os.environ)
    env["MSYS2_ARG_CONV_EXCL"] = "/sdcard"
    return subprocess.run([ADB, *a], capture_output=True, text=True, env=env, errors="replace")


def run_device(tag, epctx_path, feeds):
    d = WORK / "in"
    d.mkdir(exist_ok=True)
    files = {}
    for k, v in feeds.items():
        p = d / f"{k}.bin"
        p.write_bytes(np.ascontiguousarray(v).tobytes())
        assert adb("push", str(p), f"{DEV}/inputs/l2_{k}.bin").returncode == 0
        files[k] = f"inputs/l2_{k}.bin"
    assert adb("push", str(epctx_path), f"{DEV}/models/ops/{tag}_epctx.onnx").returncode == 0
    job = {"job_id": tag, "task": "infer", "model": f"models/ops/{tag}_epctx.onnx",
           "ep": {"provider": "QNN", "backend": "libQnnHtp.so",
                  "options": {"offload_graph_io_quantization": "0", "qnn_context_priority": "high"},
                  "session": {"intra_op_num_threads": 1, "log_severity_level": 2}},
           "output": {"dir": "out"}, "batches": [{"id": "000", "inputs": files}]}
    jp = WORK / f"{tag}.json"
    jp.write_text(json.dumps(job), encoding="utf-8")
    adb("push", str(jp), f"{DEV}/jobs/{tag}.json")
    adb("shell", f"rm -f {DEV}/out/last_result.json {DEV}/out/{tag}_000_*.bin")
    adb("shell", "am", "force-stop", "org.edgeslm.runner")
    adb("shell", "am", "start", "-n", "org.edgeslm.runner/.RunnerActivity",
        "-a", "org.edgeslm.runner.RUN", "--es", "job", f"jobs/{tag}.json")
    for _ in range(90):
        time.sleep(2)
        if "yes" in adb("shell", f"[ -f {DEV}/out/last_result.json ] && echo yes").stdout:
            break
    r = json.loads(adb("shell", f"cat {DEV}/out/last_result.json").stdout)
    assert r.get("job_id") == tag, f"別ジョブの結果を読んでいる: {r.get('job_id')}"
    if "error" in r:
        raise RuntimeError(r["error"])
    rec = r["records"][0]
    got = {}
    ld = WORK / f"dev_{tag}"
    ld.mkdir(exist_ok=True)
    for o in rec["outputs"]:
        lp = ld / (o["name"] + ".bin")
        adb("pull", f"{DEV}/{o['path']}", str(lp))
        dt = {"uint16": np.uint16, "uint8": np.uint8, "float32": np.float32}[o["type"]]
        got[o["name"]] = np.fromfile(lp, dtype=dt).reshape(o["shape"])
    return got, r["describe"].get("strict_no_cpu_fallback"), rec.get("ort_run_ms")


def compare():
    feeds = dict(np.load(WORK / "l2_inputs.npz"))
    k = "add_26_QuantizeLinear_Output"

    # baseline (l2ctl) の scale/zero は P12j 実測値そのまま使う
    m = onnx.load(str(BASELINE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    q_node = next(n for n in m.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == k)
    s_base = float(numpy_helper.to_array(init[q_node.input[1]]))
    z_base = int(numpy_helper.to_array(init[q_node.input[2]]))

    m2 = onnx.load(str(NARROW), load_external_data=False)
    init2 = {t.name: t for t in m2.graph.initializer}
    s_narrow = float(numpy_helper.to_array(init2[q_node.input[1]]))
    z_narrow = int(numpy_helper.to_array(init2[q_node.input[2]]))
    assert (s_base, z_base) == (s_narrow, z_narrow), "add_26 自体の qparam は変えていないはず"

    host_base = dict(zip([o.name for o in sess(BASELINE).get_outputs()], sess(BASELINE).run(None, feeds)))[k]
    host_narrow = dict(zip([o.name for o in sess(NARROW).get_outputs()], sess(NARROW).run(None, feeds)))[k]

    dev_base, strict_b, ms_b = run_device("l2ctl", ROOT / "work/q3_l2/l2ctl_epctx.onnx", feeds)
    dev_narrow, strict_n, ms_n = run_device("l2ctl_narrow", NARROW_EPCTX, feeds)

    def real_rel(host_code, dev_code, s, z):
        h = (host_code.astype(np.float64) - z) * s
        d = (dev_code.astype(np.float64) - z) * s
        return float(np.linalg.norm(d - h) / (np.linalg.norm(h) or 1.0)), \
            int(np.abs(dev_code.astype(np.int64) - host_code.astype(np.int64)).max())

    rel_base, max_base = real_rel(host_base, dev_base[k], s_base, z_base)
    rel_narrow, max_narrow = real_rel(host_narrow, dev_narrow[k], s_narrow, z_narrow)

    result = {
        "baseline": {"host_vs_htp_real_rel": rel_base, "code_max_diff": max_base,
                     "strict_no_cpu_fallback": strict_b, "ort_run_ms": ms_b},
        "narrow_mul41_linear20": {"host_vs_htp_real_rel": rel_narrow, "code_max_diff": max_narrow,
                                   "strict_no_cpu_fallback": strict_n, "ort_run_ms": ms_n},
        "p12j_reference_baseline_real_rel_0.2209": 0.2209,
        "improved": rel_narrow < rel_base,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "compare", **result})
    print(f"\n  baseline (l2ctl, P12j既存範囲):     Host↔HTP add_26 実数誤差 = {rel_base*100:.2f}%  (差max {max_base}段)")
    print(f"  narrow   (mul_41/linear_20 後続範囲): Host↔HTP add_26 実数誤差 = {rel_narrow*100:.2f}%  (差max {max_narrow}段)")
    print(f"  -> {'改善した' if result['improved'] else '改善しなかった'}")
    print(f"\n-> {OUT / 'compare.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "wrap", "compare"], required=True)
    a = ap.parse_args()
    {"build": build, "wrap": wrap, "compare": compare}[a.phase]()
