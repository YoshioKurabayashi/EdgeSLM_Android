# -*- coding: utf-8 -*-
"""P12s: 層2 linear_20 だけを4x range(P12rのsweetspot候補)にした層2部分モデルを実機で走らせ、
add_26 の Host↔HTP 誤差がどこまで下がるかを測る。

P12r: Hostで linear_20 のrangeを段階的に広げると、1x(narrow, margin1.5)は自由生成が壊れる
(「音楽をかけて」を無視してturn0を繰り返す)が、4x(margin6.0)からは自由生成が完全に正常化した
(3ターン自然、turn1はbaselineと完全一致)。mul_41はbaseline固定。

  変種       linear_20 scale/zero                     Host自由生成   add_26 Host↔HTP(実測)
  baseline   0.11333262920379639 / 1489 (既存)          ○           21.95%(P12m)
  1x(narrow) 0.00017944365390576422 / 21833              ×           8.61%(P12m)
  4x         0.0007177746156230569 / 21833(今回)         ○           ?(本実験)

P12mと同じ層2部分モデル(add_17→add_26、3出力のl2ctl版)を使い、mul_41は変更しない
(baseline のまま)。linear_20 だけ4xの scale/zero に差し替える。

## 使い方

  python layer2_linear20_4x_device.py --phase build     # Host側: 4x版QDQを作る
  bash scripts/build_context.sh work/q3_l2/l2ctl_4x_qdq.onnx work/q3_l2 l2ctl_4x
  python layer2_linear20_4x_device.py --phase wrap
  python layer2_linear20_4x_device.py --phase compare   # 実機で走らせ、baseline/narrowと比較
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
FOURX = WORK / "l2ctl_4x_qdq.onnx"
FOURX_EPCTX = WORK / "l2ctl_4x_epctx.onnx"
CTXDIR = WORK / "ctx_l2ctl_4x"
OUT = ROOT / "results/P12s_20260916_layer2_linear20_4x_device"
DEV = "/sdcard/Android/data/org.edgeslm.runner/files"
ADB = "D:/Android/Sdk/platform-tools/adb.exe"

# P12r build_audit_4x.json(margin=6.0)の値。mul_41は変更しない(baselineのまま)。
LINEAR20_4X = {"scale_name": "linear_20_scale", "zero_name": "linear_20_zero_point",
               "scale": 0.0007177746156230569, "zero": 21833, "dtype": "uint16"}


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def sess(p):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    q = LINEAR20_4X
    m = onnx.load(str(BASELINE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    dtype = np.dtype(q["dtype"])
    s_old = float(numpy_helper.to_array(init[q["scale_name"]]))
    z_old = int(numpy_helper.to_array(init[q["zero_name"]]))
    before = {"scale": s_old, "zero": z_old,
              "represented": [-z_old * s_old, (np.iinfo(dtype).max - z_old) * s_old]}
    s_shape = numpy_helper.to_array(init[q["scale_name"]]).shape
    z_shape = numpy_helper.to_array(init[q["zero_name"]]).shape
    init[q["scale_name"]].CopyFrom(numpy_helper.from_array(
        np.asarray(q["scale"], np.float32).reshape(s_shape), q["scale_name"]))
    init[q["zero_name"]].CopyFrom(numpy_helper.from_array(
        np.asarray(q["zero"], dtype=dtype).reshape(z_shape), q["zero_name"]))
    represented = [-q["zero"] * q["scale"], (np.iinfo(dtype).max - q["zero"]) * q["scale"]]
    onnx.save(m, str(FOURX))
    onnx.checker.check_model(str(FOURX))
    change = {"tensor": "linear_20", "before": before,
              "after": {"scale": q["scale"], "zero": q["zero"], "represented": represented}}
    (OUT / "build_audit.json").write_text(json.dumps(change, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "build", **change, "wrote": str(FOURX)})

    feeds = dict(np.load(WORK / "l2_inputs.npz"))
    rb = dict(zip([o.name for o in sess(BASELINE).get_outputs()], sess(BASELINE).run(None, feeds)))
    rn = dict(zip([o.name for o in sess(FOURX).get_outputs()], sess(FOURX).run(None, feeds)))
    k = "add_26_QuantizeLinear_Output"
    d = int(np.abs(rb[k].astype(np.int64) - rn[k].astype(np.int64)).max())
    emit({"phase": "build", "host_baseline_vs_4x_add26_max_code_diff": d})


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


def wrap():
    OUT.mkdir(parents=True, exist_ok=True)
    assert FOURX.exists(), "先に --phase build"
    bins = list(CTXDIR.glob("*.bin"))
    assert bins, f"context binary が無い。先に bash scripts/build_context.sh {FOURX} {WORK} l2ctl_4x"
    b = bins[0]
    import importlib
    wrap_epctx = importlib.import_module("wrap_epctx")
    r = wrap_epctx.run(b, FOURX, FOURX_EPCTX, "l2ctl_4x")
    (OUT / "epctx_meta.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "wrap", **{k: v for k, v in r.items() if k != "outputs"}})


def compare():
    feeds = dict(np.load(WORK / "l2_inputs.npz"))
    k = "add_26_QuantizeLinear_Output"

    m = onnx.load(str(BASELINE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    q_node = next(n for n in m.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == k)
    s_base = float(numpy_helper.to_array(init[q_node.input[1]]))
    z_base = int(numpy_helper.to_array(init[q_node.input[2]]))

    m2 = onnx.load(str(FOURX), load_external_data=False)
    init2 = {t.name: t for t in m2.graph.initializer}
    s_4x = float(numpy_helper.to_array(init2[q_node.input[1]]))
    z_4x = int(numpy_helper.to_array(init2[q_node.input[2]]))
    assert (s_base, z_base) == (s_4x, z_4x), "add_26 自体の qparam は変えていないはず"

    host_base = dict(zip([o.name for o in sess(BASELINE).get_outputs()], sess(BASELINE).run(None, feeds)))[k]
    host_4x = dict(zip([o.name for o in sess(FOURX).get_outputs()], sess(FOURX).run(None, feeds)))[k]

    dev_base, strict_b, ms_b = run_device("l2ctl", ROOT / "work/q3_l2/l2ctl_epctx.onnx", feeds)
    dev_4x, strict_4, ms_4 = run_device("l2ctl_4x", FOURX_EPCTX, feeds)

    def real_rel(host_code, dev_code, s, z):
        h = (host_code.astype(np.float64) - z) * s
        d = (dev_code.astype(np.float64) - z) * s
        return float(np.linalg.norm(d - h) / (np.linalg.norm(h) or 1.0)), \
            int(np.abs(dev_code.astype(np.int64) - host_code.astype(np.int64)).max())

    rel_base, max_base = real_rel(host_base, dev_base[k], s_base, z_base)
    rel_4x, max_4x = real_rel(host_4x, dev_4x[k], s_4x, z_4x)

    result = {
        "baseline": {"host_vs_htp_real_rel": rel_base, "code_max_diff": max_base,
                     "strict_no_cpu_fallback": strict_b, "ort_run_ms": ms_b},
        "linear20_4x": {"host_vs_htp_real_rel": rel_4x, "code_max_diff": max_4x,
                         "strict_no_cpu_fallback": strict_4, "ort_run_ms": ms_4},
        "p12m_reference_baseline_real_rel": 0.2195,
        "p12m_reference_narrow_1x_real_rel": 0.0861,
        "improved_vs_baseline": rel_4x < rel_base,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "compare", **result})
    print(f"\n  baseline:            Host<->HTP add_26 real error = {rel_base*100:.2f}%  (code max diff {max_base})")
    print(f"  linear_20 4x:        Host<->HTP add_26 real error = {rel_4x*100:.2f}%  (code max diff {max_4x})")
    print(f"  (参考) P12m narrow(1x): 8.61%")
    print(f"\n-> {OUT / 'compare.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["build", "wrap", "compare"], required=True)
    a = ap.parse_args()
    {"build": build, "wrap": wrap, "compare": compare}[a.phase]()
