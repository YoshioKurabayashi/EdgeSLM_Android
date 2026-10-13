# -*- coding: utf-8 -*-
"""層2 を切り出し、Host と実機（HTP）の整数コードが**最初に食い違う演算**を探す（P12j）。

## 背景

P12c: 層境界の hidden（逆量子化後）の Host↔HTP 誤差は 層2入口 add_17 3.14% → 層3入口 add_26 23.61%。
P12i: 残差の巨大値を作っているのは層2 MLP の down_proj 出力 `linear_20`（先頭トークンで 8141）。
→ 層2 のブロック内で何かが起きている。ただし add_26 は scale も 0.0003→0.113 に跳ねるので、
  「整数の食い違いが増えた」のか「同じ食い違いが粗い刻みで拡大された」のかを分けて見る必要がある。

## 方法

- 本番 QDQ（`final_qdq.onnx`）から add_17 → add_26 を**そのまま**切り出す（qparam は作り直さない）
- 入力: add_17（Q の手前の実数）/ past_k_2 / past_v_2 / cos / sin / attn_mask。値は P12b の step41（ホストのキャッシュ）
- 版A（観測）: 層2 内の全 QuantizeLinear 出力（uint16）＋ add_26 の Q 出力 ＋ present_k_2/v_2
- 版B（対照）: add_26 の Q 出力 ＋ present_k_2/v_2 だけ
  出力を増やすと HTP の融合が変わり数値が変わりうるので、**版A と版B で add_26 が一致するか**を先に確かめる
- 比較はグラフ順に: 整数の完全一致率 / |コード差| の RMS・最大 / scale / 逆量子化後の相対誤差

## 使い方

  python layer2_probe.py --phase prepare   # 部分モデル2つと入力 npz を作る
  （bash scripts/build_context.sh で DLC→context、wrap_epctx.py で包み、端末で infer）
  python layer2_probe.py --phase compare   # 実機出力を取り込み、比較表を出す
"""
from pathlib import Path
import argparse
import collections
import json
import os
import subprocess
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

ROOT = Path("D:/AI/source/n_chap5/EdgeSLM_Android")
SRC = ROOT / "work/ops/qwen3_dec28c256/qwen3_dec28c256_final_qdq.onnx"
HID = ROOT / "work/q3_hid/hid_qdq.onnx"          # 層0-5 の部分モデル（add_17 を取り出すのに使う）
WORK = ROOT / "work/q3_l2"
STEP41 = Path(r"C:\Users\kurab\AppData\Local\Temp\claude\C--Users-kurab\8a37dc21-8bfe-4a66-917d-b37ce5900acf\scratchpad\step41.npz")
OUT = ROOT / "results/P12j_20260916_layer2_htp_probe"
DEV = "/sdcard/Android/data/org.edgeslm.runner/files"
ADB = "D:/Android/Sdk/platform-tools/adb.exe"
INPUTS = ["add_17", "past_k_2", "past_v_2", "cos", "sin", "attn_mask"]


def sess(p):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def layer2_q_outputs():
    """add_17 → add_26 の間にある QuantizeLinear の出力名（グラフ順）と、その scale/zero point。"""
    m = onnx.load(str(SRC), load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    inits = {t.name: t for t in g.initializer}
    anc, st = set(), ["add_26"]
    while st:
        t = st.pop()
        if t in anc or t in inits or t not in prod:
            continue
        anc.add(t)
        st += list(prod[t].input)
    desc, st = set(), ["add_17"]
    while st:
        t = st.pop()
        if t in desc:
            continue
        desc.add(t)
        for n in cons[t]:
            st += list(n.output)
    inside = anc & desc
    qs = [n for n in g.node if n.op_type == "QuantizeLinear" and (n.input[0] in inside)]
    if not any(n.input[0] == "add_26" for n in qs):  # add_26 自体は inside に入るので通常は既にある
        qs.append(next(n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == "add_26"))
    info = []
    for n in qs:
        s = numpy_helper.to_array(inits[n.input[1]]).astype(np.float64).ravel()[0]
        z = numpy_helper.to_array(inits[n.input[2]])
        info.append({"tensor": n.input[0], "q_output": n.output[0], "scale": float(s),
                     "zero_point": float(z.ravel()[0]), "dtype": str(z.dtype),
                     "producer": prod[n.input[0]].op_type if n.input[0] in prod else "input"})
    return info


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    info = layer2_q_outputs()
    (OUT / "q_tensors.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  層2 の観測点 {len(info)}（最後が add_26）")

    # グラフ入力の直後の Q（add_17）を出力にすると qairt-converter が
    # 「Squash not possible for op add_17_QuantizeLinear」で止まる（入力への量子化の畳み込みができない）。
    # 入力側の Q は観測しない。出力側も同じで、present_k_2/v_2（グラフ出力の DQ）の直前の Q を出力にすると
    # 「Squash not possible for op present_v_2_DequantizeLinear」で止まる。
    m0 = onnx.load(str(SRC), load_external_data=False)
    feeds_present = {n.input[0] for n in m0.graph.node
                     if n.op_type == "DequantizeLinear" and n.output[0] in ("present_k_2", "present_v_2")}
    del m0
    outs_a = [x["q_output"] for x in info
              if x["tensor"] not in INPUTS and x["q_output"] not in feeds_present] + ["present_k_2", "present_v_2"]
    outs_b = ["add_26_QuantizeLinear_Output", "present_k_2", "present_v_2"]
    for tag, outs in (("l2obs", outs_a), ("l2ctl", outs_b)):
        onnx.utils.extract_model(str(SRC), str(WORK / f"{tag}_qdq.onnx"),
                                 input_names=INPUTS, output_names=outs, check_model=False)
        m = onnx.load(str(WORK / f"{tag}_qdq.onnx"), load_external_data=False)
        print(f"  {tag}: ノード {len(m.graph.node)} / 入力 {[i.name for i in m.graph.input]} / 出力 {len(m.graph.output)}")

    # add_17（層2 に入る hidden の実数）を、ホストの層0-5 部分モデルから取り出す
    z = np.load(STEP41)
    m = onnx.load(str(HID), load_external_data=False)
    m.graph.output.append(helper.make_tensor_value_info("add_17", TensorProto.FLOAT, None))
    tmp = WORK / "hid_with_add17.onnx"
    onnx.save(m, str(tmp))
    s = sess(tmp)
    need = [i.name for i in s.get_inputs()]
    got = dict(zip([o.name for o in s.get_outputs()], s.run(None, {k: z[k] for k in need})))
    feeds = {"add_17": got["add_17"].astype(np.float32),
             **{k: z[k] for k in ("past_k_2", "past_v_2", "cos", "sin", "attn_mask")}}
    np.savez(WORK / "l2_inputs.npz", **feeds)
    tmp.unlink()
    print("  入力:", {k: tuple(v.shape) for k, v in feeds.items()})

    # 等価チェック: 版A/版B をホストで回し、add_26 と present が本番の部分モデルと矛盾しないこと
    ra = dict(zip([o.name for o in sess(WORK / "l2obs_qdq.onnx").get_outputs()],
                  sess(WORK / "l2obs_qdq.onnx").run(None, feeds)))
    rb = dict(zip([o.name for o in sess(WORK / "l2ctl_qdq.onnx").get_outputs()],
                  sess(WORK / "l2ctl_qdq.onnx").run(None, feeds)))
    for k in outs_b:
        print(f"  ホスト 版A↔版B {k}: 最大差 {np.abs(ra[k].astype(np.float64) - rb[k].astype(np.float64)).max():.3e}")
    # 層0-5 部分モデルが出していた add_26 の Q 出力とも照合
    hid_ref = got.get("add_26_QuantizeLinear_Output")
    if hid_ref is not None:
        d = int(np.abs(hid_ref.astype(np.int64) - rb["add_26_QuantizeLinear_Output"].astype(np.int64)).max())
        print(f"  層0-5 部分モデルの add_26 と層2 部分モデルの add_26（ホスト）: 最大コード差 {d}")


def adb(*a):
    env = dict(os.environ)
    env["MSYS2_ARG_CONV_EXCL"] = "/sdcard"
    return subprocess.run([ADB, *a], capture_output=True, text=True, env=env, errors="replace")


def run_device(tag, feeds):
    files = {}
    d = WORK / "in"
    d.mkdir(exist_ok=True)
    for k, v in feeds.items():
        p = d / f"{k}.bin"
        p.write_bytes(np.ascontiguousarray(v).tobytes())
        assert adb("push", str(p), f"{DEV}/inputs/l2_{k}.bin").returncode == 0
        files[k] = f"inputs/l2_{k}.bin"
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
    info = json.loads((OUT / "q_tensors.json").read_text(encoding="utf-8"))
    host = {}
    for tag in ("l2obs", "l2ctl"):
        s = sess(WORK / f"{tag}_qdq.onnx")
        host[tag] = dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))
    dev = {}
    for tag in ("l2ctl", "l2obs"):
        dev[tag], strict, ms = run_device(tag, feeds)
        print(f"  実機 {tag}: strict={strict} OrtRun {ms:.1f} ms  出力 {len(dev[tag])}")

    # 対照: 観測出力を増やしても実機の add_26 が変わらないか
    k = "add_26_QuantizeLinear_Output"
    d_ab = int(np.abs(dev["l2obs"][k].astype(np.int64) - dev["l2ctl"][k].astype(np.int64)).max())
    print(f"\n  対照（実機）版A↔版B の add_26 最大コード差 = {d_ab}  "
          f"{'→ 観測による数値変化なし' if d_ab == 0 else '→ ★観測出力が HTP の計算を変えている'}")

    rows = []
    H, D = host["l2obs"], dev["l2obs"]
    for x in info:
        q = x["q_output"]
        if q not in D:
            continue
        h, dv = H[q].astype(np.int64).ravel(), D[q].astype(np.int64).ravel()
        dc = dv - h
        hr = (h - x["zero_point"]) * x["scale"]
        rel = float(np.linalg.norm(dc * x["scale"]) / (np.linalg.norm(hr) or 1.0))
        rows.append({**x, "size": int(h.size), "exact": float((dc == 0).mean()),
                     "code_rms": float(np.sqrt((dc.astype(np.float64) ** 2).mean())), "code_max": int(np.abs(dc).max()),
                     "real_rel": rel, "host_real_rms": float(np.sqrt((hr ** 2).mean()))})
    for nm in ("present_k_2", "present_v_2"):
        h = H[nm][:, :, -1:, :].astype(np.float64)
        dv = D[nm][:, :, -1:, :].astype(np.float64)
        rows.append({"tensor": nm + "(新スロット)", "real_rel": float(np.linalg.norm(dv - h) / np.linalg.norm(h))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps({"control_add26_code_diff_obs_vs_ctl": d_ab, "rows": rows},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  {'#':>3} {'テンソル':24s} {'生成':12s} {'要素':>5} {'一致':>6} {'差RMS':>7} {'差max':>6} {'scale':>10} {'値RMS':>9} {'実数誤差':>9}")
    for i, r in enumerate(rows):
        if "exact" not in r:
            print(f"      {r['tensor']:24s} {'':12s} {'':>5} {'':>6} {'':>7} {'':>6} {'':>10} {'':>9} {r['real_rel']*100:8.2f}%")
            continue
        flag = " ★" if r["code_max"] > 2 else ""
        print(f"  {i:3d} {r['tensor'][:24]:24s} {r['producer'][:12]:12s} {r['size']:5d} {r['exact']*100:5.1f}% "
              f"{r['code_rms']:7.2f} {r['code_max']:6d} {r['scale']:10.3g} {r['host_real_rms']:9.4g} {r['real_rel']*100:8.2f}%{flag}")
    print(f"\n-> {OUT / 'compare.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["prepare", "compare"], required=True)
    a = ap.parse_args()
    prepare() if a.phase == "prepare" else compare()
