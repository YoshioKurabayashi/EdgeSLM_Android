# -*- coding: utf-8 -*-
"""P13d: L11-17のV(生成・write・read)を、baseline/条件A/条件Bでsame-input Host↔HTP比較する(Stage1)。

P13cで、L11-17のV量子化パラメータをbaseline量子化モデルのまま変更した条件A(range拡張)・
条件B(read側8bit→16bit)が、どちらもHost自由生成を維持できることを確認した。
次はこの変更がHost↔HTP誤差の縮小にも効くかを、実機で直接測る。

## 観測点(層L=11,14,17)

  v_proj出力(state生成①) : linear_79 / linear_100 / linear_121 (量子化前, float)
  cache write後(②)        : present_v_{L}_QuantizeLinear_Output (this stepのconcat後、量子化後)
  次のattentionでread(③)  : repeat_interleave_{2X}_QuantizeLinear_Output (8bit, C-8制約)

write→readは同一のforward呼び出し内で連続して起きる(concatされた直後のpresent_vを
そのままattentionが読む)ため、1回のデバイス呼び出しで①②③すべてを同時に観測できる。

## 部分モデル

  入力: add_98(L11入口hidden) / past_k_11..17 / past_v_11..17 / cos / sin / attn_mask
  出力: 上記の観測点 + add_161(L18入口hidden、Stage2用) + present_k/v_11..17(継続性確認用)

## 使い方

  python v_l11_17_device_probe.py --phase prepare   # 部分モデル3つ(baseline/A/B)と入力npzを作る
  bash scripts/build_context.sh work/p13d_v_probe/{tag}_qdq.onnx work/p13d_v_probe {tag}   # 各tag
  python v_l11_17_device_probe.py --phase wrap
  python v_l11_17_device_probe.py --phase compare   # 実機出力を取り込み比較
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

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, ROOT, SOURCE  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from v_l11_17_intervention import COND_A, COND_B  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import helper, TensorProto  # noqa: E402

WORK = ROOT / "work/p13d_v_probe"
OUT = ROOT / "results/P13d_20260917_v_l11_17_device_probe"
DEV = "/sdcard/Android/data/org.edgeslm.runner/files"
ADB = "D:/Android/Sdk/platform-tools/adb.exe"

LAYER_ENTRY = {11: "add_98", 14: "add_125", 17: "add_152"}
V_PROJ = {11: "linear_79", 14: "linear_100", 17: "linear_121"}
# 量子化コード(QuantizeLinearの生出力)を直接グラフ出力にするとqairt-converterの
# remove_quant_nodes最適化がQ->DQをsquashできず変換失敗する(Squash not possible for op ...)。
# そのためDequantizeLinear後のfloat値をタップし、Python側でscale/zeroから再量子化してcodeを復元する。
WRITE_Q = {11: "present_v_11", 14: "present_v_14", 17: "present_v_17"}  # 既にpresent_v_11..17に含まれる(DQ後float)
# read側はQの生コード(uint8, C-8)を直接タップする。write(present_v, Concat直後)と違い、
# repeat_interleaveのDQはこの後MatMulにしか使われないため squash 競合が起きず、実機ビルドが
# 通ることを実測で確認済み(baseline/条件AともS5-S7成功、実機出力名もQuantizeLinear_Output)。
READ_Q = {11: "repeat_interleave_23_QuantizeLinear_Output", 14: "repeat_interleave_29_QuantizeLinear_Output",
          17: "repeat_interleave_35_QuantizeLinear_Output"}
L11_ENTRY = "add_98"
L18_ENTRY_RAW = "add_161"  # Host専用のsanity check用(rawなfloat、device出力には含めない)
L18_ENTRY = "add_161_QuantizeLinear_Output"  # add_26と同じパターン: Qの生コードをデバイス出力に使う

MODELS = {"baseline": SOURCE, "condA": COND_A, "condB": COND_B}
PARTIAL_INPUTS = [L11_ENTRY] + [f"past_k_{L}" for L in range(11, 18)] + \
    [f"past_v_{L}" for L in range(11, 18)] + ["cos", "sin", "attn_mask"]


def emit(o):
    print(json.dumps(o, ensure_ascii=False), flush=True)


def sess(p):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    outs = [L18_ENTRY]
    for L in (11, 14, 17):
        outs += [READ_Q[L]]  # WRITE_Q(=present_v_{L})はpresent_v_11..17に既に含まれる
    outs += [f"present_k_{L}" for L in range(11, 18)] + [f"present_v_{L}" for L in range(11, 18)]
    # 注: V_PROJ(linear_79等、v_proj生出力=①)は device 出力に含めない。
    # HTP側でQ/DQに挟まれていない生の中間float(fp16内部計算)を新規にfp32グラフ出力として
    # 引き出すと、graph finalizeがdatatype変換ノードを作れず失敗する(baseline/条件A共通で確認済み)。
    # ユーザー指定の最重要タップは②(write)③(read)④(layer exit hidden)であり①は対象外でも進められる。
    # ①はHostのみで別途参照可能。

    for tag, path in MODELS.items():
        out_path = WORK / f"{tag}_qdq.onnx"
        onnx.utils.extract_model(str(path), str(out_path), input_names=PARTIAL_INPUTS,
                                 output_names=outs, check_model=False)
        m = onnx.load(str(out_path), load_external_data=False)
        emit({"phase": "prepare", "tag": tag, "nodes": len(m.graph.node), "outputs": len(m.graph.output),
              "wrote": str(out_path)})

    # 入力の準備: baselineモデルにadd_98を追加出力させ、実対話(点B)の41トークン目まで流して
    # 位置41直前のキャッシュ(past_k/v_11..17)と、位置41処理時のadd_98を採取する
    m = onnx.load(str(SOURCE), load_external_data=False)
    existing = {v.name for v in m.graph.output}
    if L11_ENTRY not in existing:
        m.graph.output.append(helper.make_tensor_value_info(L11_ENTRY, TensorProto.FLOAT, None))
    tmp = WORK / "baseline_with_add98.onnx"
    onnx.save(m, str(tmp))
    del m

    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "point_B_music")
    ids = p["ids"]
    pos = min(41, len(ids) - 1)
    prefix, feed_tid = ids[:pos], ids[pos]
    emit({"phase": "prepare", "input_tokens": len(ids), "capture_position": pos})

    cos, sin = tables()
    s = sess(tmp)
    r = Runner(s, cos, sin)
    for tid in prefix:
        r.feed(tid, s)
    cache_before = {k: v.copy() for k, v in r.cache.items() if 11 <= int(k.split("_")[-1]) <= 17}
    got, feed_inputs = r.feed(feed_tid, s)
    add98 = got[L11_ENTRY].astype(np.float32)

    feeds = {L11_ENTRY: add98, "cos": feed_inputs["cos"], "sin": feed_inputs["sin"],
             "attn_mask": feed_inputs["attn_mask"],
             **{k: v for k, v in cache_before.items()}}
    np.savez(WORK / "l11_inputs.npz", **feeds)
    tmp.unlink()
    emit({"phase": "prepare", "wrote": str(WORK / "l11_inputs.npz"),
          "shapes": {k: list(v.shape) for k, v in feeds.items()}})

    # Host側の健全性チェック: 3モデルとも同じadd_98/past_k/v_11..17を入れてadd_161を比較
    # (層11以前は共通のはずなので、baseline/A/Bで add_161 が近い値になることを期待。
    #  layer11-17自体のVを変えているので厳密一致はしない)
    for tag in MODELS:
        ps = sess(WORK / f"{tag}_qdq.onnx")
        r2 = dict(zip([o.name for o in ps.get_outputs()], ps.run(None, feeds)))
        code = r2[L18_ENTRY].astype(np.float64)
        emit({"phase": "prepare", "sanity_check": tag,
              "add_161_code_mean": float(code.mean()), "add_161_code_std": float(code.std())})


def adb(*a):
    env = dict(os.environ)
    env["MSYS2_ARG_CONV_EXCL"] = "/sdcard"
    return subprocess.run([ADB, *a], capture_output=True, text=True, env=env, errors="replace")


def wrap_one(tag):
    ctxdir = WORK / f"ctx_{tag}"
    bins = list(ctxdir.glob("*.bin"))
    assert bins, f"context binary が無い。先に bash scripts/build_context.sh work/p13d_v_probe/{tag}_qdq.onnx work/p13d_v_probe {tag}"
    import importlib
    wrap_epctx = importlib.import_module("wrap_epctx")
    out_path = WORK / f"{tag}_epctx.onnx"
    r = wrap_epctx.run(bins[0], WORK / f"{tag}_qdq.onnx", out_path, tag)
    emit({"phase": "wrap", "tag": tag, **{k: v for k, v in r.items() if k != "outputs"}})


def wrap():
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in MODELS:
        # 条件B は C-8（act-act MatMul は16bit不可）でビルド自体が通らないので飛ばす（2026-09-17）
        if not list((WORK / f"ctx_{tag}").glob("*.bin")):
            emit({"phase": "skip", "tag": tag, "reason": "context binary が無い（条件B は C-8 で実機不可）"})
            continue
        wrap_one(tag)


def run_device(tag, feeds):
    d = WORK / "in"
    d.mkdir(exist_ok=True)
    files = {}
    for k, v in feeds.items():
        p = d / f"{k}.bin"
        p.write_bytes(np.ascontiguousarray(v).tobytes())
        assert adb("push", str(p), f"{DEV}/inputs/l11_{k}.bin").returncode == 0
        files[k] = f"inputs/l11_{k}.bin"
    epctx = WORK / f"{tag}_epctx.onnx"
    assert adb("push", str(epctx), f"{DEV}/models/ops/{tag}_l11_epctx.onnx").returncode == 0
    job = {"job_id": tag, "task": "infer", "model": f"models/ops/{tag}_l11_epctx.onnx",
           "ep": {"provider": "QNN", "backend": "libQnnHtp.so",
                  "options": {"offload_graph_io_quantization": "0", "qnn_context_priority": "high"},
                  "session": {"intra_op_num_threads": 1, "log_severity_level": 2}},
           "output": {"dir": "out"}, "batches": [{"id": "000", "inputs": files}]}
    jp = WORK / f"{tag}_job.json"
    jp.write_text(json.dumps(job), encoding="utf-8")
    adb("push", str(jp), f"{DEV}/jobs/{tag}_job.json")
    adb("shell", f"rm -f {DEV}/out/last_result.json {DEV}/out/{tag}_000_*.bin")
    adb("shell", "am", "force-stop", "org.edgeslm.runner")
    adb("shell", "am", "start", "-n", "org.edgeslm.runner/.RunnerActivity",
        "-a", "org.edgeslm.runner.RUN", "--es", "job", f"jobs/{tag}_job.json")
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


def qparam_of(model, name):
    """nameを出力するQuantizeLinear/DequantizeLinearノード自身のscale/zero入力から量子化パラメータを読む。
    write/read側はDQ後float(present_v_L、repeat_interleave_*_DequantizeLinear_Output)、
    L18入口hiddenはadd_26と同じ方式でQの生コード(add_161_QuantizeLinear_Output)をタップしている
    (どちらもQ/DQに挟まれていない生の中間floatを新規fp32出力にするとHTP graph finalizeが失敗するため)。"""
    inits = {t.name: t for t in model.graph.initializer}
    from onnx import numpy_helper
    for n in model.graph.node:
        if n.op_type in ("DequantizeLinear", "QuantizeLinear") and n.output[0] == name:
            s = float(numpy_helper.to_array(inits[n.input[1]]))
            z = numpy_helper.to_array(inits[n.input[2]])
            return s, int(z.ravel()[0]), str(z.dtype)
    return None


def to_code(value, scale, zero, dtype):
    qmax = {"uint8": 255, "uint16": 65535}[dtype]
    code = np.round(value.astype(np.float64) / scale) + zero
    return np.clip(code, 0, qmax).astype(np.int64)


def compare():
    OUT.mkdir(parents=True, exist_ok=True)
    feeds = dict(np.load(WORK / "l11_inputs.npz"))
    obs_names = []
    for L in (11, 14, 17):
        obs_names += [WRITE_Q[L], READ_Q[L]]

    result = {}
    for tag in MODELS:
        # 条件B は v68 の C-8（act-act MatMul は16bit不可）でビルド自体が通らない。
        # context binary が無い tag は実機比較の対象から外す（2026-09-17）
        if not list((WORK / f"ctx_{tag}").glob("*.bin")):
            emit({"phase": "skip", "tag": tag, "reason": "context binary が無い（条件B は C-8 で実機不可）"})
            continue
        m = onnx.load(str(WORK / f"{tag}_qdq.onnx"), load_external_data=False)
        s = sess(WORK / f"{tag}_qdq.onnx")
        host = dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))
        dev, strict, ms = run_device(tag, feeds)
        emit({"phase": "compare", "tag": tag, "strict_no_cpu_fallback": strict, "ort_run_ms": ms})

        rows = {}
        for L in (11, 14, 17):
            for kind, name in (("write", WRITE_Q[L]), ("read", READ_Q[L])):
                qp = qparam_of(m, name)
                if qp is None or name not in dev:
                    rows[f"L{L}_{kind}"] = {"error": f"missing {name}"}
                    continue
                s_, z_, dt_ = qp
                if kind == "write":
                    # write(present_v_L)はDQ後float。scale/zeroから再量子化してcodeを復元する
                    hr = host[name].astype(np.float64).ravel()
                    dr = dev[name].astype(np.float64).ravel()
                    h = to_code(hr, s_, z_, dt_)
                    d = to_code(dr, s_, z_, dt_)
                else:
                    # read(repeat_interleave)はQの生コードを直接タップ(host/devとも整数)
                    h = host[name].astype(np.int64).ravel()
                    d = dev[name].astype(np.int64).ravel()
                    hr, dr = (h - z_) * s_, (d - z_) * s_
                diff = d - h
                rel = float(np.linalg.norm(dr - hr) / (np.linalg.norm(hr) or 1.0))
                cos_sim = float(np.dot(dr, hr) / ((np.linalg.norm(dr) * np.linalg.norm(hr)) or 1.0))
                qmax = {"uint8": 255, "uint16": 65535}[dt_]
                clip_rate = float(((h == 0) | (h == qmax)).mean())
                rows[f"L{L}_{kind}"] = {"scale": s_, "zero": z_, "dtype": dt_,
                                        "exact_rate": float((diff == 0).mean()),
                                        "within1_rate": float((np.abs(diff) <= 1).mean()),
                                        "code_max_diff": int(np.abs(diff).max()),
                                        "code_rms_diff": float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
                                        "real_rel_l2": rel, "cosine": cos_sim,
                                        "clip_rate_host": clip_rate}

        # L18入口hidden: add_26と同じ方式でQの生コードをタップ(host/devとも直接int)
        qp = qparam_of(m, L18_ENTRY)
        if qp is not None and L18_ENTRY in host and L18_ENTRY in dev:
            s_, z_, dt_ = qp
            h = host[L18_ENTRY].astype(np.int64).ravel()
            d = dev[L18_ENTRY].astype(np.int64).ravel()
            diff = d - h
            hr = (h - z_) * s_
            dr = (d - z_) * s_
            rel = float(np.linalg.norm(dr - hr) / (np.linalg.norm(hr) or 1.0))
            cos_sim = float(np.dot(dr, hr) / ((np.linalg.norm(dr) * np.linalg.norm(hr)) or 1.0))
            qmax = {"uint8": 255, "uint16": 65535}[dt_]
            clip_rate = float(((h == 0) | (h == qmax)).mean())
            rows["L18_entry_hidden"] = {"scale": s_, "zero": z_, "dtype": dt_,
                                        "exact_rate": float((diff == 0).mean()),
                                        "within1_rate": float((np.abs(diff) <= 1).mean()),
                                        "code_max_diff": int(np.abs(diff).max()),
                                        "code_rms_diff": float(np.sqrt((diff.astype(np.float64) ** 2).mean())),
                                        "real_rel_l2": rel, "cosine": cos_sim,
                                        "clip_rate_host": clip_rate}
        else:
            rows["L18_entry_hidden"] = {"error": f"missing {L18_ENTRY}"}
        result[tag] = rows
        del m, s
        print(f"\n  [{tag}] strict={strict}")
        for k, v in rows.items():
            if "error" in v:
                print(f"    {k}: {v['error']}")
            else:
                print(f"    {k:16s} real_rel={v['real_rel_l2']*100:6.2f}%  cos={v.get('cosine', float('nan')):.5f}  "
                      f"code_max_diff={v.get('code_max_diff','-')}")

    (OUT / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "compare", "wrote": str(OUT / "compare.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["prepare", "wrap", "compare"], required=True)
    a = ap.parse_args()
    {"prepare": prepare, "wrap": wrap, "compare": compare}[a.phase]()


if __name__ == "__main__":
    main()
