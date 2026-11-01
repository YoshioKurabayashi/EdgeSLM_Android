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
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# ★2026-10-18 訂正: docstring の観測点の表の「cache write後(②) : present_v_{L}_QuantizeLinear_Output」は古い。
#   実際のタップは present_v_{L}（DequantizeLinear の後の float）で、compare() が scale/zero から整数コードに
#   戻している（下の WRITE_Q とその上のコメント）。量子化コードのまま出力にすると qairt-converter が Q→DQ を
#   畳めずに変換が落ちたため。read 側（③）は QuantizeLinear の生コード（uint8）を直接タップしている。
# どの段か: StudyMemo 11.6 節「2つの軸を混同しない」・11.7 節 (a)・11.8 節 (2)。
# 結果（results/P13d_20260917_v_l11_17_device_probe.md）:
#   条件B は C-8 で context binary が作れない（node_matmul_23 が "Value 68, expected >= 73" で落ちる）。
#   条件A は Host↔HTP の差を縮めなかった（L17 write 1.74% → 2.09%、L18 入口 8.23% → 8.92%）。
#   そもそも層11-17 の V は Host↔HTP では最初からほとんどずれていなかった（read 1.14〜3.04%・write 0.07〜1.74%・
#   一致率 99.6% 以上）。ずれは層の境目の残差（L18 入口、scale 0.113）に出ている。
# 入力:  baseline（SOURCE）と P13c の条件A・条件B のモデル、評価点「点B（音楽）」のトークン列（block_revert_ab.eval_points）。
# 出力:  work/p13d_v_probe/（部分モデル・入力 npz・context binary・EPContext・実機の出力）、
#        results/P13d_20260917_v_l11_17_device_probe/compare.json。
# ---------------------------------------------------------------------------
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

# 層の入口の残差と、v_proj の出力の名前（ONNX のノード名。層ごとに番号がずれる）。
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

# 比べる 3 つのモデルと、部分モデルの入力（層11 の入口の残差・層11-17 の KV・cos/sin/mask）。
MODELS = {"baseline": SOURCE, "condA": COND_A, "condB": COND_B}
PARTIAL_INPUTS = [L11_ENTRY] + [f"past_k_{L}" for L in range(11, 18)] + \
    [f"past_v_{L}" for L in range(11, 18)] + ["cos", "sin", "attn_mask"]


def emit(o):
    """途中経過を 1 行の JSON で出す。"""
    print(json.dumps(o, ensure_ascii=False), flush=True)


def sess(p):
    """CPU の ORT セッション（Host 側の計算）を作る。"""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def prepare():
    """3 つのモデルから層11-17 だけの部分モデルを切り出し、実機に与える入力（l11_inputs.npz）を作る。

    入力は「点B（音楽をかけて）」の対話の位置 41 を処理する時のもの: その直前までのキャッシュ（past_k/v_11..17）と、
    位置 41 の層11 の入口の残差（add_98）。出力: work/p13d_v_probe/{tag}_qdq.onnx・l11_inputs.npz。
    """
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

    # ① onnx.utils.extract_model で、指定した入力から指定した出力までの部分だけを切り出す（ノードは書き換えない）。
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

    # ② baseline を 1 トークンずつ流して位置 41 の直前で止め、その時のキャッシュと次の 1 歩の入力を保存する。
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
    # 層11-17 の past_k/past_v（各 [1, KV, 256, D]）だけを残す。
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
    # ③ Host での確認: 3 モデルに同じ入力を与え、層18 の入口のコードの平均・ばらつきを出す。
    for tag in MODELS:
        ps = sess(WORK / f"{tag}_qdq.onnx")
        r2 = dict(zip([o.name for o in ps.get_outputs()], ps.run(None, feeds)))
        code = r2[L18_ENTRY].astype(np.float64)
        emit({"phase": "prepare", "sanity_check": tag,
              "add_161_code_mean": float(code.mean()), "add_161_code_std": float(code.std())})


def adb(*a):
    """adb を呼ぶ。Git Bash から呼んだ時に /sdcard が Windows のパスに書き換えられないよう MSYS2_ARG_CONV_EXCL を付ける。"""
    env = dict(os.environ)
    env["MSYS2_ARG_CONV_EXCL"] = "/sdcard"
    return subprocess.run([ADB, *a], capture_output=True, text=True, env=env, errors="replace")


def wrap_one(tag):
    """build_context.sh が work/p13d_v_probe/ctx_{tag}/ に作った context binary を EPContext の ONNX で包む。

    embed を指定していないので wrap_epctx.run の既定（embed=True、context binary を ONNX の中に入れる）になる。
    """
    ctxdir = WORK / f"ctx_{tag}"
    bins = list(ctxdir.glob("*.bin"))
    assert bins, f"context binary が無い。先に bash scripts/build_context.sh work/p13d_v_probe/{tag}_qdq.onnx work/p13d_v_probe {tag}"
    import importlib
    wrap_epctx = importlib.import_module("wrap_epctx")
    out_path = WORK / f"{tag}_epctx.onnx"
    r = wrap_epctx.run(bins[0], WORK / f"{tag}_qdq.onnx", out_path, tag)
    emit({"phase": "wrap", "tag": tag, **{k: v for k, v in r.items() if k != "outputs"}})


def wrap():
    """context binary がある tag だけを包む（条件B は C-8 で作れないので飛ばす）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in MODELS:
        # 条件B は C-8（act-act MatMul は16bit不可）でビルド自体が通らないので飛ばす（2026-09-17）
        if not list((WORK / f"ctx_{tag}").glob("*.bin")):
            emit({"phase": "skip", "tag": tag, "reason": "context binary が無い（条件B は C-8 で実機不可）"})
            continue
        wrap_one(tag)


def run_device(tag, feeds):
    """入力を .bin にして push し、job（task=infer）で実機のアプリに推論させ、出力の .bin を pull して返す。

    戻り値: (出力名 → 配列, strict_no_cpu_fallback（全ノード NPU で開けたか）, OrtRun の ms)。
    """
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
    # 前回の結果を消してからアプリを止めて起動し直し、last_result.json ができるまで 2 秒ごとに見る（最大 180 秒）。
    # 読んだ結果の job_id が自分のものかを確かめる（別のジョブの結果を読まないため）。
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
    (どちらもQ/DQに挟まれていない生の中間floatを新規fp32出力にするとHTP graph finalizeが失敗するため)。
    ★2026-10-18 訂正: 旧「write/read側はDQ後float(…repeat_interleave_*_DequantizeLinear_Output)」のうち read 側は古い。
    今の READ_Q は repeat_interleave_*_QuantizeLinear_Output（Q の生コード、uint8）をタップしている（READ_Q の上のコメント）。
    この関数は Q の出力名にも DQ の出力名にも使え、そのノードの (scale, zero, zero の dtype 名) を返す。見つからなければ None。"""
    inits = {t.name: t for t in model.graph.initializer}
    from onnx import numpy_helper
    for n in model.graph.node:
        if n.op_type in ("DequantizeLinear", "QuantizeLinear") and n.output[0] == name:
            s = float(numpy_helper.to_array(inits[n.input[1]]))
            z = numpy_helper.to_array(inits[n.input[2]])
            return s, int(z.ravel()[0]), str(z.dtype)
    return None


def to_code(value, scale, zero, dtype):
    """float の値を、scale・zero で整数コードに戻す（round して 0〜qmax に収める）。戻り値は int64 の配列。"""
    qmax = {"uint8": 255, "uint16": 65535}[dtype]
    code = np.round(value.astype(np.float64) / scale) + zero
    return np.clip(code, 0, qmax).astype(np.int64)


def compare():
    """Host と実機に同じ入力を与え、層11・14・17 の V の write/read と層18 の入口を整数コードで比べる。

    指標: 一致率・±1 段以内の率・コードの最大差・コード差の RMS・実数の相対 L2（分母は Host の実数値）・
    cos 類似度・Host のコードが 0 か最大（飽和）の率。出力: compare.json。
    """
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
                # 同じ入力での整数コードの差（実機 − Host）。実数の誤差は scale を掛けて実数に戻してから測る（StudyMemo 11.8 (1)）。
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

        # 層18 の入口（層17 の出口）の残差。scale は 0.113 で、P12j の add_26 と同じ粗い格子（記録で 8.23%）。
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
    """--phase prepare / wrap / compare のどれかを走らせる（間に build_context.sh を手で流す）。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["prepare", "wrap", "compare"], required=True)
    a = ap.parse_args()
    {"prepare": prepare, "wrap": wrap, "compare": compare}[a.phase]()


if __name__ == "__main__":
    main()
