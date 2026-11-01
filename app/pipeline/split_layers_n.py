r"""24層 decode を **層ごとに切り出し、本物の入力で** ホスト↔実機を比べる準備をする。

## なぜ

C-13 のあとも 24層では ホスト↔実機 が 16〜18% 残る。2層では 2.43% なので
**層数に伴って増えている**。だが層ごとの誤差を並べただけでは

  - 前の層から伝わってきた誤差
  - その層自身が作った誤差

が混ざって分けられない（2026-09-14 に一度この読み違いをした）。

**同じ入力を与えて層単体で比べれば分離できる。** 入力は実データでなければならない
（Qwen2.5 は層によって activation のレンジが桁違いで、乱数では再現しない）。

## パイプラインのどこで使うか

パイプラインの外。**層ごとの誤差を「伝播」と「その層が作った分」に分ける道具**である。

## 使い方

    python -m app.pipeline.split_layers_n --layers 0 4 8 12 16 20 23
    # work/ops/declayer<N>/ に層ごとの fp32 モデルと spec、
    # work/qwen_dec24/{calib,eval}_layer<N>.npz に**その層の実際の入力**ができる

**注意**: 切り出した単層は C-8（`u16×u8` のみ）の壁で context 生成が通らないことがある
（2026-09-14 時点で未解決）。

## やること

  1. fp32 の 24層モデルから **各層の入力 hidden** を取り出す（グラフ出力に足して実行）
  2. 指定した層だけを切り出す（`past_k_N`/`past_v_N`/`cos`/`sin`/`attn_mask` も入力になる）
  3. その層の**実際の入力分布**で較正データを作り、spec を書く
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 3 章（層ごとの入力の大きさ。今のソースで同じ種類の数字を出すのはこの道具）・9 章。
# 記録: results/P9_20260914_perchannel_device.md 35〜36 節。層ごとの入力の最大は層 8 で 1646.73、層 23 で 71.76 に下がる。
#   同一入力での層単体の比較は、切り出した 1 層だけだと C-8 の壁（"Value 68, expected >= 73"）で context が作れず未完。
# 入力:  work/qwen_dec24/qwen05b_decode_L24_C64.onnx（Qwen2.5 の 24 層 decode）、--eval-npz・--calib-npz。
# 出力:  work/ops/declayer<N>/（fp32 モデルと spec）、work/qwen_dec24/{calib,eval}_layer<N>.npz。
#   途中で work/qwen_dec24/_expose.onnx（層の境目を出力に足したもの）も書く。
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# ROOT はプロジェクトの根。SRC は切る元の 24 層 decode（キャッシュ長 64）。
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "work/qwen_dec24/qwen05b_decode_L24_C64.onnx"
OPS = ROOT / "work/ops"


def layer_boundaries(model: onnx.ModelProto) -> list[str]:
    """各層の**出力**（= 次の層の入力）になっている残差 Add を層順に返す。"""
    g = model.graph
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    stream = []
    for n in g.node:                       # ノードは位相順に並んでいる
        if n.op_type != "Add":
            continue
        ops = {c.op_type for c in cons[n.output[0]]}
        if "Pow" in ops:                   # 次の RMSNorm へ入る = 残差ストリーム
            stream.append(n.output[0])
    # （RMSNorm の始まりは Pow（x²）なので、Pow へ入る Add ＝ 次の RMSNorm へ入る残差、と見分けている）
    # attn 残差 と mlp 残差 が交互に並ぶので、奇数番目（mlp 側）が層の出口
    return stream[1::2]


def main() -> None:
    """層の入口・出口を見つけ、実データで各層の本当の入力を取り出し、層ごとに切り出して spec を書く。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 23])
    ap.add_argument("--eval-npz", default="work/qwen_dec24/eval_dec24_big.npz")
    ap.add_argument("--calib-npz", default="work/qwen_dec24/calib_dec24.npz")
    ap.add_argument("--limit", type=int, default=48)
    args = ap.parse_args()
    import os
    os.chdir(ROOT)

    m = onnx.load(str(SRC), load_external_data=False)
    outs = layer_boundaries(m)
    print(f"層の出口テンソル {len(outs)} 個: {outs[:3]} … {outs[-2:]}")
    assert len(outs) >= 24, f"24 層ぶん見つからない（{len(outs)}）"
    outs = outs[:24]
    # 層 N の入力 = 層 N-1 の出口。層0 の入力は埋め込みの出力
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    first = outs[0]
    # （注: 下の n0 と cons は、その後で使われていない。層0 の入力は結局 Gather（埋め込み）の出力で決めている）
    n0 = prod[first]                       # 層0 の mlp 残差 Add
    # 層0 の入力は「層0 の attn 残差 Add の入力のうち、Add でない側」を辿る
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    gather = next(n for n in g.node if n.op_type == "Gather")
    emb = gather.output[0]
    ins = [emb] + outs[:-1]
    print(f"層の入口テンソル: 層0={ins[0]}  層1={ins[1]}  層23={ins[23]}")

    # ---- 実データで各層の入力を取り出す ----
    want = sorted({ins[L] for L in args.layers} | {outs[L] for L in args.layers})
    # 欲しい境目のテンソルをグラフ出力に足した写しを作り、CPU で実データを流して値を集める（--limit 件まで）。
    # 2GB を超えうるので外部データ形式で保存する。
    m2 = onnx.load(str(SRC))
    have = {o.name for o in m2.graph.output}
    for t in want:
        if t not in have:
            m2.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    tmp = ROOT / "work/qwen_dec24/_expose.onnx"
    onnx.save(m2, str(tmp), save_as_external_data=True, location="_expose.onnx.data")

    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    onames = [o.name for o in s.get_outputs()]
    inames = [i.name for i in s.get_inputs()]
    for tag, npz_path in (("eval", args.eval_npz), ("calib", args.calib_npz)):
        d = np.load(npz_path)
        n = min(args.limit, d[inames[0]].shape[0])
        acc = collections.defaultdict(list)
        for i in range(n):
            vals = s.run(None, {k: d[k][i:i + 1] for k in inames})
            for name, v in zip(onames, vals):
                if name in want:
                    acc[name].append(v)
        for L in args.layers:
            store = {ins[L]: np.concatenate(acc[ins[L]], 0)}
            # その層の単体モデルに要る他の入力（RoPE・マスク・その層の KV キャッシュ）も同じ件数だけ写す。
            for k in ("cos", "sin", "attn_mask", f"past_k_{L}", f"past_v_{L}"):
                if k in d:
                    store[k] = d[k][:n]
            p = ROOT / f"work/qwen_dec24/{tag}_layer{L}.npz"
            np.savez(p, **store)
            if tag == "eval":
                x = store[ins[L]]
                print(f"  層{L:2d}  入力 {x.shape}  |値| p50 {np.percentile(np.abs(x),50):8.4f}"
                      f"  p99.9 {np.percentile(np.abs(x),99.9):10.2f}  最大 {np.abs(x).max():10.2f}")

    # ---- 層を切り出して spec を書く ----
    for L in args.layers:
        nm = f"declayer{L}"
        dst = OPS / nm / f"{nm}_fp32.onnx"
        dst.parent.mkdir(parents=True, exist_ok=True)
        onnx.utils.extract_model(
            str(SRC), str(dst),
            [ins[L], "cos", "sin", "attn_mask", f"past_k_{L}", f"past_v_{L}"],
            [outs[L], f"present_k_{L}", f"present_v_{L}"])
        gg = onnx.load(str(dst), load_external_data=False).graph
        spec = {"name": nm, "note": f"24層 decode の層{L}（本物の入力で較正）",
                "priority": "Gate", "standard_opset": True, "checker_passed": True,
                "model": f"work/ops/{nm}/{nm}_fp32.onnx",
                "inputs": [{"name": i.name,
                            "shape": [dd.dim_value for dd in i.type.tensor_type.shape.dim],
                            "dtype": "float"} for i in gg.input],
                "outputs": [{"name": o.name,
                             "shape": [dd.dim_value for dd in o.type.tensor_type.shape.dim],
                             "dtype": "float"} for o in gg.output],
                "calib_npz": str((ROOT / f"work/qwen_dec24/calib_layer{L}.npz").as_posix()),
                "swap_matmuls": []}
        (OPS / nm / f"{nm}.spec.json").write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {nm}: {len(gg.node)} ノード  in={[i.name for i in gg.input]}")


if __name__ == "__main__":
    main()
