r"""本物のデコーダ層1枚を **Attention 側と FFN 側に割る**。

## なぜ必要か

実機とホストで同じ量子化グラフを走らせると、**層1枚で出力が 16.80% 変わる**
（2026-09-14 実測）。どの演算がそれを作っているかを知りたい。

既存の21個の単体 op モデルは **合成形状（hidden 128 / 4ヘッド）で較正データも無い**
ので使えない。range に起因する問題はランダム入力では出ない。
そこで**本物の層を切って、本物の activation で較正する**。

## パイプラインのどこで使うか

パイプラインの外。**誤差がどこで出ているかを調べる道具**である。

    本物の24層モデル → ここで層を Attention 側 / FFN 側に割る
                     → それぞれ量子化して context を作り、実機で比べる

## 使い方

    python -m app.pipeline.split_layer
    # work/ops/layer1_attn/ と work/ops/layer1_ffn/ に fp32 モデルと spec ができる
    # そのあとは通常どおり p2b_ops → build_context.sh → device_accuracy

## 切る場所

Qwen のデコーダ層は

    x → input_layernorm → attn → +x → h      ← ここが `add_4`
    h → post_attention_layernorm → mlp → +h → out

なので `add_4` で割ると、残差を含んだ意味のある2つの半分になる。

    attn 側: hidden_states -> add_4
    ffn  側: add_4         -> out
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help 用ではない（引数は無い）が、補足はコメントにまとめた。
# どの段か: StudyMemo 9.2 節「二分探索」。記録: results/P9_20260914_perchannel_device.md 17 節。
# 結果: デコーダ層 1 枚・同一入力・11 サンプルで、ホスト↔実機は FFN 側全体 14.37%、うち RMSNorm だけで 13.53%、
#   MLP だけでは 0.21%。→ RMSNorm が実機で崩れていると絞り込み、真因 C-13（gamma が符号付き 16bit で頭打ち）へ進んだ。
#   実機の数値は results/P9_20260914_ffn*_device.json など。
# 入力:  work/qwen/qwen05b_layer1_S16.onnx（Qwen2.5 の層 1 枚、系列長 16）、work/qwen/calib_layer1.npz・eval_layer1.npz。
# 出力:  work/ops/layer1_attn/・layer1_ffn/ に fp32 モデルと spec.json、work/qwen/{calib,eval}_layer1_ffn.npz。
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# ROOT はこのファイルから 2 つ上（プロジェクトの根）。SRC は切る元、CUT は割る場所（attn 側の残差 Add の出力）。
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "work/qwen/qwen05b_layer1_S16.onnx"
CUT = "add_4"
OPS = ROOT / "work/ops"


def extract(dst: Path, ins: list[str], outs: list[str]) -> None:
    """SRC から、入力 ins から出力 outs までの部分を切り出して dst に保存し、ノード数と入出力の名前を出す。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(SRC), str(dst), ins, outs)
    m = onnx.load(str(dst), load_external_data=False)
    print(f"  {dst.name}: {len(m.graph.node)} ノード  "
          f"in={[i.name for i in m.graph.input]} out={[o.name for o in m.graph.output]}")


def main() -> None:
    """層を 2 つに割り、FFN 側の較正・評価データを本物の activation で作り、2 つの spec を書く。"""
    import os
    os.chdir(ROOT)
    A = OPS / "layer1_attn/layer1_attn_fp32.onnx"
    B = OPS / "layer1_ffn/layer1_ffn_fp32.onnx"
    print("切り出し:")
    extract(A, ["hidden_states"], [CUT])
    extract(B, [CUT], ["out"])

    # FFN 側の入力は「Attention 側を fp32 で通した結果」＝本物の activation
    so = ort.SessionOptions(); so.log_severity_level = 3
    sa = ort.InferenceSession(str(A), so, providers=["CPUExecutionProvider"])
    for tag, src in (("calib", "work/qwen/calib_layer1.npz"),
                     ("eval", "work/qwen/eval_layer1.npz")):
        x = np.load(src)["hidden_states"]
        h = np.concatenate([sa.run(None, {"hidden_states": x[i:i+1]})[0]
                            for i in range(len(x))], 0)
        out = ROOT / f"work/qwen/{tag}_layer1_ffn.npz"
        np.savez(out, **{CUT: h})
        print(f"  {out.name}: {h.shape}  |値| p99.9 = {np.percentile(np.abs(h), 99.9):.2f}")

    # QK^T（Softmax へ届く MatMul）の名前を拾って swap 対象にする
    # Softmax から最大 5 段さかのぼって最初の MatMul を QK^T とみなし、左右入替（C-8 の swap）の対象にする。
    m = onnx.load(str(A), load_external_data=False)
    prod = {o: n for n in m.graph.node for o in n.output}
    qk = None
    for n in m.graph.node:
        if n.op_type != "Softmax":
            continue
        cur = n.input[0]
        for _ in range(5):
            p = prod.get(cur)
            if p is None:
                break
            if p.op_type == "MatMul":
                qk = p.name or "(名前なし)"
                break
            cur = p.input[0]
    print(f"  QK^T の MatMul: {qk}")

    # p2b_ops prepare が読む spec を書く（形は Qwen2.5 の [1, 16, 896]）。
    specs = {
        "layer1_attn": {"name": "layer1_attn", "note": "本物のデコーダ層の Attention 側（残差まで）",
                        "priority": "Gate", "standard_opset": True, "checker_passed": True,
                        "model": "work/ops/layer1_attn/layer1_attn_fp32.onnx",
                        "inputs": [{"name": "hidden_states", "shape": [1, 16, 896], "dtype": "float"}],
                        "outputs": [{"name": CUT, "shape": [1, 16, 896], "dtype": "float"}],
                        "calib_npz": str((ROOT / "work/qwen/calib_layer1.npz").as_posix()),
                        "swap_matmuls": [qk] if qk else []},
        "layer1_ffn": {"name": "layer1_ffn", "note": "本物のデコーダ層の FFN 側（残差まで）",
                       "priority": "Gate", "standard_opset": True, "checker_passed": True,
                       "model": "work/ops/layer1_ffn/layer1_ffn_fp32.onnx",
                       "inputs": [{"name": CUT, "shape": [1, 16, 896], "dtype": "float"}],
                       "outputs": [{"name": "out", "shape": [1, 16, 896], "dtype": "float"}],
                       "calib_npz": str((ROOT / "work/qwen/calib_layer1_ffn.npz").as_posix()),
                       "swap_matmuls": []},
    }
    for nm, sp in specs.items():
        p = OPS / nm / f"{nm}.spec.json"
        p.write_text(json.dumps(sp, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
