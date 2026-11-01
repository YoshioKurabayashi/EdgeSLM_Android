"""P2 の足場となる小さな fp32 ONNX を生成する。

なぜ自作の小モデルから始めるのか（基本設計② 2.3節・8.1節と同じ「変数を1つに絞る」原則）:
  パイプライン自体が未検証の段階で、いきなり実モデル（BERT や Qwen）を投入すると、
  失敗したときに「パイプラインの問題」か「モデル固有の問題」か切り分けられない。
  まず**中身を完全に把握している小モデル**で S1〜S9 を通し、パイプラインの疑いを消す。

このモデルに入れてある要素と狙い:
  - Conv + BatchNormalization  → C-5（BN 畳み込み）の経路を必ず通す
  - Relu / MaxPool             → 素直に量子化される op
  - MatMul（Gemm ではなく）     → HTP の MatMul 対応型（uint8/uint16 の組合せ）を踏む
  - 固定形状のみ                → QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: P2「パイプラインを端から端まで通す」（results/P2_20260912_pipeline_e2e.md）。この testnet で
#   実機 median 0.277 ms（DoC/中間まとめ_20260912.md）。
# 出力:  --out（既定 work/testnet/testnet_fp32.onnx）。入力 [1, 3, size, size]、出力 [1, 4]、opset 17・IR 9。
# 動かし方: & $PY2 -m app.pipeline.make_test_model --out work/testnet/testnet_fp32.onnx
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    """配列を float32 の初期化子（重み）にする。"""
    return numpy_helper.from_array(arr.astype(np.float32), name)


def build(out_path: Path, size: int = 32, in_ch: int = 3, hidden: int = 8, classes: int = 4) -> None:
    """小さな CNN（Conv×2・BN×2・ReLU・MaxPool・GAP・MatMul・Add）を組み立て、検査して保存する。

    重みは乱数（seed 0 で固定。毎回同じモデルになる）。精度ではなく「変換の経路が通るか」を見るためのもの。
    """
    rng = np.random.default_rng(0)

    # Conv -> BN -> Relu -> MaxPool -> Conv -> BN -> Relu -> GlobalAvgPool -> MatMul -> Add
    inits = [
        _init("conv1_w", rng.normal(0, 0.1, (hidden, in_ch, 3, 3))),
        _init("bn1_scale", rng.uniform(0.8, 1.2, (hidden,))),
        _init("bn1_b", rng.normal(0, 0.05, (hidden,))),
        _init("bn1_mean", rng.normal(0, 0.05, (hidden,))),
        _init("bn1_var", rng.uniform(0.5, 1.5, (hidden,))),
        _init("conv2_w", rng.normal(0, 0.1, (hidden, hidden, 3, 3))),
        _init("bn2_scale", rng.uniform(0.8, 1.2, (hidden,))),
        _init("bn2_b", rng.normal(0, 0.05, (hidden,))),
        _init("bn2_mean", rng.normal(0, 0.05, (hidden,))),
        _init("bn2_var", rng.uniform(0.5, 1.5, (hidden,))),
        _init("fc_w", rng.normal(0, 0.1, (hidden, classes))),
        _init("fc_b", rng.normal(0, 0.05, (classes,))),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "conv1_w"], ["c1"], "conv1",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("BatchNormalization",
                         ["c1", "bn1_scale", "bn1_b", "bn1_mean", "bn1_var"], ["b1"], "bn1",
                         epsilon=1e-5),
        helper.make_node("Relu", ["b1"], ["r1"], "relu1"),
        helper.make_node("MaxPool", ["r1"], ["p1"], "pool1",
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Conv", ["p1", "conv2_w"], ["c2"], "conv2",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("BatchNormalization",
                         ["c2", "bn2_scale", "bn2_b", "bn2_mean", "bn2_var"], ["b2"], "bn2",
                         epsilon=1e-5),
        helper.make_node("Relu", ["b2"], ["r2"], "relu2"),
        helper.make_node("GlobalAveragePool", ["r2"], ["g1"], "gap"),
        # (1, hidden, 1, 1) -> (1, hidden)
        helper.make_node("Reshape", ["g1", "flat_shape"], ["f1"], "flatten"),
        helper.make_node("MatMul", ["f1", "fc_w"], ["m1"], "matmul"),
        helper.make_node("Add", ["m1", "fc_b"], ["output"], "bias_add"),
    ]
    inits.append(numpy_helper.from_array(np.array([1, hidden], dtype=np.int64), "flat_shape"))

    graph = helper.make_graph(
        nodes,
        "edgeslm_p2_testnet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, in_ch, size, size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, classes])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))

    n_bn = sum(1 for n in model.graph.node if n.op_type == "BatchNormalization")
    print(f"WROTE {out_path}  nodes={len(model.graph.node)}  BatchNormalization={n_bn}")
    print(f"  input : [1,{in_ch},{size},{size}] float32")
    print(f"  output: [1,{classes}] float32")


def main() -> None:
    """引数を読んで build を呼ぶ。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work/testnet/testnet_fp32.onnx"))
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    build(args.out, size=args.size)


if __name__ == "__main__":
    main()
