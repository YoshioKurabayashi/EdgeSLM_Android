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

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr.astype(np.float32), name)


def build(out_path: Path, size: int = 32, in_ch: int = 3, hidden: int = 8, classes: int = 4) -> None:
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work/testnet/testnet_fp32.onnx"))
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    build(args.out, size=args.size)


if __name__ == "__main__":
    main()
