"""QDQ モデルを直接書き換えて `uint16 × uint8` の MatMul を作る（C-8 の再検証）

## なぜ必要か

QNN HTP の MatMul 対応は **uint16×uint16 だけが不可**で、
**uint16×uint8 / uint8×uint16 は可**とされている。もしそうなら、
attention の QK^T を

    Q(uint16) × K(uint8) -> スコア(uint16)

にできる。**スコアを 16bit に保てる**ので、C-8 の代償
（スコアが 8bit になり softmax が壊れる）を根本から避けられる。

## なぜ ORT ではできないのか

`get_qnn_qdq_config` は MatMul の入力型を揃えようとする:
- 既定（`add_qtype_converts=True`）… 片側だけ u8 を指定すると、**もう片方にも
  変換ノードを挿して u8 へ落とす**（実測: `add_1_prescaled_convert_...` が出現）
- `add_qtype_converts=False` … **指定そのものが無視され**、全部 u16 に戻る

つまり **ORT の量子化器は u16×u8 の MatMul を表現できない**。
これは**ツールの制約**であって、**ハードウェアの制約とは別**である。
両者を混同したまま「v68 では無理」と結論しないために、ここで切り分ける。

## やること

素の A16W8 QDQ モデル（全部 uint16）を入力に、
活性化×活性化 MatMul の**片側入力だけ** Q/DQ を uint8 に作り替える。
スケールは既存のレンジから再計算するので、表している値域は変わらない。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

U16_MAX = 65535
U8_MAX = 255


def _range_of(scale: np.ndarray, zp: np.ndarray, qmax: int) -> tuple[float, float]:
    s = float(np.asarray(scale).reshape(-1)[0])
    z = float(np.asarray(zp).reshape(-1)[0])
    lo = (0 - z) * s
    hi = (qmax - z) * s
    return lo, hi


def force(src: Path, dst: Path, side: str = "rhs") -> dict:
    model = onnx.load(str(src))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    arr = {k: numpy_helper.to_array(v) for k, v in inits.items()}
    prod = {o: n for n in g.node for o in n.output}

    def upstream_q(tensor: str):
        """DQ をたどって、その値を作っている QuantizeLinear を返す。"""
        n = prod.get(tensor)
        while n is not None and n.op_type == "DequantizeLinear":
            dq = n
            n = prod.get(n.input[0])
            if n is not None and n.op_type == "QuantizeLinear":
                return n, dq
        return None, None

    changed = []
    for node in g.node:
        if node.op_type != "MatMul":
            continue
        # 両辺が活性化のものだけが対象。
        # ★QDQ モデルでは**重みも DequantizeLinear 経由**で入ってくるので、
        #   「入力名が initializer か」では判定できない。DQ/Q を辿って
        #   大元が initializer かどうかを見る（一度これで重み側まで書き換えた）。
        def is_weight(t: str) -> bool:
            seen = 0
            while t in prod and seen < 4:
                n2 = prod[t]
                if n2.op_type not in ("DequantizeLinear", "QuantizeLinear"):
                    return False
                t = n2.input[0]
                seen += 1
            return t in inits
        if any(is_weight(i) for i in node.input):
            continue
        target = node.input[1] if side == "rhs" else node.input[0]
        q, dq = upstream_q(target)
        if q is None or len(q.input) < 3:
            continue
        s_name, z_name = q.input[1], q.input[2]
        if z_name not in arr or arr[z_name].dtype not in (np.uint16, np.int16):
            continue  # すでに 8bit か、扱えない形

        lo, hi = _range_of(arr[s_name], arr[z_name], U16_MAX)
        # 同じ値域を uint8 で表す。値域は変えない（変えると別のモデルになる）
        new_scale = np.float32((hi - lo) / U8_MAX)
        new_zp = np.uint8(int(round(-lo / float(new_scale))))

        s2, z2 = f"{s_name}_u8", f"{z_name}_u8"
        g.initializer.append(numpy_helper.from_array(np.array(new_scale, dtype=np.float32), s2))
        g.initializer.append(numpy_helper.from_array(np.array(new_zp, dtype=np.uint8), z2))
        q.input[1], q.input[2] = s2, z2
        dq.input[1], dq.input[2] = s2, z2

        changed.append({
            "matmul": node.name or "(no name)",
            "tensor": target,
            "side": side,
            "range": [round(lo, 4), round(hi, 4)],
            "u16_step": round((hi - lo) / U16_MAX, 6),
            "u8_step": round(float(new_scale), 6),
        })

    # 値の型が変わるので value_info の型注釈は消しておく（残すと checker が矛盾を指摘する）
    del g.value_info[:]
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return {"changed": changed, "out": str(dst).replace("\\", "/")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="素の A16W8 QDQ モデル")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side", choices=["rhs", "lhs"], default="rhs")
    args = ap.parse_args()

    r = force(args.src, args.out, args.side)
    print(f"{len(r['changed'])} 箇所を uint16 -> uint8 に書き換えた（{args.side} 側）")
    for c in r["changed"]:
        print(f"  {c['matmul']:22s} {c['tensor'][:26]:28s} "
              f"range={c['range']}  step {c['u16_step']} -> {c['u8_step']}")
    print(f"  -> {r['out']}")


if __name__ == "__main__":
    main()
