r"""QDQ の RMSNorm を **融合されない形** に書き換える（数学的には同一）。

## なぜ

`qairt-converter` は C-11 で書き換えた13ノードの RMSNorm を
**`RMSNormalization` 1個に畳んでしまう**（2026-09-14 に DLC で確認）。
畳まれると、こちらが設定した中間の量子化レンジは捨てられ、
HTP 自前のカーネルが走る。同じ入力でホストと **13.53%** ずれる。

融合を壊せば、こちらが書いた手順がそのまま実行されるはずである。
**数学的に等価なまま、パターンに一致しない形へ変える。**

手は2つ用意した。

  `--mode pow`   : `Pow(x, 2)` -> `Mul(x, x)`。**これでは崩れなかった**（converter が
                   Mul(x,x) も二乗と認識する）。記録として残す
  `--mode expose`: **中間テンソルをグラフ出力にも足す**。融合は普通「中間が単一消費者」を
                   要求するので、消費者が増えると畳めなくなる。
                   ★**数値は1ビットも変わらない**のが利点である

出力テンソル名は変えないので、前後の Q/DQ はそのまま使える。

## 使い方

    python -m app.pipeline.break_rmsnorm_fusion --qdq <in.onnx> --out <out.onnx>
## パイプラインのどこで使うか

パイプラインの外。**融合が原因かを確かめようとした道具**である。

**結論から言うと崩せなかった。** `Pow→Mul` も「中間テンソルをグラフ出力に足す」も
効かず、DLC には `RmsNorm` が残った（2026-09-14）。
真因は融合そのものではなく **gamma のエンコーディング不整合（C-13）** だったので、
この道具は使わずに済んだ。**同じ道を辿らないための記録として残している。**
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def expose(src: Path, dst: Path, names: list[str]) -> dict:
    """中間テンソルをグラフ出力にも追加して、融合パターンを崩す。

    計算は一切変えない。**融合されたかどうかだけを変える**ので、
    「融合が原因か」を確かめる A/B に向く。
    """
    import onnx
    from onnx import helper, TensorProto
    m = onnx.load(str(src))
    g = m.graph
    have = {o.name for o in g.output}
    produced = {o for n in g.node for o in n.output}
    added = []
    for nm in names:
        cand = nm if nm in produced else f"{nm}_QuantizeLinear_Output"
        # 量子化後のテンソルではなく **DQ を通した float** を出したいので、
        # `<名前>_DequantizeLinear_Output` があればそれを使う
        for c in (f"{nm}_DequantizeLinear_Output", cand):
            if c in produced and c not in have:
                g.output.append(helper.make_tensor_value_info(c, TensorProto.FLOAT, None))
                added.append(c)
                break
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst), save_as_external_data=False)
    onnx.checker.check_model(str(dst))
    return {"added": added}


def run(src: Path, dst: Path) -> dict:
    m = onnx.load(str(src))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    import numpy as np
    from onnx import numpy_helper as nh

    # Pow の指数が 2 のものだけを Mul(x, x) に置き換える
    prod = {o: n for n in g.node for o in n.output}

    def const_of(name: str):
        """DQ を1段だけ辿って定数を取る（QDQ では指数も量子化されている）。"""
        if name in inits:
            return nh.to_array(inits[name])
        n = prod.get(name)
        if n is not None and n.op_type == "DequantizeLinear" and n.input[0] in inits:
            q = nh.to_array(inits[n.input[0]]).astype(np.float64)
            sc = nh.to_array(inits[n.input[1]]).astype(np.float64)
            zp = nh.to_array(inits[n.input[2]]).astype(np.float64) if len(n.input) > 2 else 0.0
            return (q - zp) * sc
        return None

    changed = []
    for n in g.node:
        if n.op_type != "Pow" or len(n.input) != 2:
            continue
        e = const_of(n.input[1])
        if e is None or not np.allclose(np.ravel(e), 2.0, atol=1e-3):
            continue
        x = n.input[0]
        n.op_type = "Mul"
        del n.input[:]
        n.input.extend([x, x])      # ★同じテンソルを両辺に。値は完全に同じ
        changed.append(n.output[0])

    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst), save_as_external_data=False)
    onnx.checker.check_model(str(dst))
    return {"changed": changed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["pow", "expose"], default="expose")
    ap.add_argument("--expose", nargs="*", default=["pow_2", "mean_1", "add_5", "val_120"],
                    help="グラフ出力にも足す中間テンソル")
    args = ap.parse_args()
    if args.mode == "pow":
        r = run(args.qdq, args.out)
        print(f"Pow(x,2) -> Mul(x,x) に変えた箇所: {len(r['changed'])}")
    else:
        r = expose(args.qdq, args.out, args.expose)
        print(f"グラフ出力に足した中間テンソル: {len(r['added'])}")
        for c in r["added"]:
            print(f"   {c}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
