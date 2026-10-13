r"""左右入替の対象（QK^T の MatMul）をグラフから自動判定する。

## なぜ必要か

C-8 により v68 の MatMul は `u16 × u8` しか通らないので、**8bit にできるのは右辺だけ**。
Attention の `MatMul(Q, K^T)` では右辺が K で、K の方がレンジが広い（±130 に対し
Q は [-10, 6]、1段階で15倍の差）。`A × B = (B^T × A^T)^T` で入れ替えると
**細かい方（Q）を 8bit 側に置ける**（[DoC/StudyMemo/05_GraphRewrite.md] 5.3節）。

入れ替えるのは **QK^T だけ**で、P×V（softmax の出力 × V）は対象外である。
ところが **ノード名は export ごとに変わる**:

    qwen_full（HF の forward 由来）   : node_matmul_1, node_matmul_3, ...（奇数）
    qwen_pm  （自前の QwenPrefill）   : node_matmul,   node_matmul_2, ...（偶数）

手で書き写すと**取り違えても気付かない**（P×V を入れ替えても動いてしまい、
精度だけ落ちる）。グラフの形から判定する。

## 判定のしかた

| 種別 | 特徴 |
|---|---|
| **QK^T** | 出力が `Mul`/`Add` を経て **`Softmax` に届く** |
| P×V | 入力のどちらかが **`Softmax` から来ている** |

重み固定の MatMul（projection / FFN / lm_head）は `Transpose` 等を辿って
初期化子に着くかで除外する。**「入力名が初期化子か」だけでは判定できない**——
fp32 では重みが `Transpose` 経由で入るため（2026-09-12 / 09-13 に2回踏んだ）。

## 使い方

    python -m app.pipeline.find_swap_matmuls work/qwen_pm64/qwen05b_prefillmask_L24_S64_last.onnx
    python -m app.pipeline.find_swap_matmuls <model> --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import onnx

# 形だけ変える op。重みを辿るときと Softmax を辿るときに透過させる。
PASS_THROUGH = ("Transpose", "Reshape", "Identity", "Cast", "Unsqueeze", "Squeeze")


def classify(model_path: str | Path) -> dict[str, list[str]]:
    m = onnx.load(str(model_path), load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    inits = {i.name for i in g.initializer}

    def is_weight(name: str, depth: int = 0) -> bool:
        if name in inits:
            return True
        if depth > 6:
            return False
        n = prod.get(name)
        if n is None:
            return False
        if n.op_type in PASS_THROUGH:
            return is_weight(n.input[0], depth + 1)
        return False

    def reaches_softmax(name: str, depth: int = 0) -> bool:
        if depth > 5:
            return False
        for c in cons.get(name, []):
            if c.op_type == "Softmax":
                return True
            # スケール乗算とマスク加算だけを跨ぐ。他の op を跨ぐと誤判定しうる。
            if c.op_type in ("Mul", "Add") and reaches_softmax(c.output[0], depth + 1):
                return True
        return False

    def from_softmax(name: str, depth: int = 0) -> bool:
        if depth > 4:
            return False
        n = prod.get(name)
        if n is None:
            return False
        if n.op_type == "Softmax":
            return True
        if n.op_type in PASS_THROUGH:
            return from_softmax(n.input[0], depth + 1)
        return False

    qk, pv, other, weight = [], [], [], []
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        if is_weight(n.input[0]) or is_weight(n.input[1]):
            weight.append(n.name)
        elif reaches_softmax(n.output[0]):
            qk.append(n.name)
        elif from_softmax(n.input[0]) or from_softmax(n.input[1]):
            pv.append(n.name)
        else:
            other.append(n.name)
    return {"qk": qk, "pv": pv, "other": other, "weight": weight}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--json", type=Path, help="判定結果を書き出す")
    args = ap.parse_args()

    r = classify(args.model)
    print(f"MatMul の内訳: 重み固定 {len(r['weight'])} / "
          f"QK^T {len(r['qk'])} / P×V {len(r['pv'])} / 不明 {len(r['other'])}")
    print(f"  QK^T（入替の対象）: {r['qk'][:4]} ...")
    print(f"  P×V （対象外）    : {r['pv'][:4]} ...")
    if r["other"]:
        # 分類できないものがあれば**黙って進めない**。入替漏れは精度にだけ出る。
        print(f"  ★不明が {len(r['other'])} 件ある。手で確認すること: {r['other'][:5]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
