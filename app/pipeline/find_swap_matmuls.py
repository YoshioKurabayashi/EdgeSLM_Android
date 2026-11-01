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

## 入力と出力（2026-10-18 追記）

  入力 : 前処理の前の fp32 の ONNX（export したもの）。重みは読まない（load_external_data=False）。
  出力 : 表示と（--json で）{"qk": [...], "pv": [...], "other": [...], "weight": [...]}（MatMul のノード名）。
         "qk" の名前を spec の `swap_matmuls` に書くと、p2b_ops prepare（方式名に swap を含むとき）が
         preprocess.run(..., swap_matmuls=…) に渡し、swap_actact_matmul_operands() が入れ替える。
  検証 : seq16 の手作業のリストと完全に一致することを確かめてある（要件定義書 改訂 3.5 の ⑦）。
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
    """モデルの MatMul を「重み固定 / QK^T / P×V / 不明」の 4 つに分ける。

    引数: model_path（ONNX のパス）。戻り値: {"qk": [...], "pv": [...], "other": [...], "weight": [...]}
    （どれも MatMul のノード名のリスト。グラフの並び順）。
    判定の順は ① 重み固定 → ② 出力が Softmax に届く（QK^T）→ ③ 入力が Softmax から来る（P×V）→ ④ 不明。
    QK^T を先に判定するので、両方に当たる MatMul があれば QK^T に入る。
    """
    # 準備: テンソル名 → それを出すノード（prod）、テンソル名 → それを使うノード（cons）、初期化子の名前
    m = onnx.load(str(model_path), load_external_data=False)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    inits = {i.name for i in g.initializer}

    def is_weight(name: str, depth: int = 0) -> bool:
        """name が初期化子か、形だけ変える op（PASS_THROUGH）を 6 段まで遡って初期化子に着くなら True。
        fp32 の export では重みが Transpose 経由で MatMul に入るため、遡らないと見落とす。"""
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
        """name を下流へたどり、Mul（スケール）・Add（マスク）だけを経て 5 段以内に Softmax に届けば True（QK^T の印）。"""
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
        """name を上流へたどり、形だけ変える op を 4 段まで遡って Softmax の出力なら True（P×V の印）。"""
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

    # 判定の本体: MatMul を 1 つずつ見て、上の 3 つの関数で振り分ける
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
    """コマンドの入口。classify() の結果の本数と先頭の名前を表示し、--json があれば書き出す。"""
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
