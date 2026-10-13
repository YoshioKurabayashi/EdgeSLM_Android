r"""重みを per-channel で量子化させる指定を作り、**生成物を検査する**。

## なぜ必要か

`get_qnn_qdq_config(..., per_channel=True)` を渡していたのに、
生成された QDQ は **390本すべて per-tensor** だった（2026-09-13 に判明）。
誤差の主因はこれで、`MatMul` の重みを per-channel にすると
重みの相対L2 が中央値 3.79% → 0.95%（3.8倍）まで下がる見込みである。

ORT 1.30.0 の `qdq_quantizer.py` を読むと:

  - per-channel の可否は `is_tensor_per_channel()` が決めるが、
    **`MatMul` ではそこへ到達しない**
  - `QDQOpTypePerChannelSupportToAxis` は**軸の既定値を差し替えるだけ**で、
    per-channel を有効化する経路ではない（最小モデルで確認: per-tensor のまま）
  - **テンソル単位の `TensorQuantOverrides` に `axis` を書くと per-channel になる**
    （`has_per_channel_overrides()` は `"axis" in overrides_list[0]` で判定する）

書式は `{"重みの名前": [{"axis": 1}]}`。リストの中は**辞書**である
（二重リストにすると `ValueError: ... are not in a dict`）。

## パイプラインのどこで使うか

    S2 前処理 → **S4 量子化（ここ）** → S5 DLC変換 → S6 context生成 → 実機

`p2b_ops.py` の S4 から呼ばれる。spec に `per_channel_weights: true` があるとき、
量子化の**前**に `build_overrides()` で指定を作り、**直後**に `verify()` で
本当にそうなったかを確かめる。**約1時間の context 生成に入る前に止めるのが目的。**

## 使い方

    # 量子化の対象になる重みを数える（量子化前の prepped モデル）
    python -m app.pipeline.weight_percg --prepped work/ops/<name>/<name>_prepped.onnx

    # できあがった QDQ を検査する（期待と違えば異常終了）
    python -m app.pipeline.weight_percg --qdq work/ops/<name>/<name>_final_qdq.onnx --expect 168

## 軸の決め方

`MatMul(x[..., K], W[K, N])` の出力チャンネルは N なので **axis = 1**。
最小モデルで axis=1（出力の相対L2 0.626%）が axis=0（0.783%）より良く、
生成物の scale 数が N と一致することも確認した。

## ★設定ではなく生成物を見る

`verify()` は**できあがった QDQ** を開いて per-channel 本数・scale の形・axis を数える。
期待と違えば例外を投げて止める。**「per_channel=True を渡した」は根拠にならない。**
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import onnx

# per-channel にしても意味がない／対応していない op
SKIP_OPS = ("Gather",)


def matmul_weight_axes(model_path: str | Path, min_elems: int = 1024) -> dict[str, int]:
    """`MatMul` の重み（初期化子）と、その channel axis を返す。

    2次元で、MatMul の**右辺**に入っている初期化子だけを対象にする。
    左辺に重みが来る形（左右入替の結果など）は対象外にして取りこぼしを明示する。
    """
    m = onnx.load(str(model_path), load_external_data=False)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    out: dict[str, int] = {}

    def trace_init(name: str, depth: int = 0) -> str | None:
        """Transpose 等を辿って初期化子に着けばその名前を返す。"""
        if name in inits:
            return name
        if depth > 4:
            return None
        n = prod.get(name)
        if n is None or n.op_type not in ("Transpose", "Reshape", "Identity", "Cast"):
            return None
        return trace_init(n.input[0], depth + 1)

    # 初期化子ごとに「最終的に何に使われているか」を集める
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    PASS = ("Transpose", "Reshape", "Identity", "Cast")

    def terminal_ops(name: str, depth: int = 0) -> set[str]:
        if depth > 4:
            return {"?"}
        got: set[str] = set()
        for c in cons.get(name, []):
            if c.op_type in PASS:
                got |= terminal_ops(c.output[0], depth + 1)
            else:
                got.add(c.op_type)
        return got

    skipped: dict[str, set[str]] = {}
    for n in g.node:
        if n.op_type != "MatMul" or len(n.input) < 2:
            continue
        nm = trace_init(n.input[1])
        if nm is None:
            continue
        t = inits[nm]
        if len(t.dims) != 2 or int(np.prod(t.dims)) < min_elems:
            continue
        # ★**MatMul の右辺としてのみ使われる**初期化子に限る。
        #   `tie_word_embeddings` のため `step.lm_head.weight` は
        #   `Gather`（埋め込み表引き）と lm_head の両方で共有されている。
        #   そこへ axis を付けると、活性化側の `embedding` にも per-axis が及んで
        #   実行時に落ちる（2026-09-13 に実際に踏んだ）:
        #     "For per axis quantization, scale must be 1D tensor with size 64"
        uses = terminal_ops(nm)
        if uses != {"MatMul"}:
            skipped[nm] = uses
            continue
        out[nm] = 1          # [K, N] の N が出力チャンネル
    if skipped:
        for nm, uses in list(skipped.items())[:3]:
            print(f"  per-channel から除外: {nm}（{sorted(uses)} でも使われている）")
    return out


def build_overrides(model_path: str | Path, base: dict | None = None,
                    min_elems: int = 1024) -> tuple[dict, dict[str, int]]:
    """`TensorQuantOverrides` に per-channel の指定を足して返す。"""
    axes = matmul_weight_axes(model_path, min_elems)
    ov = dict(base or {})
    for nm, ax in axes.items():
        if nm in ov:
            # 既に別の指定がある場合は上書きしない（意図が競合するため）
            continue
        ov[nm] = [{"axis": ax}]
    return ov, axes


def inspect_weights(qdq_path: str | Path) -> dict:
    """できあがった QDQ の重み量子化を数える。**これが真実。**"""
    m = onnx.load(str(qdq_path), load_external_data=False)
    inits = {t.name: t for t in m.graph.initializer}
    TYPE = {2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32"}
    rows = []
    for nd in m.graph.node:
        if nd.op_type != "DequantizeLinear" or nd.input[0] not in inits:
            continue
        w = inits[nd.input[0]]
        sc = inits.get(nd.input[1])
        n_sc = int(np.prod(sc.dims)) if sc is not None and sc.dims else 1
        zp = inits.get(nd.input[2]) if len(nd.input) > 2 else None
        n_zp = int(np.prod(zp.dims)) if zp is not None and zp.dims else 1
        ax = next((a.i for a in nd.attribute if a.name == "axis"), None)
        rows.append({"name": nd.input[0], "dtype": TYPE.get(w.data_type, str(w.data_type)),
                     "dims": [int(d) for d in w.dims], "n_scale": n_sc,
                     "n_zero_point": n_zp, "axis": ax,
                     "per_channel": n_sc > 1,
                     "elems": int(np.prod(w.dims)) if w.dims else 1})
    big = [r for r in rows if r["elems"] >= 1024]
    return {
        "total": len(rows),
        "per_channel": sum(1 for r in rows if r["per_channel"]),
        "per_tensor": sum(1 for r in rows if not r["per_channel"]),
        "big_total": len(big),
        "big_per_channel": sum(1 for r in big if r["per_channel"]),
        "big_per_tensor": sum(1 for r in big if not r["per_channel"]),
        "by_dtype": dict(collections.Counter(r["dtype"] for r in rows)),
        "axes": dict(collections.Counter(str(r["axis"]) for r in big)),
        "mismatched_zp": [r["name"] for r in rows if r["n_scale"] != r["n_zero_point"]][:5],
        "rows": rows,
    }


def verify(qdq_path: str | Path, expect_per_channel: int,
           expect_axis: int = 1, strict: bool = True) -> dict:
    """期待どおり per-channel になっているかを確かめる。違えば例外。

    **context 生成（約1時間）へ進む前にここで止める。**
    """
    info = inspect_weights(qdq_path)
    problems = []
    if info["big_per_channel"] != expect_per_channel:
        problems.append(f"per-channel の大きい重みが {info['big_per_channel']} 本"
                        f"（期待 {expect_per_channel} 本）")
    bad_axis = {k: v for k, v in info["axes"].items() if k not in (str(expect_axis), "None")}
    if bad_axis:
        problems.append(f"想定外の axis: {bad_axis}")
    if info["mismatched_zp"]:
        problems.append(f"scale と zero_point の数が違う: {info['mismatched_zp']}")
    info["problems"] = problems
    if problems and strict:
        raise SystemExit("重み量子化の検査に失敗した:\n  - " + "\n  - ".join(problems))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdq", type=Path, help="できあがった QDQ を検査する")
    ap.add_argument("--prepped", type=Path, help="per-channel 指定の対象を数える")
    ap.add_argument("--expect", type=int, help="期待する per-channel 本数（検査用）")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out = {}
    if args.prepped:
        ov, axes = build_overrides(args.prepped)
        print(f"per-channel 指定の対象: {len(axes)} 本（すべて axis=1）")
        for nm in list(axes)[:5]:
            print(f"   {nm}")
        if len(axes) > 5:
            print(f"   …（ほか {len(axes)-5} 本）")
        out["targets"] = list(axes)
    if args.qdq:
        info = verify(args.qdq, args.expect, strict=args.expect is not None) \
            if args.expect is not None else inspect_weights(args.qdq)
        print(f"\n=== 生成物の検査: {args.qdq.name} ===")
        print(f"  重みの DQ     {info['total']} 本（うち 1024要素以上 {info['big_total']} 本）")
        print(f"  per-channel  {info['big_per_channel']} 本 / per-tensor {info['big_per_tensor']} 本"
              f"（1024要素以上のみ）")
        print(f"  型の内訳      {info['by_dtype']}")
        print(f"  axis の内訳   {info['axes']}")
        if info.get("problems"):
            print("  ★問題: " + " / ".join(info["problems"]))
        out["inspect"] = {k: v for k, v in info.items() if k != "rows"}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
