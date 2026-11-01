r"""act × act の MatMul の**右辺だけ uint8 に落とす**（C-8 の対応）。

## なぜ必要か

v68 の MatMul は組合せ依存で、**`u16×u8` は通るが `u16×u16` は通らない**（C-8）。
左右入替（swap）だけでは型は変わらないので、**右辺を 8bit にする**必要がある。

`p2b_ops` の `a16w8_rhs8` は `TensorQuantOverrides` で指定する作りだが、
`get_qnn_qdq_config` は活性化の型をこちらの指定どおりにしてくれないことがある
（2026-09-14 実測: 指定しても uint16 のまま出てきた）。

そこで**できあがった QDQ を後から書き換える**。実機で動いている 24層の
`final_qdq` も同じ形になっている（同じ範囲を uint8 で表し直しただけ）。

    a16w8_rms_swap : dtype=uint16 scale=2.489e-04 zp=40473  範囲 [-10.0753, 6.2389]
    final_qdq      : dtype=uint8  scale=6.398e-02 zp=  157  範囲 [-10.0444, 6.2697]

## パイプラインのどこで使うか

    S4 量子化 → **ここ（後処理）** → S5 DLC変換 → S6 context生成 → 実機

`p2b_ops.py` の出力（`<name>_<scheme>_qdq.onnx`）に掛けて `<name>_final_qdq.onnx` を作る。
これを飛ばすと context 生成が `expected >= 73` で落ちる。

## 何をするか

対象テンソルの `QuantizeLinear`/`DequantizeLinear` の scale と zero_point を、
**同じ実数範囲を uint8 で表す値**に入れ替える。重みは触らない。

    python -m app.pipeline.downgrade_actact_rhs --qdq <in> --prepped <prepped> --out <out>

## 入力と出力（2026-10-18 追記）

  --qdq     : S4 の出力（<name>_<scheme>_qdq.onnx）。書き換える元
  --prepped : S2 の出力（<name>_prepped.onnx）。act×act の MatMul を探すのに使う
              （量子化の前のグラフなら Q/DQ が無く、MatMul の入力名がそのまま読める）
  --out     : 書き換えた QDQ（<name>_final_qdq.onnx）。1.8GB を超えるときは .data を伴う

解説は StudyMemo 2.7 節（u16 × u8 の意味）・5.3 節（左右入替）・7.8 節、要件定義書 C-8。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh


def run(qdq: Path, prepped: Path, out: Path) -> dict:
    """act×act の MatMul の右辺の Q/DQ を、同じ実数範囲を表す uint8 の scale・zero point に差し替える。

    引数: qdq（S4 の QDQ）、prepped（S2 の出力。対象の MatMul を探す）、out（書き出し先）。
    戻り値: {"done": [{name, old_scale, new_scale, range, new_zp}, ...],
             "skipped": [Q が無い・uint16 でないテンソル名], "n_rhs": 右辺の本数}
    右辺だけにするのは、v68 が通すのが u16（左）× u8（右）だけで、u8 × u16 は通らないため（C-8）。
    左辺と出力（attention のスコア）は 16bit のまま残るので、スコアの分解能を保てる。
    """
    # ① act×act の MatMul の左右の入力名を p2b_ops の関数で拾う。
    #   `python -m app.pipeline.…` で動かしたとき app/ を探索パスに足して pipeline.p2b_ops として読む
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.p2b_ops import act_act_matmul_sides

    lhs, rhs, outs = act_act_matmul_sides(str(prepped))
    # ② QDQ を読み、初期化子（scale・zero point を含む）を名前で引けるようにする
    m = onnx.load(str(qdq))
    inits = {t.name: t for t in m.graph.initializer}
    done, skipped = [], []
    for name in rhs:
        # ③ ORT の QDQ では、テンソル x は QuantizeLinear(x) → 出力 "x_QuantizeLinear_Output"
        #   → DequantizeLinear という並びになる。その Q と DQ の両方を拾って同じ値に差し替える
        # その名前を量子化している Q/DQ を集める
        nodes = [n for n in m.graph.node
                 if n.op_type in ("QuantizeLinear", "DequantizeLinear")
                 and (n.input[0] == name
                      or n.input[0] == f"{name}_QuantizeLinear_Output")]
        q = next((n for n in nodes if n.op_type == "QuantizeLinear"), None)
        # Q が無い（量子化されていない）か zero point を持たないものは対象外として記録だけする
        if q is None or len(q.input) < 3:
            skipped.append(name)
            continue
        # ④ いまの scale s と zero point z（uint16）を読み、同じ実数範囲 [lo, hi] を
        #   0〜255 の 256 段で表す scale・zero point を計算し直す（活性化は per-tensor なので値は 1 個）。
        #   範囲は変えないので、分解能が 65535/255 ≒ 257 倍粗くなるだけで、はみ出しは起きない
        s = float(np.ravel(nh.to_array(inits[q.input[1]]))[0])
        zarr = nh.to_array(inits[q.input[2]])
        if zarr.dtype != np.uint16:
            skipped.append(name)
            continue
        z = float(np.ravel(zarr)[0])
        lo, hi = (0 - z) * s, (65535 - z) * s          # いま表している実数範囲
        new_s = (hi - lo) / 255.0
        new_z = int(np.clip(round(-lo / new_s), 0, 255))

        # ★scale / zero_point の初期化子は **他のノードと共有されている**。
        #   その場で書き換えると無関係なテンソルまで巻き添えになる
        #   （2026-09-14: lm_head の DQ が uint16/uint8 混在になりロードできなくなった）。
        #   **新しい初期化子を作って、このノードだけ差し替える。**
        sn, zn = f"{name}_scale_u8", f"{name}_zp_u8"
        if sn not in inits:
            t = nh.from_array(np.float32(new_s), sn); m.graph.initializer.append(t); inits[sn] = t
            t = nh.from_array(np.uint8(new_z), zn); m.graph.initializer.append(t); inits[zn] = t
        for n in nodes:
            n.input[1] = sn
            n.input[2] = zn
        done.append({"name": name, "old_scale": s, "new_scale": new_s,
                     "range": [lo, hi], "new_zp": new_z})

    # ⑤ 保存。protobuf は 1 ファイル 2GB が上限なので、1.8GB を超えたら重みを
    #   <out>.data に出す外部データ形式にする
    out.parent.mkdir(parents=True, exist_ok=True)
    big = m.ByteSize() > 1_800_000_000
    onnx.save(m, str(out), save_as_external_data=big,
              location=(out.name + ".data") if big else None)
    return {"done": done, "skipped": skipped, "n_rhs": len(rhs)}


def main() -> None:
    """コマンドの入口。--qdq / --prepped / --out を受け、run() の結果（落とした本数と先頭 3 本）を表示する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--prepped", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = run(args.qdq, args.prepped, args.out)
    print(f"  act-act MatMul の右辺 {r['n_rhs']} 本中 {len(r['done'])} 本を uint8 に落とした"
          f"（対象外 {len(r['skipped'])} 本）")
    for d in r["done"][:3]:
        print(f"    {d['name']}: scale {d['old_scale']:.4e} -> {d['new_scale']:.4e}"
              f"  範囲 [{d['range'][0]:.3f}, {d['range'][1]:.3f}]")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
