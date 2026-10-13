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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh


def run(qdq: Path, prepped: Path, out: Path) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.p2b_ops import act_act_matmul_sides

    lhs, rhs, outs = act_act_matmul_sides(str(prepped))
    m = onnx.load(str(qdq))
    inits = {t.name: t for t in m.graph.initializer}
    done, skipped = [], []
    for name in rhs:
        # その名前を量子化している Q/DQ を集める
        nodes = [n for n in m.graph.node
                 if n.op_type in ("QuantizeLinear", "DequantizeLinear")
                 and (n.input[0] == name
                      or n.input[0] == f"{name}_QuantizeLinear_Output")]
        q = next((n for n in nodes if n.op_type == "QuantizeLinear"), None)
        if q is None or len(q.input) < 3:
            skipped.append(name)
            continue
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

    out.parent.mkdir(parents=True, exist_ok=True)
    big = m.ByteSize() > 1_800_000_000
    onnx.save(m, str(out), save_as_external_data=big,
              location=(out.name + ".data") if big else None)
    return {"done": done, "skipped": skipped, "n_rhs": len(rhs)}


def main() -> None:
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
