r"""KVキャッシュの入口(past_*)と出口(present_*)の量子化パラメータを揃える。

## なぜ必要か

decode はキャッシュを `present_* → past_*` と毎ステップ往復させる。
`--preserve_io_datatype` なので境界は float32 で、出口で dequantize、入口で quantize される。

**scale と zero point が同じなら、この往復は量子化格子の上の恒等写像で無損失**である。
ところが ORT は `past_k_0` と `present_k_0` を**別のテンソルとして較正する**ので、
表せる範囲が食い違う。24層モデルで実測すると:

    48本中42本で出口の範囲が入口をはみ出す（最大 15.5%）
    past_k_10 : 入口 [-10.13, 5.48] / 出口 [-12.48, 5.99]
    past_v_5  : 入口 [-1.955, 1.906] / 出口 [-1.955, 2.497]

はみ出した分は **clip される**。丸め（1段階ぶん）と違い、値そのものを失う。
1往復の試算で相対L2 最大 9.5%、clip 率 15.5%。

**単発の推論では出ない**（往復しないため）。**ループで効く。**

## 何をするか

各 `past_X` / `present_X` の組について、**両方の範囲を含む共通の (scale, zero point)** を
作り、入口の Q/DQ と出口の Q/DQ の両方に付け替える。
既存の初期化子は他と共有されている可能性があるので**上書きせず新しく作る**。

## 使い方

    python -m app.pipeline.fix_cache_qparams \
        --src work/ops/qwen_dec24/qwen_dec24_final_qdq.onnx \
        --out work/ops/qwen_dec24/qwen_dec24_unified_qdq.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

QMAX = 65535


def _maps(g):
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return prod, cons


def run(src: Path, out: Path, in_prefix: str = "past_",
        out_prefix: str = "present_") -> dict:
    model = onnx.load(str(src))
    g = model.graph
    prod, cons = _maps(g)
    inits = {i.name: i for i in g.initializer}

    def val(name):
        t = inits.get(name)
        return None if t is None else numpy_helper.to_array(t).item()

    def q_after(tensor):
        q = next((c for c in cons.get(tensor, []) if c.op_type == "QuantizeLinear"), None)
        if q is None:
            return None, None
        dq = next((c for c in cons.get(q.output[0], []) if c.op_type == "DequantizeLinear"), None)
        return q, dq

    def dq_before(tensor):
        dq = prod.get(tensor)
        if dq is None or dq.op_type != "DequantizeLinear":
            return None, None
        q = prod.get(dq.input[0])
        if q is None or q.op_type != "QuantizeLinear":
            return None, None
        return q, dq

    new_inits = []
    changed = []
    ins = [v.name for v in g.input if v.name.startswith(in_prefix)]
    for pin in ins:
        pout = out_prefix + pin[len(in_prefix):]
        qi, dqi = q_after(pin)
        qo, dqo = dq_before(pout)
        if not (qi and dqi and qo and dqo):
            continue
        s_i, z_i = val(qi.input[1]), val(qi.input[2])
        s_o, z_o = val(qo.input[1]), val(qo.input[2])
        if None in (s_i, z_i, s_o, z_o):
            continue

        lo = min((0 - z_i) * s_i, (0 - z_o) * s_o)
        hi = max((QMAX - z_i) * s_i, (QMAX - z_o) * s_o)
        # uint16 asymmetric。0 が格子に乗るよう zero point は丸める
        s = (hi - lo) / QMAX
        z = int(np.clip(round(-lo / s), 0, QMAX))
        # 丸めた zero point に合わせて scale を取り直す（両端を必ず含めるため）
        if z > 0:
            s = max(s, -lo / z) if lo < 0 else s
        if z < QMAX:
            s = max(s, hi / (QMAX - z))

        base = pin + "_unified"
        sn, zn = base + "_scale", base + "_zp"
        new_inits.append(numpy_helper.from_array(np.array(s, dtype=np.float32), sn))
        new_inits.append(helper.make_tensor(zn, TensorProto.UINT16, [], [z]))
        for node in (qi, dqi, qo, dqo):
            node.input[1] = sn
            node.input[2] = zn
        changed.append({"tensor": pin,
                        "in": [(0 - z_i) * s_i, (QMAX - z_i) * s_i],
                        "out": [(0 - z_o) * s_o, (QMAX - z_o) * s_o],
                        "unified": [(0 - z) * s, (QMAX - z) * s]})

    g.initializer.extend(new_inits)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out), save_as_external_data=True,
              location=out.name + ".data", all_tensors_to_one_file=True)
    return {"changed": len(changed), "detail": changed, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--in-prefix", default="past_")
    ap.add_argument("--out-prefix", default="present_")
    args = ap.parse_args()

    r = run(args.src, args.out, args.in_prefix, args.out_prefix)
    print(f"{r['changed']} 本のキャッシュで入口と出口の量子化パラメータを揃えた")
    for d in r["detail"][:5]:
        print(f"  {d['tensor']:<14} 入口 [{d['in'][0]:8.4f},{d['in'][1]:8.4f}] "
              f"出口 [{d['out'][0]:8.4f},{d['out'][1]:8.4f}] "
              f"-> 統一 [{d['unified'][0]:8.4f},{d['unified'][1]:8.4f}]")
    print(f"-> {r['out']}")


if __name__ == "__main__":
    main()
