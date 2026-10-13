"""oracle 実験: どのテンソルの量子化を外すと精度が戻るかを調べる（P3 診断）

## これは本番実装ではない

量子化を外すと float セグメントが残り、**C-1 違反で v68 では動かない**。
これは**原因を確定させるためのホスト上の診断**である。

## なぜ必要か

24層が壊れる原因を「残差の massive activations」と推定したが、**推定のままである**。
「どこの量子化を外すと戻るか」を見れば、**原因を消去法で確定できる**。

    全部量子化          top-1 ?%
    残差だけ非量子化     top-1 ?%   ← ここで大きく戻れば残差が主因
    RMSNormだけ非量子化  top-1 ?%
    両方               top-1 ?%

戻り方が小さければ「残差が主因」という仮説は**外れ**であり、
別の場所を疑い直す必要がある。**対策を実装する前にここを踏む。**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from app.pipeline import calib, quantize_qdq

# 除外パターン。op_type で選ぶ（ノード名はモデルごとに違うため）
PATTERNS = {
    "none": [],
    "residual": ["Add"],
    "rmsnorm": ["Pow", "ReduceMean", "Sqrt", "Reciprocal"],
    "residual+rmsnorm": ["Add", "Pow", "ReduceMean", "Sqrt", "Reciprocal"],
    "matmul_act": [],       # 特別扱い（下記）
}


def nodes_of(prepped: Path, pattern: str) -> list[str]:
    model = onnx.load(str(prepped), load_external_data=False)
    inits = {i.name for i in model.graph.initializer}
    if pattern == "matmul_act":
        return [n.name for n in model.graph.node
                if n.op_type == "MatMul" and not any(i in inits for i in n.input)]
    ops = set(PATTERNS[pattern])
    return [n.name for n in model.graph.node if n.op_type in ops and n.name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepped", type=Path, required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--calib-npz", type=Path, required=True)
    ap.add_argument("--eval-npz", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--patterns", nargs="*", default=list(PATTERNS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from app.pipeline.accuracy_logits import _run, metrics

    z = np.load(args.calib_npz)
    cname = list(z.files)[0]
    n = z[cname].shape[0]

    ze = np.load(args.eval_npz)
    ename = list(ze.files)[0]
    feeds = [{ename: ze[ename][i:i + 1]} for i in range(ze[ename].shape[0])]
    ref = _run(args.fp32, feeds)

    args.workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pat in args.patterns:
        excl = nodes_of(args.prepped, pat)
        out = args.workdir / f"oracle_{pat.replace('+', '_')}_qdq.onnx"
        reader = calib.MultiReader([{cname: z[cname][i:i + 1]} for i in range(n)])
        try:
            quantize_qdq.run(args.prepped, out, scheme="a16w8", reader=reader,
                             nodes_to_exclude=excl or None)
            m = metrics(np.concatenate(ref, 0), np.concatenate(_run(str(out), feeds), 0))
        except Exception as e:  # noqa: BLE001
            print(f"{pat:18s} 失敗: {type(e).__name__}: {str(e)[:120]}")
            continue
        m["pattern"] = pat
        m["excluded_nodes"] = len(excl)
        rows.append(m)
        print(f"{pat:18s} 除外 {len(excl):4d} ノード  "
              f"top-1 {m['top1_agreement']:6.1%}  KL {m['kl_fp32_to_quant']:8.4f}  "
              f"相対L2 {m['logits_rel_l2']:7.2%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
