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
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 4.1・4.2 節「Oracle 実験」（3 章の表・12 章「量子化に耐えるか」でも使う道具）。
# 結果（results/P3_oracle6.json、Qwen2.5 の 6 層・ホスト）: top-1 は 全部量子化 6.2% → 残差（Add）だけ外す 6.2%
#   → RMSNorm だけ外す 74.4% → 両方 80.1%、活性化×活性化 MatMul を外しても 6.2%。
#   → 犯人は残差ではなく RMSNorm（の中の x²）だと消去法で確定し、C-11（RMSNorm を max で正規化）へ進んだ。
# 入力:  --prepped（S2 前処理後の ONNX）・--fp32（基準の ONNX）・--calib-npz（較正データ）・--eval-npz（評価データ）。
# 出力:  --workdir に oracle_<パターン>_qdq.onnx、--out にパターンごとの指標の JSON。
# 動かし方: & $PY2 -m app.pipeline.oracle_exclude --prepped … --fp32 … --calib-npz … --eval-npz … --workdir … --out …
# 注意: QDQ を「作り直す」版。できあがった QDQ から Q/DQ を外す版は oracle_qdq.py（多入力のモデルにも使える）。
#   npz の最初の配列を入力として使うので、入力が 1 つのモデル（input_ids だけの prefill）向け。
# ---------------------------------------------------------------------------

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
    """パターンに当たるノードの名前の一覧を返す（量子化から外すノード）。

    matmul_act は「両方の入力が初期化子（重み）でない MatMul」＝ 活性化×活性化 の MatMul（QK^T・注意×V）。
    名前の無いノードは除外の指定に使えないので数えない。
    """
    model = onnx.load(str(prepped), load_external_data=False)
    inits = {i.name for i in model.graph.initializer}
    if pattern == "matmul_act":
        return [n.name for n in model.graph.node
                if n.op_type == "MatMul" and not any(i in inits for i in n.input)]
    ops = set(PATTERNS[pattern])
    return [n.name for n in model.graph.node if n.op_type in ops and n.name]


def main() -> None:
    """パターンごとに、そのノードを除外して量子化し直し、fp32 と logits を比べる。"""
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

    # ① 較正データと評価データを読む（どちらも npz の最初の配列を使う）。評価は 1 サンプルずつ流す。
    #    fp32 の logits（基準）は 1 回だけ計算しておく。
    z = np.load(args.calib_npz)
    cname = list(z.files)[0]
    n = z[cname].shape[0]

    ze = np.load(args.eval_npz)
    ename = list(ze.files)[0]
    feeds = [{ename: ze[ename][i:i + 1]} for i in range(ze[ename].shape[0])]
    ref = _run(args.fp32, feeds)

    # ② パターンごとに量子化をやり直す（A16W8）。失敗したパターンは理由を出して次へ進む。
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
