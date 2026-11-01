"""end-to-end の精度を logits で測る（P3）

## なぜ層の L2 誤差では判断できないか

層1枚の出力の相対 L2 誤差は、**最終的にどのトークンを選ぶか**とは別物である。
- 誤差が層をまたいで打ち消し合うこともあれば、増幅されることもある
- logits は最後に argmax / softmax を通るので、**順位さえ保たれていれば L2 が大きくても実害は小さい**
- 逆に L2 が小さくても、上位トークンの順位が入れ替われば出力は変わる

P3 の層1枚では相対 L2 が 30.4% だったが、それだけで「使えない」とは言えない。
**見る場所を間違えない**ために、ここでは生成に直結する指標で測る。

## 指標

| 指標 | 意味 |
|---|---|
| **top-1 一致率** | 次トークンの予測が fp32 と同じ割合。**最も直接的** |
| top-5 重なり | 上位5候補の集合がどれだけ重なるか。貪欲でない生成での余裕を見る |
| **KL(fp32 ‖ 量子化)** | 確率分布としてのずれ。0 に近いほどよい |
| logits 相対L2 | 参考。層の測定と繋げるため |
| 平均順位ずれ | fp32 の top-1 が量子化側で何位になったか |

## 使い方（2026-10-18 追記）

- 段: P3 の end-to-end（24 層まとめた logits）。結果は results/P3_logits_full24.json・results/P3_logits_prefill2.json
  （説明は results/P3_20260913_full24.md）。StudyMemo 4.4 節・8.7 節の「指標」の実物
- 入力: 正解の fp32 ONNX（出力 1 本目が logits [1, 系列長, 語彙数]）・評価データの npz・比べる QDQ ONNX
- 出力: 画面の表（top-1 一致の高い順）と、--out を付ければ方式ごとの指標の JSON
- 動かし方（プロジェクトの直下）:
    python -m app.pipeline.accuracy_logits --fp32 … --eval-npz … --qdq ラベル=パス … --topk 5 --out results/….json
- 注意: 指標の名前は「top{k}_overlap」で、--topk を変えると鍵の名前も変わる。cos 類似度は返さない
  （cos を含む 5 つの指標を返すのは ab_host_device.py の metrics。StudyMemo 8 章の訂正を参照）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _run(model: str, feeds: list[dict]) -> np.ndarray:
    """ONNX を CPU EP で 1 件ずつ実行し、各件の最初の出力（logits）を先頭の次元でつないで返す。

    feeds  : 1 件ごとの {入力名: [1, …] の配列}
    戻り値 : [件数, 系列長, 語彙数]（件ごとの [1, 系列長, 語彙数] を縦に重ねたもの）
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    return np.concatenate([sess.run(None, {n: f[n] for n in names})[0] for f in feeds], 0)


def _softmax(x: np.ndarray) -> np.ndarray:
    """最後の軸で softmax を取る。先に最大値を引くのは、exp があふれないようにするため（結果は同じ）。"""
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def metrics(ref: np.ndarray, test: np.ndarray, k: int = 5) -> dict:
    """fp32 の logits（ref）と量子化側の logits（test）から、生成に直結する指標を出す。

    ref, test : 同じ形の logits（最後の軸が語彙）。中で [位置の数, 語彙数] に平らにして float64 で計算する
    k         : top-k の重なりの k（既定 5）
    戻り値    : top1_agreement（次の語の予測が一致した位置の割合）・top{k}_overlap・
                kl_fp32_to_quant（位置ごとの KL(fp32‖量子化) の平均。log の 0 を避けて eps=1e-12 を足す）・
                rank_of_fp32_top1_mean / _p95（fp32 の 1 位が量子化側で何位か。1 が最良）・
                logits_rel_l2（全体の ‖ref−test‖/‖ref‖）・positions（比べた位置の数）
    なぜ: 層の相対L2 は「どの語を選ぶか」とは別物なので、選ぶ語そのもので測る（上の説明）
    """
    r = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
    t = test.reshape(-1, test.shape[-1]).astype(np.float64)

    top1 = float((r.argmax(-1) == t.argmax(-1)).mean())

    # 上位 k 個の語の番号（並びは順不同。集合として重なりを数えるだけなので並べ替えは要らない）
    rk = np.argpartition(-r, k, axis=-1)[:, :k]
    tk = np.argpartition(-t, k, axis=-1)[:, :k]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(rk, tk)]))

    pr, pt = _softmax(r), _softmax(t)
    eps = 1e-12
    kl = float(np.mean((pr * (np.log(pr + eps) - np.log(pt + eps))).sum(-1)))

    # fp32 の top-1 が量子化側で何位か
    best = r.argmax(-1)
    # 量子化側で、fp32 の 1 位の語より **厳密に大きい** logit を持つ語の数 + 1 = 順位（同点は上に数えない）
    rank = (t > t[np.arange(len(t)), best][:, None]).sum(-1) + 1

    return {
        "top1_agreement": top1,
        f"top{k}_overlap": overlap,
        "kl_fp32_to_quant": kl,
        "rank_of_fp32_top1_mean": float(rank.mean()),
        "rank_of_fp32_top1_p95": float(np.percentile(rank, 95)),
        "logits_rel_l2": float(np.linalg.norm(r - t) / np.linalg.norm(r)),
        "positions": int(r.shape[0]),
    }


def main() -> None:
    """コマンドの入口。評価データを 1 件ずつ流し、fp32 と各 QDQ の logits を metrics で比べて表と JSON にする。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--eval-npz", required=True)
    ap.add_argument("--qdq", nargs="+", required=True, help="`ラベル=パス`")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    # ① npz の各入力を 1 件ずつに切る（[N, 系列長] → N 個の [1, 系列長]）
    z = np.load(args.eval_npz)
    names = list(z.files)
    n = z[names[0]].shape[0]
    feeds = [{k: z[k][i:i + 1] for k in names} for i in range(n)]
    print(f"評価 {n} 系列 x {z[names[0]].shape[1]} トークン  入力 {names}")

    ref = _run(args.fp32, feeds)
    print(f"fp32 logits: shape={ref.shape}\n")

    # ② 各 QDQ を流して比べる。「ラベル=パス」の形。動かないモデルは理由を出して飛ばす
    rows = []
    for spec in args.qdq:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).stem, spec
        try:
            m = metrics(ref, _run(path, feeds), args.topk)
        except Exception as e:  # noqa: BLE001
            print(f"{label}: 実行できず {type(e).__name__}: {e}")
            continue
        m["label"] = label
        rows.append(m)

    # ③ top-1 一致の高い順に表を出す
    hdr = (f"{'方式':30s} {'top-1一致':>10s} {'top5重なり':>11s} "
           f"{'KL':>10s} {'fp32top1の順位':>14s} {'logits相対L2':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(rows, key=lambda r: -r["top1_agreement"]):
        print(f"{m['label']:30s} {m['top1_agreement']:9.1%} "
              f"{m[f'top{args.topk}_overlap']:10.1%} {m['kl_fp32_to_quant']:10.4f} "
              f"{m['rank_of_fp32_top1_mean']:13.2f} {m['logits_rel_l2']:11.2%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
