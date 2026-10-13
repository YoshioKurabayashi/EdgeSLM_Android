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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _run(model: str, feeds: list[dict]) -> np.ndarray:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    return np.concatenate([sess.run(None, {n: f[n] for n in names})[0] for f in feeds], 0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def metrics(ref: np.ndarray, test: np.ndarray, k: int = 5) -> dict:
    r = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
    t = test.reshape(-1, test.shape[-1]).astype(np.float64)

    top1 = float((r.argmax(-1) == t.argmax(-1)).mean())

    rk = np.argpartition(-r, k, axis=-1)[:, :k]
    tk = np.argpartition(-t, k, axis=-1)[:, :k]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(rk, tk)]))

    pr, pt = _softmax(r), _softmax(t)
    eps = 1e-12
    kl = float(np.mean((pr * (np.log(pr + eps) - np.log(pt + eps))).sum(-1)))

    # fp32 の top-1 が量子化側で何位か
    best = r.argmax(-1)
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--eval-npz", required=True)
    ap.add_argument("--qdq", nargs="+", required=True, help="`ラベル=パス`")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    z = np.load(args.eval_npz)
    names = list(z.files)
    n = z[names[0]].shape[0]
    feeds = [{k: z[k][i:i + 1] for k in names} for i in range(n)]
    print(f"評価 {n} 系列 x {z[names[0]].shape[1]} トークン  入力 {names}")

    ref = _run(args.fp32, feeds)
    print(f"fp32 logits: shape={ref.shape}\n")

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
