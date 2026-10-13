"""量子化の精度を測る（P3）

## 何と何を比べるか

fp32 の出力を正解とし、QDQ モデルの出力との差を見る。QDQ モデルを CPU EP で走らせると
**Q/DQ ノードが実際に量子化・逆量子化を行う**ので、量子化による情報の損失がそのまま出る。

これは **NPU 上の実際の誤差とは別物**である点に注意する:
  - ここで測れるもの … 量子化そのものによる損失（精度指定・較正の良し悪し）
  - ここで測れないもの … HTP の演算順序・丸め・累積の違い
実機との差は runner に出力を返す機能（v0.3）を足してから測る。
**したがって本モジュールの数字は「下限」**であって、実機はこれ以上に悪くなりうる。

## なぜ相対誤差と cos 類似度の両方を見るか

層1枚の出力は次の層への入力になるので、**方向が合っているか（cos）**と
**大きさが合っているか（相対L2）**の両方が要る。cos が 1.0 に近くても
スケールがずれていれば、24層積むうちに発散する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _run(model: str, feeds_list: list[dict]) -> list[np.ndarray]:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    out = []
    for f in feeds_list:
        out.append(sess.run(None, {n: f[n] for n in names})[0])
    return out


def compare(ref: list[np.ndarray], test: list[np.ndarray]) -> dict:
    rel, cos, mx = [], [], []
    for a, b in zip(ref, test):
        a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        denom = np.linalg.norm(a64)
        rel.append(float(np.linalg.norm(a64 - b64) / denom) if denom else 0.0)
        na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
        cos.append(float(a64 @ b64 / (na * nb)) if na and nb else 0.0)
        mx.append(float(np.abs(a64 - b64).max()))
    return {
        "n": len(rel),
        "rel_l2_mean": float(np.mean(rel)),
        "rel_l2_max": float(np.max(rel)),
        "cosine_mean": float(np.mean(cos)),
        "cosine_min": float(np.min(cos)),
        "max_abs_err": float(np.max(mx)),
        "ref_abs_max": float(np.max([np.abs(a).max() for a in ref])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True, help="正解となる fp32 ONNX")
    ap.add_argument("--eval-npz", required=True)
    ap.add_argument("--qdq", nargs="+", required=True,
                    help="比較する QDQ モデル。`ラベル=パス` の形で渡せる")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    z = np.load(args.eval_npz)
    names = list(z.files)
    n = z[names[0]].shape[0]
    feeds = [{k: z[k][i:i + 1] for k in names} for i in range(n)]
    print(f"評価サンプル {n} 件  入力 {names}")

    ref = _run(args.fp32, feeds)
    print(f"fp32 出力: shape={ref[0].shape} 値域 [{min(r.min() for r in ref):.4f}, "
          f"{max(r.max() for r in ref):.4f}]\n")

    rows = []
    for spec in args.qdq:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).stem, spec
        try:
            m = compare(ref, _run(path, feeds))
        except Exception as e:  # noqa: BLE001
            print(f"{label:26s} 実行できず: {type(e).__name__}: {e}")
            continue
        m["label"] = label
        m["model"] = path
        rows.append(m)

    print(f"{'方式':26s} {'相対L2誤差':>12s} {'cos類似度':>12s} {'最大絶対誤差':>14s}")
    print("-" * 68)
    for m in sorted(rows, key=lambda r: r["rel_l2_mean"]):
        print(f"{m['label']:26s} {m['rel_l2_mean']:11.4%} "
              f"{m['cosine_mean']:12.6f} {m['max_abs_err']:14.5f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
