"""実機の誤差を「量子化」と「HTP側」に分解する（P3）

## 考え方

これまで「fp32 vs 実機」しか見ておらず、**誤差の内訳が分からなかった**。
中間に「ホスト CPU 上で同じ QDQ モデルを走らせた結果」を挟めば分解できる:

    fp32  ──[ 量子化誤差 ]──>  ホストQDQ  ──[ HTP側の誤差 ]──>  実機

- **量子化誤差**   … 値を離散化したことによる損失。較正・精度指定・グラフの書き方で変わる
- **QNN-HTP 追加分** … **まったく同じ量子化モデル**を実機で実行した結果のずれ。
                     HTP の積和精度・丸めだけでなく、**QNN への lowering・op fusion・
                     QDQ 境界の変更・requantization・layout 変換・kernel 選択**も含む。
                     「HTP の演算誤差」と呼ぶと狭すぎる。**ホストでは絶対に見えない分**

**同じ QDQ モデル・同じスケールを使うので、差分は演算の違いだけ**である。

## なぜ切り分けたいのか

実機で測ったところ、**量子化誤差を小さくした方が実機での低下が大きい**という
逆説的な傾向が出た（両側A8 は -3.4pt、左右入替版は -17.6pt）。
これが正しければ **HTP 側の誤差が下限を作っている**ことになり、
ホスト上で量子化を作り込む価値には上限がある。**どこで手を止めるかの判断**に直結する。

## 学習用の補足（2026-10-18 追記）

- 段: P3 の実機の精度（results/P3_20260913_error_decompose.md・results/P3_error_decompose.json）。
  StudyMemo 1 章（ホストの CPU EP で QDQ を実行）・4.5 節（罠 1 = 分母）・6.1 節の例 5・7 章の分解の表の実物
- 入力: fp32 の ONNX・実機に載せたのと同じ QDQ の ONNX・評価データの npz（device_accuracy.py に渡したのと同じもの）・
  device_accuracy.py が端末から持ち帰った出力の .bin のフォルダ（ファイル名は <job_id>_<件の番号>_<出力名>.bin）
- 出力: 画面の 3 行（量子化・QNN-HTP 追加分・合計）と、--out を付ければ JSON の表（同じ label の行は置き換える）
- 動かし方:
    python -m app.pipeline.error_decompose --fp32 … --qdq … --eval-npz … --device-dir work/dev24 --out-name logits
        --label qwen24 --out results/P3_error_decompose.json
  （フォルダ名・出力名は例。--out-name はモデルの出力の名前）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

# 型の名前 → numpy の型（★2026-10-18 注: このファイルの中では使っていない。実機の .bin は fp32 の出力と同じ型として読む）
DTYPE = {"float32": np.float32, "float16": np.float16, "int64": np.int64,
         "int32": np.int32, "uint8": np.uint8, "int8": np.int8,
         "uint16": np.uint16, "int16": np.int16}


def _run(model: str, feeds: list[dict]) -> list[np.ndarray]:
    # （2026-10-18 追記）ONNX を CPU EP で 1 件ずつ実行し、各件の最初の出力を返す（件数ぶんのリスト）。
    # QDQ ならホストでも Q/DQ が本当に量子化するので、ここが「ホストQDQ」の値になる
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    return [sess.run(None, {n: f[n] for n in names})[0] for f in feeds]


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    """絶対 L2 距離。"""
    return float(np.linalg.norm(a.astype(np.float64).ravel() - b.astype(np.float64).ravel()))


def _rel(a: np.ndarray, b: np.ndarray, denom: np.ndarray | None = None) -> float:
    """相対 L2。**分母は常に fp32 参照のノルム**にする。

    ★当初 `‖a-b‖ / ‖a‖` としていたため、
      量子化誤差は `/‖fp32‖`、HTP 差は `/‖ホストQDQ‖` と**分母が揃っておらず**、
      「どちらが大きい」を直接比較できない数字を並べていた（2026-09-13 に指摘され訂正）。
      なお L2 はベクトルなので **3つの値は加算関係にない**（直交に近ければ二乗和）。
    """
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.linalg.norm((denom if denom is not None else a).astype(np.float64).ravel())
    return float(np.linalg.norm(x - y) / d) if d else 0.0


def _top1(a: np.ndarray, b: np.ndarray) -> float | None:
    """logits（[1, 系列長, 語彙数]）なら、位置ごとの 1 位が一致する割合。logits でなければ None。

    「最後の軸が 10000 以上なら語彙」と見なす簡易な判定（層 1 枚の出力 [1, 16, 896] などは対象外にする）
    """
    if a.ndim != 3 or a.shape[-1] < 10000:
        return None
    return float((a.argmax(-1) == b.argmax(-1)).mean())


def main() -> None:
    """① 実機の .bin を集める → ② 同じ件を fp32・ホストQDQ で流す → ③ 3 つの差を出す → ④ 表示と保存。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--qdq", required=True, help="実機に載せたのと同じ QDQ モデル")
    ap.add_argument("--eval-npz", type=Path, required=True)
    ap.add_argument("--device-dir", type=Path, required=True,
                    help="device_accuracy.py が pull した .bin のあるディレクトリ")
    ap.add_argument("--out-name", default="out", help="出力テンソル名（.bin の命名に使われている）")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    # ① 実機の出力の .bin。ファイル名の後ろから 2 つ目の欄を「何件目か」として読む。
    #    （2026-10-18 注）出力名に _ が入っている（例 present_k_0）と欄がずれて、件の番号を読み違える
    bins = sorted(args.device_dir.glob(f"*_{args.out_name}.bin"))
    if not bins:
        raise SystemExit(f"{args.device_dir} に *_{args.out_name}.bin が無い")

    z = np.load(args.eval_npz)
    names = list(z.files)
    ids = [int(p.stem.split("_")[-2]) for p in bins]
    feeds = [{k: z[k][i:i + 1] for k in names} for i in ids]

    # ② 同じ件を fp32 とホストQDQ で流し、実機のバイト列を fp32 の出力と同じ型・形の配列に戻す
    ref = _run(args.fp32, feeds)
    host = _run(args.qdq, feeds)
    dev = [np.frombuffer(p.read_bytes(), dtype=ref[0].dtype).reshape(ref[0].shape)
           for p in bins]

    # **分母はすべて ‖fp32‖ に揃える**
    q_err = [_rel(r, h, r) for r, h in zip(ref, host)]     # 量子化
    h_err = [_rel(h, d, r) for r, h, d in zip(ref, host, dev)]  # QNN-HTP 追加分
    t_err = [_rel(r, d, r) for r, d in zip(ref, dev)]      # 合計
    q_abs = [_l2(r, h) for r, h in zip(ref, host)]
    h_abs = [_l2(h, d) for h, d in zip(host, dev)]
    t_abs = [_l2(r, d) for r, d in zip(ref, dev)]

    row = {
        "label": args.label or Path(args.qdq).stem,
        "samples": len(bins),
        "quant_rel_l2": float(np.mean(q_err)),
        "qnn_htp_rel_l2": float(np.mean(h_err)),
        "total_rel_l2": float(np.mean(t_err)),
        "quant_abs_l2": float(np.mean(q_abs)),
        "qnn_htp_abs_l2": float(np.mean(h_abs)),
        "total_abs_l2": float(np.mean(t_abs)),
        "ref_norm": float(np.mean([np.linalg.norm(r.astype(np.float64)) for r in ref])),
        "quadrature_check": float(np.sqrt(np.mean(q_err) ** 2 + np.mean(h_err) ** 2)),
    }
    # top-1 の一致（logits のときだけ）: fp32↔ホスト・fp32↔実機・ホスト↔実機
    t1 = [_top1(r, h) for r, h in zip(ref, host)]
    t2 = [_top1(r, d) for r, d in zip(ref, dev)]
    if t1[0] is not None:
        row["top1_host"] = float(np.mean(t1))
        row["top1_device"] = float(np.mean(t2))
        row["top1_host_vs_device"] = float(np.mean(
            [_top1(h, d) for h, d in zip(host, dev)]))

    print(f"=== {row['label']}（{row['samples']} サンプル）===")
    print(f"  （分母はすべて ‖fp32‖。L2 なので3つは加算関係にない）")
    print(f"  量子化           fp32 → ホストQDQ : {row['quant_rel_l2']:8.4%}  (abs {row['quant_abs_l2']:.4g})")
    print(f"  QNN-HTP 追加分   ホストQDQ → 実機 : {row['qnn_htp_rel_l2']:8.4%}  (abs {row['qnn_htp_abs_l2']:.4g})")
    print(f"  合計             fp32 → 実機      : {row['total_rel_l2']:8.4%}  (abs {row['total_abs_l2']:.4g})")
    print(f"  参考: 二乗和の平方根 {row['quadrature_check']:.4%} "
          f"（合計に近ければ2つの誤差はほぼ直交）")
    if "top1_host" in row:
        print(f"  top-1  ホスト {row['top1_host']:.1%} / 実機 {row['top1_device']:.1%} "
              f"/ ホストと実機の一致 {row['top1_host_vs_device']:.1%}")

    # ④ 既にある JSON を読み、同じ label の行を差し替えて書き直す（モデルごとに 1 行ずつ足していける）
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else []
        prev = [r for r in prev if r.get("label") != row["label"]]
        prev.append(row)
        args.out.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
