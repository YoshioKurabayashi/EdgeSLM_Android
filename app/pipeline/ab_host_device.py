r"""同じ評価セットで **fp32 / ホストQDQ / 実機HTP** を3者比較する。

## パイプラインのどこで使うか

    実機で device_accuracy を回す → **ここで3者比較（fp32 / ホストQDQ / 実機）**

`device_accuracy.py` が保存した実機の生出力を読むので、**先にそちらを回しておく**。

## 使い方

    # 先に実機で測る（work/devacc_<id>/ に生出力が残る）
    python -m app.pipeline.device_accuracy --model models/ops/<m>.onnx         --eval-npz <eval.npz> --fp32 <fp32.onnx> --job-id <id>         --limit 48 --positions all --work work/devacc_<id> --out results/<r>.json

    # 3者を突き合わせる（ラベル=QDQパス=実機workディレクトリ=job_id）
    python -m app.pipeline.ab_host_device --fp32 <fp32.onnx> --eval-npz <eval.npz>         --limit 48 --label "C-13後=<qdq.onnx>=work/devacc_<id>=<id>"

## 主指標は「ホスト QDQ ↔ 実機 HTP」

fp32 との差だけを見ると、量子化誤差と HTP 実装差が混ざる。
**同じ量子化グラフをホストと実機で走らせて直接引く**のが、HTP 側の問題を見る主指標。
二乗差による推定（quadrature）は直交を仮定するので使わない。

## 出す指標

相対L2 / cos / KL / fp32 の1位の順位 / top-1 一致率。
decode なら `present_k`・`present_v` は **新しい1スロットだけ**でも出す
（64スロットのうち63は past の写しなので、全体で測ると希釈される）。

## 入力と出力（2026-10-18 追記）

- 入力: fp32 の ONNX・評価セットの npz（入力名ごとに [N, …]）・量子化した QDQ の ONNX・
  device_accuracy.py が端末から持ち帰った出力（`<work>/<job_id>_<番号>_<出力名>.bin`。
  名前の付け方は android_runner の job.cpp。中身は float32 の生のバイト列）
- 出力: 画面の表と、`--out` を付けたときの JSON（例: results/P9_20260914_dec24_c13_ab.json）
- 使った節: StudyMemo 6.3（道具の一覧）・9.1「まず指標を変えた」・8 章（5 つの指標）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _softmax(x):
    """最後の軸で softmax を取る（最大値を引いてから exp。桁あふれを防ぐ定石）。"""
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """a を b と比べる（b が基準）。logits を想定。

    引数: a, b は [N, …] の同じ形（N はサンプル数）。2 つめ以降の軸は 1 本に平らにして比べる。
    戻り値: 5 つの指標の dict
      rel_l2           全体の相対 L2 = ‖a−b‖ / ‖b‖（全サンプルまとめて 1 つ）
      cosine           サンプルごとの cos 類似度の平均
      kl               KL(b‖a) の平均（b の確率分布を正として、a がどれだけずれたか）
      top1_agreement   1 位の語が一致した割合
      rank_of_ref_top1 基準の 1 位が a では何位か（1 が最良。top-1 が外れたときの「惜しさ」を見る）
    第 2 引数が基準なので、fp32 との比較でもホスト QDQ との比較でも同じ関数を使える（StudyMemo 8 章）。
    """
    A = a.reshape(a.shape[0], -1).astype(np.float64)
    B = b.reshape(b.shape[0], -1).astype(np.float64)
    out = {"rel_l2": float(np.linalg.norm(A - B) / np.linalg.norm(B))}
    out["cosine"] = float(np.mean([np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                                   for x, y in zip(A, B)]))
    pa, pb = _softmax(A), _softmax(B)
    out["kl"] = float(np.mean((pb * (np.log(pb + 1e-12) - np.log(pa + 1e-12))).sum(-1)))
    top_b = B.argmax(-1)
    out["top1_agreement"] = float(np.mean(A.argmax(-1) == top_b))
    # 基準の1位が、比べる側で何位か
    ranks = [int((A[i] > A[i, top_b[i]]).sum()) + 1 for i in range(len(A))]
    out["rank_of_ref_top1"] = float(np.mean(ranks))
    return out


def load_device(work: Path, prefix: str, name: str, shape) -> np.ndarray:
    """端末が書いた出力 .bin を全サンプル分読み、[N, *shape] に積む。

    引数: work = device_accuracy の --work のフォルダ、prefix = job_id、name = 出力名（logits など）、
          shape = 1 サンプルの形（fp32 の出力の shape[1:] を渡す）。
    ファイル名は job.cpp が付ける「<job_id>_<番号3桁>_<出力名>.bin」。番号の順に並べるため sorted する。
    中身は float32 と決め打ちしている（QDQ モデルの入出力は float のままなので、端末も float32 で書く。
    device_accuracy.py の方は型を端末の返す JSON（describe）から取るので決め打ちしていない）。
    """
    fs = sorted(work.glob(f"{prefix}_*_{name}.bin"))
    if not fs:
        raise SystemExit(f"実機の出力が無い: {work}/{prefix}_*_{name}.bin")
    return np.stack([np.fromfile(f, dtype=np.float32).reshape(shape) for f in fs])


def run_onnx(path: str, npz, n: int) -> dict:
    """ONNX を PC の CPU（ORT）で先頭 n サンプル分走らせ、出力名 → [n, …] の dict を返す。

    1 サンプルずつ（バッチ 1 で）流すのは、モデルの入力の形がバッチ 1 で固定だから。
    fp32 のモデルと QDQ のモデルの両方をこの関数で走らせる（QDQ は Q/DQ の丸めを CPU で真似る「ホスト QDQ」）。
    """
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    nm = [i.name for i in s.get_inputs()]; on = [o.name for o in s.get_outputs()]
    acc = {o: [] for o in on}
    for i in range(n):
        for o, v in zip(on, s.run(None, {k: npz[k][i:i+1] for k in nm})):
            acc[o].append(v)
    return {o: np.concatenate(v, 0) for o, v in acc.items()}


def main() -> None:
    """3 者比較の本体。fp32 を基準に、ラベルごとにホスト QDQ と実機の出力を比べて表と JSON を出す。

    段: ① 引数と評価セット → ② fp32 を走らせる → ③ ラベルごとにホスト QDQ を走らせ実機の出力を読む
        → ④ KV は新しい 1 スロットだけ相対 L2 → ⑤ 主出力で 5 指標 → ⑥ 表示と保存
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--eval-npz", required=True)
    ap.add_argument("--label", nargs="+", required=True,
                    help="`名前=QDQパス=実機workディレクトリ=job_id`")
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--primary", default="logits")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    # ① 評価セットを読み、件数を --limit で切る（npz の先頭のキーの 1 軸目がサンプル数）
    d = np.load(args.eval_npz)
    n = min(args.limit, d[list(d.files)[0]].shape[0])
    print(f"評価 {n} 件\n基準 fp32 を実行中…", flush=True)
    # ② 基準の fp32 を走らせる（全ラベルで共通なので 1 回だけ）
    F = run_onnx(args.fp32, d, n)
    P = args.primary
    res = {}
    # ③ ラベル「名前=QDQパス=実機workディレクトリ=job_id」ごとに、ホスト QDQ を走らせ、実機の出力を読む
    for spec in args.label:
        tag, qdq, work, jid = spec.split("=")
        print(f"{tag}: ホスト QDQ を実行中…", flush=True)
        H = run_onnx(qdq, d, n)
        D = {P: load_device(Path(work), jid, P, F[P].shape[1:])}
        # ④ KV キャッシュの出力（present_k_* / present_v_*）は、新しく書いた最後の 1 スロット [:, :, -1:, :] だけで比べる。
        #   残りの L−1 スロットは入力の past を左へずらした写しなので、全体で測ると誤差が薄まって見える（冒頭の説明）。
        #   層ごとに相対 L2 を出し、層の平均を記録する。
        for key in ("present_k", "present_v"):
            names = [k for k in F if k.startswith(key)]
            if not names:
                continue
            hs, ds, hd = [], [], []
            for nm in names:
                dd = load_device(Path(work), jid, nm, F[nm].shape[1:])
                f_, h_, d_ = (F[nm][:, :, -1:, :], H[nm][:, :, -1:, :], dd[:, :, -1:, :])
                r = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
                hs.append(r(h_, f_)); ds.append(r(d_, f_)); hd.append(r(d_, h_))
            res.setdefault(tag, {})[key] = {"fp32_host": float(np.mean(hs)),
                                            "fp32_device": float(np.mean(ds)),
                                            "host_device": float(np.mean(hd))}
        # ⑤ 主出力（既定 logits）で 5 指標。★host_device は「同じ量子化グラフを PC と端末で走らせた差」で、
        #   量子化の誤差が混ざらない＝端末（HTP）側の問題だけを見る主指標（StudyMemo 9.1・results/P9_20260914_dec24_c13_ab.json）
        res.setdefault(tag, {})["fp32_host"] = metrics(H[P], F[P])
        res[tag]["fp32_device"] = metrics(D[P], F[P])
        res[tag]["host_device"] = metrics(D[P], H[P])

    # ⑥ ラベルごとに 3 行（fp32↔ホスト・fp32↔実機・ホスト↔実機）の表を出し、--out があれば JSON に保存する
    hdr = f"{'':26s} {'相対L2':>9} {'cos':>9} {'KL':>9} {'基準1位の順位':>13} {'top-1一致':>10}"
    for tag, r in res.items():
        print(f"\n=== {tag} ===")
        print(hdr); print("-" * 82)
        for k, lab in (("fp32_host", "fp32 ↔ ホストQDQ"),
                       ("fp32_device", "fp32 ↔ 実機HTP"),
                       ("host_device", "★ホストQDQ ↔ 実機HTP")):
            m = r[k]
            print(f"{lab:26s} {m['rel_l2']*100:8.2f}% {m['cosine']:9.5f} {m['kl']:9.4f} "
                  f"{m['rank_of_ref_top1']:13.2f} {m['top1_agreement']*100:9.1f}%")
        for key in ("present_k", "present_v"):
            if key in r:
                c = r[key]
                print(f"{key+'（新スロット）':26s} "
                      f"fp32↔ホスト {c['fp32_host']*100:6.2f}%  "
                      f"fp32↔実機 {c['fp32_device']*100:6.2f}%  "
                      f"★ホスト↔実機 {c['host_device']*100:6.2f}%")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
