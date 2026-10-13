r"""同じ DLC を **QNN CPU backend** で走らせ、ホスト QDQ・実機 HTP と比べる。

## なぜこれをやるか

実機 decode の誤差が 35% で、重みを per-channel にしても動かなかった。
ホスト（ORT CPU EP）では 31.19% → 14.06% と半分以下になるので、
**差は「ORT より後ろ」のどこか**にある。そこは2つに分かれる。

    ONNX(QDQ) ──[ qairt-converter ]──> DLC ──[ context生成 ]──> HTP で実行
                    ↑ lowering / encoding        ↑ kernel / fusion

**同じ DLC を CPU backend で実行**すれば、この2つを切り分けられる。

    CPU も悪い  → lowering / encoding 側（DLC を作る時点で壊れている）
    CPU は良い  → HTP の kernel / fusion 側

CPU backend は HTP と別実装なので**完全な対照実験ではない**（同じ DLC を
読むだけで、演算そのものは別物）。それでも「DLC まで無事か」は判定できる。

## パイプラインのどこで使うか

パイプラインの外。**「DLC までは無事か」を切り分ける道具**である。

**ただし Qwen2.5 / Qwen3 では使えない。** QNN の CPU backend は `Gather` も `RmsNorm` も
持っておらず、グラフを組めない（2026-09-14 に確認）。**記録として残している。**

## 使い方

    python -m app.pipeline.qnn_cpu_compare --dlc work/ops/qwen_dec24pc/ctx/qwendec24pc.dlc \
        --fp32 work/qwen_dec24/qwen05b_decode_L24_C64.onnx \
        --eval-npz work/qwen_dec24/eval_dec24_big.npz --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QAIRT = Path(os.environ.get("QAIRT_ROOT", r"D:\QAIRT\v2.48.0.260626\qairt\2.48.0.260626"))
BIN = QAIRT / "bin/x86_64-windows-msvc"
LIB = QAIRT / "lib/x86_64-windows-msvc"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dlc", type=Path, required=True)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--eval-npz", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--backend", default="QnnCpu.dll")
    ap.add_argument("--work", type=Path, default=Path("work/qnncpu"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    os.chdir(ROOT)

    import onnxruntime as ort

    d = np.load(args.eval_npz)
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(str(args.fp32), so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in s.get_inputs()]
    out_names = [o.name for o in s.get_outputs()]
    n = min(args.limit, d[in_names[0]].shape[0])
    print(f"入力 {len(in_names)} / 出力 {len(out_names)} / {n} サンプル")

    # ---- fp32 参照 ----
    print("fp32 参照を計算中…", flush=True)
    ref = {o: [] for o in out_names}
    for i in range(n):
        for o, v in zip(out_names, s.run(None, {k: d[k][i:i+1] for k in in_names})):
            ref[o].append(v)
    ref = {o: np.concatenate(v, 0) for o, v in ref.items()}

    # ---- qnn-net-run 用の raw を書く ----
    raw = args.work / "raw"; raw.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        parts = []
        for k in in_names:
            p = raw / f"{k}_{i}.raw"
            d[k][i:i+1].astype(np.float32).tofile(p)
            # qnn-net-run は `名前:=パス` 形式（相対パスは入力リストからの相対）
            parts.append(f"{k}:={p.resolve()}")
        lines.append(" ".join(parts))
    ilist = args.work / "input_list.txt"
    ilist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"入力リスト: {ilist}（{n} 行）")

    # ---- 実行 ----
    outdir = args.work / "out"
    env = dict(os.environ)
    env["PATH"] = f"{LIB};{BIN};" + env.get("PATH", "")
    cmd = [str(BIN / "qnn-net-run.exe"),
           "--backend", str(LIB / args.backend),
           "--dlc_path", str(args.dlc.resolve()),
           "--input_list", str(ilist.resolve()),
           "--output_dir", str(outdir.resolve())]
    print("実行:", " ".join(cmd[:3]), "…", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
    print(f"  {time.time()-t0:.1f}s  exit={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); raise SystemExit("qnn-net-run 失敗")

    # ---- 突き合わせ ----
    res = {}
    for o in out_names:
        got, exp = [], []
        for i in range(n):
            f = outdir / f"Result_{i}" / f"{o}.raw"
            if not f.is_file():
                cand = list((outdir / f"Result_{i}").glob("*.raw"))
                print(f"  {o} が見つからない。Result_{i} の中身: {[c.name for c in cand][:5]}")
                break
            got.append(np.fromfile(f, dtype=np.float32).reshape(ref[o][i].shape))
            exp.append(ref[o][i])
        if not got:
            continue
        g, e = np.array(got), np.array(exp)
        res[o] = float(np.linalg.norm(g - e) / np.linalg.norm(e))

    print("\n=== QNN CPU backend vs fp32（相対L2）===")
    if "logits" in res:
        print(f"  logits      {res['logits']*100:6.2f}%")
    for key in ("present_k", "present_v"):
        vals = [v for k, v in res.items() if k.startswith(key)]
        if vals:
            print(f"  {key} 平均 {np.mean(vals)*100:6.2f}%")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
