r"""RoPE の cos/sin テーブルを作る。

## パイプラインのどこで使うか

パイプラインの外。**端末へ置く RoPE のテーブルを作る道具**である。
モデルごとに `head_dim` が違うので、**モデルを変えたら作り直す**
（Qwen2.5 は 64、Qwen3 は 128）。

## 使い方

    # Qwen2.5（head_dim 64）
    python -m app.pipeline.make_rope_table --rows 2048 --out-dir work/rope         --verify-against work/_cos64.bin

    # Qwen3（head_dim 128）
    python -m app.pipeline.make_rope_table --rows 2048 --head-dim 128 --out-dir work/rope_q3

    adb push work/rope/cos_table_l2048.bin /sdcard/.../files/inputs/

**必ず既存のテーブルか HF の rotary と突き合わせてから使う。**

## 行数は KVキャッシュの容量とは別物

  容量 L        … 何トークンぶん覚えるか（モデルに焼かれている）
  テーブル行数  … 位置を何番まで表せるか（アプリが選ぶ）

窓が一杯になっても**位置は増え続ける**ので、行数は「生成したいトークン数」で決める。
Qwen2.5 は `max_position_embeddings=32768` なので、そこまでは理屈の上で伸ばせる。
検証では 2048 行あれば足りる。

端末に入っていた 64行のテーブルと、この式の出力が最大差 2.1e-06 で一致することを
確認済み（2026-09-14）。

## 出力（2026-10-18 追記）

`<out-dir>/cos_table_l<rows>.bin`・`sin_table_l<rows>.bin`: float32 の [rows, head_dim] をヘッダ無しで並べたもの。
行 p が位置 p の cos / sin。端末は 1 行（head_dim 個）を読んで decode の cos / sin 入力 [1,1,1,hd] に渡す。
解説: StudyMemo（EdgeSLM_Android）10.3「RoPE の位置」。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build(rows: int, head_dim: int = 64, theta: float = 1000000.0):
    """位置 0〜rows−1 の cos / sin の表を作る（HF の Qwen2 の rotary と同じ式）。

    周波数 inv[j] = 1 / theta^(2j/head_dim)（j = 0〜head_dim/2−1）、角度 = 位置 × inv。
    前半と後半に同じ値を並べる（rotate_half が前半と後半を組にして回すため）。
    計算は float64 で行い、最後に float32 にする。theta は Qwen2.5 の rope_theta（1e6）。
    戻り値: (cos, sin)。どちらも [rows, head_dim] の float32。
    """
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.arange(rows, dtype=np.float64)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1).astype(np.float32)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1).astype(np.float32)
    return cos, sin


def main() -> None:
    """表を作って .bin に書き、--verify-against があれば既存の表（先頭の行）との最大差を表示する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--theta", type=float, default=1000000.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--verify-against", type=Path,
                    help="既存のテーブルと先頭行を突き合わせる")
    args = ap.parse_args()
    cos, sin = build(args.rows, args.head_dim, args.theta)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for nm, a in (("cos", cos), ("sin", sin)):
        p = args.out_dir / f"{nm}_table_l{args.rows}.bin"
        a.tofile(p)
        print(f"  {p}  {a.shape}  {p.stat().st_size:,} bytes")
    if args.verify_against and args.verify_against.is_file():
        old = np.fromfile(args.verify_against, dtype=np.float32).reshape(-1, args.head_dim)
        n = len(old)
        print(f"  既存 {args.verify_against.name}（{n}行）との最大差: "
              f"{np.abs(cos[:n] - old).max():.3e}")


if __name__ == "__main__":
    main()
