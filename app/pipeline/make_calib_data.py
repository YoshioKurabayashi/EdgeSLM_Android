"""本物のキャリブレーションデータを作る（P3）

## なぜ乱数ではいけないか

P2b までのキャリブレーションは乱数と合成パターンだった。「op が載るか」を見るなら
それで足りるが、**精度を測る段では致命的**になる。量子化は
「キャリブレーションで見た値域を N 段階に割る」処理なので、**値域が実際と違えば
量子化そのものが的外れになる**。乱数 ±1 で較正したモデルに、実際は ±30 の値が
来れば全部クリップされる。

## 何を作るか

`--mode layer` の export はデコーダ層1枚で、入力は **hidden_states**（埋め込み後の
ベクトル）である。したがって実テキストをトークン化し、**実際の埋め込み表を通した
hidden_states** を集める。これが層1枚から見た「本物の入力」になる。

24層のうち0層目だけを見ている点は承知のうえである。深い層ほど値域が広がることが
知られているので、**層ごとに較正が必要かどうか**は P4 で別途確認する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# 較正用の文。長さと内容に幅を持たせる（1種類の文だけだと値域が偏る）。
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "人工知能の研究は、計算機科学の中でも特に発展の速い分野である。",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Q: What is the capital of France?\nA: The capital of France is Paris.",
    "スマートフォンに搭載された NPU は、行列演算を低消費電力で実行する。",
    "1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987",
    "In a hole in the ground there lived a hobbit.",
    "量子化とは、連続的な値を有限個の離散値で近似することをいう。",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1;",
    "こんにちは。今日はいい天気ですね。散歩にでも行きませんか。",
    "The mitochondrion is the powerhouse of the cell.",
    "エッジ環境での推論は、通信を介さずに完結する点に価値がある。",
]


# 評価用。**較正に使った文とは別のもの**を使う。
# 同じ文で較正して同じ文で評価すると、量子化誤差を過小評価する。
EVAL_TEXTS = [
    "It was the best of times, it was the worst of times.",
    "機械学習モデルの推論を端末側で完結させる利点は、応答の速さと秘匿性にある。",
    "for i in range(10):\n    print(i * i)",
    "The capital of Japan is Tokyo, which has a population of about 14 million.",
    "半導体の微細化が限界に近づくなか、専用回路による効率化が進んでいる。",
    "2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096",
    "She sells seashells by the seashore.",
    "推論サービスを C++ で実装する場合、メモリ管理の設計が性能を左右する。",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("work/qwen/calib_layer1.npz"))
    ap.add_argument("--input-name", default="hidden_states")
    ap.add_argument("--ids", action="store_true",
                    help="hidden_states ではなく input_ids を出す（フルモデル用）")
    ap.add_argument("--eval", action="store_true",
                    help="評価用データを作る（較正とは別の文を使う）")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.float32, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()

    embed = model.model.embed_tokens
    texts = EVAL_TEXTS if args.eval else TEXTS
    samples: list[np.ndarray] = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids
            # 長さを seq に揃える。足りなければ繰り返して埋める
            # （pad トークンで埋めると値域が pad に引っ張られる）。
            flat = ids[0]
            while flat.numel() < args.seq:
                flat = torch.cat([flat, flat])
            for start in range(0, max(1, flat.numel() - args.seq + 1), args.seq):
                chunk = flat[start:start + args.seq]
                if chunk.numel() < args.seq:
                    break
                if args.ids:
                    # フルモデル（input_ids 入力）用。int64 のまま出す。
                    # トークン ID は量子化されないが、較正では**実際の文を通して
                    # 活性化の値域を観測させる**ことが目的なので実 ID が要る。
                    samples.append(chunk.unsqueeze(0).numpy().astype(np.int64))
                else:
                    h = embed(chunk.unsqueeze(0)).float().numpy()
                    samples.append(h.astype(np.float32))

    arr = np.concatenate(samples, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{args.input_name: arr})

    print(f"WROTE {args.out}  {arr.shape[0]} サンプル  shape={arr.shape[1:]} dtype={arr.dtype}")
    if arr.dtype != np.float32:
        print(f"  トークンID 値域 [{arr.min()}, {arr.max()}]")
        args.out.with_suffix(".meta.json").write_text(
            json.dumps({"samples": int(arr.shape[0]), "shape": list(arr.shape[1:]),
                        "dtype": str(arr.dtype)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return
    print(f"  値域   min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  分布   mean={arr.mean():.4f}  std={arr.std():.4f}")
    p = np.percentile(np.abs(arr), [50, 99, 99.9, 100])
    print(f"  |x| の分位 p50={p[0]:.4f} p99={p[1]:.4f} p99.9={p[2]:.4f} max={p[3]:.4f}")
    # 乱数 ±1 との比較。ここが大きく違うなら乱数較正は的外れだったということ。
    print(f"  ※ P2b までの乱数較正は ±1.0 の一様分布だった")

    meta = {"model_id": args.model_id, "seq": args.seq,
            "input_name": args.input_name, "samples": int(arr.shape[0]),
            "shape": list(arr.shape[1:]),
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std())}
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
