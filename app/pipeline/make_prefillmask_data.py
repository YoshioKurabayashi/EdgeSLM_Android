r"""マスク入力版 prefill（seq=S）の較正／評価データを作る。

## 何を入れるか

入力は4本（`input_ids` / `cos` / `sin` / `attn_mask`）。**右詰め**で、
有効トークンより前は `attn_mask` で塞ぐ（`export_qwen_decode.build_mask` と同じ約束）。

較正データは「実行時の入力分布を模倣する」ものでなければ意味がない
（[DoC/StudyMemo/02_量子化の基礎.md] 2.2節。乱数 ±1.0 で較正して誤差が
30.4% → 38.9% に悪化した実例がある）。窓ずらし生成では

  - プロンプトが窓より短い間 … パディングあり（有効長が短い）
  - 生成が進んで窓が埋まった後 … **パディング無し（有効長 = S）**

の両方が起きる。**後者の方が回数は多い**ので、較正には
**有効長を散らしつつ満了ケースを厚めに**入れる。

## 使い方

    python -m app.pipeline.make_prefillmask_data --seq 64 --out work/qwen_pm64/calib_pm64.npz
    python -m app.pipeline.make_prefillmask_data --seq 64 --eval \
        --out work/qwen_pm64/eval_pm64.npz

`--eval` は**較正と別の文**を使う。同じ文で較正して同じ文で評価すると
量子化誤差を過小評価する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.export_qwen_decode import build_mask, load, rope_cos_sin  # noqa: E402
from pipeline.make_calib_data import EVAL_TEXTS, TEXTS  # noqa: E402

PAD_ID = 0

# 生成で実際に来る短いプロンプト。有効長のばらつきを作るために使う。
PROMPTS = [
    "The capital city of France is",
    "The capital city of Japan is Tokyo. The capital city of France is",
    "Tokyo is",
    "Water boils at",
    "The largest planet in our solar system is",
    "Two plus two equals",
    "The sun rises in the",
    "Python is a programming",
    "The first president of the United States was",
    "The chemical symbol for water is",
]
EVAL_PROMPTS = [
    "The capital of Italy is",
    "The tallest mountain in the world is",
    "A group of lions is called a",
    "The speed of light is approximately",
    "An atom consists of protons, neutrons and",
    "The longest river in Africa is",
    "Three times four equals",
    "Java is a programming",
]


def windows_from_text(ids: list[int], seq: int) -> list[list[int]]:
    """長い文から**満了した窓**を切り出す（パディング無しのケース）。"""
    out = []
    for start in range(0, max(1, len(ids) - seq + 1), max(1, seq // 2)):
        w = ids[start:start + seq]
        if len(w) == seq:
            out.append(w)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval", action="store_true", help="評価用（較正と別の文）")
    ap.add_argument("--max-samples", type=int, default=0, help="0 で制限なし")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = load(args.model_id)
    S = args.seq

    # cos/sin は位置だけで決まるので全サンプル共通（窓の先頭を 0 とする）
    cos, sin = rope_cos_sin(model, list(range(S)))

    texts = EVAL_TEXTS if args.eval else TEXTS
    prompts = EVAL_PROMPTS if args.eval else PROMPTS

    rows: list[tuple[str, list[int], int]] = []   # (種別, 窓のID列, 有効長)

    # ① 短いプロンプト（パディングあり）。生成の最初の数ステップがこれ。
    for p in prompts:
        ids = tok(p, add_special_tokens=False)["input_ids"][-S:]
        rows.append(("pad", [PAD_ID] * (S - len(ids)) + ids, len(ids)))

    # ② 窓に満たない文は右詰めで1件ずつ（有効長のばらつきを作る）
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) < S:
            rows.append(("pad", [PAD_ID] * (S - len(ids)) + ids, len(ids)))
        else:
            for w in windows_from_text(ids, S):
                rows.append(("full", w, S))

    # ③ 満了した窓（パディング無し）。**生成の大半がこれ**なので厚めに入れる。
    #
    # 手元の文は1本ずつだと 64 トークンに届かないので、**連結した長い列から
    # 窓をずらして切り出す**。連結の境目は人工的だが、窓ずらし生成でも
    # 「直前の文脈が途中から始まる」状況は普通に起きるので許容する。
    joined = []
    for t in texts:
        joined += tok(t, add_special_tokens=False)["input_ids"]
    for w in windows_from_text(joined, S):
        rows.append(("full", w, S))

    # ④ 中間の有効長（窓の 1/4・1/2・3/4 程度）。①と③の間を埋める。
    for n in (S // 4, S // 2, 3 * S // 4):
        n = max(1, n)
        ids = joined[:n]
        rows.append(("pad", [PAD_ID] * (S - n) + ids, n))

    if args.max_samples:
        rows = rows[:args.max_samples]

    ids_a = np.array([r[1] for r in rows], dtype=np.int64)
    msk_a = np.stack([build_mask(r[2], S)[0] for r in rows]).astype(np.float32)
    cos_a = np.repeat(cos, len(rows), axis=0).astype(np.float32)
    sin_a = np.repeat(sin, len(rows), axis=0).astype(np.float32)
    nv_a = np.array([r[2] for r in rows], dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, input_ids=ids_a, cos=cos_a, sin=sin_a,
             attn_mask=msk_a, n_valid=nv_a)

    meta = {"seq": S, "samples": len(rows), "pad_id": PAD_ID,
            "kinds": {k: sum(1 for r in rows if r[0] == k) for k in ("pad", "full")},
            "n_valid": sorted(set(int(x) for x in nv_a)),
            "eval": bool(args.eval)}
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"-> {args.out}")
    print(f"   {len(rows)} サンプル（満了 {meta['kinds']['full']} / "
          f"パディングあり {meta['kinds']['pad']}）")
    print(f"   有効長: {meta['n_valid']}")
    for name, a in (("input_ids", ids_a), ("cos", cos_a), ("sin", sin_a),
                    ("attn_mask", msk_a)):
        print(f"   {name:10s} {a.shape} {a.dtype}")


if __name__ == "__main__":
    main()
