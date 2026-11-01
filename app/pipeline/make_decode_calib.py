"""P4: decode モデルの較正データを作る（キャッシュ込み）

## 難しいところ

decode モデルの入力は 52 本あり、うち 48 本が **KV キャッシュ**である。
キャッシュの中身は**前のステップの出力**なので、乱数やゼロで埋めると
実行時とまったく違う分布になる。P3 で「乱数較正は実データ較正より 8.5pt 悪い」
ことを測っているので、ここは手を抜けない。

## やること

**fp32 の decode モデルを実テキストで1トークンずつ回し、各ステップの入力一式を記録する。**
キャッシュは前ステップの出力をそのまま渡すので、**実行時と同じ分布**になる。

    トークン列 →  step 0: (ids, cos, sin, mask, 全部ゼロのキャッシュ)
                 step 1: (ids, cos, sin, mask, step0 の present)
                 step 2: ...

ゼロ埋めのキャッシュ（最初の数ステップ）も実行時に現れるので**含めてよい**。

## 入力と出力・動かし方（2026-10-18 追記）

    python -m app.pipeline.make_decode_calib --model <fp32 の decode.onnx> --cache-len 512 --steps 6 --out work/calib/<名前>.npz
    （--eval を付けると評価用の文 EVAL_TEXTS で作る。EdgeSLM_AndroidHead では scripts/run_p2.ps1 の 2) がこの道具を呼ぶ）

- 入力: export_qwen_decode が書いた fp32 の decode ONNX（入力は input_ids・cos・sin・attn_mask・past_* の順）、HF のトークナイザ
- 出力: 入力名ごとに全ステップを積んだ npz（例 past_k_0 は [文の数×steps, KV, L, hd]）と、同じ名前の .meta.json（ステップ数・文・L）
- 使う所: make_spec の --calib に渡し、p2b_ops prepare（量子化）の較正に使う。解説は StudyMemo（EdgeSLM_Android）8 章「比較を成立させる」
- ★入力の名前は input_ids 決め打ち。--embeds-input で書き出した模型（第 1 入力が inputs_embeds）には使えない
  （EdgeSLM_AndroidScale の 1.5B は make_calib_15b.py を使う）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# 空きスロットを塞ぐ値（export_qwen_decode.MASK_NEG と同じ値にしておくこと）
MASK_NEG = -60.0

# 較正に使う文（日本語・英語・数字の並び・コード）。評価用の文 EVAL_TEXTS は較正と重ならない別の文にする（StudyMemo 8 章）
TEXTS = [
    "日本の首都は",
    "The capital of France is",
    "1, 2, 3, 4,",
    "人工知能とは",
    "def add(a, b):",
]
EVAL_TEXTS = [
    "アメリカの首都は",
    "The largest planet is",
    "10, 20, 30,",
    "機械学習とは",
]


def main() -> None:
    """fp32 の decode を ORT で 1 トークンずつ回し、各ステップの入力一式を記録して npz にする。

    段: ① HF（トークナイザと RoPE の表のため）と ORT のセッションを用意 → ② 文ごとにキャッシュを 0 から始め、
        --steps ステップ進めながら入力を記録 → ③ 全ステップを積んで npz と meta を書き、主な入力の値域を表示
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="fp32 の decode ONNX")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cache-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6, help="各文で進めるステップ数")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.pipeline.export_qwen_decode import rope_cos_sin

    tok = AutoTokenizer.from_pretrained(args.model_id)
    try:
        hf = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    except TypeError:
        hf = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    hf.eval()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    cache_names = [n for n in in_names if n.startswith("past_")]
    # 入力の 5 番目（past_k_0）の形 [1, KV, L, hd] から KV のヘッド数と hd を取る（入力の並びが決まっている前提）
    nkv, hd = sess.get_inputs()[4].shape[1], sess.get_inputs()[4].shape[3]

    texts = EVAL_TEXTS if args.eval else TEXTS
    acc: dict[str, list[np.ndarray]] = {n: [] for n in in_names}

    for t in texts:
        ids = tok(t, return_tensors="np").input_ids[0].tolist()
        # 生成も少し混ぜる: 実際の decode では自分の出力を食べるため
        caches = {n: np.zeros((1, nkv, args.cache_len, hd), dtype=np.float32)
                  for n in cache_names}
        seq = list(ids)
        # ② s 番目のステップ: 文の語が残っていればそれを、尽きたら直前に予測した語を入れる（範囲は結果として --steps と同じ）
        for s in range(min(args.steps, len(ids) + args.steps)):
            if s < len(ids):
                cur = ids[s]
            else:
                cur = seq[-1]            # 直前の予測トークン
            n_valid = s + 1
            cos, sin = rope_cos_sin(hf, [s])
            mask = np.full((1, 1, 1, args.cache_len), MASK_NEG, dtype=np.float32)
            mask[0, 0, 0, args.cache_len - min(n_valid, args.cache_len):] = 0.0

            # 実行時と同じ入力を作って記録してから動かし、出力 present_* を次の past_* に回す
            feeds = {"input_ids": np.array([[cur]], dtype=np.int64),
                     "cos": cos, "sin": sin, "attn_mask": mask, **caches}
            for n in in_names:
                acc[n].append(feeds[n])

            outs = sess.run(None, feeds)
            got = dict(zip(out_names, outs))
            caches = {f"past_{k[len('present_'):]}": got[k]
                      for k in out_names if k.startswith("present_")}
            if s >= len(ids) - 1:
                seq.append(int(got["logits"][0, -1].argmax()))

    # ③ 入力名ごとに全ステップを 1 軸目で連結する（較正の道具は 1 軸目をサンプルとして読む）
    arrays = {n: np.concatenate(v, 0) for n, v in acc.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    n = arrays["input_ids"].shape[0]
    print(f"WROTE {args.out}  {n} ステップ分 / 入力 {len(arrays)} 本")
    for k in ["input_ids", "cos", "attn_mask", "past_k_0", "past_k_23"]:
        if k in arrays:
            a = arrays[k]
            print(f"  {k:12s} {str(a.shape):22s} 値域 [{a.min():.4f}, {a.max():.4f}]")
    (args.out.with_suffix(".meta.json")).write_text(json.dumps(
        {"steps": n, "texts": texts, "cache_len": args.cache_len}, ensure_ascii=False,
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
