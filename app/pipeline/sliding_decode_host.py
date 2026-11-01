r"""ホストで **端末と同じ sliding-window decode** を回す。

## なぜ必要か

端末で 600 トークン生成すると、窓（256）が埋まるあたりから文章が崩れる。
だが fp32 を**窓なし**で回すと崩れない。したがって原因は

  (a) 256 の窓で古い文脈を捨てること自体
  (b) 量子化や HTP への移植

のどちらか。**同じ窓・同じ位置の付け方**で fp32 と QDQ を回せば分けられる。

## パイプラインのどこで使うか

パイプラインの外。**端末の C++ と同じ生成ループをホストで再現する道具**である。
「実機がおかしいのか、方式がおかしいのか」を分けるのに使う。

## 使い方

    python -m app.pipeline.sliding_decode_host         --model work/qwen_dec24c256/qwen05b_decode_L24_C256.onnx --max-new 600

## 端末（job.cpp `generate_kv`）と同じにしていること

  - キャッシュ更新はモデル内の `cat(past, k)[:, :, 1:, :]`（毎ステップ最古を捨てる）
  - マスクは末尾 `n_valid` 個だけ開ける（`n_valid` は L で頭打ち）
  - 位置は **絶対位置のまま増やす**（2026-09-14 に clamp を外したのに合わせる）
  - 貪欲（argmax）
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 10.1〜10.4 節（窓 256 が埋まる所の崖）・8.8 節（無効化して既知の方式と一致するか）。
#   記録: results/P10_20260914_context_length.md（道具の表にこのファイルがある）。
# 入力:  --model（固定長キャッシュの decode の ONNX。fp32 でも QDQ でもよい）、--prompt（英文）、
#        トークナイザ Qwen/Qwen2.5-0.5B-Instruct（HF の名前で読む）。
# 出力:  画面に区間ごとの生成文（序盤・満杯直前・満杯直後・終盤）、--out に生成した ID の JSON。
# 注意: docstring の「使い方」の行の途中にある長い空白は、元は行の継続（\ と改行）だったものと推定。1 行で打てばよい。
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def rope_tables(rows: int, head_dim: int = 64, theta: float = 1000000.0):
    """RoPE の cos/sin 表を作る。戻り値: (cos [rows, head_dim], sin [rows, head_dim])、float32。

    theta 1e6 は Qwen2.5 の rope_theta。前半と後半に同じ角度を並べる形（HF の rotate_half と同じ並び）。
    """
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.arange(rows, dtype=np.float64)[:, None] * inv[None, :]
    return (np.concatenate([np.cos(ang), np.cos(ang)], -1).astype(np.float32),
            np.concatenate([np.sin(ang), np.sin(ang)], -1).astype(np.float32))


def generate(model: str, prompt_ids: list[int], max_new: int,
             mask_neg: float = -60.0, progress: int = 100) -> dict:
    """端末と同じ生成ループで、prompt_ids を流してから greedy で max_new トークン作る。

    引数: model = ONNX のパス、prompt_ids = プロンプトの ID、mask_neg = マスクの値（−60、C-9）。
    戻り値: {"new_ids": 生成した ID のリスト, "cache_len": キャッシュ長 L}。
    入力の形（L・head_dim）はモデルの入力から読むので、Qwen2.5（64）でも Qwen3（128）でも動く。
    """
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in s.get_inputs()}
    outs = [o.name for o in s.get_outputs()]
    L = ins["attn_mask"].shape[-1]
    HD = ins["cos"].shape[-1]
    cos_t, sin_t = rope_tables(len(prompt_ids) + max_new + 8, HD)

    # ① 入力をすべてゼロで用意する（KV キャッシュは空から始まる）。動的な次元は 1 にする。
    feed = {}
    for nm, vi in ins.items():
        if nm == "input_ids":
            feed[nm] = np.zeros([1, 1], dtype=np.int64)
        else:
            feed[nm] = np.zeros([d if isinstance(d, int) else 1 for d in vi.shape],
                                dtype=np.float32)
    present = {o: outs.index(o) for o in outs if o.startswith("present_")}
    li = outs.index("logits")

    new_ids, last, n_valid = [], prompt_ids[0], 0
    total = len(prompt_ids) + max_new
    for t in range(total):
        feed["input_ids"][0, 0] = prompt_ids[t] if t < len(prompt_ids) else last
        feed["cos"][:] = cos_t[t].reshape(feed["cos"].shape)
        feed["sin"][:] = sin_t[t].reshape(feed["sin"].shape)
        n_valid = min(n_valid + 1, L)
        m = np.full((L,), mask_neg, dtype=np.float32)
        m[L - n_valid:] = 0.0
        feed["attn_mask"][:] = m.reshape(feed["attn_mask"].shape)

        got = s.run(None, feed)
        for o, oi in present.items():
            feed["past_" + o[len("present_"):]][:] = got[oi]
        last = int(np.argmax(got[li].reshape(-1)))
        if t >= len(prompt_ids) - 1:
            new_ids.append(last)
        if progress and (t + 1) % progress == 0:
            print(f"    {t+1}/{total}", flush=True)
    return {"new_ids": new_ids[:max_new], "cache_len": L}


def main() -> None:
    """プロンプトを ID にして generate を回し、窓が埋まる前後の区間の文を並べて表示する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The history of computing begins with the abacus. "
                                        "Later, mechanical calculators were built by Pascal and Leibniz. ")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ids = tk(args.prompt)["input_ids"]
    print(f"  プロンプト {len(ids)} トークン → 生成 {args.max_new}")
    r = generate(args.model, ids, args.max_new)
    g = r["new_ids"]
    print(f"  窓 {r['cache_len']} / 生成 {len(g)}\n")
    for lo, hi, lab in [(0,60,"序盤 0-60"),(200,260,"200-260"),
                        (260,320,"★満杯直後 260-320"),(540,600,"終盤 540-600")]:
        seg = g[lo:hi]
        if seg:
            print(f"--- {lab} ---\n  " + tk.decode(seg).replace("\n", " ⏎ ")[:300] + "\n")
    if args.out:
        args.out.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
