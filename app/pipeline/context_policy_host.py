r"""固定サイズ KV キャッシュの **文脈保持の方針** を fp32 だけで比べる。

## パイプラインのどこで使うか

パイプラインの外。`sink_compare_host` の上位版で、**基準の系列を自分で作る**。

## 使い方

    python -m app.pipeline.context_policy_host         --model work/qwen_dec24c256/qwen05b_decode_L24_C256.onnx         --pins 0 4 --max-new 400 --prompt "..." --label "系列1"

## 比べるもの

先頭 N スロットを固定し、残りを rolling window にする。N の意味づけが2通りある。

| N | 呼び方 | 狙い |
|---:|---|---|
| 1〜8 | attention sink | 先頭トークンへ集まる注意を保つ（StreamingLLM） |
| 32〜64 | **固定 prefix 保持** | **system prompt や指示を守る**（車載エージェント等の実用形） |

仕組みは同じ（先頭 N を固定 + cache内位置 + 再回転）だが、**測り方が違う**。
sink は「先頭を捨てると数値が壊れるか」、prefix は「**守るべき情報が消えると困るか**」
なので、**先頭に守る価値のある内容があるプロンプト**で測らないと意味がない。

## 指標

teacher-forced NLL（小さいほど良い）と top-1 一致率。基準は**窓なし fp32**。

## 実装の注意（2026-09-14 に2回間違えた）

  - cache内位置では新トークンは **常に位置 L-1** で入る（`min(t, L-1)` ではない）
  - 再回転は **最新スロットも含めた全スロット**に掛ける（次ステップで全部が左へ動く）
  - 検算: N=0 なら cache内位置と絶対位置は一致するはず（NLL 差 0.000003 を確認済み）
## 学習用の補足（2026-10-18 追記）

- 段: P10 の文脈の長さの検証（Qwen2.5-0.5B の decode・KV 256。StudyMemo 10.4〜10.6 節）。
  結果は StudyMemo 10.5 節の表（3 系列とも「先頭 4 固定」は「固定なし」と同等か悪い →
  Qwen2.5-0.5B は先頭に強い attention sink を持たない）。関係する記録は results/P10_20260914_context_length.md
- 入力: decode の fp32 ONNX（--model。export_qwen_decode.py の出力）・HF のモデル ID（--model-id。基準の系列を作る）・プロンプト
- 出力: 画面の表（絶対位置の帯ごとの NLL の平均。窓なし fp32 と、--pins の各 N）と、--out を付ければ JSON
  （prompt_len・n と、N ごとの nll・ranks）
- 中身の計算（キャッシュの持ち方・位置の付け方）は sink_compare_host.run_mode が行う。ここは基準の系列を作り、帯ごとにまとめるだけ
- 使い方の行は、元は複数行に分けたコマンドを 1 行につないだもの（推定）。PowerShell でも bash でも 1 行で打てる
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# run_mode(モデル, 系列, N, mode): 系列を 1 トークンずつ流し（teacher forcing）、位置ごとの NLL・順位と cache_len を返す。
# mode は "absolute"（絶対位置。N=0 のとき）か "cache_local"（cache 内位置 + 再回転。先頭 N を固定するとき）
from app.pipeline.sink_compare_host import run_mode


def reference(model_id: str, text: str, max_new: int):
    """窓なし fp32 で系列を作り、その teacher-forced NLL も返す（これが上限）。"""
    # （2026-10-18 追記）model_id: HF のモデル（transformers で読む。torch を使う）、text: プロンプト、max_new: 足す語の数。
    # 戻り値: ids（プロンプト + 生成した語の全系列）・prompt_len・full_nll（各位置で次の語を当てる NLL。長さ n−1）・
    #         full_ranks（次の語の順位。1 が最良）
    # ① greedy で max_new 語ちょうど生成する（下の ★）→ ② 全系列を 1 回で流し直して、位置ごとの NLL と順位を出す
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tk = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    ids = tk(text)["input_ids"]
    with torch.no_grad():
        # ★長さを強制する。EOS で止まると窓（256）をほとんど超えられず、
        #   「満杯後どうなるか」が測れない。
        seq = m.generate(torch.tensor([ids]), max_new_tokens=max_new,
                         min_new_tokens=max_new, do_sample=False,
                         num_beams=1, pad_token_id=tk.eos_token_id)[0].tolist()
        lg = m(torch.tensor([seq])).logits[0].float()
    # ② 位置 i の logits で、次の語 seq[i+1] を当てる NLL と順位（それより大きい logit の数 + 1）
    tgt = torch.tensor(seq[1:])
    nll = torch.nn.functional.cross_entropy(lg[:-1], tgt, reduction="none").numpy()
    rk = [int((lg[i] > lg[i, seq[i + 1]]).sum()) + 1 for i in range(len(seq) - 1)]
    return {"ids": seq, "prompt_len": len(ids), "full_nll": nll.tolist(), "full_ranks": rk}


def bands_of(n: int, L: int):
    """結果をまとめる絶対位置の帯: [0, L)（窓に収まる間）・[L, L+32)・[L+32, L+96)・[L+96, n)（窓が埋まった後）。
    n は系列の長さ、L はキャッシュの長さ（256 など）。
    """
    return [(0, L), (L, L + 32), (L + 32, L + 96), (L + 96, n)]


def main() -> None:
    """① 基準の系列を作る → ② --pins の各 N で run_mode を回す → ③ 帯ごとの NLL の平均を表にする。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--pins", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    # ① 窓なし fp32 の基準（これより良くはならない上限。StudyMemo の「上限（fp32）を測る」）
    ref = reference(args.model_id, args.prompt, args.max_new)
    seq = ref["ids"]
    fn = np.array(ref["full_nll"]); fr = np.array(ref["full_ranks"])
    res = {"prompt_len": ref["prompt_len"], "n": len(seq)}
    print(f"\n########## {args.label or args.prompt[:40]}")
    print(f"  プロンプト {ref['prompt_len']} トークン / 全長 {len(seq)}")

    rows = {}
    for n in args.pins:
        mode = "absolute" if n == 0 else "cache_local"
        r = run_mode(args.model, seq, n, mode)
        rows["窓なし" if False else (f"固定なし（recent のみ）" if n == 0 else f"先頭{n}固定")] = r
        res[str(n)] = {"nll": r["nll"], "ranks": r["ranks"]}
    L = rows[list(rows)[0]]["cache_len"]
    pos = np.arange(len(fn)) + 1
    bands = bands_of(len(seq), L)

    # ③ 帯ごとの表。pos は「当てられる語」の絶対位置（1 から）。窓が埋まる帯（lo == L）に印を付ける
    print(f"\n{'絶対位置':>13} {'窓なし':>9} " + " ".join(f"{k:>12}" for k in rows))
    print("-" * (14 + 10 + 13 * len(rows)))
    for lo, hi in bands:
        sel = (pos >= lo) & (pos < hi)
        if sel.sum() == 0:
            continue
        line = f"{lo:5d}〜{hi:5d} {fn[sel].mean():9.3f} "
        for k, r in rows.items():
            nl = np.array(r["nll"])
            line += f"{nl[sel].mean():12.3f} "
        mark = "  ← 窓が埋まる" if lo == L else ""
        print(line + mark)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
