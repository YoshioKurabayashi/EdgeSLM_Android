# -*- coding: utf-8 -*-
"""層2 MLP 出力（linear_20 = down_proj）の巨大値が、どのトークン・位置・チャンネルで出るかを確かめる（P12k-1）。

「FP32 で固定 prefix の KV を事前計算し、端末は先頭トークンを計算しない」方式が成立する条件は、
**巨大値が固定 prefix の中（位置0）でしか出ない**こと。入力内容しだいで後続にも出るなら成立しない。

公式 HF の FP32 で層2 の mlp 出力を hook し、プロンプトごとに
  - |値| の上位（位置・チャンネル・値・トークン）
  - 位置0 と位置1以降それぞれの最大 |値|
を出す。あわせて、層3 に入る残差（hidden_states[3]）でも同じ位置・チャンネルかを確認する。

【補足（学習用、2026-10-18 追記）】
どの段か: StudyMemo 11.4 節「なぜ scale がこんなに粗いのか — 位置0 の巨大活性」の表（9 通りのプロンプト）の出どころ。
    記録は results/P12k_20260916_fp32_prefix.md（P12k「FP32 で固定 prefix を事前計算する案」の 1 段目）。
結果: 9 通りすべてで位置0 の最大はチャンネル35（<|im_start|> で始まるものは 8142.7、生テキストは 5749.7〜7259.2）。
    位置1以降の最大は 2.50〜4.79。2 ターン目以降の <|im_start|>（位置0 でない）は 1.25〜1.75 にとどまる。
    → 巨大値はトークンの種類ではなく「位置0」で出る。
入力:  Qwen3-0.6B の公式スナップショット（HF のキャッシュ。ネットには出ない）。
出力:  results/P12k_20260916_fp32_prefix/massive_origin.json（プロンプトごとの記録）と画面の 1 行要約。
動かし方: & $PY2 -B app/pipeline/massive_origin_check.py（引数なし。パスは Windows の絶対パスで固定）
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("D:/AI/source/n_chap5/EdgeSLM_Android")
SNAP = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
OUT = ROOT / "results/P12k_20260916_fp32_prefix/massive_origin.json"


def main():
    """9 通りのプロンプトを公式 HF の FP32 に通し、層2 の MLP 出力と層3 に入る残差の大きな値の場所を記録する。"""
    tok = AutoTokenizer.from_pretrained(str(SNAP), local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(str(SNAP), dtype=torch.float32, local_files_only=True).eval()
    # ① 層2 の MLP の出力をフックで写す（値は変えない）。ONNX の linear_20（down_proj の出力）に当たる。
    cap = {}
    m.model.layers[2].mlp.register_forward_hook(lambda mod, a, o: cap.__setitem__("mlp2", o.detach().numpy()))  # MLP の出力はテンソル [1,T,1024]

    # ② チャットの形のプロンプトをトークン列にする（思考は無し）。生テキストはそのまま encode する。
    def chat(msgs):
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False))["input_ids"]

    # ③ 比べるプロンプト: system あり・英語・長い system・system 無し・2 ターン分の履歴・生テキスト 3 本・改行で始まる文。
    cases = {
        "chat_ja_car": chat([{"role": "system", "content": "あなたは車載アシスタントです。短く答えてください。"},
                             {"role": "user", "content": "エアコンを22度にして"}]),
        "chat_en_banana": chat([{"role": "system", "content": 'Always answer exactly "BANANA". Nothing else.'},
                                {"role": "user", "content": "What is 2+2?"}]),
        "chat_long_system": chat([{"role": "system", "content": "You are an in-car assistant. Tools: set_temperature(celsius), "
                                   "play_music(genre), navigate(destination). Never discuss the weather. Address the driver as Captain."},
                                  {"role": "user", "content": "Navigate to Tokyo Station."}]),
        "chat_no_system": chat([{"role": "user", "content": "音楽をかけて"}]),
        "chat_two_turns": chat([{"role": "system", "content": "短く答えて。"}, {"role": "user", "content": "こんにちは"},
                                {"role": "assistant", "content": "こんにちは！"}, {"role": "user", "content": "さっき何て言った？"}]),
        "raw_ja": tok("日本の首都は東京です。")["input_ids"],
        "raw_en": tok("The capital of France is Paris.")["input_ids"],
        "raw_code": tok("def add(a, b):\n    return a + b")["input_ids"],
        "raw_starts_with_newline": tok("\n\nこんにちは")["input_ids"],
    }
    res = {}
    for name, ids in cases.items():
        with torch.no_grad():
            o = m(torch.tensor([ids]), output_hidden_states=True)
        y = cap["mlp2"][0]                                  # [T,1024] 層2 MLP 出力 = linear_20
        h3 = o.hidden_states[3][0].numpy()                  # 層3 に入る残差 = add_26
        a = np.abs(y)
        flat = np.argsort(-a.ravel())[:5]
        top = [{"pos": int(i // a.shape[1]), "ch": int(i % a.shape[1]), "value": float(y.ravel()[i]),
                "token": tok.decode([ids[i // a.shape[1]]])} for i in flat]
        later = a[1:].max() if len(ids) > 1 else 0.0
        lp = int(np.unravel_index(a[1:].argmax(), a[1:].shape)[0] + 1) if len(ids) > 1 else None
        # 2ターン目の <|im_start|> など、位置0以外の特殊トークンの値
        specials = [{"pos": p, "token": tok.decode([t]), "max_abs": float(a[p].max())}
                    for p, t in enumerate(ids) if t in (151644, 151645) and p > 0]
        res[name] = {"n_tokens": len(ids), "first_token": tok.decode([ids[0]]), "first_token_id": ids[0],
                     "pos0_max_abs": float(a[0].max()), "pos0_argmax_ch": int(a[0].argmax()),
                     "later_max_abs": float(later), "later_argmax_pos": lp,
                     "h3_pos0_ch35": float(h3[0, 35]), "h3_later_max_abs": float(np.abs(h3[1:]).max()) if len(ids) > 1 else 0.0,
                     "top5": top, "special_tokens_after_pos0": specials}
        r = res[name]
        print(f"  {name:22s} 先頭 {r['first_token']!r:14s} 位置0 最大 {r['pos0_max_abs']:8.1f}(ch{r['pos0_argmax_ch']})  "
              f"位置1以降 最大 {r['later_max_abs']:7.2f}(pos{r['later_argmax_pos']})  "
              f"層3残差 pos0/ch35 {r['h3_pos0_ch35']:8.1f} 後続最大 {r['h3_later_max_abs']:6.2f}"
              + ("  | 位置0以外の特殊トークン: " + ", ".join(f"pos{s['pos']}{s['token']!r}={s['max_abs']:.2f}" for s in specials) if specials else ""))
    # ④ 全プロンプトの記録を JSON に書く。
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
