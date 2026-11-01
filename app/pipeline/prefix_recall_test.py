r"""**先頭に置いた事実を、窓が埋まった後で思い出せるか**を測る。

## パイプラインのどこで使うか

パイプラインの外。**固定 prefix に意味があるかを判定する道具**である。

`context_policy_host` は「全体の予測しやすさ」を見るが、こちらは
**答えのトークン1つの順位**だけを見る。system prompt のように
「守らないと困る情報」があるかを測るにはこちらが要る。

## 使い方

    python -m app.pipeline.prefix_recall_test         --model work/qwen_dec24c256/qwen05b_decode_L24_C256.onnx         --pins 0 16 32 64

## なぜこの形にしたか

「system prompt を固定する」効果を測ろうとして、チャット形式で NLL を比べたが
**実験が成立していなかった**（2026-09-14）。窓なし fp32 の生成に、system prompt で
指示したマーカーが **0回**しか出ず、**先頭に守るべき情報が存在しなかった**。
prefix を固定しても直近トークンを捨てるだけになり、当然悪化する。

そこで**先頭の情報が後半の予測に必ず効く**形にする。

    [先頭] 事実（合言葉は 8317）
    [中間] 無関係な長文（窓 256 を超えさせる）
    [末尾] 質問「合言葉は？」 → 答えのトークンを何位に置くか

**答えのトークンの順位と NLL だけ**を見る。中間の文の予測しやすさは関係ない。
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 10.6 節「固定 prefix は効く。ただしサイズを中身に合わせる」（Qwen2.5-0.5B）。
#   記録: 10.6 節の表（先頭 16 固定で窓なしとほぼ同じ所まで戻る。再回転の有無は 1 本では優劣が付かなかった）。
#   この道具の結果ファイルは results/ には見当たらない（--out を付けて流した記録が残っていない）。
# 入力:  --model（固定長キャッシュの decode の ONNX）、--model-id（トークナイザと窓なし fp32 の HF モデル）。
# 出力:  画面の表（方式ごとの答えの順位と NLL）、--out に JSON。
# 注意: docstring の「使い方」の行の途中にある長い空白は、元は行の継続（\ と改行）だったものと推定。
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.pipeline.sink_compare_host import rope_tables, rot_back_one

# 先頭の事実・質問・答え・無関係な文（窓 256 を超えさせるために --filler-repeat 回くり返す）。
FACT = "Remember this fact carefully: the secret access code is 8317. "
QUESTION = " Question: what is the secret access code? Answer: the secret access code is"
ANSWER = " 8317"
FILLER = ("The weather in the mountains changes quickly during autumn. "
          "Hikers often carry extra layers of clothing for warmth. "
          "Rivers run faster after the rain, and the paths become slippery. "
          "Many birds migrate south before the first frost arrives. "
          "The forest floor is covered with fallen leaves of many colours. ")


def score(model: str, seq: list[int], pin: int, answer_at: int,
          absolute_prefix: bool = False) -> dict:
    """`absolute_prefix=True` なら **絶対 RoPE のまま prefix を固定する**。

    StreamingLLM の sink は「注意の逃げ場を近くに置き続ける」ために cache内位置と
    再回転を使うが、**意味のある固定 prefix は目的が違う**。窓なしモデルでも
    prefix は位置0付近、現在トークンは位置700付近で注意しているので、
    **絶対位置のまま置く方が自然**かもしれない。これなら**再回転が要らない**。
    """
    """teacher forcing で流し、位置 `answer_at` の次トークンの順位と NLL を返す。"""
    # 【説明】上の 2 つ目の文字列は docstring にはならない（関数の先頭の 1 つ目だけが docstring）。中身の説明として読む。
    #   引数: model = ONNX、seq = 全トークン列（答えまで含む）、pin = 先頭何トークンを固定するか（0 = 固定なし）、
    #   answer_at = 答えを予測する位置。戻り値: {"ranks": [順位…], "nll": [NLL…]}（answer_at 以降の位置ぶん）。
    #   方式: pin=0 は絶対位置・固定なし。pin>0 は absolute_prefix=True なら絶対位置のまま先頭を固定、
    #   False なら cache 内位置＋再回転（sink_compare_host.run_mode と同じやり方）。
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in s.get_inputs()}
    outs = [o.name for o in s.get_outputs()]
    L = ins["attn_mask"].shape[-1]; HD = ins["cos"].shape[-1]
    cos_t, sin_t = rope_tables(max(len(seq) + 8, L + 8), HD)
    cos1, sin1 = cos_t[1].astype(np.float32), sin_t[1].astype(np.float32)
    feed = {nm: (np.zeros([1, 1], np.int64) if nm == "input_ids"
                 else np.zeros([d if isinstance(d, int) else 1 for d in vi.shape], np.float32))
            for nm, vi in ins.items()}
    pres = {o: outs.index(o) for o in outs if o.startswith("present_")}
    kn = [o for o in pres if o.startswith("present_k")]
    li = outs.index("logits")
    # ① cache 内位置では、新しいトークンは常に位置 L−1 で入る（sink_compare_host.py の ★ の説明）。
    mode = "absolute" if (pin == 0 or absolute_prefix) else "cache_local"
    sink_kv: dict[str, np.ndarray] = {}
    n_valid = 0
    out = {}
    for t in range(len(seq) - 1):
        feed["input_ids"][0, 0] = seq[t]
        pos = t if mode == "absolute" else (L - 1)
        feed["cos"][:] = cos_t[pos].reshape(feed["cos"].shape)
        feed["sin"][:] = sin_t[pos].reshape(feed["sin"].shape)
        n_valid = min(n_valid + 1, L)
        mk = np.full((L,), -60.0, np.float32); mk[L - n_valid:] = 0.0
        feed["attn_mask"][:] = mk.reshape(feed["attn_mask"].shape)
        got = s.run(None, feed)
        pinned = pin > 0 and t >= L - 1
        for o, oi in pres.items():
            cur = got[oi]
            if absolute_prefix and pin > 0:
                # 再回転なし。**元の絶対位置の回転のまま**固定する
                cur = cur.copy()
                if t == L - 1:
                    sink_kv[o] = cur[:, :, :pin, :].copy()
                if pinned and o in sink_kv:
                    cur[:, :, :pin, :] = sink_kv[o]
            elif mode == "cache_local":
                cur = cur.copy()
                if o in kn:
                    lo = pin if pinned else 0
                    cur[:, :, lo:L, :] = rot_back_one(cur[:, :, lo:L, :], cos1, sin1)
                if pin > 0 and t == L - 1:
                    sink_kv[o] = cur[:, :, :pin, :].copy()
                if pinned and o in sink_kv:
                    cur[:, :, :pin, :] = sink_kv[o]
            feed["past_" + o[len("present_"):]][:] = cur
        if t >= answer_at:
            lg = got[li].reshape(-1).astype(np.float64)
            nxt = seq[t + 1]
            z = lg - lg.max()
            out.setdefault("ranks", []).append(int((lg > lg[nxt]).sum()) + 1)
            out.setdefault("nll", []).append(float(-(z[nxt] - np.log(np.exp(z).sum()))))
    return out


def main() -> None:
    """事実 + 無関係な文 + 質問 + 答え の系列を作り、窓なし fp32（上限）と、固定の方式ごとの答えの順位・NLL を表にする。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--pins", type=int, nargs="+", default=[0, 16, 32, 64])
    ap.add_argument("--filler-repeat", type=int, default=12)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model_id)
    fact = tk(FACT)["input_ids"]
    filler = tk(FILLER * args.filler_repeat)["input_ids"]
    q = tk(QUESTION)["input_ids"]
    ans = tk(ANSWER)["input_ids"]
    # ★答えのトークンを**系列に入れる**。入れないと「質問文の末尾を当てられるか」を
    #   測ってしまう（2026-09-14 に一度やった）。
    answer_at = len(fact) + len(filler) + len(q) - 1
    seq = fact + filler + q + ans
    print(f"  事実 {len(fact)} + 無関係な文 {len(filler)} + 質問 {len(q)} = {len(seq)} トークン")
    print(f"  （窓 256 を {len(seq)-256} トークン超える → 事実は追い出される）")
    print(f"  答えのトークン: {tk.convert_ids_to_tokens(ans)}")

    # 窓なし fp32（上限）
    # ② 窓なし fp32（HF のモデルに全系列を一度に通す）の順位と NLL。これが上限（StudyMemo 一番大事なこと 9）。
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).eval()
    with torch.no_grad():
        lg = m(torch.tensor([seq])).logits[0].float()
    fr = [int((lg[i] > lg[i, seq[i+1]]).sum())+1 for i in range(answer_at, len(seq)-1)]
    fn = [float(torch.nn.functional.cross_entropy(lg[i:i+1], torch.tensor([seq[i+1]])))
          for i in range(answer_at, len(seq)-1)]
    print(f"\n{'方式':>22} {'答えの順位':>12} {'答えの NLL':>12}")
    print("-" * 50)
    print(f"{'窓なし fp32（上限）':>22} {fr[0]:12d} {fn[0]:12.3f}")
    res = {"full": {"ranks": fr, "nll": fn}}
    # ③ pin ごとに、固定長キャッシュの ONNX で方式を変えて測る（pin=0 は固定なしだけ）。
    for pin in args.pins:
        for absol in ((False,) if pin == 0 else (True, False)):
            r = score(args.model, seq, pin, answer_at, absolute_prefix=absol)
            if pin == 0:
                lab = "固定なし（recent のみ）"
            else:
                lab = f"先頭{pin}固定・" + ("絶対位置（再回転なし）" if absol else "cache内位置（再回転）")
            print(f"{lab:>34} {r['ranks'][0]:12d} {r['nll'][0]:12.3f}")
            res[f"{pin}_{'abs' if absol else 'local'}"] = r
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
