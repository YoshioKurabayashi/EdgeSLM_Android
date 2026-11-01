r"""ホストの fp32 だけで **attention sink が効くか**を確かめる。

## なぜ実機の前にこれをやるか

窓256 で位置256 を境に top-1 一致が 75% → 44% に落ちた。だがその低下には

  ① attention sink を捨てたことによる不安定化
  ② **単に必要な過去情報が消えたこと**

が混ざっている。②だけでも一致率は落ちるので、**崖＝sink が原因とは言えない**。
sink の有無だけを変えて比べれば分けられる。**NPU の再生成は要らない。**

## パイプラインのどこで使うか

パイプラインの外。**ホストの fp32 だけで attention sink の有効性を判定する道具**である。
NPU の再生成（1本1時間）に入る前にここで決める。

## 使い方

    # 先に基準の系列を作る（窓なし fp32 が書いた文章）
    # → work/ref_seq.json（ids / full_nll / full_ranks）
    python -m app.pipeline.sink_compare_host         --model work/qwen_dec24c256/qwen05b_decode_L24_C256.onnx         --ref-json work/ref_seq.json --sinks 0 1 4 8

## 3つの方式（前提が違う）

| | Q の位置 | K の位置 | |
|---|---|---|---|
| 旧実装 | cache内（L-1 で clamp） | 絶対 | **不整合** |
| naive | 絶対 | 絶対 | 整合。いまの実装 |
| sink | cache内 | cache内（**再回転**） | StreamingLLM |

## 再回転（rebase）

RoPE は回転なので合成できる。位置 p の K を位置 p-1 として使うには `R(-1)` を掛ける。

    R(-1): K' = K * cos(θ) - rot_half(K) * sin(θ)      （θ は位置1ぶんの角度）
    rot_half(x) = concat(-x[d/2:], x[:d/2])

キャッシュは毎ステップ1つ左へずれるので、**保持する K に毎回 R(-1) を掛ける**。
sink スロットは動かさないので掛けない。

## 指標

top-1 一致だけでなく **NLL** と **順位** も出す。一致率は「情報が消えたから変わった」
だけでも落ちるが、NLL が跳ねるなら**分布そのものが壊れている**ことを示す。
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: StudyMemo 10.5 節「attention sink は効かなかった」（Qwen2.5-0.5B、3 系列・fp32・teacher forcing）。
#   結果: 3 系列とも sink を残しても同等か悪化（先頭 1 固定は NLL 差 0.0007）。結果ファイルは results/ には見当たらない。
#   prefix_recall_test.py は rope_tables と rot_back_one をここから import して使う。
# 入力:  --model（固定長キャッシュの decode の ONNX）、--ref-json（窓なし fp32 が作った系列。ids を使う。
#        context_policy_host.reference が作る形）。
# 出力:  画面に位置の帯ごとの top-1 率・順位の中央値・NLL の平均、--out に JSON。
# 注意: docstring の「使い方」の行の途中にある長い空白は、元は行の継続（\ と改行）だったものと推定。
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def rope_tables(rows: int, head_dim: int = 64, theta: float = 1000000.0):
    """RoPE の cos/sin 表 [rows, head_dim] を作る（theta 1e6 は Qwen2.5 の値。前半と後半に同じ角度を並べる）。"""
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.arange(rows, dtype=np.float64)[:, None] * inv[None, :]
    return (np.concatenate([np.cos(ang), np.cos(ang)], -1).astype(np.float32),
            np.concatenate([np.sin(ang), np.sin(ang)], -1).astype(np.float32))


def rot_back_one(k: np.ndarray, cos1: np.ndarray, sin1: np.ndarray) -> np.ndarray:
    """位置を1つ戻す回転 `R(-1)` を掛ける。k の最後の次元が head_dim。"""
    d = k.shape[-1]
    h = d // 2
    rot = np.concatenate([-k[..., h:], k[..., :h]], axis=-1)
    return k * cos1 - rot * sin1


def run_mode(model: str, ref: list[int], sink: int, mode: str,
             mask_neg: float = -60.0) -> dict:
    """基準の系列 ref を teacher forcing で 1 トークンずつ流し、各位置の次トークンの順位と NLL を返す。

    引数: sink = 先頭何スロットを固定するか、mode = "absolute"（絶対位置・固定なし）か "cache_local"（cache 内位置＋再回転）。
    戻り値: {"ranks": [...], "nll": [...], "cache_len": L}。
    """
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in s.get_inputs()}
    outs = [o.name for o in s.get_outputs()]
    L = ins["attn_mask"].shape[-1]
    HD = ins["cos"].shape[-1]
    cos_t, sin_t = rope_tables(max(len(ref) + 8, L + 8), HD)
    cos1 = cos_t[1].astype(np.float32)
    sin1 = sin_t[1].astype(np.float32)

    feed = {nm: (np.zeros([1, 1], np.int64) if nm == "input_ids"
                 else np.zeros([d if isinstance(d, int) else 1 for d in vi.shape], np.float32))
            for nm, vi in ins.items()}
    pres = {o: outs.index(o) for o in outs if o.startswith("present_")}
    li = outs.index("logits")
    kn = [o for o in pres if o.startswith("present_k")]

    # ★cache内位置では「新トークンは **常に位置 L-1** で入り、毎ステップ1つ下がる」。
    #   `pos = min(t, L-1)` にすると序盤のトークンが位置 t で入ってから下げられ、
    #   実効位置が `2*entry_t - current_t` というでたらめな値になる（最初これで測って
    #   「sink は逆効果」という誤った結論を出しかけた）。
    #
    #   sink スロットの固定は **t == L-1 の瞬間**に行う。そのとき slot j にはちょうど
    #   トークン j が入っていて、回転も位置 j ぶんまで下がっている（entry 位置 L-1 から
    #   (L-1-j) 回下げられているため）。**その値をそのまま固定すればよい。**
    sink_kv: dict[str, np.ndarray] = {}
    ranks, nll = [], []
    n_valid = 0
    for t in range(len(ref) - 1):
        feed["input_ids"][0, 0] = ref[t]
        pos = t if mode == "absolute" else (L - 1)
        feed["cos"][:] = cos_t[pos].reshape(feed["cos"].shape)
        feed["sin"][:] = sin_t[pos].reshape(feed["sin"].shape)
        n_valid = min(n_valid + 1, L)
        mk = np.full((L,), mask_neg, np.float32); mk[L - n_valid:] = 0.0
        feed["attn_mask"][:] = mk.reshape(feed["attn_mask"].shape)

        got = s.run(None, feed)
        pinned = sink > 0 and t >= L - 1
        for o, oi in pres.items():
            cur = got[oi]
            if mode == "cache_local":
                cur = cur.copy()
                if o in kn:
                    # ★**最新スロットも含めて**回す。`present` は次ステップで
                    #   **全スロットが左へ1つ動く**ので、全部の位置が1つ下がるため。
                    #   `[lo:L-1]` にすると最新トークンだけ位置が下がらず、
                    #   次ステップで「スロットは動いたのに位置はそのまま」になる。
                    #   （最初これで測って sink が逆効果に見えた。sink なしの
                    #    cache内位置が絶対位置と一致しないことで気付いた）
                    lo = sink if pinned else 0
                    cur[:, :, lo:L, :] = rot_back_one(cur[:, :, lo:L, :], cos1, sin1)
                if sink > 0 and t == L - 1:
                    sink_kv[o] = cur[:, :, :sink, :].copy()   # ここで固定する
                if pinned and o in sink_kv:
                    cur[:, :, :sink, :] = sink_kv[o]
            feed["past_" + o[len("present_"):]][:] = cur

        lg = got[li].reshape(-1).astype(np.float64)
        nxt = ref[t + 1]
        ranks.append(int((lg > lg[nxt]).sum()) + 1)
        z = lg - lg.max()
        nll.append(float(-(z[nxt] - np.log(np.exp(z).sum()))))
    return {"ranks": ranks, "nll": nll, "cache_len": L}


def summarize(name: str, r: dict, bands) -> None:
    """位置の帯（[lo, hi)）ごとに top-1 率・順位の中央値・NLL の平均を表で出す。位置は予測する側の番号（t+1）。"""
    rk = np.array(r["ranks"]); nl = np.array(r["nll"])
    pos = np.arange(len(rk)) + 1
    print(f"\n=== {name} ===")
    print(f"{'絶対位置':>13} {'top-1率':>8} {'順位中央':>9} {'NLL 平均':>9}")
    print("-" * 44)
    for lo, hi in bands:
        sel = (pos >= lo) & (pos < hi)
        if sel.sum() == 0:
            continue
        print(f"{lo:5d}〜{hi:5d} {(rk[sel]==1).mean()*100:7.1f}% "
              f"{np.median(rk[sel]):9.1f} {nl[sel].mean():9.3f}")


def main() -> None:
    """--sinks の数ごとに run_mode を回し、帯ごとの表を出す（0 は naive＝絶対位置・sink なし）。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref-json", type=Path, required=True, help="基準の系列（窓なし fp32）")
    ap.add_argument("--sinks", type=int, nargs="+", default=[0, 1, 4, 8])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    ref = json.loads(args.ref_json.read_text(encoding="utf-8"))["ids"]
    bands = [(0,128),(128,256),(256,272),(272,320),(320,400),(400,500),(500,len(ref))]
    res = {}
    for sk in args.sinks:
        mode = "absolute" if sk == 0 else "cache_local"
        label = "naive（sink なし・絶対位置）" if sk == 0 else f"sink {sk}（cache内位置・再回転）"
        print(f"\n実行中: {label} …", flush=True)
        r = run_mode(args.model, ref, sk, mode)
        res[label] = r
        summarize(label, r, bands)
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
