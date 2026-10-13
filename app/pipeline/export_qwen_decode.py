"""Qwen の decode グラフ（seq=1 + 固定長KVキャッシュ）を export する

## 対応しているモデル

  Qwen2.5-0.5B … 24層 / hidden 896 / KV 2 / head_dim 64  / RMSNorm 49個
  Qwen3-0.6B   … 28層 / hidden 1024 / KV 8 / head_dim 128 / RMSNorm 113個

Qwen3 で違うのは2つ（2026-09-14 に対応）。

  - `head_dim` を config が持ち、**`hidden/heads` と一致しない**（1024/16=64 だが実際は128）
  - projection の直後に **ヘッドごとの RMSNorm**（`q_norm` / `k_norm`）が入る

## なぜ HF の forward をそのまま使わないのか

transformers の KV キャッシュは `Cache` オブジェクトで管理され、**長さが動的に伸びる**。
QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）ので、
**キャッシュを明示的なテンソル入出力として固定長で持つ**形に組み直す必要がある。

ただし**重みは本物をそのまま使う**。組み直すのは配線だけである。
配線が正しいことは、**HF の `generate` と logits を突き合わせて確認する**
（自分で書いた attention が HF と一致するか、推測で済ませない）。

## 固定長キャッシュの方式: 左シフト + マスク

形状を静的に保つため、長さ L を固定し**毎ステップ1つ左へずらして末尾に書く**:

    present_k = Concat( past_k[:, :, 1:, :], new_k )      # [1, KV, L, D] のまま

新しいトークンが常に**末尾**に来るので、有効範囲は「後ろ n 個」になる。
空きスロットは `attn_mask`（ホストが毎ステップ作る [1,1,1,L]）で塞ぐ。

`ScatterND` で位置指定して書く方式もあるが、左シフトなら
**`Slice` と `Concat` だけ**で済む。どちらも P2b で v68 動作を確認済みである。

## 入出力

    入力:  input_ids   [1, 1]        int64
           cos / sin   [1,1,1,hd]    float32   （ホストが計算して渡す。
                                                 端末で Cos/Sin を使わないため）
           attn_mask   [1, 1, 1, L]  float32   （0 または -60。C-9 に従い -inf は使わない）
           past_k_{i}  [1, KV, L, D] float32   × 24
           past_v_{i}  [1, KV, L, D] float32   × 24
    出力:  logits      [1, 1, V]     float32
           present_k_{i} / present_v_{i}       × 24

入出力が 51 / 49 本になるので、**runner の多入力・多出力対応（v0.2 / v0.3）が効く**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

MASK_NEG = -60.0   # C-9: -inf は量子化できない


class QwenDecodeStep(torch.nn.Module):
    """1トークン分の decode。**重みは本物、配線だけ自前。**"""

    def __init__(self, model, cache_len: int, layers: int | None = None):
        super().__init__()
        cfg = model.config
        self.cfg = cfg
        self.L = cache_len
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        # ★Qwen3 は `head_dim` を config に持ち、`hidden/heads` と**一致しない**。
        #   Qwen3-0.6B: hidden 1024 / heads 16 だが head_dim は **128**（割り算だと 64）。
        #   割り算で出すと q_proj の出力（2048）を割り切れず、形が崩れる。
        self.hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
        self.rep = self.nh // self.nkv
        self.scale = self.hd ** -0.5

        self.embed = model.model.embed_tokens
        self.rotary = model.model.rotary_emb
        self.layers = torch.nn.ModuleList(
            list(model.model.layers[:layers]) if layers else list(model.model.layers))
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def _rope(self, x, cos, sin):
        # HF と同じ rotate_half
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def forward(self, input_ids, cos, sin, attn_mask, *caches):
        """cos / sin は**ホストが計算して渡す**。

        position_ids を入力にすると `Cos` / `Sin` がグラフに残るが、
        これらは v68 で未検証の op である。RoPE の cos/sin は位置だけで決まる
        表引きなので、**端末で計算する理由がない**。ホストで作って渡す。
        （`rope_cos_sin()` が HF の rotary と同じ値を作る。）
        """
        h = self.embed(input_ids)                        # [1,1,H]

        presents = []
        for i, layer in enumerate(self.layers):
            past_k, past_v = caches[2 * i], caches[2 * i + 1]
            attn = layer.self_attn
            res = h
            x = layer.input_layernorm(h)

            # ★Qwen3 は projection の直後に **ヘッドごとの RMSNorm**（q_norm / k_norm）を掛ける。
            #   最後の次元（head_dim）に対する正規化なので transpose の前後どちらでも同じ。
            #   Qwen2.5 には無いので `hasattr` で分岐する。
            q = attn.q_proj(x).view(1, 1, self.nh, self.hd)
            k = attn.k_proj(x).view(1, 1, self.nkv, self.hd)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = attn.v_proj(x).view(1, 1, self.nkv, self.hd).transpose(1, 2)

            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)

            # 左シフトして末尾に書く（長さ L のまま）。
            #
            # ★「連結してから捨てる」順にしているのには理由がある（2026-09-13）。
            #
            #   書きたい式      : cat(past[:, :, 1:, :], k)     ← past が Slice に入る
            #   こう書いている  : cat(past, k)[:, :, 1:, :]     ← past が Concat に入る
            #
            # どちらも結果は同じ（[past の2番目以降, k] で長さ L）。違うのは
            # **past_k / past_v というグラフ入力が最初に入る op** である。
            #
            # ORT の QDQ 量子化器は **`Slice` に直接入る float のグラフ入力を
            # 量子化しない**。`tensor_quant_overrides` で型を指定しても効かない
            # （最小モデルで確認: MatMul と Concat なら量子化される / Slice は float のまま）。
            # その結果 C-1 違反になり、QNN 側で
            #   "QNN_DATATYPE_FLOAT_32_converted_past_k_0_QNN_DATATYPE_FLOAT_16"
            #     generated: could not create op
            # となって context binary が作れない（v68 の skel は fp16 非対応）。
            #
            # **ハードの制約ではなくツールの制約**なので、順序を入れ替えて避ける。
            # 2026-09-13 以前はこれが decode の本当の行き止まりだった
            # （`rms_norm` の融合が原因だと誤って記録していた）。
            pk = torch.cat((past_k, k), dim=2)[:, :, 1:, :]
            pv = torch.cat((past_v, v), dim=2)[:, :, 1:, :]
            presents += [pk, pv]

            # GQA: kv ヘッドを query ヘッド数へ
            kk = pk.repeat_interleave(self.rep, dim=1)
            vv = pv.repeat_interleave(self.rep, dim=1)

            scores = torch.matmul(q, kk.transpose(2, 3)) * self.scale   # [1,nh,1,L]
            scores = scores + attn_mask
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, vv)                                # [1,nh,1,hd]
            ctx = ctx.transpose(1, 2).reshape(1, 1, self.nh * self.hd)
            h = res + attn.o_proj(ctx)

            res = h
            x = layer.post_attention_layernorm(h)
            h = res + layer.mlp(x)

        h = self.norm(h)
        return (self.lm_head(h), *presents)


class QwenPrefill(torch.nn.Module):
    """seq=S の prefill。**attention マスクを入力で受け取る。**

    ★なぜ作り直したか（2026-09-13）:
      最初の prefill export は HF の forward をそのまま使ったため、
      **因果マスクが定数として焼き込まれ、パディング用のマスク入力が無かった**。
      そのため窓（16トークン）に足りないプロンプトを pad で埋めると
      **pad トークンも attention され、出力が崩れた**。

      fp32 で確認した結果（ホスト）:

          プロンプト                      真値(pad無し)   pad+mask      pad+maskなし
          日本の首都は東京です。フランスの首都は  、東京の南      、東京の南     ？ - お笑
          The capital city of France is   Paris. It is    同じ          Paris, and it
          日本の首都は                     、その都心部    同じ          、2018

      **mask を渡せば pad 無しと完全に一致する。** Qwen 本来の制約ではなく、
      こちらの export の作りの問題だった。

      配線は `QwenDecodeStep` と同じもの（HF と 3.3e-05 で一致を確認済み）を
      KVキャッシュ無し・seq=S に広げただけである。
    """

    def __init__(self, model, seq: int, layers: int | None = None,
                 last_only: bool = False):
        super().__init__()
        self.step = QwenDecodeStep(model, seq, layers)   # 部品を共有する
        self.seq = seq
        # last_only: 最終段の norm と lm_head を**最終位置だけ**に掛ける。
        #
        # 生成では最終位置の logits しか使わないので、他の位置を計算する意味がない。
        # 窓を広げると効果が大きい（S に比例して lm_head が重くなる）:
        #   S=16 : lm_head は 16×896×151936、出力 logits  9.7MB
        #   S=64 : 同 64×896×151936（4倍）、出力 logits 38.9MB
        # 最終位置だけなら S に依らず 1×896×151936 / 0.6MB で済む。
        #
        # 数学的には「最終位置の出力」は他の位置に依存しない（因果マスクがあるため）
        # ので、切る位置をここにしても答えは変わらない。fp32 で確認する。
        self.last_only = last_only

    def forward(self, input_ids, cos, sin, attn_mask):
        st = self.step
        h = st.embed(input_ids)                                  # [1,S,H]
        for layer in st.layers:
            attn = layer.self_attn
            res = h
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(1, self.seq, st.nh, st.hd)
            k = attn.k_proj(x).view(1, self.seq, st.nkv, st.hd)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = attn.v_proj(x).view(1, self.seq, st.nkv, st.hd).transpose(1, 2)
            q = st._rope(q, cos, sin)
            k = st._rope(k, cos, sin)
            kk = k.repeat_interleave(st.rep, dim=1)
            vv = v.repeat_interleave(st.rep, dim=1)
            scores = torch.matmul(q, kk.transpose(2, 3)) * st.scale   # [1,nh,S,S]
            scores = scores + attn_mask
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, vv).transpose(1, 2).reshape(1, self.seq, st.nh * st.hd)
            h = res + attn.o_proj(ctx)
            res = h
            h = res + layer.mlp(layer.post_attention_layernorm(h))
        if self.last_only:
            h = h[:, -1:, :]
        return st.lm_head(st.norm(h))


def build_mask(n_valid: int, seq: int, neg: float = MASK_NEG) -> np.ndarray:
    """因果 + パディングを合わせた 4D マスクを作る（右詰め前提）。

    有効トークンは**後ろ n_valid 個**に置く約束。
    それより前（パディング）は全行で塞ぐ。
    """
    m = np.full((1, 1, seq, seq), neg, dtype=np.float32)
    off = seq - n_valid
    for i in range(off, seq):
        m[0, 0, i, off:i + 1] = 0.0        # 自分自身まで見える（因果）
    # パディング行（前半）は何も見えないが、softmax が NaN にならないよう自分だけ開ける
    for i in range(off):
        m[0, 0, i, i] = 0.0
    return m


def verify_prefill(pf: QwenPrefill, model, seq: int, n_check: int = 5) -> dict:
    # pf が last_only なら出力は [1,1,V]。どちらでも `[:, -1:, :]` で最終位置が取れる。
    """**HF に attention_mask を渡した結果**と突き合わせる。"""
    tok_ids = [785, 6722, 3283, 315, 9625, 374][:n_check + 1]
    saved = model.model.layers
    model.model.layers = pf.step.layers
    pad = 0
    diffs, t1 = [], []
    with torch.no_grad():
        ids = tok_ids
        n = len(ids)
        w = [pad] * (seq - n) + ids
        am = [0] * (seq - n) + [1] * n
        ref = model(input_ids=torch.tensor([w]),
                    attention_mask=torch.tensor([am])).logits[:, -1:, :]
        cos, sin = rope_cos_sin(model, list(range(seq)))
        mask = build_mask(n, seq)
        mine = pf(torch.tensor([w]), torch.from_numpy(cos), torch.from_numpy(sin),
                  torch.from_numpy(mask))[:, -1:, :]
        diffs.append(float((ref - mine).abs().max()))
        t1.append(bool(ref.argmax(-1).item() == mine.argmax(-1).item()))
    model.model.layers = saved
    return {"max_abs_diff": max(diffs), "top1_match": all(t1)}


def rope_cos_sin(model, positions) -> tuple[np.ndarray, np.ndarray]:
    """HF の rotary_emb と同じ cos/sin を作る（形は [1,1,1,hd]）。

    自前で三角関数を書かず **HF のモジュールを呼ぶ**。
    式を写し間違えると静かにずれるため。
    """
    with torch.no_grad():
        dummy = torch.zeros(1, len(positions), model.config.hidden_size)
        cos, sin = model.model.rotary_emb(dummy, torch.tensor([positions]))
    return (cos.unsqueeze(1).numpy().astype(np.float32),
            sin.unsqueeze(1).numpy().astype(np.float32))


def load(model_id: str):
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                 attn_implementation="eager")
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    m.eval()
    return m


def verify(step: QwenDecodeStep, model, cache_len: int, n_check: int = 6) -> dict:
    """**自前の配線が HF と一致するか確かめる。**

    HF に同じトークン列を1つずつ食わせ、各ステップの logits を比べる。
    ここが合わなければ、以降の精度議論はすべて無意味になる。
    """
    tok_ids = [9707, 11, 1879, 0, 3555, 374]          # 適当な実トークン列
    tok_ids = tok_ids[:n_check]
    nl, nkv, hd = len(step.layers), step.nkv, step.hd
    # ★HF 側も同じ層数にする。
    #   これを忘れて「自前2層 vs HF24層」を比べ、配線バグと誤判定した（2026-09-13）。
    saved = model.model.layers
    model.model.layers = step.layers
    caches = [torch.zeros(1, nkv, cache_len, hd) for _ in range(2 * nl)]
    diffs, t1 = [], []

    with torch.no_grad():
        # HF 側は素の forward に全履歴を渡して最後の位置の logits を取る
        for n in range(1, len(tok_ids) + 1):
            ids = torch.tensor([tok_ids[:n]])
            ref = model(input_ids=ids, use_cache=False).logits[:, -1:, :]

            # 自前側は1トークンずつ
            c, sn = rope_cos_sin(model, [n - 1])
            mask = torch.full((1, 1, 1, cache_len), MASK_NEG)
            mask[0, 0, 0, cache_len - n:] = 0.0        # 有効なのは後ろ n 個
            out = step(torch.tensor([[tok_ids[n - 1]]]),
                       torch.from_numpy(c), torch.from_numpy(sn), mask, *caches)
            mine, caches = out[0], list(out[1:])

            d = float((ref - mine).abs().max())
            diffs.append(d)
            t1.append(bool(ref.argmax(-1).item() == mine.argmax(-1).item()))

    model.model.layers = saved
    return {"max_abs_diff": max(diffs), "per_step": [round(d, 5) for d in diffs],
            "top1_match": all(t1), "steps": len(diffs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cache-len", type=int, default=128)
    ap.add_argument("--layers", type=int, default=0, help="0 で全層")
    ap.add_argument("--opset", type=int, default=21)
    ap.add_argument("--out", type=Path, default=Path("work/qwen_decode"))
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode",
                    help="prefill: KVキャッシュ無し・seq=S・マスク入力あり")
    ap.add_argument("--seq", type=int, default=16, help="prefill の窓")
    ap.add_argument("--last-only", action="store_true",
                    help="最終段の norm と lm_head を最終位置だけに掛ける（窓を広げるとき必須級）")
    args = ap.parse_args()

    print(f"=== load {args.model_id} ===")
    model = load(args.model_id)
    step = QwenDecodeStep(model, args.cache_len, args.layers or None)
    nl = len(step.layers)
    print(f"  layers {nl} / cache_len {args.cache_len} / "
          f"kv_heads {step.nkv} / head_dim {step.hd}")

    if args.mode == "prefill":
        pf = QwenPrefill(model, args.seq, args.layers or None,
                         last_only=args.last_only)
        nl = len(pf.step.layers)
        print(f"  [prefill] layers {nl} / seq {args.seq}"
              f"{' / last_only' if args.last_only else ''}")
        print("=== 自前の配線を HF（attention_mask 付き）と突き合わせる ===")
        v = verify_prefill(pf, model, args.seq)
        print(f"  最大絶対差 {v['max_abs_diff']:.3e}  top-1 一致 {v['top1_match']}")
        if not v["top1_match"] or v["max_abs_diff"] > 1e-2:
            raise SystemExit("配線が HF と一致しない。export を中止する")
        if args.verify_only:
            return
        args.out.mkdir(parents=True, exist_ok=True)
        ids = torch.zeros((1, args.seq), dtype=torch.long)
        c0, s0 = rope_cos_sin(model, list(range(args.seq)))
        mk = torch.from_numpy(build_mask(args.seq, args.seq))
        tag = f"L{nl}_S{args.seq}" + ("_last" if args.last_only else "")
        path = args.out / f"qwen05b_prefillmask_{tag}.onnx"
        print(f"=== export -> {path} ===")
        with torch.no_grad():
            torch.onnx.export(
                pf, (ids, torch.from_numpy(c0), torch.from_numpy(s0), mk), str(path),
                input_names=["input_ids", "cos", "sin", "attn_mask"],
                output_names=["logits"], dynamic_axes=None,
                opset_version=args.opset, do_constant_folding=True, dynamo=True)
        import onnx, collections
        m2 = onnx.load(str(path), load_external_data=False)
        c = collections.Counter(n.op_type for n in m2.graph.node)
        meta = {"mode": "prefill_mask", "layers": nl, "seq": args.seq,
                "inputs": ["input_ids", "cos", "sin", "attn_mask"],
                "outputs": ["logits"], "nodes": len(m2.graph.node),
                "op_types": dict(sorted(c.items(), key=lambda kv: -kv[1])),
                "verify": v, "mask_neg": MASK_NEG, "last_only": args.last_only,
                "path": str(path).replace("\\", "/")}
        (args.out / f"prefillmask_{tag}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {len(m2.graph.node)} ノード / {len(c)} 種")
        return

    print("=== 自前の配線を HF と突き合わせる ===")
    v = verify(step, model, args.cache_len)
    print(f"  最大絶対差 {v['max_abs_diff']:.3e}  top-1 全一致 {v['top1_match']}")
    print(f"  各ステップ {v['per_step']}")
    if not v["top1_match"] or v["max_abs_diff"] > 1e-2:
        raise SystemExit("配線が HF と一致しない。export を中止する")
    if args.verify_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    ids = torch.zeros((1, 1), dtype=torch.long)
    c0, s0 = rope_cos_sin(model, [0])
    cos_t, sin_t = torch.from_numpy(c0), torch.from_numpy(s0)
    mask = torch.zeros((1, 1, 1, args.cache_len), dtype=torch.float32)
    caches = [torch.zeros(1, step.nkv, args.cache_len, step.hd) for _ in range(2 * nl)]

    in_names = (["input_ids", "cos", "sin", "attn_mask"]
                + [f"past_{'kv'[j]}_{i}" for i in range(nl) for j in (0, 1)])
    out_names = (["logits"]
                 + [f"present_{'kv'[j]}_{i}" for i in range(nl) for j in (0, 1)])

    path = args.out / f"qwen05b_decode_L{nl}_C{args.cache_len}.onnx"
    print(f"=== export -> {path} ===")
    with torch.no_grad():
        torch.onnx.export(step, (ids, cos_t, sin_t, mask, *caches), str(path),
                          input_names=in_names, output_names=out_names,
                          dynamic_axes=None, opset_version=args.opset,
                          do_constant_folding=True, dynamo=True)

    import onnx
    m = onnx.load(str(path), load_external_data=False)
    import collections
    c = collections.Counter(n.op_type for n in m.graph.node)
    meta = {"model_id": args.model_id, "layers": nl, "cache_len": args.cache_len,
            "kv_heads": step.nkv, "head_dim": step.hd, "vocab": step.cfg.vocab_size,
            "inputs": in_names, "outputs": out_names,
            "nodes": len(m.graph.node), "op_types": dict(sorted(c.items(), key=lambda kv: -kv[1])),
            "verify": v, "mask_neg": MASK_NEG, "path": str(path).replace("\\", "/")}
    (args.out / f"decode_L{nl}_C{args.cache_len}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  {len(m.graph.node)} ノード / {len(c)} 種")
    print(f"  入力 {len(in_names)} 本 / 出力 {len(out_names)} 本")
    print(f"  -> {args.out / f'decode_L{nl}_C{args.cache_len}.meta.json'}")


if __name__ == "__main__":
    main()
