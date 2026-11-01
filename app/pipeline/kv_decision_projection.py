# -*- coding: utf-8 -*-
"""KV 誤差が「音−エ」の判定に効く向きを向いているかを、層ごとの一次近似で測る（P12h 続き）。

kv_error_direction.py の結果、層ごとの大きさ・attention への効き方（Δscore、KL、ΣaΔv、出力誤差）では
focused の方がほぼ全層で baseline より良かった。それでも答えを壊すので、
**判定そのものに対する向き**を見る。

  g_L  = ∂(logit音 − logitエ) / ∂(層L の self_attn 出力の最終位置)   … FP32 の公式モデルで逆伝播
  Δy_L = 量子化 KV（最新スロットと q は FP32）で計算した attention 出力 − FP32 の attention 出力
  予測寄与_L = g_L · Δy_L                                           … 他の層の KV は正しいとした一次近似

K だけ・V だけ・両方の3通りを出す。層群ごとの合計を P12g の KV 差し替え結果
（focused の KV・最終ステップ FP32: 層0-6 +0.09 / 7-13 +0.25 / 14-20 +1.99 / 21-27 +1.01、全体 −3.43）
と照合し、一次近似が当てになるかを確かめる。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）。kv_error_direction.py（P12h）の続き。記録は results/P12h_20260915_kv_error_direction.md の 2 節
- 結果: 全層の合計は baseline −2.23 / focused −3.71 で、P12g の差し替えの実測 −1.46 / −3.43 とおおむね合う。差の大半は層15〜21。
  focused はどの層でも誤差の大きさ |Δy| は小さいのに、判定を悪くする向きの成分が増えていた（|cos| はすべて 0.14 未満）
- 入力: kv_error_direction.py と同じ（ONNX の 3 モデルの KV・公式 HF の Qwen3-0.6B。torch で逆伝播する）
- 出力: --out（既定 results/P12h_20260915_kv_error_direction/）の decision_projection.json と画面の表。
  フォルダは作らないので、先に kv_error_direction.py を流しておく
- 動かし方: python -m app.pipeline.kv_decision_projection [--out …]
- main の中の swap の表の数（P12g の回復量）は、results/P12g_20260915_kv_step_swap.md の「層0-6 +0.09 …」をそのまま書いたもの
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from kv_error_direction import onnx_caches, attention, load_json, SNAP, REP, ROOT, NL
import numpy as np
import torch

# 判定に使う 2 つの語の ID: 「音」（FP32 の 1 位）と「エ」（focused が選んでしまう語）
ON, E = 78685, 75639


def grads(ids):
    """FP32 の公式モデルで、各層 self_attn 出力（最終位置）に対する「音−エ」の勾配を取る。"""
    # （2026-10-18 追記）ids: 入力トークン列（点B の 80 個）。戻り値: (FP32 の「音−エ」の値, {層: 勾配 g_L [hidden 1024]})
    # しかけ: 各層の self_attn の出力の最終位置に、0 で始めた変数 delta を足す hook を付けて流し、margin を逆伝播する。
    #   delta の勾配が、そのまま「attention 出力の最終位置に対する margin の勾配」になる
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(SNAP), dtype=torch.float32, attn_implementation="eager",
                                             local_files_only=True).eval()
    deltas = {L: torch.zeros(1, 1, m.config.hidden_size, requires_grad=True) for L in range(NL)}

    def hook(mod, a, o, L):
        h = o[0]
        pad = torch.zeros_like(h[:, :-1, :])
        return (h + torch.cat([pad, deltas[L]], dim=1),) + tuple(o[1:])

    hs = [m.model.layers[L].self_attn.register_forward_hook(lambda mod, a, o, L=L: hook(mod, a, o, L))
          for L in range(NL)]
    lg = m(torch.tensor([ids])).logits[0, -1]
    margin = lg[ON] - lg[E]
    margin.backward()
    for h in hs:
        h.remove()
    return float(margin), {L: deltas[L].grad[0, 0].double().numpy() for L in range(NL)}


def main():
    """① 点B の入力 → ② ONNX の KV・HF の q/K/V・勾配を取る → ③ 層ごとに Δy と g·Δy（K だけ・V だけ・両方）→ ④ 層群の合計と P12g の照合。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12h_20260915_kv_error_direction")
    args = ap.parse_args()

    d = load_json(ROOT / "results/P12d_20260915_dialogue_activation_ranges/chat_original.json")
    chunks = [[r["token_id"] for r in d["rows"] if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    f0 = load_json(REP / "focused_free_generation.json")["focused_split"][0]["generated_ids"]
    ids = chunks[0] + f0 + chunks[1]
    body = ids[:-1]
    n = len(body)

    caches = onnx_caches(body)
    from kv_error_direction import hf_state
    q, Khf, Vhf, _, o_proj = hf_state(ids)
    margin, g = grads(ids)
    print(f"\n  FP32 の「音−エ」= {margin:+.3f}（P12g の +5.07 と一致するはず）", flush=True)

    # ③ 層ごとに: 量子化の KV（最新スロットだけ FP32）で attention を計算し直し、FP32 との差 Δy を勾配 g に射影する
    rows = []
    for L in range(NL):
        Kf, Vf = Khf[L], Vhf[L]
        y_ref = attention(q[L], Kf, Vf, o_proj[L])[0]
        row = {"layer": L, "grad_norm": float(np.linalg.norm(g[L]))}
        for m in ("baseline", "focused"):
            Kb, Vb = caches[m][L]
            Kq = np.concatenate([Kb, Kf[:, n:]], axis=1)
            Vq = np.concatenate([Vb, Vf[:, n:]], axis=1)
            dK = attention(q[L], Kq, Vf, o_proj[L])[0] - y_ref
            dV = attention(q[L], Kf, Vq, o_proj[L])[0] - y_ref
            dKV = attention(q[L], Kq, Vq, o_proj[L])[0] - y_ref
            row[m] = {"K": float(g[L] @ dK), "V": float(g[L] @ dV), "KV": float(g[L] @ dKV),
                      "dy_norm": float(np.linalg.norm(dKV)),
                      "cos_with_grad": float(g[L] @ dKV / (np.linalg.norm(g[L]) * np.linalg.norm(dKV) + 1e-30))}
        rows.append(row)

    # ④ 層群ごとに足して、P12g の差し替えの回復量（符号を反転して並べる）と比べる
    groups = {"層0-6": range(0, 7), "層7-13": range(7, 14), "層14-20": range(14, 21), "層21-27": range(21, 28),
              "全層": range(0, 28)}
    swap = {"層0-6": 0.09, "層7-13": 0.25, "層14-20": 1.99, "層21-27": 1.01, "全層": 3.43}  # P12g: 差し替えで戻った量
    print(f"\n  層ごとの一次近似の寄与（「音−エ」への変化。負 = 「エ」寄りに押す）")
    print(f"  {'層':>3} {'|g|':>7} {'baseline KV':>12} {'focused KV':>12} {'(K / V)':>18} {'|Δy| b→f':>16} {'cos(Δy,g) b→f':>16}")
    for r in rows:
        b, f = r["baseline"], r["focused"]
        mark = " ◆" if 14 <= r["layer"] <= 20 else ""
        print(f"  {r['layer']:3d} {r['grad_norm']:7.3f} {b['KV']:+12.3f} {f['KV']:+12.3f} "
              f"({f['K']:+7.3f} / {f['V']:+7.3f}) {b['dy_norm']:7.3f}→{f['dy_norm']:6.3f} "
              f"{b['cos_with_grad']:+7.3f}→{f['cos_with_grad']:+7.3f}{mark}")
    summary = {}
    print(f"\n  {'層群':10s} {'baseline 予測':>12} {'focused 予測':>12} {'差 f−b':>9} {'P12g 差し替え(f, FP32最終)':>26}")
    for k, rg in groups.items():
        b = sum(rows[L]["baseline"]["KV"] for L in rg)
        f = sum(rows[L]["focused"]["KV"] for L in rg)
        summary[k] = {"baseline_pred": b, "focused_pred": f, "focused_swap_recovery_P12g": swap[k]}
        print(f"  {k:10s} {b:+12.3f} {f:+12.3f} {f - b:+9.3f} {'−' + format(swap[k], '.2f') + '（回復量の符号反転）':>26}")
    (args.out / "decision_projection.json").write_text(json.dumps(
        {"fp32_margin": margin, "rows": rows, "groups": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out / 'decision_projection.json'}")


if __name__ == "__main__":
    main()
