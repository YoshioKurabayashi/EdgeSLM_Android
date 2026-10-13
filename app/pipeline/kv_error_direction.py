# -*- coding: utf-8 -*-
"""KV 誤差の「向き」を測る（P12h）。点B（「音楽をかけて」直後・prefix 80）の KV を使う。

P12g で、focused の KV は全層で baseline より FP32 に近い（L2 が小さい）のに答えを壊し、
効いているのは層14-20 の KV だと分かった。そこで「誤差の大きさ」ではなく「向き」を測る。

各層・K と V 別に、79スロットの誤差 e_t = KV_QDQ,t − KV_FP32,t（8ヘッド×128次元を1024次元に並べる）について:

  ① bias ratio   = ||mean_t e_t|| / sqrt(mean_t ||e_t||^2)
                   独立な雑音なら約 1/sqrt(79) ≈ 0.11、全スロット同じ誤差なら 1
  ② スロット間の誤差 cosine の平均（対角を除く）
  ③ PCA 第1主成分の寄与率（中心化した誤差行列）
  ④ attention に対する実効（最終ステップの q は **FP32** のものを使い、KV の違いだけを見る）
       K: Δscore = q·Δk/√d の RMS（FP32 の score のばらつきとの比）と、attention 重みの KL(FP32‖量子化K)
       V: FP32 の attention 重みで重み付けした ||Σ a_t Δv_t|| / ||Σ a_t v_t||
       KV 両方: attention 出力（o_proj 後）の相対誤差

q は公式 HF の FP32 から計算する。その前に2つの等価チェックを通す:
  - HF の KV（位置0-78）と ONNX FP32 の KV が一致すること
  - 自前の attention 計算（FP32 の q/K/V）が HF の self_attn 出力と一致すること
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, session, ROOT, OPS

import numpy as np
import torch

ART = Path("D:/AI/artifacts/EdgeSLM_Android/P12e_20260915_temporal_qdq")
REP = ROOT / "results/P12e_20260915_temporal_qdq"
SNAP = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
NL, NH, NKV, HD = 28, 16, 8, 128


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def onnx_caches(body):
    cos, sin = tables()
    paths = {"fp32": (OPS / "qwen3_dec28c256_prepped.onnx",) * 2,
             "baseline": (ART / "baseline.onnx",) * 2,
             "focused": (ART / "focused_shared.onnx", ART / "focused_later.onnx")}
    out = {}
    for name, (p1, p2) in paths.items():
        s1 = session(p1)
        s2 = s1 if p2 == p1 else session(p2)
        r = Runner(s1, cos, sin)
        for pos, tid in enumerate(body):
            r.feed(tid, s1 if pos == 0 else s2)
        n = r.position
        # 左シフト式なので有効スロットは末尾 n 個（古い→新しい順）
        out[name] = {L: (r.cache[f"past_k_{L}"][0, :, -n:, :].astype(np.float64),
                         r.cache[f"past_v_{L}"][0, :, -n:, :].astype(np.float64)) for L in range(NL)}
        print(f"  ONNX の KV を積んだ: {name}", flush=True)
        del r, s1, s2
    return out


def hf_state(ids):
    """FP32 の公式モデルで全 prefix を1回流し、各層の q（最終位置）・K/V・attention 出力を取る。"""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    m = AutoModelForCausalLM.from_pretrained(str(SNAP), dtype=torch.float32, attn_implementation="eager",
                                             local_files_only=True).eval()
    attn_out = {}
    hooks = [m.model.layers[L].self_attn.register_forward_hook(
        lambda mod, a, o, L=L: attn_out.__setitem__(L, o[0][0, -1].detach().double().numpy()))
        for L in range(NL)]
    with torch.no_grad():
        o = m(torch.tensor([ids]), output_hidden_states=True, use_cache=True)
    for h in hooks:
        h.remove()
    pkv = o.past_key_values
    q, K, V = {}, {}, {}
    pos = torch.tensor([[len(ids) - 1]])
    with torch.no_grad():
        for L in range(NL):
            lay = m.model.layers[L]
            x = lay.input_layernorm(o.hidden_states[L][:, -1:, :])
            qq = lay.self_attn.q_norm(lay.self_attn.q_proj(x).view(1, 1, NH, HD)).transpose(1, 2)
            cos, sin = m.model.rotary_emb(x, pos)
            qq, _ = apply_rotary_pos_emb(qq, qq, cos, sin)
            q[L] = qq[0, :, 0, :].double().numpy()                      # [16,128]
            if hasattr(pkv, "layers"):
                k, v = pkv.layers[L].keys, pkv.layers[L].values
            else:
                k, v = pkv.key_cache[L], pkv.value_cache[L]
            K[L], V[L] = k[0].double().numpy(), v[0].double().numpy()   # [8,80,128]
    o_proj = {L: m.model.layers[L].self_attn.o_proj.weight.detach().double().numpy() for L in range(NL)}
    return q, K, V, attn_out, o_proj


def attention(q, K, V, o_proj):
    """q [16,128], K/V [8,T,128] → (o_proj 後の出力 [1024], 重み [16,T])。GQA は kv を2回ずつ使う。"""
    Kr, Vr = np.repeat(K, NH // NKV, axis=0), np.repeat(V, NH // NKV, axis=0)
    s = np.einsum("hd,htd->ht", q, Kr) / np.sqrt(HD)
    s = s - s.max(axis=1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(axis=1, keepdims=True)
    y = np.einsum("ht,htd->hd", a, Vr).reshape(-1)
    return o_proj @ y, a, s


def direction_stats(E):
    """E [8,T,128] の誤差 → スロットごとの1024次元ベクトルにして ①②③。"""
    X = E.transpose(1, 0, 2).reshape(E.shape[1], -1)                  # [T,1024]
    T = X.shape[0]
    bias = float(np.linalg.norm(X.mean(0)) / np.sqrt((np.linalg.norm(X, axis=1) ** 2).mean()))
    U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    C = U @ U.T
    cos_mean = float((C.sum() - np.trace(C)) / (T * (T - 1)))
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False) ** 2
    return {"bias_ratio": bias, "slot_cos_mean": cos_mean, "pc1_ratio": float(sv[0] / sv.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12h_20260915_kv_error_direction")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load_json(ROOT / "results/P12d_20260915_dialogue_activation_ranges/chat_original.json")
    chunks = [[r["token_id"] for r in d["rows"] if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    f0 = load_json(REP / "focused_free_generation.json")["focused_split"][0]["generated_ids"]
    ids = chunks[0] + f0 + chunks[1]
    body = ids[:-1]

    caches = onnx_caches(body)
    q, Khf, Vhf, attn_hf, o_proj = hf_state(ids)
    n = len(body)

    # 等価チェック1: HF の KV（0-78）と ONNX FP32 の KV
    kmax = max(np.linalg.norm(caches["fp32"][L][0] - Khf[L][:, :n]) / np.linalg.norm(Khf[L][:, :n]) for L in range(NL))
    vmax = max(np.linalg.norm(caches["fp32"][L][1] - Vhf[L][:, :n]) / np.linalg.norm(Vhf[L][:, :n]) for L in range(NL))
    # 等価チェック2: 自前 attention（HF の q/K/V 全80）と HF の self_attn 出力
    amax = max(np.linalg.norm(attention(q[L], Khf[L], Vhf[L], o_proj[L])[0] - attn_hf[L]) / np.linalg.norm(attn_hf[L])
               for L in range(NL))
    print(f"\n  等価チェック: HF↔ONNX FP32 の KV 最大相対L2  K {kmax:.2e} / V {vmax:.2e}  |  "
          f"自前 attention ↔ HF self_attn 最大相対L2 {amax:.2e}", flush=True)
    # ONNX FP32 は前処理（畳み込み・クランプ）済みの prepped.onnx で、公式 HF と最大 0.224% ずれることが
    # P12e で記録済み（初回実行では K 0.097% / V 0.195%）。ここで比べる量子化誤差は 20〜70% なので、
    # KV は 0.5% までを許す。attention の自前計算は同じ HF 同士なので厳しく見る。
    assert kmax < 5e-3 and vmax < 5e-3 and amax < 1e-4, "等価チェック失敗"

    rows = []
    for L in range(NL):
        Kf, Vf = Khf[L], Vhf[L]                                          # FP32 の全80（最新スロット含む）
        y_ref, a_ref, s_ref = attention(q[L], Kf, Vf, o_proj[L])
        s_std = float(s_ref.std(axis=1).mean())
        row = {"layer": L}
        for m in ("baseline", "focused"):
            Kq_body, Vq_body = caches[m][L]
            EK, EV = Kq_body - Kf[:, :n], Vq_body - Vf[:, :n]
            Kq = np.concatenate([Kq_body, Kf[:, n:]], axis=1)             # 最新スロットは FP32（KV の違いだけを見る）
            Vq = np.concatenate([Vq_body, Vf[:, n:]], axis=1)
            # ④ K: score のずれと attention 重みの変化（V は FP32）
            yK, aK, _ = attention(q[L], Kq, Vf, o_proj[L])
            ds = np.einsum("hd,htd->ht", q[L], np.repeat(EK, NH // NKV, axis=0)) / np.sqrt(HD)
            klK = float(np.mean(np.sum(a_ref * (np.log(a_ref + 1e-30) - np.log(aK + 1e-30)), axis=1)))
            # ④ V: FP32 の重みで重み付けした V 誤差
            wv = np.einsum("ht,htd->hd", a_ref[:, :n], np.repeat(EV, NH // NKV, axis=0))
            v_sum = np.einsum("ht,htd->hd", a_ref, np.repeat(Vf, NH // NKV, axis=0))
            yKV, _, _ = attention(q[L], Kq, Vq, o_proj[L])
            row[m] = {
                "K": {"rel_l2": float(np.linalg.norm(EK) / np.linalg.norm(Kf[:, :n])), **direction_stats(EK),
                      "dscore_rms": float(np.sqrt((ds ** 2).mean())), "dscore_over_score_std": float(np.sqrt((ds ** 2).mean()) / s_std),
                      "attn_kl": klK, "attn_out_rel_err_K_only": float(np.linalg.norm(yK - y_ref) / np.linalg.norm(y_ref))},
                "V": {"rel_l2": float(np.linalg.norm(EV) / np.linalg.norm(Vf[:, :n])), **direction_stats(EV),
                      "attn_weighted_rel": float(np.linalg.norm(wv) / np.linalg.norm(v_sum))},
                "attn_out_rel_err_KV": float(np.linalg.norm(yKV - y_ref) / np.linalg.norm(y_ref)),
            }
        rows.append(row)

    (args.out / "report.json").write_text(json.dumps({"prefix_len": len(ids), "equivalence": {
        "kv_hf_vs_onnx_fp32_max_rel_l2": {"K": kmax, "V": vmax}, "attention_self_vs_hf_max_rel_l2": amax},
        "random_bias_ratio_reference": 1 / np.sqrt(n), "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    def line(L, m, x):
        r = rows[L][m]
        return (f"  層{L:2d} {m:9s} {x} L2 {r[x]['rel_l2']*100:6.2f}%  bias {r[x]['bias_ratio']:.3f}  "
                f"cos {r[x]['slot_cos_mean']:+.3f}  PC1 {r[x]['pc1_ratio']:.3f}  ")
    print(f"\n  ※ 独立な雑音の bias ratio の目安 1/√{n} = {1/np.sqrt(n):.3f}")
    for x in ("K", "V"):
        print(f"\n  ===== {x} =====")
        for L in range(NL):
            for m in ("baseline", "focused"):
                r = rows[L][m]
                extra = (f"Δscore/σ {r['K']['dscore_over_score_std']:.3f}  attnKL {r['K']['attn_kl']:.4f}  出力誤差(Kのみ) {r['K']['attn_out_rel_err_K_only']*100:6.2f}%"
                         if x == "K" else f"ΣaΔv/Σav {r['V']['attn_weighted_rel']*100:6.2f}%  出力誤差(KV) {r['attn_out_rel_err_KV']*100:6.2f}%")
                mark = " ◆" if 14 <= L <= 20 else ""
                print(line(L, m, x) + extra + mark)
    print(f"\n-> {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
