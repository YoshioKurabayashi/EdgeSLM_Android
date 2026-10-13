r"""decode ループの誤差が「いつ・どこで」増えるかを測る（teacher forcing）。

## なぜ teacher forcing なのか

生成は1トークンでも分岐するとその後が別の文になる。分岐した2つのキャッシュを
比べても、差が **①量子化誤差 ②キャッシュ誤差 ③そもそも入力トークン列が違う**
のどれなのか分けられない（2026-09-13 にこれで誤った結論を出した）。

**fp32 が選んだトークンを QDQ 側にも強制的に入れる。** こうすると入力列が揃うので、
残る差は量子化とキャッシュの誤差だけになる。

## 何を出すか

各ステップで、

  - 層ごとの `present_k_i` / `present_v_i` の相対L2 …… **層方向**
  - `logits` の相対L2 / KL / cos / fp32 の1位の順位 …… **時間方向**

**トークン一致率は主指標にしない。** logits が僅差のときは小さな誤差で1位が入れ替わり、
差が大きいときは大きな誤差でも変わらないので、指標として暴れる。
一致率は最後にユーザー体験を確認するために使う。

## 読み方

    層方向が特定の層に集中 → そこを Oracle で叩く
    全層で滑らかに増える   → 特定 op ではなく方式（A16W8 / キャッシュ量子化）の限界

## 使い方

    python -m app.pipeline.decode_error_profile \
        --fp32 work/qwen_dec24/qwen05b_decode_L24_C64.onnx \
        --qdq  work/ops/qwen_dec24/qwen_dec24_final_qdq.onnx \
        --layers 24 --cache-len 64 --steps 24 \
        --out results/P8_decode_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.export_qwen_decode import MASK_NEG, load, rope_cos_sin  # noqa: E402

PROMPTS = [
    "The capital city of Japan is Tokyo. The capital city of France is",
    "The largest planet in our solar system is",
    "Water boils at one hundred degrees Celsius. Ice melts at",
    "Python is a programming language that is widely used for",
]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """b を基準にした相対L2。分母は必ず fp32 側に揃える。"""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def logit_metrics(q: np.ndarray, r: np.ndarray) -> dict:
    q = q.astype(np.float64).ravel()
    r = r.astype(np.float64).ravel()

    def softmax(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    pr, pq = softmax(r), softmax(q)
    top = int(r.argmax())
    # fp32 の1位が QDQ 側で何位か（1 が最良）
    rank = int((q > q[top]).sum()) + 1
    return {
        "rel_l2": rel_l2(q, r),
        "kl": float(np.sum(pr * (np.log(pr + 1e-30) - np.log(pq + 1e-30)))),
        "cos": float(q @ r / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-30)),
        "rank_of_fp32_top1": rank,
        "top1_same": bool(int(q.argmax()) == top),
        # 1位と2位の差（小さいほど、わずかな誤差で入れ替わる）
        "fp32_margin": float(np.sort(r)[-1] - np.sort(r)[-2]),
    }


class Runner:
    def __init__(self, path: Path, layers: int, cache_len: int):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.s.get_inputs()]
        self.out_names = [o.name for o in self.s.get_outputs()]
        self.li = self.out_names.index("logits")
        self.L = cache_len
        self.NL = layers
        self.reset()

    def reset(self):
        shp = None
        for i in self.s.get_inputs():
            if i.name.startswith("past_"):
                shp = tuple(i.shape)
                break
        self.cache = {f"past_{k}_{i}": np.zeros(shp, np.float32)
                      for i in range(self.NL) for k in ("k", "v")}
        self.n_valid = 0

    def step(self, tid: int, cos: np.ndarray, sin: np.ndarray):
        self.n_valid = min(self.n_valid + 1, self.L)
        m = np.full((1, 1, 1, self.L), MASK_NEG, np.float32)
        m[0, 0, 0, self.L - self.n_valid:] = 0.0
        feed = {"input_ids": np.array([[tid]], np.int64),
                "cos": cos, "sin": sin, "attn_mask": m}
        feed.update(self.cache)
        out = self.s.run(None, {k: feed[k] for k in self.in_names})
        present = {}
        for i, nm in enumerate(self.out_names):
            if nm.startswith("present_"):
                present[nm] = out[i]
                self.cache["past_" + nm[len("present_"):]] = out[i]
        return out[self.li][0, -1], present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--cache-len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    hf = load(args.model_id)
    ca, sa = rope_cos_sin(hf, list(range(args.cache_len)))
    cs = [(ca[:, :, p:p + 1, :].copy(), sa[:, :, p:p + 1, :].copy())
          for p in range(args.cache_len)]

    NL, S = args.layers, args.steps
    # [prompt][step][layer]
    acc_k = np.zeros((args.prompts, S, NL))
    acc_v = np.zeros((args.prompts, S, NL))
    lg_rows: list[list[dict]] = []

    for pi, prompt in enumerate(PROMPTS[:args.prompts]):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ref = Runner(args.fp32, NL, args.cache_len)
        test = Runner(args.qdq, NL, args.cache_len)
        print(f"[{pi}] {prompt!r}  プロンプト {len(ids)} トークン")

        rows = []
        forced: list[int] = []
        si = 0
        for t in range(len(ids) + S):
            # ★入力トークンは常に fp32 側が選んだもの（teacher forcing）
            tid = ids[t] if t < len(ids) else forced[t - len(ids)]
            cos, sin = cs[min(t, args.cache_len - 1)]
            r_log, r_pres = ref.step(tid, cos, sin)
            q_log, q_pres = test.step(tid, cos, sin)
            if t >= len(ids) - 1:
                forced.append(int(r_log.argmax()))
                if si < S:
                    for i in range(NL):
                        acc_k[pi, si, i] = rel_l2(q_pres[f"present_k_{i}"],
                                                  r_pres[f"present_k_{i}"])
                        acc_v[pi, si, i] = rel_l2(q_pres[f"present_v_{i}"],
                                                  r_pres[f"present_v_{i}"])
                    rows.append(logit_metrics(q_log, r_log))
                    si += 1
                if si >= S:
                    break
        lg_rows.append(rows)
        print(f"      fp32 の続き: {tok.decode(forced[:S])!r}")

    mk = acc_k.mean(axis=0)      # [step][layer]
    mv = acc_v.mean(axis=0)

    # ---- 時間方向 ----
    print("\n=== 時間方向（全プロンプト平均） ===")
    print(f"{'step':>5} {'K cache':>9} {'V cache':>9} {'logits L2':>10} "
          f"{'KL':>8} {'cos':>7} {'1位の順位':>10} {'top1一致':>8}")
    print("-" * 72)
    shown = [s for s in (0, 1, 2, 3, 5, 7, 11, 15, 19, S - 1) if s < S]
    for s in shown:
        lm = [r[s] for r in lg_rows if len(r) > s]
        print(f"{s:>5} {mk[s].mean():8.2%} {mv[s].mean():8.2%} "
              f"{np.mean([x['rel_l2'] for x in lm]):9.2%} "
              f"{np.mean([x['kl'] for x in lm]):8.3f} "
              f"{np.mean([x['cos'] for x in lm]):7.4f} "
              f"{np.mean([x['rank_of_fp32_top1'] for x in lm]):10.1f} "
              f"{np.mean([x['top1_same'] for x in lm]):8.1%}")

    # ---- 層方向 ----
    print("\n=== 層方向: present_v の相対L2（step ごと） ===")
    hdr = "  ".join(f"L{i:02d}" for i in range(0, NL, 2))
    print(f"{'step':>5}  {hdr}")
    print("-" * (7 + len(hdr)))
    for s in shown:
        row = "  ".join(f"{mv[s, i] * 100:3.0f}" for i in range(0, NL, 2))
        print(f"{s:>5}  {row}")

    print("\n=== 層方向: present_k の相対L2（step ごと） ===")
    print(f"{'step':>5}  {hdr}")
    print("-" * (7 + len(hdr)))
    for s in shown:
        row = "  ".join(f"{mk[s, i] * 100:3.0f}" for i in range(0, NL, 2))
        print(f"{s:>5}  {row}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "fp32": str(args.fp32), "qdq": str(args.qdq),
            "layers": NL, "cache_len": args.cache_len, "steps": S,
            "prompts": PROMPTS[:args.prompts],
            "k_cache_rel_l2": acc_k.tolist(), "v_cache_rel_l2": acc_v.tolist(),
            "logits": lg_rows,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
