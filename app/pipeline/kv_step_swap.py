# -*- coding: utf-8 -*-
"""点B（「音楽をかけて」直後）の逆転が、**積み上がった KV** 由来か **最後の1ステップ** 由来かを分ける（P12g）。

prefix の最後の1トークンの手前まで各モデルで KV を積み、
最後の1ステップだけを別のモデルで計算する。KV の出どころ × 最終ステップのモデル を 3×3 で総当たりする。

  KV \\ 最終step | FP32 | baseline | focused
  FP32          | 基準 | 最終stepだけの効果 | 最終stepだけの効果
  baseline      | KVだけの効果 | 基準(自前) | 入れ替え
  focused       | KVだけの効果 | 入れ替え | 壊れるケース(自前)

focused の最終ステップは位置0でないので focused_later.onnx を使う（P12e の実行方式と同じ）。
判定は L2 ではなく **FP32 1位（「音」）の順位 / 「音」−「エ」の差 / KL** を主にする（P12f の学び）。
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS

import numpy as np

ART = Path("D:/AI/artifacts/EdgeSLM_Android/P12e_20260915_temporal_qdq")
REP = ROOT / "results/P12e_20260915_temporal_qdq"


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12g_20260915_kv_step_swap")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    cos, sin = tables()

    # 点B の prefix（P12f と同じ組み立て）
    d = load_json(ROOT / "results/P12d_20260915_dialogue_activation_ranges/chat_original.json")
    chunks = [[r["token_id"] for r in d["rows"] if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    f0 = load_json(REP / "focused_free_generation.json")["focused_split"][0]["generated_ids"]
    ids = chunks[0] + f0 + chunks[1]
    body, last = ids[:-1], ids[-1]

    paths = {"fp32": (OPS / "qwen3_dec28c256_prepped.onnx",) * 2,
             "baseline": (ART / "baseline.onnx",) * 2,
             "focused": (ART / "focused_shared.onnx", ART / "focused_later.onnx")}  # (位置0用, 以降用)

    # 1) 各モデルで、最後の1トークンの手前まで KV を積む
    caches = {}
    for name, (p_first, p_later) in paths.items():
        s_first = session(p_first)
        s_later = s_first if p_later == p_first else session(p_later)
        r = Runner(s_first, cos, sin)
        for pos, tid in enumerate(body):
            r.feed(tid, s_first if pos == 0 else s_later)
        caches[name] = ({k: v.copy() for k, v in r.cache.items()}, r.position)
        print(f"  KV を積んだ: {name}  位置 {r.position}", flush=True)
        del r, s_first, s_later

    # 2) 最後の1ステップだけを、各モデルで計算する
    ONG, E = tok.encode("音")[0], tok.encode("エ")[0]
    results, ref = {}, None
    for step_name, (_, p_later) in paths.items():
        s = session(p_later)  # 最終ステップは位置0ではない
        for kv_name, (cache, position) in caches.items():
            r = Runner(s, cos, sin)
            r.cache = {k: v.copy() for k, v in cache.items()}
            r.position = position
            got, _ = r.feed(last, s)
            results[(kv_name, step_name)] = got["logits"].ravel().astype(np.float64)
        del s

    ref = results[("fp32", "fp32")]
    t1 = int(ref.argmax())
    p_ref = softmax(ref)
    rows = []
    print(f"\n  FP32 1位 = {tok.decode([t1])!r}（id {t1}）。表の「音−エ」は logit の差（正なら「音」が上）\n")
    print(f"  {'KV の出どころ':14s} {'最終step':10s} {'1位':>6} {'FP32 1位の順位':>14} {'音−エ':>8} {'KL':>8}  上位3")
    for kv_name in paths:
        for step_name in paths:
            lg = results[(kv_name, step_name)]
            q = softmax(lg)
            row = {"kv": kv_name, "final_step": step_name,
                   "top1": tok.decode([int(lg.argmax())]), "top1_id": int(lg.argmax()),
                   "fp32_top1_rank": int((lg > lg[t1]).sum()) + 1,
                   "logit_on_minus_e": float(lg[ONG] - lg[E]),
                   "logit_on": float(lg[ONG]), "logit_e": float(lg[E]),
                   "kl_fp32_to_q": float((p_ref * (np.log(p_ref + 1e-30) - np.log(q + 1e-30))).sum()),
                   "top3": [(tok.decode([int(i)]), float(lg[i])) for i in np.argsort(-lg)[:3]]}
            rows.append(row)
            t3 = " ".join(f"{a!r}:{b:.2f}" for a, b in row["top3"])
            print(f"  {kv_name:14s} {step_name:10s} {row['top1']!r:>6} {row['fp32_top1_rank']:14d} "
                  f"{row['logit_on_minus_e']:8.2f} {row['kl_fp32_to_q']:8.4f}  {t3}")
        print()
    (args.out / "report.json").write_text(json.dumps(
        {"prefix_len": len(ids), "last_token": last, "on_id": ONG, "e_id": E, "rows": rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
