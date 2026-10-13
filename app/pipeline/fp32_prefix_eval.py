# -*- coding: utf-8 -*-
"""FP32 で事前計算した prefix KV ＋ 後続用範囲の QDQ を Host で評価する（P12k）。

P12j: Host↔HTP の差は、層2 MLP（mul_41 / linear_20）の scale が位置0 の巨大値に合わせて粗くなり、
後続トークンがほぼ3値でしか表されないことから生じる。
P12k-1（massive_origin_check.py）: 巨大値は**トークンによらず位置0 だけ**で出る。チャットでは位置0 は常に
`<|im_start|>` で、system prompt に関係なく値は同一（8142.7）。→ 位置0 の KV は全会話で共通。

そこで「端末は位置0 を計算しない（FP32 で事前計算した KV を持たせる）」構成が成立するかを確かめる。

## 変種

  baseline        : 全位置 baseline QDQ
  focused_split   : 位置0 は focused_shared、以降 focused_later（P12e）
  D0_fp32pos0_base: 位置0 だけ FP32 事前計算、以降 baseline QDQ（事前計算だけの効果）
  D1_fp32pos0_later: 位置0 だけ FP32 事前計算、以降 focused_later（本命・system prompt によらず使える）
  D2_fp32sys_later : system ブロック（最初の <|im_end|>\\n まで）を FP32 事前計算、以降 focused_later（製品固定 prefix 版）

## 評価

  P12i と同じ評価点（日英の回答位置 52、点B、点A、BANANA/バナナ）と margin/KL 指標
  ＋ 元の日本語3ターン対話の自由生成（最大32トークン）
  ＋ 事前計算した FP32 KV が、past_k/past_v 入力の量子化範囲からはみ出さないか（層ごと）
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS
from block_revert_ab import eval_points, score, load_json, ART, REP, P12D

import numpy as np
import onnx
from onnx import numpy_helper

STOP = (151645, 151643)


def prefix_len(ids, mode):
    if mode == "pos0":
        return 1
    if mode == "sys":
        i = ids.index(151645)            # 最初の <|im_end|>
        return i + 2                     # <|im_end|> と改行まで
    return 0


class Pipeline:
    """位置 < P は FP32、以降は first/later の QDQ で食べさせる。"""

    def __init__(self, fp32, first, later, mode):
        self.fp32, self.first, self.later, self.mode = fp32, first, later, mode
        self.cos, self.sin = tables()

    def start(self, ids):
        self.P = prefix_len(ids, self.mode)
        self.r = Runner(self.first, self.cos, self.sin)
        self.fp = Runner(self.fp32, self.cos, self.sin) if self.P else None

    def feed(self, tid):
        pos = self.r.position
        if pos < self.P:
            got, _ = self.fp.feed(tid, self.fp32)
            self.r.cache = self.fp.cache
            self.r.position = self.fp.position
            return got
        s = self.first if (pos == 0 and self.P == 0) else self.later
        got, _ = self.r.feed(tid, s)
        return got


def run_points(pipe, pts):
    out = {}
    for p in pts:
        pipe.start(p["ids"])
        want = set(p["eval_positions"])
        assert min(want) >= pipe.P, f"{p['name']}: 評価位置が事前計算範囲に入っている"
        got_l = []
        for pos, tid in enumerate(p["ids"]):
            got = pipe.feed(tid)
            if pos in want:
                got_l.append(got["logits"].ravel().astype(np.float64))
        out[p["name"]] = np.stack(got_l)
    return out


def free_generation(pipe, tok, max_new=32):
    rows = load_json(P12D / "chat_original.json")["rows"]
    chunks = [[r["token_id"] for r in rows if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    pipe.start(chunks[0])
    turns, got = [], None
    for t, ids in enumerate(chunks):
        for tid in ids:
            got = pipe.feed(tid)
        out, stop = [], None
        for _ in range(max_new):
            nid = int(got["logits"].ravel().argmax())
            if nid in STOP:
                stop = nid
                break
            out.append(nid)
            got = pipe.feed(nid)
        turns.append({"turn": t, "text": tok.decode(out), "generated_ids": out, "stop_id": stop})
    return turns


def kv_input_clip(fp32, pts):
    """事前計算した FP32 KV（位置0 と system ブロック）が past_k/v 入力の Q 範囲に収まるか。"""
    m = onnx.load(str(OPS / "qwen3_dec28c256_final_qdq.onnx"), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    rng = {}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0].startswith("past_"):
            s = float(numpy_helper.to_array(init[n.input[1]]).ravel()[0])
            z = numpy_helper.to_array(init[n.input[2]])
            qmax = 65535 if z.dtype == np.uint16 else 255
            z = float(z.ravel()[0])
            rng[n.input[0]] = ((0 - z) * s, (qmax - z) * s, s)
    del m
    cos, sin = tables()
    res = {}
    for p in pts:
        if p["name"] not in ("chat_original", "chat_english", "sys_en_BANANA"):
            continue
        r = Runner(fp32, cos, sin)
        P = prefix_len(p["ids"], "sys")
        for tid in p["ids"][:P]:
            r.feed(tid, fp32)
        worst = {"outside": 0, "total": 0, "max_overflow_steps": 0.0}
        per = {}
        for name, a in r.cache.items():
            lo, hi, s = rng[name]
            v = a[:, :, -P:, :]
            out = (v < lo) | (v > hi)
            over = float(np.maximum(v - hi, lo - v)[out].max() / s) if out.any() else 0.0
            per[name] = {"outside": int(out.sum()), "pos0_outside": int(out[:, :, 0, :].sum()),
                         "range": [lo, hi], "value_min": float(v.min()), "value_max": float(v.max())}
            worst["outside"] += int(out.sum())
            worst["total"] += int(v.size)
            worst["max_overflow_steps"] = max(worst["max_overflow_steps"], over)
        res[p["name"]] = {"prefix_len": P, **worst, "per_tensor": per}
        print(f"  KV 入力の飽和確認 {p['name']:14s} prefix {P:2d}: はみ出し {worst['outside']}/{worst['total']} 要素"
              f"（最大 {worst['max_overflow_steps']:.1f} 段）", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12k_20260916_fp32_prefix")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "progress.log", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    tok = tokenizer()
    pts = eval_points(tok)
    fp32 = session(OPS / "qwen3_dec28c256_prepped.onnx")
    report = {"kv_input_clip": kv_input_clip(fp32, pts)}

    t0 = time.time()
    ref = run_points(Pipeline(fp32, fp32, fp32, "none"), pts)
    say(f"FP32 基準 {time.time() - t0:.0f}s")
    report["free_generation"] = {"fp32": free_generation(Pipeline(fp32, fp32, fp32, "none"), tok)}

    base = session(ART / "baseline.onnx")
    f_first, f_later = session(ART / "focused_shared.onnx"), session(ART / "focused_later.onnx")
    variants = [("baseline", base, base, "none"), ("focused_split", f_first, f_later, "none"),
                ("D0_fp32pos0_base", base, base, "pos0"), ("D1_fp32pos0_later", f_later, f_later, "pos0"),
                ("D2_fp32sys_later", f_later, f_later, "sys")]
    report["scores"] = {}
    for name, first, later, mode in variants:
        t0 = time.time()
        pipe = Pipeline(fp32, first, later, mode)
        sc = score(run_points(pipe, pts), ref)
        gen = free_generation(pipe, tok)
        report["scores"][name] = sc
        report["free_generation"][name] = gen
        a = sc["ALL"]
        say(f"{name:20s} 1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  最小margin {a['margin_min']:+6.2f}  "
            f"KL {a['kl_mean']:.3f} | 点B {sc['point_B_music']['margins'][0]:+.2f} 点A {sc['point_A_first_div']['margins'][0]:+.2f} "
            f"BANANA {sc['sys_en_BANANA']['margins'][0]:+.2f} バナナ {sc['sys_ja_banana']['margins'][0]:+.2f}  ({time.time() - t0:.0f}s)")
        for t in gen:
            say(f"      turn{t['turn']}: {t['text']!r}")
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"-> {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
