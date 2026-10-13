# -*- coding: utf-8 -*-
"""focused の範囲変更を「層ブロック × テンソル種別」ごとに baseline へ戻し、判定マージンで評価する（P12i）。

P12h で、focused の害は誤差の大きさではなく「判定を悪くする向きの成分」だと分かった。
168本を一括で変えたので、有益な変更と有害な変更が混ざっている。そこで1ブロックだけ戻して比べる。

## 変種（結果を見る前に固定）

  ブロック = 層群 {0-13, 14-20, 21-27} × 種別 {残差 add, RMSNorm 補助 node_pow_*, その他}
  各変種は focused から**そのブロックだけ** baseline の scale/zero point に戻す（先頭用・後続用の両モデルで）。
  対照: focused（何も戻さない）/ 全168本を戻す（= baseline と一致するはず）/ baseline

  層の割り当て: グラフの位置で決める。層L の入力残差 add（層L の input_layernorm に入る Add）より
  後ろで作られたテンソルを層L とする（その add 自体は層L−1 の出力）。

## 評価点（結果を見る前に固定）

  1. 元の日本語3ターン（P12d の FP32 軌跡）の回答トークン位置 26
  2. 英語3ターンの回答トークン位置 26
  3. 点B: focused の1ターン目回答を履歴にした「音楽をかけて」直後（既知の崩れ）
  4. 点A: 1ターン目で baseline と focused が分岐する直前
  5. System Prompt: 英「BANANA」/ 日「バナナ」の最初の回答トークン

## 指標（L2 は使わない）

  margin = logit[FP32 1位] − そのモデルでの最強の対抗 logit（負なら 1位を失った）
  - FP32 1位の保持率 / margin 保持率（margin ÷ FP32 の margin）の平均 / **最小 margin（worst case）** / KL 平均
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, ROOT, OPS

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ART = Path("D:/AI/artifacts/EdgeSLM_Android/P12e_20260915_temporal_qdq")
REP = ROOT / "results/P12e_20260915_temporal_qdq"
P12D = ROOT / "results/P12d_20260915_dialogue_activation_ranges"


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sess_from(obj):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(obj if isinstance(obj, (bytes, str)) else str(obj), so,
                                providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------- ブロック分け
def layer_of_tensors(model, names):
    g = model.graph
    idx = {o: i for i, n in enumerate(g.node) for o in n.output}
    prod = {o: n for n in g.node for o in n.output}
    inits = {t.name for t in g.initializer}
    bounds = []
    for L in range(1, 28):
        w = f"layers.{L}.input_layernorm.weight_DequantizeLinear_Output"
        mul = [n for n in g.node if w in n.input][0]
        cur = [i for i in mul.input if i != w][0]
        for _ in range(20):
            n = prod[cur]
            src = [i for i in n.input if i not in inits]
            if n.op_type == "Add" and len(src) == 2:
                bounds.append(idx[n.output[0]])
                break
            cur = src[0]
        else:
            raise RuntimeError(f"層{L} の入力残差 add が見つからない")
    assert bounds == sorted(bounds), "層境界がグラフ順に並んでいない"
    return {t: sum(1 for b in bounds if idx[t] > b) for t in names}


def kind_of(src):
    if re.fullmatch(r"add_\d+", src):
        return "residual"
    if re.fullmatch(r"node_pow_\d+_(abs|max|maxr)", src):
        return "rmsnorm_aux"
    return "other"


# ---------------------------------------------------------------- 評価点
def eval_points(tok):
    pts = []
    for case in ("chat_original", "chat_english"):
        rows = load_json(P12D / f"{case}.json")["rows"]
        ids = [r["token_id"] for r in rows]
        answer_pos = [p for p in range(len(rows) - 1) if rows[p + 1]["phase"] == "generated"]
        pts.append({"name": case, "ids": ids, "eval_positions": answer_pos})
    d = load_json(P12D / "chat_original.json")["rows"]
    chunks = [[r["token_id"] for r in d if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    gen = {**load_json(REP / "free_generation.json"), **load_json(REP / "focused_free_generation.json")}
    b0, f0 = gen["baseline"][0]["generated_ids"], gen["focused_split"][0]["generated_ids"]
    k = next(i for i, (x, y) in enumerate(zip(b0, f0)) if x != y)
    for name, ids in (("point_B_music", chunks[0] + f0 + chunks[1]), ("point_A_first_div", chunks[0] + b0[:k])):
        pts.append({"name": name, "ids": ids, "eval_positions": [len(ids) - 1]})
    for name, sysmsg, user in (("sys_en_BANANA", 'Always answer exactly "BANANA". Nothing else.', "What is 2+2?"),
                               ("sys_ja_banana", "どんな質問にも必ず「バナナ」とだけ答えてください。他には何も言わないこと。", "2たす2は？")):
        text = tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text)["input_ids"]
        pts.append({"name": name, "ids": ids, "eval_positions": [len(ids) - 1]})
    return pts


def run_points(first, later, pts):
    """各評価点の logits（評価位置のみ）を返す。点ごとに KV を空から積む。"""
    cos, sin = tables()
    out = {}
    for p in pts:
        r = Runner(first, cos, sin)
        want = set(p["eval_positions"])
        got_l = []
        for pos, tid in enumerate(p["ids"]):
            got, _ = r.feed(tid, first if pos == 0 else later)
            if pos in want:
                got_l.append(got["logits"].ravel().astype(np.float64))
        out[p["name"]] = np.stack(got_l)
    return out


def score(logits, ref):
    res, allm = {}, []
    for name, L in logits.items():
        R = ref[name]
        rows = []
        for lg, rf in zip(L, R):
            t1 = int(rf.argmax())
            m_ref = rf[t1] - np.max(np.delete(rf, t1))
            m = lg[t1] - np.max(np.delete(lg, t1))
            p = np.exp(rf - rf.max()); p /= p.sum()
            q = np.exp(lg - lg.max()); q /= q.sum()
            rows.append({"keep_top1": bool(m > 0), "margin": float(m), "margin_ref": float(m_ref),
                         "retention": float(m / m_ref), "kl": float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())})
        allm += rows
        res[name] = {"n": len(rows), "top1_keep": float(np.mean([r["keep_top1"] for r in rows])),
                     "retention_mean": float(np.mean([r["retention"] for r in rows])),
                     "margin_min": float(min(r["margin"] for r in rows)),
                     "kl_mean": float(np.mean([r["kl"] for r in rows])),
                     "margins": [r["margin"] for r in rows]}
    res["ALL"] = {"n": len(allm), "top1_keep": float(np.mean([r["keep_top1"] for r in allm])),
                  "retention_mean": float(np.mean([r["retention"] for r in allm])),
                  "margin_min": float(min(r["margin"] for r in allm)),
                  "kl_mean": float(np.mean([r["kl"] for r in allm]))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12i_20260915_block_revert_ab")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "progress.log", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    tok = tokenizer()
    pts = eval_points(tok)
    say("評価点: " + ", ".join(f"{p['name']}({len(p['eval_positions'])})" for p in pts))

    audit = load_json(REP / "focused_build_audit.json")
    base_m = onnx.load(str(OPS / "qwen3_dec28c256_final_qdq.onnx"), load_external_data=False)
    base_init = {t.name: numpy_helper.to_array(t) for t in base_m.graph.initializer
                 if t.name.endswith(("_scale", "_zero_point"))}
    changes = audit["shared"]["changes"]
    srcs = [c["sources"][0] for c in changes]
    layer = layer_of_tensors(base_m, srcs)
    del base_m
    blocks = {}
    for c in changes:
        s = c["sources"][0]
        L = layer[s]
        grp = "L0-13" if L <= 13 else ("L14-20" if L <= 20 else "L21-27")
        blocks.setdefault(f"{grp}/{kind_of(s)}", []).append(c)
    say("ブロック: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(blocks.items())))
    (args.out / "blocks.json").write_text(json.dumps(
        {k: [{"source": c["sources"][0], "layer": layer[c["sources"][0]], "scale_name": c["scale_name"]} for c in v]
         for k, v in blocks.items()}, ensure_ascii=False, indent=1), encoding="utf-8")

    # 基準（FP32）と baseline
    t0 = time.time()
    s = sess_from(OPS / "qwen3_dec28c256_prepped.onnx")
    ref = run_points(s, s, pts)
    del s
    say(f"FP32 基準 {time.time() - t0:.0f}s")
    results = {}
    s = sess_from(ART / "baseline.onnx")
    results["baseline"] = score(run_points(s, s, pts), ref)
    del s

    # focused のモデルを読み、ブロックを戻して評価する
    fm = {"shared": onnx.load(str(ART / "focused_shared.onnx")), "later": onnx.load(str(ART / "focused_later.onnx"))}
    finit = {ph: {t.name: t for t in m.graph.initializer} for ph, m in fm.items()}

    def variant(revert):
        saved = {}
        for ph in fm:
            for c in revert:
                for nm in (c["scale_name"], c["zero_name"]):
                    t = finit[ph][nm]
                    saved[(ph, nm)] = onnx.TensorProto(); saved[(ph, nm)].CopyFrom(t)
                    b = base_init[nm]
                    t.CopyFrom(numpy_helper.from_array(b.astype(numpy_helper.to_array(t).dtype).reshape(b.shape), nm))
        sess = {ph: sess_from(fm[ph].SerializeToString()) for ph in fm}
        for (ph, nm), t in saved.items():  # 元に戻す
            finit[ph][nm].CopyFrom(t)
        return sess["shared"], sess["later"]

    plan = [("focused", [])] + [(f"revert {k}", v) for k, v in sorted(blocks.items())] + [("revert ALL168", changes)]
    for name, rv in plan:
        t0 = time.time()
        s1, s2 = variant(rv)
        results[name] = score(run_points(s1, s2, pts), ref)
        del s1, s2
        a = results[name]["ALL"]
        say(f"{name:32s} 1位保持 {a['top1_keep']*100:5.1f}%  margin保持 {a['retention_mean']*100:6.1f}%  "
            f"最小margin {a['margin_min']:+6.2f}  KL {a['kl_mean']:.3f}  |  点B {results[name]['point_B_music']['margins'][0]:+.2f}  "
            f"BANANA {results[name]['sys_en_BANANA']['margins'][0]:+.2f} バナナ {results[name]['sys_ja_banana']['margins'][0]:+.2f}  "
            f"({time.time() - t0:.0f}s)")
        (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    b, ra = results["baseline"], results["revert ALL168"]
    same = all(np.allclose(b[k]["margins"], ra[k]["margins"], atol=1e-6) for k in b if k != "ALL")
    say(f"\n対照: 全168本を戻したものと baseline の margin 一致 = {same}")
    say(f"-> {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
