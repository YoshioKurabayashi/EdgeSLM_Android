# -*- coding: utf-8 -*-
"""自由生成が**最初に分岐する瞬間**を、同一 prefix の teacher forcing で解剖する（P12f）。

## なぜこの形にするか

P12e の focused_split は、固定入力の一致率が上がったのに自由生成で
「音楽をかけて」にエアコンの説明を繰り返した。壊れた**後**の出力を集計しても、
一度違う語を選んだ後は KV も入力も別物になるので原因と結果が混ざる。
そこで、分岐する**直前まで全モデルに同じトークン列を食べさせ**（各モデルは自前の KV を積む）、
その1点で次を並べる。

  - top-10 logits / FP32 1位の順位 / KL(FP32‖量子化) / logits 相対L2
  - 範囲を変えた 168 本の量子化入力が、そのモデル自身の表現範囲から**はみ出した率**（clip 率）
  - prefix 全体での累積 clip（どの位置から始まったか）

## 判定の読み方

  - focused だけ clip があり、baseline は 0% → clip 仮説を**強く支持**
  - 168 本すべて clip 0% なのに分岐 → clip 仮説は**棄却**。格子/zero point の変更や相互作用を疑う
  - FP32 自身も同じ語を選ぶ（点B）→ 2ターン目の量子化ではなく、1ターン目の回答内容が原因

## モデル

  baseline      : P12e の観測出力付き既存 QDQ（範囲は既存のまま）
  focused_split : position 0 は focused_shared.onnx、以降は focused_later.onnx
  FP32          : work/ops/qwen3_dec28c256/qwen3_dec28c256_prepped.onnx（外部 .data はハードリンクで共有）

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）。P12e（temporal_qdq_experiment.py・focused_qdq_experiment.py）の直後
- 結果（results/P12f_20260915_divergence_probe.md）: 崩れの原因は clip ではない（168 本すべて、2 つの分岐点の prefix の全位置で
  clip 0。FP32 の値を focused の範囲に当てる陽性対照では 168 本とも検出できた）。点B で FP32 は「音」を 2 位と 4.62 差で選ぶが、
  focused_split は「エ」16.11・「音」15.73 に逆転。誤差の大きさではなく「ずれの向き」が効いている → P12h（kv_error_direction.py）へ
- 入力: P12e の成果物（ART）と記録（REP の focused_build_audit.json・free_generation.json・focused_free_generation.json）、
  P12d の対話（chat_original.json）、work/ops/qwen3_dec28c256/ の final_qdq.onnx・prepped.onnx
- 出力: --out（既定 results/P12f_20260915_divergence_probe/）の report.json。
  観測用の複製は ART/probe/ に作る（あれば作り直さない）
- 動かし方: python -m app.pipeline.divergence_probe [--out …]
"""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS  # 同じ実行方式を使う

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ART = Path("D:/AI/artifacts/EdgeSLM_Android/P12e_20260915_temporal_qdq")
REP = ROOT / "results/P12e_20260915_temporal_qdq"


def load_json(p):
    """UTF-8 の JSON ファイルを読む。"""
    return json.loads(Path(p).read_text(encoding="utf-8"))


def changed_groups():
    """P12e で範囲を変えた 168 本。量子化される**入力テンソル名**と、各相の表現範囲。"""
    # （2026-10-18 追記）戻り値: [{source（量子化される入力テンソル名）, scale_name, zero_name,
    #   shared_range（位置0 用モデルでの表現範囲 [lo, hi]）, later_range（位置1 以降用での表現範囲）}, …]
    a = load_json(REP / "focused_build_audit.json")
    later = {c["scale_name"]: c for c in a["later"]["changes"]}
    return [{"source": c["sources"][0], "scale_name": c["scale_name"], "zero_name": c["zero_name"],
             "shared_range": c["represented"], "later_range": later[c["scale_name"]]["represented"]}
            for c in a["shared"]["changes"]]


def baseline_ranges(groups):
    """既存 QDQ の同じ scale/zero point から、既存の表現範囲を復元する。"""
    # （2026-10-18 追記）表現範囲 = (0 − zp) × scale 〜 (qmax − zp) × scale。qmax は uint16 なら 65535、uint8 なら 255。
    # 戻り値: {source: [lo, hi]}
    m = onnx.load(str(OPS / "qwen3_dec28c256_final_qdq.onnx"), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    out = {}
    for g in groups:
        s = float(numpy_helper.to_array(init[g["scale_name"]]).ravel()[0])
        z = numpy_helper.to_array(init[g["zero_name"]])
        qmax = 65535 if z.dtype == np.uint16 else 255
        z = float(z.ravel()[0])
        out[g["source"]] = [(0 - z) * s, (qmax - z) * s]
    return out


def probe_model(src, dst, names, data_link=None):
    """観測したいテンソルを出力に足した複製を作る（元モデルは触らない）。"""
    # （2026-10-18 追記）src → dst に、names のうちグラフの中で作られているものを出力として足して保存する。
    #   dst が既にあれば何もしない（names を変えても作り直さないので、変えたときは手で消す。2026-10-18 注）。
    #   data_link を渡すと、重みの .data を dst と同じフォルダにハードリンクする（数 GB をコピーしないため）
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if data_link:  # 外部データはコピーせずハードリンクで共有（同一ボリューム）
        link = dst.parent / data_link.name
        if not link.exists():
            os.link(data_link, link)
    m = onnx.load(str(src), load_external_data=False)
    have = {o.name for o in m.graph.output}
    produced = {o for n in m.graph.node for o in n.output}
    for n in names:
        if n in produced and n not in have:
            m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
    onnx.save(m, str(dst))
    return dst


def softmax(x):
    """1 次元の softmax（最大値を引いてからあふれないように exp を取る）。"""
    e = np.exp(x - x.max())
    return e / e.sum()


def run_prefix(first, later, ids, groups, ranges_for):
    """ids を順に食べさせ、最後の位置の出力と、全位置の clip 記録を返す。"""
    # （2026-10-18 追記）first は位置0、later は位置1 以降のセッション。ranges_for(g, pos) はその位置での表現範囲を返す関数。
    # 戻り値: (最後の位置の出力の dict, 位置ごとの {source: clip の記録}（clip が無い本は入れない）)
    cos, sin = tables()
    r = Runner(first, cos, sin)
    clip_by_pos = []
    got = None
    for pos, tid in enumerate(ids):
        got, _ = r.feed(tid, first if pos == 0 else later)
        rec = {}
        for g in groups:
            v = got.get(g["source"])
            if v is None:
                continue
            lo, hi = ranges_for(g, pos)
            out = (v < lo) | (v > hi)
            n = int(out.sum())
            if n:
                over = float(np.maximum(v - hi, lo - v)[out].max())
                rec[g["source"]] = {"clipped": n, "size": int(v.size), "max_overflow": over,
                                    "range": [lo, hi], "value_min": float(v.min()), "value_max": float(v.max())}
        clip_by_pos.append(rec)
    return got, clip_by_pos


def summarize_logits(name, lg, ref, tok):
    """1 つのモデルの最後の位置の logits を、FP32（ref）と比べてまとめる。

    戻り値: top10（語・logit・確率）・fp32_top1_rank・top1_top2_margin（そのモデル自身の 1 位と 2 位の差）・
            kl_fp32_to_q・logits_rel_l2
    """
    p, q = softmax(ref.astype(np.float64)), softmax(lg.astype(np.float64))
    t1 = int(ref.argmax())
    top = np.argsort(-lg)[:10]
    return {"model": name,
            "top10": [{"id": int(i), "text": tok.decode([int(i)]), "logit": float(lg[i]), "p": float(q[i])}
                      for i in top],
            "fp32_top1_rank": int((lg > lg[t1]).sum()) + 1,
            "top1_top2_margin": float(np.sort(lg)[-1] - np.sort(lg)[-2]),
            "kl_fp32_to_q": float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum()),
            "logits_rel_l2": float(np.linalg.norm(lg - ref) / np.linalg.norm(ref))}


def clip_summary(clip):
    """最終位置の clip と、prefix 全体（先頭を除く）で一度でも clip したテンソルの累積。"""
    cum, first_pos = {}, None
    for pos, rec in enumerate(clip):
        if pos == 0:
            continue  # 先頭は別の範囲。問題にしているのは後続
        for src, v in rec.items():
            c = cum.setdefault(src, {"positions": 0, "clipped": 0, "max_overflow": 0.0, "first_pos": pos})
            c["positions"] += 1
            c["clipped"] += v["clipped"]
            c["max_overflow"] = max(c["max_overflow"], v["max_overflow"])
            first_pos = pos if first_pos is None else first_pos
    return {"last_position_clipped_tensors": len(clip[-1]), "last_position": clip[-1],
            "prefix_clipped_tensors": len(cum), "prefix_first_clip_position": first_pos,
            "prefix_cumulative": dict(sorted(cum.items(), key=lambda kv: -kv[1]["clipped"]))}


def main():
    """① 2 つの分岐点の入力を作る → ② 観測用の複製を作る → ③ 点ごとに FP32・baseline・focused_split を流して比べる。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12f_20260915_divergence_probe")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()

    # 対話の入力（P12d/P12e と同じ ID 列）と、各モデルの自由生成
    d = load_json(ROOT / "results/P12d_20260915_dialogue_activation_ranges/chat_original.json")
    chunks = [[r["token_id"] for r in d["rows"] if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    gen = {**load_json(REP / "free_generation.json"), **load_json(REP / "focused_free_generation.json")}
    b0, f0 = gen["baseline"][0]["generated_ids"], gen["focused_split"][0]["generated_ids"]
    k = next(i for i, (x, y) in enumerate(zip(b0, f0)) if x != y)
    points = {
        # A: 最初の分岐。1ターン目の回答 k トークン目を選ぶ直前
        "A_first_divergence": chunks[0] + b0[:k],
        # B: 壊れる瞬間。focused 自身の1ターン目回答を履歴にして「音楽をかけて」を入れた直後
        "B_music_turn": chunks[0] + f0 + chunks[1],
    }

    # ② 範囲を変えた 168 本と logits を出力させた複製を、4 つのモデル（baseline・focused 2 つ・FP32）について作る
    groups = changed_groups()
    base_rng = baseline_ranges(groups)
    obs = ["logits"] + [g["source"] for g in groups]

    P = ART / "probe"
    m_base = probe_model(ART / "baseline.onnx", P / "baseline_probe.onnx", obs)
    m_fs = probe_model(ART / "focused_shared.onnx", P / "focused_shared_probe.onnx", obs)
    m_fl = probe_model(ART / "focused_later.onnx", P / "focused_later_probe.onnx", obs)
    m_fp = probe_model(OPS / "qwen3_dec28c256_prepped.onnx", P / "fp32_probe.onnx", obs,
                       data_link=OPS / "qwen3_dec28c256_prepped.onnx.data")

    def split_range(g, pos):
        return g["shared_range"] if pos == 0 else g["later_range"]

    # ③ 点ごとに 3 通りを流す。FP32 は「focused の範囲に当てたら clip するか」を見る陽性対照も兼ねる
    report = {"divergence_index_turn1": k, "changed_groups": len(groups), "points": {}}
    for pname, ids in points.items():
        print(f"\n=== {pname}  prefix {len(ids)} トークン ===", flush=True)
        s = session(m_fp)
        fp_last, fp_clip = run_prefix(s, s, ids, groups, split_range)  # FP32 の値を focused の範囲に当てたら
        del s
        s = session(m_base)
        b_last, b_clip = run_prefix(s, s, ids, groups, lambda g, pos: base_rng[g["source"]])
        del s
        s1, s2 = session(m_fs), session(m_fl)
        f_last, f_clip = run_prefix(s1, s2, ids, groups, split_range)
        del s1, s2

        ref = fp_last["logits"].ravel()
        res = {"prefix_len": len(ids),
               "logits": [summarize_logits("fp32", ref, ref, tok),
                          summarize_logits("baseline", b_last["logits"].ravel(), ref, tok),
                          summarize_logits("focused_split", f_last["logits"].ravel(), ref, tok)],
               "clip": {"fp32_on_focused_ranges": clip_summary(fp_clip),
                        "baseline": clip_summary(b_clip),
                        "focused_split": clip_summary(f_clip)}}
        # 観測テンソルの誤差（FP32 にも同名がある分だけ）
        err = {}
        for g in groups:
            src = g["source"]
            if src in fp_last and src in b_last and src in f_last:
                r0 = fp_last[src].astype(np.float64).ravel()
                den = float(np.linalg.norm(r0)) or 1.0
                err[src] = {"baseline": float(np.linalg.norm(b_last[src].ravel() - r0) / den),
                            "focused_split": float(np.linalg.norm(f_last[src].ravel() - r0) / den)}
        res["activation_rel_l2_last_position"] = err
        report["points"][pname] = res

        for L in res["logits"]:
            t = " ".join(f"{x['text']!r}:{x['logit']:.2f}" for x in L["top10"][:5])
            print(f"  {L['model']:14s} FP32 1位の順位 {L['fp32_top1_rank']:5d}  KL {L['kl_fp32_to_q']:.4f}  "
                  f"relL2 {L['logits_rel_l2'] * 100:6.2f}%  | {t}")
        for m, c in res["clip"].items():
            print(f"  clip[{m:22s}] 最終位置 {c['last_position_clipped_tensors']:3d} 本 / "
                  f"prefix で一度でも {c['prefix_clipped_tensors']:3d} 本 / 最初の clip 位置 {c['prefix_first_clip_position']}")
        for src, v in list(res["clip"]["focused_split"]["prefix_cumulative"].items())[:8]:
            print(f"      {src:24s} {v['positions']:3d} 位置 / 計 {v['clipped']:5d} 要素 / "
                  f"最大はみ出し {v['max_overflow']:.3g} / 初出 {v['first_pos']}")
        print(f"  観測テンソルの FP32 比誤差: FP32 と同名 {len(err)} 本", flush=True)
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
