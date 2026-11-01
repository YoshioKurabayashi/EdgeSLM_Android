# -*- coding: utf-8 -*-
"""点B の逆転を作っている KV が**どの深さの層**のものかを、層ごとの KV 差し替えで特定する（P12g 続き）。

P12g の 3×3 で、逆転の主因は focused が積み上げた KV（差 5.07→1.64）だと分かった。
そこで focused の KV のうち**一部の層だけ**を FP32 / baseline の KV に置き換え、
最後の1ステップを FP32 と focused で計算して「音」−「エ」の差がどれだけ戻るかを見る。
**モデルを作り直さずに**「どの深さの範囲変更が効いているか」を切り分けられる。

  - 最終ステップ FP32 : KV の違いだけの効果（純粋な帰属）
  - 最終ステップ focused : 実際に壊れる条件での効果

あわせて、各層の KV（有効な79スロット）の FP32 比誤差を baseline / focused で並べる。

## 学習用の補足（2026-10-18 追記）

- 段: Qwen3-0.6B の P12 系（StudyMemo 11.5 節）。kv_step_swap.py（P12g の 3×3）の続き。記録は results/P12g_20260915_kv_step_swap.md
- 結果（同じ md の結論 2）: 効いているのは層14〜20 の KV。focused の KV のうち層14〜20 だけを FP32 に戻すと +1.64 → +3.63。
  層0〜6 は +0.09、層7〜13 は +0.25、層21〜27 は +1.01 で、4 群の合計 3.34 は全体の 3.43 とほぼ一致（ほぼ足し算どおり）
- 入力: work/ops/qwen3_dec28c256/ の prepped.onnx（FP32）、P12e の成果物（ART）と記録（REP）、P12d の対話
- 出力: --out（既定 results/P12g_20260915_kv_step_swap/）の layer_attrib.json と画面の表
- 動かし方: python -m app.pipeline.kv_layer_attrib [--out …]
"""
from pathlib import Path
import argparse
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS

import numpy as np

# P12e の成果物・記録の置き場と、Qwen3-0.6B の層の数
ART = Path("D:/AI/artifacts/EdgeSLM_Android/P12e_20260915_temporal_qdq")
REP = ROOT / "results/P12e_20260915_temporal_qdq"
NL = 28


def load_json(p):
    """UTF-8 の JSON ファイルを読む。"""
    return json.loads(Path(p).read_text(encoding="utf-8"))


def softmax(x):
    """1 次元の softmax（最大値を引いてから exp）。"""
    e = np.exp(x - x.max())
    return e / e.sum()


def layer_of(name):
    """past_k_12 のような名前の末尾の数（層の番号）を返す。"""
    return int(re.search(r"_(\d+)$", name).group(1))


def main():
    """① 3 モデルで点B の KV を積む → ② 層ごとの KV の誤差 → ③ 層群ごとに KV を差し替えて最後の 1 歩の「音−エ」を測る → ④ 表示と保存。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/P12g_20260915_kv_step_swap")
    args = ap.parse_args()
    tok = tokenizer()
    cos, sin = tables()

    d = load_json(ROOT / "results/P12d_20260915_dialogue_activation_ranges/chat_original.json")
    chunks = [[r["token_id"] for r in d["rows"] if r["turn"] == i and r["phase"] == "prompt"] for i in range(3)]
    f0 = load_json(REP / "focused_free_generation.json")["focused_split"][0]["generated_ids"]
    ids = chunks[0] + f0 + chunks[1]
    body, last = ids[:-1], ids[-1]
    ON, E = 78685, 75639  # 「音」「エ」（P12g で確認済み）

    # ① 各モデルに body（位置 0〜78）を流して KV を積む。focused は位置0 を shared、位置1 以降を later で流す
    paths = {"fp32": (OPS / "qwen3_dec28c256_prepped.onnx",) * 2,
             "baseline": (ART / "baseline.onnx",) * 2,
             "focused": (ART / "focused_shared.onnx", ART / "focused_later.onnx")}
    caches = {}
    for name, (p_first, p_later) in paths.items():
        s1 = session(p_first)
        s2 = s1 if p_later == p_first else session(p_later)
        r = Runner(s1, cos, sin)
        for pos, tid in enumerate(body):
            r.feed(tid, s1 if pos == 0 else s2)
        caches[name] = {k: v.copy() for k, v in r.cache.items()}
        position = r.position
        print(f"  KV を積んだ: {name}", flush=True)
        del r, s1, s2

    # 層ごとの KV 誤差（有効スロット = 末尾 position 個）
    n = position
    kv_err = []
    for L in range(NL):
        row = {"layer": L}
        for x in ("k", "v"):
            ref = caches["fp32"][f"past_{x}_{L}"][:, :, -n:, :].astype(np.float64)
            den = np.linalg.norm(ref)
            for m in ("baseline", "focused"):
                row[f"{x}_{m}"] = float(np.linalg.norm(caches[m][f"past_{x}_{L}"][:, :, -n:, :] - ref) / den)
        kv_err.append(row)

    # ③ 最後の 1 歩を計算するモデル（FP32 か focused_later）
    steps = {"fp32": session(paths["fp32"][1]), "focused": session(paths["focused"][1])}

    def margin(cache, step):
        # cache を与えて最後の語を step のモデルで流し、(logits, 「音」−「エ」) を返す
        r = Runner(steps[step], cos, sin)
        r.cache = cache
        r.position = position
        got, _ = r.feed(last, steps[step])
        lg = got["logits"].ravel().astype(np.float64)
        return lg, float(lg[ON] - lg[E])

    ref_lg, _ = margin(caches["fp32"], "fp32")
    p_ref = softmax(ref_lg)

    def mixed(layers, donor):
        # focused の KV を写し、layers に入る層だけ donor（fp32 か baseline）の KV に置き換える
        c = {k: v for k, v in caches["focused"].items()}
        for k in c:
            if layer_of(k) in layers:
                c[k] = caches[donor][k]
        return c

    groups = {"なし（focused の KV のまま）": [],
              "層0-6": range(0, 7), "層7-13": range(7, 14), "層14-20": range(14, 21), "層21-27": range(21, 28),
              "最後4層(24-27)": range(24, 28), "最後8層(20-27)": range(20, 28), "最後12層(16-27)": range(16, 28),
              "最初14層(0-13)": range(0, 14), "全層": range(0, 28)}
    rows = []
    print(f"\n  focused の KV の一部の層を差し替え →「音」−「エ」の差（FP32 は +5.07、負で逆転）")
    print(f"  {'差し替えた層':24s} {'FP32KV/最終FP32':>16} {'FP32KV/最終focused':>18} {'baseKV/最終focused':>18}")
    for label, layers in groups.items():
        layers = set(layers)
        out = {"layers": sorted(layers)}
        for donor in ("fp32", "baseline"):
            for step in ("fp32", "focused"):
                if donor == "baseline" and step == "fp32":
                    continue
                lg, mg = margin(mixed(layers, donor), step)
                q = softmax(lg)
                out[f"{donor}_kv__{step}_step"] = {
                    "on_minus_e": mg, "top1": tok.decode([int(lg.argmax())]),
                    "kl": float((p_ref * (np.log(p_ref + 1e-30) - np.log(q + 1e-30))).sum())}
        rows.append({"label": label, **out})
        a, b, c = (out["fp32_kv__fp32_step"], out["fp32_kv__focused_step"], out["baseline_kv__focused_step"])
        print(f"  {label:24s} {a['on_minus_e']:+9.2f} {a['top1']!r:>5} {b['on_minus_e']:+11.2f} {b['top1']!r:>5}"
              f" {c['on_minus_e']:+11.2f} {c['top1']!r:>5}", flush=True)

    # ④ focused の誤差が baseline より 1 割以上大きい層に ★ を付ける
    print(f"\n  層ごとの KV 誤差（FP32 比・有効 {n} スロット）  K: baseline→focused / V: baseline→focused")
    for r in kv_err:
        mark = " ★" if r["k_focused"] > r["k_baseline"] * 1.1 or r["v_focused"] > r["v_baseline"] * 1.1 else ""
        print(f"    層{r['layer']:2d}  K {r['k_baseline']*100:6.2f}% → {r['k_focused']*100:6.2f}%   "
              f"V {r['v_baseline']*100:6.2f}% → {r['v_focused']*100:6.2f}%{mark}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "layer_attrib.json").write_text(json.dumps({"rows": rows, "kv_error_by_layer": kv_err},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {args.out / 'layer_attrib.json'}")


if __name__ == "__main__":
    main()
