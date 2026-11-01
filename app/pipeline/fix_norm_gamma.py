r"""RMSNorm の gamma を **符号付き16bit に収まる形** で量子化し直す（C-13）。

## 見つけた不具合（2026-09-14）

`qairt-converter` は RMSNorm を融合 op `RmsNorm` にする（`qairt-dlc-info` で確認）。
そのとき **gamma を `sFxp_16`（符号付き）として扱う**が、ONNX 側の QDQ は
**uint16**（zero_point=0、scale で [0, 1.99] を表現）である。

その結果、**量子化値が 32767 を超える要素が黙って頭打ちになる**。

    実測: 倍率 × gamma = 0.9961 = 32767 × 3.03988e-05  ← 全列で一定
    896要素中 33要素が該当。RMSNorm 単体の相対L2 13.54%
    直したら 0.0342%（cos 1.000000）

## 直し方

gamma を **符号付き対称**（zero_point=0、`|gamma|max / 32767` を scale）で入れ直す。
符号付きとして読まれても値が変わらない。表現力もほぼ落ちない（再量子化誤差 0.002%）。

## パイプラインのどこで使うか

    S2 前処理 → **S4 量子化の直後（ここ）** → S5 DLC変換 → S6 context生成 → 実機

`p2b_ops.py` の S4 のあとに**自動で**呼ばれる（spec に `skip_c13: true` があれば飛ばす）。
はみ出しが1つでもあれば直し、直ったことを確かめてから次へ進む。

## 使い方

    python -m app.pipeline.fix_norm_gamma --qdq <in> --audit          # 検査だけ
    python -m app.pipeline.fix_norm_gamma --qdq <in> --out <out>      # 直す

## 入力と出力（2026-10-18 追記）

  入力 : S4 の QDQ（`<name>_<scheme>_qdq.onnx`）。
  出力 : `--audit` なら表示だけ（`--json` で記録）。`--out` なら gamma を入れ直した QDQ。
         p2b_ops からは同じファイルへ上書きで呼ばれる（`run(qdq, qdq)`）。
  解説 : StudyMemo 9.4 節「真因 — C-13」・9.7 節（自動化）、演習 ex04_c13_signed_gamma.py。
         24層の prefill / decode とも 49 本中 49 本が該当していた（9.4 節）。
  ONNX 側の QDQ を見るだけでは気付けない（融合されると ONNX のノードは消える）ので、
  DLC 側は dlc_audit.py で確かめる（9.6 節 ①）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh

# ★`*norm.weight` を**すべて**拾う。
#   最初は `input_layernorm` / `post_attention_layernorm` / 最終 `norm` だけを
#   列挙していたが、**Qwen3 の `q_norm` / `k_norm`（ヘッドごとの RMSNorm）が漏れた**
#   （2026-09-14: 2層で 9本あるはずが 5本しか拾えなかった）。
#   Qwen3-0.6B は 28層 × 4 + 1 = **113本**。列挙ではなく「norm.weight を含む」で拾う。
DEFAULT_PATTERN = r"norm\.weight"
# 符号付き 16bit（int16 / QNN の sFxp_16）で表せる最大値。uint16 の量子化値がこれを超えると頭打ちになる
SIGNED_MAX = 32767


def _targets(model: onnx.ModelProto, pattern: str):
    """RMSNorm の gamma を量子化している DequantizeLinear を 1 つずつ返す（ジェネレータ）。

    戻り値: (DQ ノード, {初期化子名: TensorProto}) の組。DQ の入力 0 が初期化子（＝重み）で、
    その名前から ORT が付ける接尾辞 "_quantized" を外した元の名前が pattern に合うものだけ。
    """
    inits = {t.name: t for t in model.graph.initializer}
    rx = re.compile(pattern)
    for nd in model.graph.node:
        if nd.op_type != "DequantizeLinear" or nd.input[0] not in inits:
            continue
        base = nd.input[0].replace("_quantized", "")
        if not rx.search(base):
            continue
        yield nd, inits


def audit(src: Path, pattern: str = DEFAULT_PATTERN) -> dict:
    """**生成物を開いて数える。** 設定値は見ない。

    引数: src（QDQ の ONNX）、pattern（対象の重み名の正規表現）。
    戻り値: {"count": 対象の本数, "rows": [本ごとの型・scale・zero point・量子化値の最小最大・
             gamma の範囲・32767 超の要素数], "n_unsafe": はみ出しのある本数, "total_over": 要素数の合計}
    gamma は「(量子化値 - zero point) × scale」で実数に戻して範囲を出す。
    """
    m = onnx.load(str(src))
    rows = []
    for nd, inits in _targets(m, pattern):
        q = nh.to_array(inits[nd.input[0]])
        sc = np.asarray(nh.to_array(inits[nd.input[1]])).astype(np.float64)
        zp = (np.asarray(nh.to_array(inits[nd.input[2]])).astype(np.float64)
              if len(nd.input) > 2 else np.zeros_like(sc))
        qf = q.astype(np.float64)
        g = (qf - zp) * np.ravel(sc)[0]
        rows.append({
            "name": nd.input[0],
            "dtype": str(q.dtype),
            "signed": bool(np.issubdtype(q.dtype, np.signedinteger)),
            "n": int(qf.size),
            "scale": float(np.ravel(sc)[0]),
            "zero_point": float(np.ravel(zp)[0]),
            "q_min": float(qf.min()), "q_max": float(qf.max()),
            "gamma_min": float(g.min()), "gamma_max": float(g.max()),
            "over_signed_max": int((qf > SIGNED_MAX).sum()),
        })
    bad = [r for r in rows if r["over_signed_max"] > 0]
    return {"count": len(rows), "rows": rows,
            "n_unsafe": len(bad),
            "total_over": sum(r["over_signed_max"] for r in rows)}


def run(src: Path, dst: Path, pattern: str = DEFAULT_PATTERN) -> dict:
    """対象の gamma をすべて「符号付き 16bit・対称（zero point 0）」で量子化し直し、dst に保存する。

    引数: src（元の QDQ）、dst（書き出し先。src と同じでもよい）、pattern（対象の名前）。
    戻り値: {"fixed": [{name, over_signed_max（直す前の 32767 超の数）, old_scale, new_scale,
             requant_rel_l2（入れ直しによる相対 L2 誤差）}, ...]}
    はみ出しの有無にかかわらず、pattern に合う gamma は全部入れ直す（p2b_ops は 1 本でも
    はみ出しがあるときだけ呼ぶ）。gamma がすべて 0 だと scale が 0 になり割り算が成り立たない
    （RMSNorm の gamma では起きない前提。コードからの推定）。
    """
    m = onnx.load(str(src))
    fixed = []
    for nd, inits in _targets(m, pattern):
        w, s_n = nd.input[0], nd.input[1]
        z_n = nd.input[2] if len(nd.input) > 2 else None
        q = nh.to_array(inits[w]).astype(np.float64)
        sc = np.asarray(nh.to_array(inits[s_n])).astype(np.float64)
        zp = (np.asarray(nh.to_array(inits[z_n])).astype(np.float64)
              if z_n else np.zeros_like(sc))
        # ① いまの量子化値を実数の gamma に戻す（per-tensor なので scale・zero point は 1 個）
        g = (q - np.ravel(zp)[0]) * np.ravel(sc)[0]

        # ② 符号付き対称で入れ直す: scale = |gamma| の最大 / 32767、量子化値は int16。
        #   符号付きとして読まれても値が変わらず、分解能は元の約半分（再量子化誤差 0.002%、9.4 節）
        new_sc = float(np.abs(g).max()) / SIGNED_MAX
        new_q = np.clip(np.rint(g / new_sc), -32768, SIGNED_MAX).astype(np.int16)
        err = float(np.linalg.norm(new_q * new_sc - g) / np.linalg.norm(g))

        # ③ 同じ名前の初期化子を中身ごと差し替える（形は元のまま、型は int16 に変わる）。
        #   zero point も int16 の 0 にする（DequantizeLinear は量子化値と zero point の型がそろう必要がある）
        inits[w].CopyFrom(nh.from_array(new_q.reshape(nh.to_array(inits[w]).shape), w))
        inits[s_n].CopyFrom(nh.from_array(np.float32(new_sc), s_n))
        if z_n:
            inits[z_n].CopyFrom(nh.from_array(np.array(0, dtype=np.int16), z_n))
        fixed.append({"name": w, "over_signed_max": int((q > SIGNED_MAX).sum()),
                      "old_scale": float(np.ravel(sc)[0]), "new_scale": new_sc,
                      "requant_rel_l2": err})

    # ④ 保存。初期化子の生データの合計が 1.8GB を超えたら外部データ形式（<dst>.data）にする
    #   （protobuf の 1 ファイル 2GB の上限を避ける）
    dst.parent.mkdir(parents=True, exist_ok=True)
    big = sum(t.raw_data.__len__() for t in m.graph.initializer) > 1.8e9
    onnx.save(m, str(dst), save_as_external_data=big,
              location=dst.name + ".data" if big else None)
    return {"fixed": fixed}


def _print_audit(a: dict, title: str) -> None:
    """audit() の結果を人が読む形で表示する（本数・型・zero point・gamma と scale の範囲・はみ出しの本数）。"""
    print(f"\n=== {title} ===")
    print(f"  対象 RMSNorm gamma        {a['count']} 本")
    if not a["count"]:
        return
    r0 = a["rows"][0]
    dts = {r["dtype"] for r in a["rows"]}
    print(f"  QDQ の dtype              {dts}"
          f"  （符号付き: {sorted({r['signed'] for r in a['rows']})}）")
    print(f"  zero_point                {sorted({r['zero_point'] for r in a['rows']})}")
    gmin = min(r["gamma_min"] for r in a["rows"]); gmax = max(r["gamma_max"] for r in a["rows"])
    print(f"  gamma の範囲（全体）      [{gmin:.4f}, {gmax:.4f}]")
    smin = min(r["scale"] for r in a["rows"]); smax = max(r["scale"] for r in a["rows"])
    print(f"  scale の範囲              [{smin:.6g}, {smax:.6g}]")
    qmax = max(r["q_max"] for r in a["rows"])
    print(f"  量子化値の最大            {qmax:.0f}  （符号付き16bitの上限 {SIGNED_MAX}）")
    mark = "★危険" if a["n_unsafe"] else "OK"
    print(f"  {mark}  32767 を超える本数    {a['n_unsafe']} 本 / {a['count']} 本"
          f"（要素数の合計 {a['total_over']}）")


def main() -> None:
    """コマンドの入口。修正前を検査して表示し、--audit（または --out なし）ならそこで終わる。
    --out があれば run() で直し、修正後を検査し直して、まだはみ出しがあれば止まる。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audit", action="store_true", help="検査だけして直さない")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    before = audit(args.qdq, args.pattern)
    _print_audit(before, f"修正前: {args.qdq.name}")
    out = {"before": {k: v for k, v in before.items() if k != "rows"}}
    if args.audit or not args.out:
        if args.json:
            args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    r = run(args.qdq, args.out, args.pattern)
    worst = max((f["requant_rel_l2"] for f in r["fixed"]), default=0.0)
    print(f"\n  直した本数 {len(r['fixed'])}  再量子化の誤差 最大 {worst*100:.4f}%")
    after = audit(args.out, args.pattern)
    _print_audit(after, f"修正後: {args.out.name}")
    if after["n_unsafe"]:
        raise SystemExit("★修正後もまだ 32767 を超えている本がある")
    out["after"] = {k: v for k, v in after.items() if k != "rows"}
    out["worst_requant_rel_l2"] = worst
    print(f"\n-> {args.out}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
