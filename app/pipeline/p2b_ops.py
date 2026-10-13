"""P2b: 単体 op モデルをパイプラインに通し、**どの段で落ちたか**を記録する

基本設計② 6.1節の判定は「①そのまま通る ②分解すれば通る ③通らない」の3値だが、
③ をそのまま記録しても次の手が決まらない。**どの段で落ちたのか**まで残す:

  S2 前処理     … ONNX の形が悪い（形状推論・最適化で落ちる）
  S4 量子化     … 量子化器が op を知らない / QDQ を挿せない
  S5 変換       … qairt-converter が op を知らない
  S6 context生成 … HTP コンパイラが op を受け付けない  ← v68 の対応可否はここ
  端末          … ロード or 実行で落ちる                ← C-1〜C-7 の世界

同じ「通らない」でも S5 と S6 では対処がまったく違う（前者は ONNX の書き換え、
後者は op の分解や精度指定）。だから段を分けて記録する。

ホスト側の Python 環境が2つに割れている（NFR-5）ため、本モジュールは
**本体 .venv で動く部分だけ**を担当する:
  prepare  : S2 前処理 → S3 キャリブレーション → S4 量子化 → C-1/C-5 検証
  finalize : S8 EPContext ラッパー → C-2/C-3/C-4 検証
間の S5〜S7 は scripts/build_context.sh（QAIRT 環境）が担当する。
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

import onnx

from app.pipeline import calib, preprocess, quantize_qdq, verify_artifacts, wrap_epctx


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _spec(work: Path, name: str) -> dict:
    return json.loads((work / name / f"{name}.spec.json").read_text(encoding="utf-8"))


def _int_max(spec: dict) -> dict[str, int]:
    return {i["name"]: i["int_max"] for i in spec["inputs"] if "int_max" in i}


def act_act_matmul_sides(prepped: Path) -> tuple[list[str], list[str], list[str]]:
    """活性化×活性化 MatMul の (左入力, 右入力, 出力) を別々に返す。

    QNN HTP の MatMul 対応は **uint16×uint16 だけが不可**で、
    **uint16×uint8 / uint8×uint16 は可**とされている。つまり
    「両方 8bit に落とす」以外に「**片側だけ 8bit**」という選択肢がある。
    片側を 16bit に残せれば、出力も 16bit に保てる可能性がある——
    これが効けば attention スコアの分解能問題（C-8 の代償）が根本から消える。
    """
    model = onnx.load(str(prepped))
    inits = {i.name for i in model.graph.initializer}
    lhs, rhs, outs = [], [], []
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        if any(i in inits for i in n.input):
            continue
        lhs.append(n.input[0])
        rhs.append(n.input[1])
        outs.extend(n.output)
    return lhs, rhs, outs


def act_act_matmul_tensors(prepped: Path, inputs_only: bool = False) -> list[str]:
    """**両辺が活性化**の MatMul に関わるテンソル名を集める。

    v68 の HTP は活性化×活性化の MatMul を 16bit で受け付けない
    （`has incorrect Value 68, expected >= 73`）。一方 A8 なら通る。
    そこで「そこだけ 8bit に落とす」ための対象を機械的に拾う。

    重みが initializer 側にある MatMul（FFN / projection）は 16bit のままでよい。
    **両者を区別せずに全部 8bit へ落とすと、精度を無駄に捨てる。**

    `inputs_only=True` なら**入力だけ**を 8bit にし、**出力は 16bit に残す**。
    C-8 が禁じているのが「16bit × 16bit の MatMul」だとすれば、出力まで落とす必要はない。
    これは実モデルで決定的に効く（2026-09-12 実測）:
      QK^T の出力（attention スコア）は実測で **[-23, +1564]** に広がっており、
      8bit だと 1段階 6.2 になる。softmax の入力で 6.2 の差は確率で約 500倍で、
      **attention が壊れる**。16bit なら 1段階 0.024 で済む。
    """
    model = onnx.load(str(prepped))
    inits = {i.name for i in model.graph.initializer}
    names: list[str] = []
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        if any(i in inits for i in n.input):
            continue  # 重み固定側。16bit のままで通る
        names.extend(list(n.input))
        if not inputs_only:
            names.extend(list(n.output))
    # 重複を保ったまま順序を残す（後で読むとき対応が追いやすい）
    seen, out = set(), []
    for t in names:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def prepare(work: Path, name: str, scheme: str, n_random: int) -> dict:
    """S2 → S4。段ごとに成否を記録し、落ちても例外を外へ投げない。"""
    d = work / name
    spec = _spec(work, name)
    st: dict = {"name": name, "scheme": scheme, "stages": {},
                # 外部から持ち込んだモデル（実 export など）は make_op_models が
                # 付ける項目を持たないことがあるので、無い分は落とす。
                "spec": {k: spec[k] for k in ("note", "priority", "op_types",
                                              "domains", "nodes", "standard_opset",
                                              "calib_npz", "shape_config")
                         if k in spec},
                "timestamp": datetime.now().isoformat(timespec="seconds")}

    src = Path(spec["model"])
    prepped = d / f"{name}_prepped.onnx"
    qdq = d / f"{name}_{scheme}_qdq.onnx"

    # --- S2 前処理 ---
    try:
        # scheme 名に "_swap" が付いていたら QK^T の左右を入れ替える（C-8 対策）。
        # 対象は「右辺のレンジが左辺より広い MatMul」だが、前処理の時点では
        # レンジが分からないので、**名前で指定する**形にしてある。
        swap = spec.get("swap_matmuls") if "swap" in scheme else None
        r = preprocess.run(src, d / f"{name}_folded.onnx", prepped,
                           swap_matmuls=set(swap) if swap else None,
                           norm_rmsnorm="rms" in scheme)
        m = onnx.load(str(prepped))
        st["stages"]["S2_preprocess"] = {
            "ok": True,
            "nodes": len(m.graph.node),
            "op_types": sorted({n.op_type for n in m.graph.node}),
            "batchnorm_after": r["batchnorm_after_preprocess"],
        }
    except Exception as e:  # noqa: BLE001 - 段の失敗は記録して次へ進めたい
        st["stages"]["S2_preprocess"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc(limit=3)}
        st["verdict"] = "fail_S2_preprocess"
        return st

    # --- S3/S4 キャリブレーション + 量子化 ---
    overrides = None
    base_scheme = scheme
    converts = True

    # ★グラフ入力が形状 op（Slice / Unsqueeze 等）に直接入ると、量子化器は
    #   それを素通し扱いにして **float のまま残す**（C-1 違反）。
    #   decode モデルの KV キャッシュ入力 48本がまさにこれだった（2026-09-13）。
    #   明示的に型を指定して Q/DQ を挿させる。
    def _float_graph_inputs(path: Path) -> list[str]:
        m = onnx.load(str(path), load_external_data=False)
        names = {i.name for i in m.graph.initializer}
        prod = {o: n for n in m.graph.node for o in n.output}
        cons: dict[str, list] = {}
        for n in m.graph.node:
            for i in n.input:
                cons.setdefault(i, []).append(n)
        out = []
        for vi in m.graph.input:
            if vi.name in names:
                continue
            if vi.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                continue
            if any(c.op_type != "QuantizeLinear" for c in cons.get(vi.name, [])):
                out.append(vi.name)
        return out
    if scheme.endswith("_swap") or "rms" in scheme:
        base_scheme = "a16w8"
        # ★`rhs8` を一緒に指定できるようにする（2026-09-14）。
        #   C-8 は「act × act の MatMul は u16×u8 しか通らない」なので、
        #   swap（左右入替）だけでは足りず、**右辺を 8bit に落とす**必要がある。
        #   ここを排他の分岐にしていたため `a16w8_rms_swap` だけでは
        #   u16×u16 が残り、context 生成が `expected >= 73` で落ちていた。
        #   実機で動いている 24層の `final_qdq` は右辺が uint8 になっている。
        if "rhs8" in scheme:
            from onnxruntime.quantization import QuantType
            lhs, rhs, outs = act_act_matmul_sides(prepped)
            overrides = {t: [{"quant_type": QuantType.QUInt8}] for t in rhs}
            converts = False          # ORT に型を揃えさせない
            st["mixed_precision"] = {"downgraded_to_u8": rhs, "count": len(rhs),
                                     "strategy": scheme, "add_qtype_converts": False,
                                     "kept_u16": lhs + outs}
    elif scheme in ("a16w8_rhs8", "a16w8_lhs8"):
        # ★片側だけ 8bit にする（ChatGPT の指摘で試験）。
        #   QNN HTP は uint16×uint8 を許すとされるので、
        #   スコア（出力）を 16bit に残せる可能性がある。
        from onnxruntime.quantization import QuantType
        base_scheme = "a16w8"
        lhs, rhs, outs = act_act_matmul_sides(prepped)
        side = rhs if scheme == "a16w8_rhs8" else lhs
        overrides = {t: [{"quant_type": QuantType.QUInt8}] for t in side}
        converts = False   # ORT に型を揃えさせない
        st["mixed_precision"] = {"downgraded_to_u8": side, "count": len(side),
                                 "strategy": scheme, "add_qtype_converts": False,
                                 "kept_u16": (lhs if scheme == "a16w8_rhs8" else rhs) + outs}
    elif scheme in ("a16w8_mixed", "a16w8_in8", "a16w8_out16"):
        inputs_only = (scheme in ("a16w8_in8", "a16w8_out16"))
        # A16 を基本にしつつ、活性化×活性化の MatMul だけ A8 へ落とす。
        from onnxruntime.quantization import QuantType
        base_scheme = "a16w8"
        targets = act_act_matmul_tensors(prepped, inputs_only=inputs_only)
        overrides = {t: [{"quant_type": QuantType.QUInt8}] for t in targets}
        st["mixed_precision"] = {"downgraded_to_u8": targets, "count": len(targets),
                                 "inputs_only": inputs_only}
        if scheme == "a16w8_out16":
            # 出力を**明示的に** 16bit へ固定する。
            # 何も指定しないと、量子化器は「入力が uint8 なら出力も uint8」と
            # 揃えてしまい、入力だけ落としたつもりが出力まで 8bit になる
            # （2026-09-12 に実際に踏んだ。md5 は変わるのに精度が1ビットも変わらず気付いた）。
            outs = [t for t in act_act_matmul_tensors(prepped, inputs_only=False)
                    if t not in targets]
            for t in outs:
                overrides[t] = [{"quant_type": QuantType.QUInt16}]
            st["mixed_precision"]["pinned_to_u16"] = outs


    from onnxruntime.quantization import QuantType as _QT
    # ★重みを per-channel で量子化させる（spec に per_channel_weights: true があるとき）。
    #
    #   `get_qnn_qdq_config(per_channel=True)` だけでは **MatMul の重みに効かない**。
    #   ORT は `TensorQuantOverrides` に `axis` が書かれているかで per-channel を決める
    #   （`qdq_quantizer.has_per_channel_overrides`）。2026-09-13 まで
    #   **390本すべてが per-tensor**になっていて、それが誤差の主因だった。
    if spec.get("per_channel_weights"):
        from app.pipeline import weight_percg
        overrides, axes = weight_percg.build_overrides(prepped, overrides)
        st["per_channel_weights"] = {"requested": len(axes), "axis": 1}
        print(f"  重みを per-channel にする指定: {len(axes)} 本（axis=1）")

    gin = _float_graph_inputs(prepped)
    if gin:
        overrides = dict(overrides or {})
        for t in gin:
            overrides.setdefault(t, [{"quant_type": _QT.QUInt16}])
        st["forced_input_quant"] = gin

    try:
        if spec.get("calib_npz"):
            # ★実データ較正。乱数較正との違いは致命的になりうる:
            #   本物の hidden_states は ±0.09（std 0.0156）なのに対し、
            #   P2b までの乱数較正は ±1.0 だった（約11倍広い）。
            #   A16 なら耐えるが A8（256段階）では実データが ±6段階にしか
            #   収まらず、量子化が意味をなさなくなる。
            import numpy as _np
            z = _np.load(spec["calib_npz"])
            # ★npz のキーを**モデルの入力名で絞る**。
            #   較正データに評価用のメタ情報（有効トークン数 `n_valid` など）を
            #   同梱できるようにするため。絞らずに渡すと存在しない入力名を
            #   較正器へ食わせることになる。
            want = [sp["name"] for sp in calib.all_input_specs(str(prepped))]
            names = [k for k in z.files if k in want]
            extra = [k for k in z.files if k not in want]
            missing = [k for k in want if k not in names]
            if missing:
                raise RuntimeError(
                    f"較正データに足りない入力がある: {missing}（npz のキー: {list(z.files)}）")
            n = z[names[0]].shape[0]
            batches = [{k: z[k][i:i + 1] for k in names} for i in range(n)]
            reader = calib.MultiReader(batches)
            st["calibration"] = {"source": spec["calib_npz"], "samples": n,
                                 "inputs": names, "ignored_keys": extra}
            if extra:
                print(f"  較正データの非入力キーは無視する: {extra}")
        else:
            reader = calib.build_multi_reader(str(prepped), n_random=n_random,
                                              int_max=_int_max(spec))
        r = quantize_qdq.run(prepped, qdq, scheme=base_scheme, reader=reader,
                             init_overrides=overrides, add_qtype_converts=converts)

        # ★**設定ではなく生成物を見る。** per-channel を頼んだなら、
        #   本当にそうなっているかをここで確かめる（約1時間の context 生成の前に止める）。
        if spec.get("per_channel_weights"):
            from app.pipeline import weight_percg
            info = weight_percg.verify(qdq, expect_per_channel=st["per_channel_weights"]["requested"],
                                       expect_axis=1, strict=True)
            st["per_channel_weights"]["verified"] = {
                "per_channel": info["big_per_channel"], "per_tensor": info["big_per_tensor"],
                "axes": info["axes"], "by_dtype": info["by_dtype"]}
            print(f"  検査OK: per-channel {info['big_per_channel']} 本 / "
                  f"per-tensor {info['big_per_tensor']} 本 / axis {info['axes']}")

        # ★C-13: RMSNorm の gamma を**符号付き16bitに収める**。
        #   融合 op `RmsNorm` の重みは DLC で `sFxp_16` として渡されるのに、
        #   ORT は uint16（zero_point 0）で量子化する。量子化値が 32767 を超える要素は
        #   **黙って頭打ちになる**（2026-09-14: RMSNorm 単体で相対L2 13.54%）。
        #   spec に `skip_c13: true` があるときだけ飛ばす。
        if not spec.get("skip_c13"):
            from app.pipeline import fix_norm_gamma
            before = fix_norm_gamma.audit(qdq)
            if before["n_unsafe"]:
                fix_norm_gamma.run(qdq, qdq)
                after = fix_norm_gamma.audit(qdq)
                if after["n_unsafe"]:
                    raise RuntimeError("C-13: 修正後も 32767 超が残っている")
                st["c13_norm_gamma"] = {"targets": before["count"],
                                        "unsafe_before": before["n_unsafe"],
                                        "elements_over": before["total_over"],
                                        "unsafe_after": 0}
                print(f"  C-13: RMSNorm gamma {before['count']} 本のうち "
                      f"{before['n_unsafe']} 本が符号付き16bitからはみ出していたので直した")
            else:
                st["c13_norm_gamma"] = {"targets": before["count"], "unsafe_before": 0}
                if before["count"]:
                    print(f"  C-13: RMSNorm gamma {before['count']} 本 — はみ出しなし")

        m = onnx.load(str(qdq))
        ops = [n.op_type for n in m.graph.node]
        st["stages"]["S4_quantize"] = {
            "ok": True,
            "nodes": len(ops),
            "dequantize": ops.count("DequantizeLinear"),
            "quantize": ops.count("QuantizeLinear"),
            "calibration_samples": r["calibration_samples"],
            "qdq_path": str(qdq).replace("\\", "/"),
        }
    except Exception as e:  # noqa: BLE001
        st["stages"]["S4_quantize"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                       "trace": traceback.format_exc(limit=3)}
        st["verdict"] = "fail_S4_quantize"
        return st

    # --- C-1 / C-5（モデル単体で判定できる分）---
    c1 = verify_artifacts.check_c1_no_float_segments(qdq)
    c5 = verify_artifacts.check_c5_no_batchnorm(qdq)
    st["checks"] = {"C-1": {"ok": c1.ok, "detail": c1.get("detail", ""),
                            "violations": c1.get("violations", [])},
                    "C-5": {"ok": c5.ok, "detail": c5.get("detail", "")}}
    if not c1.ok:
        # C-1 を割ったまま先へ進めても端末で error 1002 になるだけ。ここで止める。
        st["verdict"] = "fail_C1_float_segment"
        return st

    st["verdict"] = "ready_for_convert"
    return st


def finalize(work: Path, name: str, scheme: str, ctx_dir: Path, sdk_version: str) -> dict:
    """S8 + C-2/C-3/C-4。build_context.sh が成功した場合のみ呼ばれる想定。"""
    d = work / name
    qdq = d / f"{name}_{scheme}_qdq.onnx"
    st: dict = {"name": name, "scheme": scheme, "stages": {}}

    bins = sorted(ctx_dir.glob("ctx_*/*.bin")) or sorted(ctx_dir.glob("*.bin"))
    if not bins:
        st["stages"]["S8_epctx"] = {"ok": False, "error": f"context binary が無い: {ctx_dir}"}
        st["verdict"] = "fail_S6_context"
        return st

    meta = next(iter(sorted(ctx_dir.glob("*_meta.json"))), None)
    cmds_file = next(iter(sorted(ctx_dir.glob("*_commands.txt"))), None)
    cmds = cmds_file.read_text(encoding="utf-8").splitlines() if cmds_file else []

    try:
        r = wrap_epctx.run(bins[0], qdq, d / f"{name}_{scheme}_epctx.onnx", name,
                           sdk_version=sdk_version)
        st["stages"]["S8_epctx"] = {"ok": True, "epctx_path": r["epctx_path"].replace("\\", "/"),
                                    "bytes": r["bytes"], "context_bytes": r["context_bytes"],
                                    "inputs": r["inputs"], "outputs": r["outputs"]}
    except Exception as e:  # noqa: BLE001
        st["stages"]["S8_epctx"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                    "trace": traceback.format_exc(limit=3)}
        st["verdict"] = "fail_S8_epctx"
        return st

    c1 = verify_artifacts.check_c1_no_float_segments(qdq)
    checks = {"C-2": verify_artifacts.check_c2_socmodel_zero(meta) if meta else None,
              "C-3": verify_artifacts.check_c3_no_backend_config(cmds),
              "C-4": verify_artifacts.check_c4_preserve_io(cmds, c1)}
    st["checks"] = {k: ({"ok": v.ok, "detail": v.get("detail", "")} if v is not None else
                        {"ok": False, "detail": "meta.json が無く検証不能"})
                    for k, v in checks.items()}
    st["verdict"] = ("ready_for_device"
                     if all(v["ok"] for v in st["checks"].values())
                     else "fail_constraint_check")
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "finalize"])
    ap.add_argument("--work", type=Path, default=Path("work/ops"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--scheme", default="a16w8")
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--ctx-dir", type=Path)
    ap.add_argument("--sdk-version", default="2.48.0")
    args = ap.parse_args()

    if args.stage == "prepare":
        st = prepare(args.work, args.name, args.scheme, args.n_random)
        out = args.work / args.name / f"status_prepare_{args.scheme}.json"
    else:
        if not args.ctx_dir:
            raise SystemExit("finalize には --ctx-dir が必要")
        st = finalize(args.work, args.name, args.scheme, args.ctx_dir, args.sdk_version)
        out = args.work / args.name / f"status_finalize_{args.scheme}.json"

    _write(out, st)
    print(f"[{args.name}] verdict={st['verdict']}")
    for k, v in st.get("stages", {}).items():
        mark = "OK " if v.get("ok") else "NG "
        extra = "" if v.get("ok") else f"  {v.get('error', '')[:160]}"
        print(f"  {mark}{k}{extra}")
    for k, v in st.get("checks", {}).items():
        print(f"  {'OK ' if v['ok'] else 'NG '}{k}  {v.get('detail', '')[:120]}")
    print(f"  -> {out}")

    # ready_* 以外は非ゼロ終了にして、呼び出し元のシェルが段を飛ばせるようにする。
    raise SystemExit(0 if str(st["verdict"]).startswith("ready") else 1)


if __name__ == "__main__":
    main()
