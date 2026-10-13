# -*- coding: utf-8 -*-
"""P12w: add_26 の4状態比較（FP32 / baseline / 4x / oracle）＋ decision-direction射影（①）。

P12vのoracle失敗を受けて、「baselineでは複数箇所の量子化誤差が相互に打ち消し合っており、
linear_20だけを高精度化すると誤差キャンセルのバランスが崩れる」という仮説を検証する。

まず①: 点B（「音楽をかけて」の判定位置）と同じ入力条件で、add_26（層2の出力＝層3への入力）を
FP32・baseline QDQ・4x（linear_20だけ4x range）・oracle（linear_20のQDQ除去）の4状態で比較する。

  - FP32比のL2相対誤差・コサイン類似度・最大誤差（幾何的な近さ）
  - 「decision方向への射影」: 各状態のadd_26を、層3-27+lm_headの**FP32継続モデル**に注入し、
    元のFP32 add_26を注入した場合と比べてmarginがどれだけ動くか（P32継続なので量子化ノイズを
    経由しない、純粋にadd_26の値そのものの効果を見る）

これにより2パターンに分岐する:
  A. oracleはL2ではFP32に近いのにdecision射影が悪化 → 「誤差の大きさより向き」が再確認される
  B. oracleのadd_26自体がFP32から遠い → 真犯人はlinear_20より上流

## 使い方

  python add26_causal_probe.py --phase prepare   # 4つのprefixモデル(add_26出力付き)とl3plus_fp32を作る
  python add26_causal_probe.py --phase capture    # 点Bの入力を4モデルに流し、add_26を採取
  python add26_causal_probe.py --phase compare    # L2/cosine/max + FP32継続への射影
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, session, ROOT, OPS, SOURCE, fit  # noqa: E402
from block_revert_ab import eval_points, REP  # noqa: E402
from linear20_oracle import ORACLE_FULL, bypass_linear20_qdq, WORK as ORACLE_WORK  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import helper, numpy_helper, TensorProto  # noqa: E402

OUT = ROOT / "results/P12w_20260916_add26_causal_probe"
WORK = ROOT / "work/p12w_add26_probe"

FOURX_FULL = WORK / "fourx_full28_qdq.onnx"  # linear_20だけ4x range、mul_41はbaseline(P12rと同じ値)
L3PLUS_FP32 = WORK / "fp32src" / "l3plus_fp32.onnx"  # prepped.onnx.dataと同じディレクトリに置く必要がある
L3PLUS_BASELINE = WORK / "l3plus_baseline_qdq.onnx"

L3PLUS_INPUTS = ["add_26"] + [f"past_k_{i}" for i in range(3, 28)] + [f"past_v_{i}" for i in range(3, 28)] + \
    ["cos", "sin", "attn_mask"]


def emit(o):
    print(json.dumps(o, ensure_ascii=False), flush=True)


def sess(p):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])


def add_output(model_path, out_path, tensor_name, elem_type=TensorProto.FLOAT):
    m = onnx.load(str(model_path), load_external_data=False)
    existing = {v.name for v in m.graph.output}
    if tensor_name not in existing:
        m.graph.output.append(helper.make_tensor_value_info(tensor_name, elem_type, None))
    onnx.save(m, str(out_path))


def build_fourx_full():
    """linear_20だけP12rの4x(margin=6.0)にした28層フルモデル(mul_41はbaseline)。"""
    audit = json.loads((REP / "build_audit.json").read_text(encoding="utf-8"))
    later = {c["scale_name"]: c for c in audit["later"]["changes"] if len(c["sources"]) == 1}
    calib = json.loads((REP / "calibration_ranges.json").read_text(encoding="utf-8"))
    lo, hi = calib["ranges"]["later"]["linear_20"]
    qmap = json.loads((REP / "qparam_map.json").read_text(encoding="utf-8"))
    g = next(g for g in qmap["groups"] if "linear_20" in g["sources"])
    dt = np.dtype(g["dtype"])
    qmax = np.iinfo(dt).max
    s, z = fit(lo, hi, qmax, 6.0)

    m = onnx.load(str(SOURCE), load_external_data=False)
    init = {t.name: t for t in m.graph.initializer}
    s_shape = numpy_helper.to_array(init[g["scale_name"]]).shape
    z_shape = numpy_helper.to_array(init[g["zero_name"]]).shape
    init[g["scale_name"]].CopyFrom(numpy_helper.from_array(np.asarray(s, np.float32).reshape(s_shape), g["scale_name"]))
    init[g["zero_name"]].CopyFrom(numpy_helper.from_array(np.asarray(z, dtype=dt).reshape(z_shape), g["zero_name"]))
    onnx.save(m, str(FOURX_FULL))
    onnx.checker.check_model(str(FOURX_FULL))
    return {"scale": float(s), "zero": int(z)}


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # ONNXのcheckerが、work/ops配下のprepped.onnx.dataの既存ハードリンク数を「hardlink攻撃」と
    # 誤検知するため、リンクの無いコピーを作ってから使う（extract_model・add_outputの両方で）
    fp32_copy_dir = WORK / "fp32src"
    fp32_copy_dir.mkdir(parents=True, exist_ok=True)
    fp32_copy = fp32_copy_dir / "qwen3_dec28c256_prepped.onnx"
    if not fp32_copy.exists():
        import shutil
        shutil.copyfile(OPS / "qwen3_dec28c256_prepped.onnx", fp32_copy)
        shutil.copyfile(OPS / "qwen3_dec28c256_prepped.onnx.data", fp32_copy_dir / "qwen3_dec28c256_prepped.onnx.data")

    # 4つのprefixモデル: add_26を追加出力させるだけ(構造はそれぞれ既存のまま)
    add_output(fp32_copy, fp32_copy_dir / "fp32_prefix_add26.onnx", "add_26")
    add_output(SOURCE, WORK / "baseline_prefix_add26.onnx", "add_26")
    fourx_info = build_fourx_full()
    add_output(FOURX_FULL, WORK / "fourx_prefix_add26.onnx", "add_26")
    add_output(ORACLE_FULL, WORK / "oracle_prefix_add26.onnx", "add_26")

    # 継続モデル: add_26を入力として層3-27+lm_headだけを実行
    # FP32側は重みが約2GB超あり、onnx.utils.extract_model の既定(infer_shapes=True)だと
    # 外部データを丸ごとメモリに読み込んでprotobufの2GB上限を超えて失敗する。
    # load_external_data=False のまま Extractor を直接使い、外部データ参照を保ったまま
    # コピー先ディレクトリ(fp32copyの.dataがある場所)に保存する。
    from onnx.utils import Extractor
    fp32_model = onnx.load(str(fp32_copy), load_external_data=False)
    extracted = Extractor(fp32_model).extract_model(L3PLUS_INPUTS, ["logits"])
    onnx.save(extracted, str(L3PLUS_FP32))
    onnx.checker.check_model(str(L3PLUS_FP32))

    onnx.utils.extract_model(str(SOURCE), str(L3PLUS_BASELINE),
                              input_names=L3PLUS_INPUTS, output_names=["logits"], check_model=False)
    onnx.checker.check_model(str(L3PLUS_BASELINE))

    emit({"phase": "prepare", "fourx": fourx_info,
          "l3plus_fp32_nodes": len(onnx.load(str(L3PLUS_FP32), load_external_data=False).graph.node),
          "l3plus_baseline_nodes": len(onnx.load(str(L3PLUS_BASELINE), load_external_data=False).graph.node)})


def capture():
    """点Bの入力を4モデルに流し、決定位置のadd_26を採取する。
    層3-27の文脈(past_k/v・cos/sin/attn_mask)はFP32軌跡のものを固定して使う
    （「同じ入力条件で」add_26だけを差し替えて比較するため）。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "point_B_music")
    ids, pos_want = p["ids"], p["eval_positions"][0]
    emit({"phase": "capture", "tokens": len(ids), "decision_position": pos_want})

    cos, sin = tables()
    variants = {
        "fp32": WORK / "fp32src" / "fp32_prefix_add26.onnx",
        "baseline": WORK / "baseline_prefix_add26.onnx",
        "4x": WORK / "fourx_prefix_add26.onnx",
        "oracle": WORK / "oracle_prefix_add26.onnx",
    }
    npz = {}
    l3plus_context = None
    for name, path in variants.items():
        s = sess(path)
        r = Runner(s, cos, sin)
        add26 = None
        for pos, tid in enumerate(ids):
            if pos == pos_want and name == "fp32":
                # 層3-27用の文脈(このpositionをfeedする直前のcache + cos/sin/mask)をFP32軌跡から保存
                mask = np.full((1, 1, 1, 256), -60, np.float32)
                mask[..., 256 - min(pos + 1, 256):] = 0
                l3plus_context = {
                    **{k: v for k, v in r.cache.items() if int(k.split("_")[-1]) >= 3},
                    "cos": cos[pos].reshape(1, 1, 1, 128), "sin": sin[pos].reshape(1, 1, 1, 128),
                    "attn_mask": mask,
                }
            got, _ = r.feed(tid, s)
            if pos == pos_want:
                add26 = got["add_26"].astype(np.float32).copy()
        if add26 is None:
            raise RuntimeError("decision position not reached")
        npz[f"add26_{name}"] = add26
        del s, r
    assert l3plus_context is not None
    np.savez(WORK / "add26_states.npz", **npz)
    np.savez(WORK / "l3plus_context.npz", **l3plus_context)
    emit({"phase": "capture", "wrote": str(WORK / "add26_states.npz"),
          "keys": list(npz.keys()), "l3plus_context_keys": list(l3plus_context.keys())})


def compare():
    OUT.mkdir(parents=True, exist_ok=True)
    states = dict(np.load(WORK / "add26_states.npz"))
    context = dict(np.load(WORK / "l3plus_context.npz"))
    fp32_add26 = states["add26_fp32"].astype(np.float64).ravel()

    geo = {}
    for name in ("fp32", "baseline", "4x", "oracle"):
        v = states[f"add26_{name}"].astype(np.float64).ravel()
        diff = v - fp32_add26
        cos_sim = float(np.dot(v, fp32_add26) / (np.linalg.norm(v) * np.linalg.norm(fp32_add26) + 1e-30))
        geo[name] = {"l2_rel": float(np.linalg.norm(diff) / (np.linalg.norm(fp32_add26) or 1.0)),
                     "cosine": cos_sim, "max_abs_diff": float(np.abs(diff).max())}

    s = sess(L3PLUS_FP32)
    logits = {}
    for name in ("fp32", "baseline", "4x", "oracle"):
        feeds = {**context, "add_26": states[f"add26_{name}"].astype(np.float32)}
        out = s.run(None, feeds)
        logits[name] = out[0].ravel().astype(np.float64)
    t1 = int(logits["fp32"].argmax())
    margins = {}
    for name, lg in logits.items():
        mask = np.ones(lg.size, dtype=bool)
        mask[t1] = False
        margins[name] = float(lg[t1] - lg[mask].max())

    result = {"decision_top1_token": t1, "geometry_vs_fp32": geo,
              "margin_via_fp32_continuation": margins,
              "margin_delta_vs_fp32": {k: margins[k] - margins["fp32"] for k in margins}}
    (OUT / "compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "compare", **result})
    print(f"\n  {'条件':10s} {'L2相対誤差':>10s} {'コサイン':>9s} {'最大誤差':>9s}  {'margin(FP32継続)':>16s} {'Δmargin':>9s}")
    for name in ("fp32", "baseline", "4x", "oracle"):
        g = geo[name]
        print(f"  {name:10s} {g['l2_rel']*100:9.2f}% {g['cosine']:9.5f} {g['max_abs_diff']:9.2f}  "
              f"{margins[name]:+16.3f} {margins[name]-margins['fp32']:+9.3f}")
    print(f"\n-> {OUT / 'compare.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["prepare", "capture", "compare"], required=True)
    a = ap.parse_args()
    if a.phase == "prepare":
        prepare()
    elif a.phase == "capture":
        capture()
    elif a.phase == "compare":
        compare()


if __name__ == "__main__":
    main()
