"""S4: QDQ 量子化（本パイプラインの中核）

基本設計② S4 の実装。

**`quantize_static` に op リストを手で渡す形は使わない。**
`get_qnn_qdq_config` は QNN EP 向けの制約を織り込んだ設定を生成するため、
これを通さないと既定の op 集合で float セグメントが残り **C-1 に違反**する。

本線は A16W8（QUInt16 活性化 / QInt8 per-channel 重み）。公式ドキュメントの
標準量子化例と一致する（要件定義書 3.3.2節#17b）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

from app.pipeline import calib

# 活性化の指定。v68 では 16a16w の matmul が使えない（V73以上）ため、
# 重みは常に INT8 per-channel とし、活性化だけを切り替える。
ACTIVATION = {
    "a16w8": QuantType.QUInt16,
    "a8w8": QuantType.QUInt8,
}


def run(
    prepped: Path,
    out_qdq: Path,
    scheme: str = "a16w8",
    reader=None,
    init_overrides: dict | None = None,
    add_qtype_converts: bool = True,
    calib_method: str = "minmax",
    nodes_to_exclude: list | None = None,
    percentile: float = 99.999,
    n_random: int = 32,
    seed: int = 0,
) -> dict:
    if scheme not in ACTIVATION:
        raise ValueError(f"未対応の scheme: {scheme}（{list(ACTIVATION)} のいずれか）")

    if reader is None:
        reader = calib.build_reader(str(prepped), n_random=n_random, seed=seed)

    # ★init_overrides のテンソル名は **必ず prepped（前処理後）のモデル**から取る。
    # quant_pre_process の最適化はノード名・テンソル名を変更・統合するため、
    # 元の fp32 モデルから拾うと一致せず、**上書きが黙って無効になる**
    # （エラーにならないので気付きにくい。基本設計② S2 補足3）。
    cfg = get_qnn_qdq_config(
        str(prepped),
        reader,
        # MinMax は外れ値1個にレンジ全体を決められてしまう。
        # LLM の残差ストリームは少数の次元だけ桁違いに大きい（massive activations）ため、
        # **Percentile で上位を切り捨てる**方が、層が加える小さな寄与を保てる可能性がある。
        calibrate_method=(CalibrationMethod.Percentile if calib_method == "percentile"
                          else CalibrationMethod.MinMax),
        activation_type=ACTIVATION[scheme],
        weight_type=QuantType.QInt8,
        per_channel=True,
        init_overrides=init_overrides,
        # 診断用。**除外すると float が残り C-1 違反**になるので本番では使わない
        # （app/pipeline/oracle_exclude.py の注記）。
        nodes_to_exclude=nodes_to_exclude,
        # False にすると ORT は型を揃えるための変換ノードを挿さない。
        # QNN HTP は uint16×uint8 の MatMul を許すとされるので、
        # **ORT が勝手に 8bit へ揃えてしまう**のを止められるか試すためのつまみ。
        add_qtype_converts=add_qtype_converts,
    )
    # get_qnn_qdq_config は extra_options を受け取らないので、
    # 返ってきた設定オブジェクト側に percentile の指定を差し込む。
    if calib_method == "percentile":
        cfg.extra_options["CalibPercentile"] = percentile
        cfg.extra_options["CalibMovingAverage"] = False

    # 2GB を超えるモデルは外部データ形式でしか保存できない。
    # 指定しないと `EncodeError: Failed to serialize proto` で落ちる
    # （落ちるだけまだよく、S2 の quant_pre_process は無言で空ファイルを残した）。
    if prepped.stat().st_size + sum(
            f.stat().st_size for f in prepped.parent.glob(prepped.name + ".data")
    ) > 1_000_000_000:
        cfg.use_external_data_format = True

    out_qdq.parent.mkdir(parents=True, exist_ok=True)
    quantize(str(prepped), str(out_qdq), cfg)

    # **「ファイルができた＝成功」と読まない。** 2GB 超のモデルでは
    # 保存系が例外を出さずに空の成果物を残すことがある（preprocess._assert_sane の注記）。
    import onnx as _onnx
    _m = _onnx.load(str(out_qdq), load_external_data=False)
    if len(_m.graph.node) == 0 or not _m.opset_import:
        raise RuntimeError(
            f"S4: 量子化の成果物が壊れている（nodes={len(_m.graph.node)}）: {out_qdq}")

    return {
        "scheme": scheme,
        "activation_type": ACTIVATION[scheme].name,
        "weight_type": QuantType.QInt8.name,
        "per_channel": True,
        "calibrate_method": "MinMax",
        "calibration_samples": len(reader),
        "init_overrides": init_overrides or {},
        "qdq_path": str(out_qdq),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepped", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scheme", default="a16w8", choices=sorted(ACTIVATION))
    ap.add_argument("--n-random", type=int, default=32)
    args = ap.parse_args()

    r = run(args.prepped, args.out, scheme=args.scheme, n_random=args.n_random)
    print(f"量子化完了 scheme={r['scheme']} "
          f"activation={r['activation_type']} weight={r['weight_type']} per_channel=True")
    print(f"  calibration samples: {r['calibration_samples']}")
    print(f"  {r['qdq_path']}")


if __name__ == "__main__":
    main()
