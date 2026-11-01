"""S4: QDQ 量子化（本パイプラインの中核）

基本設計② S4 の実装。

**`quantize_static` に op リストを手で渡す形は使わない。**
`get_qnn_qdq_config` は QNN EP 向けの制約を織り込んだ設定を生成するため、
これを通さないと既定の op 集合で float セグメントが残り **C-1 に違反**する。

本線は A16W8（QUInt16 活性化 / QInt8 per-channel 重み）。公式ドキュメントの
標準量子化例と一致する（要件定義書 3.3.2節#17b）。

（2026-10-18 追記: 学習用の説明）
■ 何のための道具か
  前処理（S2、preprocess.py）を済ませた fp32 の ONNX に、QuantizeLinear / DequantizeLinear
  （Q/DQ）の組を挿した「QDQ モデル」を作る。HTP（NPU）は固定小数点でしか計算しないので、
  どのテンソルをどの型・scale で表すかをこの形で書いておき、後段の qairt-converter（S5）が
  それを読んで固定小数点のグラフにする（StudyMemo 1.2・2.8 節）。
■ どの段で使うか
  S4。普段は p2b_ops.prepare() の中から呼ばれる（較正の読み手・上書き指定を渡される）。
■ 入力と出力
  入力: <名前>_prepped.onnx（S2 の出力）。出力: <名前>_<方式>_qdq.onnx（2GB 近いときは
  .data を伴う外部データ形式）。戻り値の辞書は p2b_ops の記録 JSON に入る。
■ 動かし方（単体）
  python -m app.pipeline.quantize_qdq --prepped <prepped.onnx> --out <qdq.onnx> --scheme a16w8
  （単体では乱数＋合成パターンの較正になる。LLM の本番は p2b_ops prepare を使う）
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
    """前処理後のモデル prepped を量子化し、QDQ モデルを out_qdq に書く。

    引数:
      scheme            : "a16w8"（活性化 uint16）か "a8w8"（活性化 uint8）。重みは常に int8。
      reader            : 較正の読み手（calib の ArrayReader / MultiReader）。None なら乱数＋合成パターン。
      init_overrides    : テンソルごとの上書き（ORT の TensorQuantOverrides。
                          {テンソル名: [{"quant_type": ..., "axis": 1, ...}]}）。名前は prepped のもの。
      add_qtype_converts: False なら ORT に型をそろえる変換ノードを挿させない（C-8 の試験用）。
      calib_method      : "minmax"（既定）か "percentile"（percentile で上位を切る）。
      nodes_to_exclude  : 量子化しないノード名（診断専用。C-1 違反になる）。
      n_random, seed    : reader を作るときの乱数の件数と種。
    戻り値: 方式・型・較正件数・出力パスなどの辞書。

    ★注意（2026-10-18 追記）: per_channel=True を渡しているが、MatMul の重みはこれだけでは
    per-tensor のままになる（2026-09-13 まで 390 本すべて per-tensor。StudyMemo 7.2・7.4 節）。
    per-channel にするには init_overrides に {"axis": 1} を書く（weight_percg.build_overrides が作り、
    p2b_ops が spec の per_channel_weights: true のときに渡す）。
    """
    if scheme not in ACTIVATION:
        raise ValueError(f"未対応の scheme: {scheme}（{list(ACTIVATION)} のいずれか）")

    # ① 較正の読み手が渡されなければ、単一入力向けの既定（乱数＋合成パターン）を作る
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
    # ② ここまでで「どのテンソルを何 bit で、どの較正法で」という設定オブジェクトができた
    # get_qnn_qdq_config は extra_options を受け取らないので、
    # 返ってきた設定オブジェクト側に percentile の指定を差し込む。
    if calib_method == "percentile":
        cfg.extra_options["CalibPercentile"] = percentile
        cfg.extra_options["CalibMovingAverage"] = False

    # ③ 保存形式を決める。protobuf は 1 ファイル 2GB が上限。判定のしきい値を 1GB（10^9 バイト）と
    #   低めに取っているのは、Q/DQ の分だけ出力が大きくなる余裕を見たためと推定（コードからの推定）。
    #   prepped 本体と、隣の <prepped>.data（外部データ）の合計で判定する。
    # 2GB を超えるモデルは外部データ形式でしか保存できない。
    # 指定しないと `EncodeError: Failed to serialize proto` で落ちる
    # （落ちるだけまだよく、S2 の quant_pre_process は無言で空ファイルを残した）。
    if prepped.stat().st_size + sum(
            f.stat().st_size for f in prepped.parent.glob(prepped.name + ".data")
    ) > 1_000_000_000:
        cfg.use_external_data_format = True

    # ④ 量子化の本体。reader を流して最小・最大を観測し、Q/DQ を挿したモデルを書く
    out_qdq.parent.mkdir(parents=True, exist_ok=True)
    quantize(str(prepped), str(out_qdq), cfg)

    # **「ファイルができた＝成功」と読まない。** 2GB 超のモデルでは
    # 保存系が例外を出さずに空の成果物を残すことがある（preprocess._assert_sane の注記）。
    import onnx as _onnx
    _m = _onnx.load(str(out_qdq), load_external_data=False)
    if len(_m.graph.node) == 0 or not _m.opset_import:
        raise RuntimeError(
            f"S4: 量子化の成果物が壊れている（nodes={len(_m.graph.node)}）: {out_qdq}")

    # ⑤ 結果を辞書で返す（p2b_ops が記録 JSON の S4 に写す）。
    # ★注意（2026-10-18 追記）: "calibrate_method" は "MinMax" と決め打ちで、
    #   calib_method="percentile" で動かしても "MinMax" と書かれる（記録だけの食い違い。コードは直していない）。
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
    """コマンドの入口。--prepped / --out / --scheme / --n-random を受け、run() を呼んで結果を表示する。"""
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
