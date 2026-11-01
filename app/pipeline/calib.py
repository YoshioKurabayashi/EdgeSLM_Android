"""S3: キャリブレーションデータ

基本設計② S3 の実装。

原則: **キャリブレーションは実行時の入力分布を模倣する**。
ただし実データだけでは活性化の値域の端が現れず、MinMax キャリブレーションが
過小な範囲を学習してしまう。そのため**合成パターンを必ず混ぜる**
（ZipDepth が単色・勾配・市松模様を入れているのはこの理由と解釈している）。

（2026-10-18 追記: 学習用の説明）
■ 何のための部品か
  量子化（S4）の「較正（calibration）」に使う入力を作って渡す部品。較正とは、モデルに
  入力を何件か流して各テンソルの最小値・最大値を観測し、そこから量子化の scale と
  zero point を決める作業（StudyMemo 2.2 節。s = (hi - lo) / (qmax - qmin)）。
  ORT（onnxruntime）の量子化器は「CalibrationDataReader」という形の読み手から
  入力を 1 件ずつ受け取るので、その読み手（ArrayReader / MultiReader）をここで作る。
■ どの段で使うか
  - quantize_qdq.run()：reader を渡さなかったときの既定（build_reader。単一入力の CNN 用、P2）
  - p2b_ops.prepare()：spec に calib_npz（実データ）があれば MultiReader に詰めて使い、
    無ければ build_multi_reader（乱数＋合成パターン。P2b の単体 op 用）
  - oracle_exclude.py：MultiReader を直接使う
■ 入力と出力
  入力: 量子化前（前処理後）の ONNX のパス。出力: CalibrationDataReader（get_next() が
  {入力名: np.ndarray} を 1 件ずつ返し、尽きたら None）。
■ 動かし方
  単体のコマンドは無い（他の道具から import して使う）。
■ 注意（StudyMemo 2.2 節「較正データが的外れだとどうなるか」）
  ここで作る乱数の較正は ±1.0 の一様分布だが、本物の Qwen の hidden states は ±0.09
  （std 0.0156）で、約 11 倍広かった。層 1 枚の相対 L2 は実データ較正 30.4% に対し
  乱数較正 38.9%。LLM では make_calib_data.py の実データ（calib_npz）を使うこと。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader


def model_input_spec(model_path: str) -> tuple[str, list[int], int]:
    """モデルの第1入力の (名前, 形状, elem_type) を返す。動的次元があれば例外。

    形状は [1, 3, H, W] のような int のリスト。elem_type は onnx.TensorProto の番号（1=FLOAT など）。
    初期化子（重み）も graph.input に並ぶ古い形式に備えて、初期化子の名前は除く。
    """
    model = onnx.load(model_path)
    inits = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in inits]
    if not inputs:
        raise RuntimeError("モデルに入力がない")
    vi = inputs[0]
    shape: list[int] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            shape.append(d.dim_value)
        else:
            # QNN EP は動的形状を扱えない（要件定義書 3.3.2節#11）。
            # ここで弾かないと、変換後に分かりにくい失敗になる。
            raise RuntimeError(
                f"入力 '{vi.name}' に動的次元がある: {vi.type.tensor_type.shape}. "
                "QNN EP は固定形状しか扱えないため、export 時に固定すること"
            )
    return vi.name, shape, vi.type.tensor_type.elem_type


def synthetic_patterns(shape: Sequence[int]) -> list[np.ndarray]:
    """値域の端を踏ませるための合成入力（range-pinning）。

    引数 shape: 入力の形（例 [1, 3, H, W]）。戻り値: float32 の配列のリスト（5〜8 枚）。
    中身は ① 単色 5 枚（0.0〜1.0）② 横・縦の勾配 2 枚 ③ 市松模様 1 枚（②③は最後の
    2 軸がある場合だけ）。乱数だけだと 0 や 1 ちょうどの値がほとんど出ず、較正で観測する
    最小値・最大値が実際より狭くなる（範囲の外は量子化で頭打ちになる）ため、端を必ず入れる。
    """
    out: list[np.ndarray] = []
    # 単色（0.0〜1.0）
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(np.full(shape, v, dtype=np.float32))
    # 勾配（最後の2軸方向）
    if len(shape) >= 2:
        h, w = shape[-2], shape[-1]
        # ② 勾配: 0→1 をなめらかに並べ、中間の値もまんべんなく踏ませる（横方向と縦方向の 2 枚）
        ramp_w = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        ramp_h = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
        for g in (ramp_w, ramp_h):
            out.append(np.broadcast_to(g, shape).astype(np.float32).copy())
        # 市松模様
        # ③ 1 マスの一辺は h // 8 画素（最低 1）。0 と 1 が隣り合う急な変化を作る
        chk = ((np.indices((h, w)).sum(0) // max(1, h // 8)) % 2).astype(np.float32)
        out.append(np.broadcast_to(chk, shape).astype(np.float32).copy())
    return out


def random_samples(shape: Sequence[int], n: int, seed: int = 0) -> list[np.ndarray]:
    """[0, 1) の一様乱数を n 件作る（形は shape、float32）。seed を固定して毎回同じ較正にする。"""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.0, 1.0, size=shape).astype(np.float32) for _ in range(n)]


class ArrayReader(CalibrationDataReader):
    """np 配列のリストを順に返す最小の CalibrationDataReader。

    単一入力のモデル用。samples の各配列を {input_name: 配列} の辞書にして持ち、
    get_next() で 1 件ずつ返す。
    """

    def __init__(self, input_name: str, samples: Iterable[np.ndarray]) -> None:
        self._name = input_name
        self._data = [{input_name: s} for s in samples]
        self._i = 0

    def get_next(self):
        """次の 1 件（{入力名: 配列}）を返す。尽きたら None（ORT はこれで較正を終える）。"""
        if self._i >= len(self._data):
            return None
        item = self._data[self._i]
        self._i += 1
        return item

    def rewind(self) -> None:
        """先頭へ戻す（ORT の較正器が読み手を読み直すときに呼ぶ）。"""
        self._i = 0

    def __len__(self) -> int:
        """件数。quantize_qdq.run() が結果の calibration_samples に書くのに使う。"""
        return len(self._data)


def build_reader(model_path: str, n_random: int = 32, seed: int = 0) -> ArrayReader:
    """既定のキャリブレーション: ランダム n 件 + 合成パターン。

    実データが用意できるモデル（BERT / LLM）では、本関数を使わず
    実入力から ArrayReader を組み立てること。

    引数: model_path（ONNX のパス）、n_random（乱数の件数）、seed（乱数の種）。
    戻り値: ArrayReader（乱数 n_random 件 ＋ 合成パターン 5〜8 件）。
    単一入力のモデル（P2 の CNN）向け。入力が複数あるモデルは build_multi_reader を使う。
    """
    name, shape, _ = model_input_spec(model_path)
    samples = random_samples(shape, n_random, seed) + synthetic_patterns(shape)
    return ArrayReader(name, samples)


# ------------------------------------------------------------------
# 多入力対応（P2b で追加）
#
# 単体 op の検証では attention の QK^T（両辺が活性化）や KVキャッシュ
# （past_k / past_v）のように **入力が複数ある**モデルを扱う。
# 上の単一入力 API は P2 の実績があるためそのまま残し、ここに別途足す。
# ------------------------------------------------------------------

_ELEM_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
}


def all_input_specs(model_path: str) -> list[dict]:
    """全入力の (名前, 形状, elem_type) を返す。動的次元があれば例外。

    戻り値: [{"name": 入力名, "shape": [int, ...], "elem_type": onnx の型番号}, ...]（グラフの並び順）。
    p2b_ops.prepare() は、実データの npz からモデルの入力に当たるキーだけを選ぶのにも使う。
    """
    model = onnx.load(model_path)
    inits = {i.name for i in model.graph.initializer}
    out: list[dict] = []
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        shape: list[int] = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                raise RuntimeError(
                    f"入力 '{vi.name}' に動的次元がある。QNN EP は固定形状しか扱えない")
        out.append({"name": vi.name, "shape": shape,
                    "elem_type": vi.type.tensor_type.elem_type})
    if not out:
        raise RuntimeError("モデルに入力がない")
    return out


class MultiReader(CalibrationDataReader):
    """複数入力を同時に供給する CalibrationDataReader。

    batches は [{入力名: 配列, ...}, ...]（1 要素が 1 回ぶんの入力のそろい）。
    p2b_ops は実データの npz を 1 件ずつ切り出して（形 [1, ...]）ここに詰める。
    """

    def __init__(self, batches: list[dict[str, np.ndarray]]) -> None:
        self._data = batches
        self._i = 0

    def get_next(self):
        """次の 1 件（{入力名: 配列}）を返す。尽きたら None（ORT はこれで較正を終える）。"""
        if self._i >= len(self._data):
            return None
        item = self._data[self._i]
        self._i += 1
        return item

    def rewind(self) -> None:
        """先頭へ戻す（ORT の較正器が読み手を読み直すときに呼ぶ）。"""
        self._i = 0

    def __len__(self) -> int:
        """件数。quantize_qdq.run() が結果の calibration_samples に書くのに使う。"""
        return len(self._data)


def build_multi_reader(model_path: str, n_random: int = 24, seed: int = 0,
                       int_max: dict[str, int] | None = None) -> MultiReader:
    """多入力モデル向けの既定リーダー。

    整数入力（position_ids / Gather の索引）は **値域を外すと範囲外参照**になり、
    キャリブレーション中に失敗するか無意味な統計になる。`int_max` で上限を渡す。
    ランダムではなく arange を基本にしているのは、位置索引としては
    それが実際の入力分布そのものだからである。

    引数: model_path（ONNX のパス）、n_random（乱数の件数）、seed、
          int_max（{整数入力の名前: 取りうる最大値}。p2b_ops は spec の inputs[].int_max から作る）。
    戻り値: MultiReader（件数は n_random ＋ 合成パターンの枚数）。
    float 入力は [-1, 1] の乱数と合成パターン、整数入力は arange 1 件 ＋ 乱数。
    """
    specs = all_input_specs(model_path)
    int_max = int_max or {}
    rng = np.random.default_rng(seed)

    # ① 何件そろえるか: 乱数 n_random 件 ＋ 合成パターンの枚数（先頭の入力の形で数える）
    n_synth = len(synthetic_patterns(specs[0]["shape"])) if specs[0]["shape"] else 0
    total = n_random + n_synth
    batches: list[dict[str, np.ndarray]] = []

    # ② 入力ごとに較正用の配列を作る（float は値の範囲、整数は索引の範囲を踏ませる）
    per_input: dict[str, list[np.ndarray]] = {}
    for sp in specs:
        name, shape, et = sp["name"], sp["shape"], sp["elem_type"]
        if et == onnx.TensorProto.FLOAT:
            samples = random_samples(shape, n_random, seed) + synthetic_patterns(shape)
            # 活性化の実分布は 0..1 に収まらない。負側も踏ませる
            # （0..1 だけで MinMax を取ると負の値が全部クリップされる）。
            samples = [(s * 2.0 - 1.0).astype(np.float32) for s in samples]
        else:
            # 整数入力（索引）: 1 件目は 0,1,2,… を hi で折り返した列、残りは 0..hi の乱数。
            # hi の既定は「要素数 - 1」（int_max が無いとき）
            dtype = _ELEM_NP.get(et, np.int64)
            n = int(np.prod(shape)) if shape else 1
            hi = int_max.get(name, max(0, n - 1))
            base = (np.arange(n) % (hi + 1)).reshape(shape).astype(dtype)
            samples = [base.copy()]
            for _ in range(max(0, total - 1)):
                samples.append(rng.integers(0, hi + 1, size=shape).astype(dtype))
        per_input[name] = samples

    # ③ 入力ごとの件数が違うときは短い方にそろえ、i 番目どうしを 1 回ぶんの入力にまとめる
    n = min(len(v) for v in per_input.values())
    for i in range(n):
        batches.append({name: per_input[name][i] for name in per_input})
    return MultiReader(batches)
