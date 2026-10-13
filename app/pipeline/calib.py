"""S3: キャリブレーションデータ

基本設計② S3 の実装。

原則: **キャリブレーションは実行時の入力分布を模倣する**。
ただし実データだけでは活性化の値域の端が現れず、MinMax キャリブレーションが
過小な範囲を学習してしまう。そのため**合成パターンを必ず混ぜる**
（ZipDepth が単色・勾配・市松模様を入れているのはこの理由と解釈している）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader


def model_input_spec(model_path: str) -> tuple[str, list[int], int]:
    """モデルの第1入力の (名前, 形状, elem_type) を返す。動的次元があれば例外。"""
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
    """値域の端を踏ませるための合成入力（range-pinning）。"""
    out: list[np.ndarray] = []
    # 単色（0.0〜1.0）
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(np.full(shape, v, dtype=np.float32))
    # 勾配（最後の2軸方向）
    if len(shape) >= 2:
        h, w = shape[-2], shape[-1]
        ramp_w = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        ramp_h = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
        for g in (ramp_w, ramp_h):
            out.append(np.broadcast_to(g, shape).astype(np.float32).copy())
        # 市松模様
        chk = ((np.indices((h, w)).sum(0) // max(1, h // 8)) % 2).astype(np.float32)
        out.append(np.broadcast_to(chk, shape).astype(np.float32).copy())
    return out


def random_samples(shape: Sequence[int], n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.0, 1.0, size=shape).astype(np.float32) for _ in range(n)]


class ArrayReader(CalibrationDataReader):
    """np 配列のリストを順に返す最小の CalibrationDataReader。"""

    def __init__(self, input_name: str, samples: Iterable[np.ndarray]) -> None:
        self._name = input_name
        self._data = [{input_name: s} for s in samples]
        self._i = 0

    def get_next(self):
        if self._i >= len(self._data):
            return None
        item = self._data[self._i]
        self._i += 1
        return item

    def rewind(self) -> None:
        self._i = 0

    def __len__(self) -> int:
        return len(self._data)


def build_reader(model_path: str, n_random: int = 32, seed: int = 0) -> ArrayReader:
    """既定のキャリブレーション: ランダム n 件 + 合成パターン。

    実データが用意できるモデル（BERT / LLM）では、本関数を使わず
    実入力から ArrayReader を組み立てること。
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
    """全入力の (名前, 形状, elem_type) を返す。動的次元があれば例外。"""
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
    """複数入力を同時に供給する CalibrationDataReader。"""

    def __init__(self, batches: list[dict[str, np.ndarray]]) -> None:
        self._data = batches
        self._i = 0

    def get_next(self):
        if self._i >= len(self._data):
            return None
        item = self._data[self._i]
        self._i += 1
        return item

    def rewind(self) -> None:
        self._i = 0

    def __len__(self) -> int:
        return len(self._data)


def build_multi_reader(model_path: str, n_random: int = 24, seed: int = 0,
                       int_max: dict[str, int] | None = None) -> MultiReader:
    """多入力モデル向けの既定リーダー。

    整数入力（position_ids / Gather の索引）は **値域を外すと範囲外参照**になり、
    キャリブレーション中に失敗するか無意味な統計になる。`int_max` で上限を渡す。
    ランダムではなく arange を基本にしているのは、位置索引としては
    それが実際の入力分布そのものだからである。
    """
    specs = all_input_specs(model_path)
    int_max = int_max or {}
    rng = np.random.default_rng(seed)

    n_synth = len(synthetic_patterns(specs[0]["shape"])) if specs[0]["shape"] else 0
    total = n_random + n_synth
    batches: list[dict[str, np.ndarray]] = []

    per_input: dict[str, list[np.ndarray]] = {}
    for sp in specs:
        name, shape, et = sp["name"], sp["shape"], sp["elem_type"]
        if et == onnx.TensorProto.FLOAT:
            samples = random_samples(shape, n_random, seed) + synthetic_patterns(shape)
            # 活性化の実分布は 0..1 に収まらない。負側も踏ませる
            # （0..1 だけで MinMax を取ると負の値が全部クリップされる）。
            samples = [(s * 2.0 - 1.0).astype(np.float32) for s in samples]
        else:
            dtype = _ELEM_NP.get(et, np.int64)
            n = int(np.prod(shape)) if shape else 1
            hi = int_max.get(name, max(0, n - 1))
            base = (np.arange(n) % (hi + 1)).reshape(shape).astype(dtype)
            samples = [base.copy()]
            for _ in range(max(0, total - 1)):
                samples.append(rng.integers(0, hi + 1, size=shape).astype(dtype))
        per_input[name] = samples

    n = min(len(v) for v in per_input.values())
    for i in range(n):
        batches.append({name: per_input[name][i] for name in per_input})
    return MultiReader(batches)
