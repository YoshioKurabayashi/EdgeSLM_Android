"""S2: 前処理（BatchNorm 畳み込み → quant_pre_process）

基本設計② S2 の実装。

BatchNorm 畳み込みは**最適化ではなく「通すために必須」**の措置である。
QNN HTP は signed per-channel の Conv QDQ と BatchNorm の併存を拒否し
**error 3110** になる（要件定義書 3.4節 C-5）。

推論時の BatchNorm は gamma / beta / mean / var すべて定数なので、
チャネルごとの一次式に還元できる:

    y = gamma * (x - mean) / sqrt(var + eps) + beta
      = scale * x + shift
        where scale = gamma / sqrt(var + eps)
              shift = beta - mean * scale

順序は「BN 畳み込み → quant_pre_process」。畳み込みで生まれた Mul/Add を
定数畳み込みがさらに簡約でき、形状推論は最終トポロジに対して走らせるべきため。

（2026-10-18 追記: 学習用の説明）
■ いまの S2 の中身
  BatchNorm の畳み込みに加えて、NPU（HTP v68）で通す・精度を保つための「答えは同じで途中計算だけ
  変える」書き換え（Graph Rewrite）をここで行う（StudyMemo 5 章）。run() が呼ぶ順は
    ① clamp_extreme_constants      C-9  マスクの -3.4e38 を -60 へ
    ② rewrite_expand_to_concat     C-10 Expand を Concat の繰り返しへ
    ③ fold_scale_into_matmul       スケールを MatMul の前へ（効果は無かったが等価なので既定で入る）
    ④ stabilize_softmax            既定 off（悪化したため）
    ⑤ swap_actact_matmul_operands  C-8 QK^T の左右入替（swap_matmuls に名前があるものだけ）
    ⑥ normalize_rmsnorm_input      C-11 RMSNorm を max で割ってから計算（norm_rmsnorm=True のとき）
  の後に BatchNorm の畳み込み → quant_pre_process（ORT の最適化と形状推論）。
■ 入力と出力
  入力: export した fp32 の ONNX。出力: 書き換え後（*_clamped.onnx）・BN 畳み込み後（folded）・
  前処理後（prepped。S4 の入力）の ONNX と、何を書き換えたかの要約（辞書）。
■ 動かし方
  本番は p2b_ops prepare の中から run() が呼ばれる（方式名に rms・swap があると ⑤⑥ が入る）。
  単体: python -m app.pipeline.preprocess --src <model.onnx> --workdir <出力フォルダ>
  （単体では ①〜③ だけ。⑤⑥ は入らない）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.shape_inference import quant_pre_process


def _unique(base: str, inits: dict, graph: onnx.GraphProto) -> str:
    """既存の initializer / tensor 名と衝突しない名前を返す。

    base が空いていればそのまま、使われていれば base_1, base_2, … と番号を足す。
    inits（これから足す予定の名前の辞書）・初期化子・ノードの出力名のどれとも重ならないようにする。
    """
    taken = set(inits) | {i.name for i in graph.initializer}
    for n in graph.node:
        taken.update(n.output)
    if base not in taken:
        return base
    i = 1
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


# 量子化できない極端な値の既定の置き換え先。
#
# 因果マスクは softmax の前にスコアへ足される。遮断できればよいので、
# exp(x) が実質 0 になる程度の有限値で足りる（exp(-60) ≒ 8.8e-27）。
# 小さくしすぎると遮断が甘くなり、大きくしすぎると量子化の分解能を食う:
#   本来のスコア範囲が ±20 程度のとき
#     置き換え -60  → 範囲 [-60, 20]  16bit で 1段階 0.0012
#     置き換え -1e4 → 範囲 [-1e4, 20] 16bit で 1段階 0.15    ← 実データが潰れる
MASK_REPLACE = -60.0
EXTREME = 1e30


def clamp_extreme_constants(model: onnx.ModelProto,
                            replace_with: float = MASK_REPLACE,
                            threshold: float = EXTREME) -> list[dict]:
    """initializer 中の極端な値（±inf / ±3.4e38 等）を有限値へ置き換える。

    **なぜ必要か**（2026-09-12 に本物の Qwen2.5-0.5B を export して判明）:
    HuggingFace の因果マスクは `-3.4028235e+38`（float32 の最小値）を使う。
    量子化は「最小値〜最大値を N 段階に割る」処理なので、この値が1つでも混ざると
    **全分解能がそこに食われ、実際のスコアが丸ごと潰れる**。

    グラフの意味は変わらない。softmax を通せばどちらも 0 になるためである。
    **変えたことは必ず記録する**（黙って値を書き換えるのが一番危ない）。
    """
    # 引数: model（その場で書き換える）、replace_with（負の極端値の置き換え先。正の極端値は符号を反転した値）、
    #       threshold（絶対値がこれ以上なら「極端」とみなす。既定 1e30）。戻り値: 置き換えた初期化子ごとの記録。
    # 見るのは初期化子（重み・定数の表）だけで、Constant ノードの中の値は見ない（コードからの読み取り）
    changed: list[dict] = []
    for init in model.graph.initializer:
        if init.data_type not in (onnx.TensorProto.FLOAT,
                                  onnx.TensorProto.DOUBLE,
                                  onnx.TensorProto.FLOAT16):
            continue
        arr = numpy_helper.to_array(init)
        if arr.size == 0:
            continue
        neg = np.isneginf(arr) | (arr <= -threshold)
        pos = np.isposinf(arr) | (arr >= threshold)
        if not (neg.any() or pos.any()):
            continue
        new = arr.astype(np.float64).copy()
        new[neg] = replace_with
        new[pos] = -replace_with
        new = new.astype(arr.dtype)
        init.CopyFrom(numpy_helper.from_array(new, init.name))
        changed.append({
            "initializer": init.name,
            "shape": list(arr.shape),
            "n_negative": int(neg.sum()),
            "n_positive": int(pos.sum()),
            "original_min": float(arr.min()),
            "replaced_with": replace_with,
        })
    return changed


def rewrite_expand_to_concat(model: onnx.ModelProto) -> list[dict]:
    """`Expand` を `Concat` の繰り返しへ書き換える（C-10）。

    **なぜ必要か**（2026-09-12 に実機で切り分け）:
    HTP v68 は `Expand` を含む context binary を**ロードできない**（err 1002）。
    ホスト側の `qnn-context-binary-generator` は素通しするため、
    **端末に持って行くまで分からない**のが厄介な点である。

    切り分けの根拠（単体モデルを1つずつ実機で試した）:
      unsqueeze_only  … OK      → Unsqueeze は無罪
      expand_only     … err1002 → **Expand が単独で原因**
      gqa_concat      … OK      → Concat は代替として使える

    `Expand` が「サイズ1の軸を n 倍に広げる」形なら、同じテンソルを n 個
    `Concat` するのと**結果は完全に同じ**である。GQA のヘッド複製
    （HuggingFace の export が `Unsqueeze`→`Expand`→`Reshape` で出す形）が
    まさにこれに当たる。

    ブロードキャストが「サイズ1の軸を広げる」以外の形（次元数そのものが増える等）
    の場合は**書き換えず残す**。黙って別の意味のグラフにするより、
    端末で落ちて気付くほうがよい。
    """
    # 引数: model（その場で書き換える）。戻り値: 書き換えた Expand ごとの記録（入力の形・目標・Concat の軸と回数）。
    # ① 形状推論で各テンソルの形を集める。失敗したら元のモデルのまま（グラフの入出力と初期化子の形だけ）で進む。
    #   ★注意（2026-10-18 追記。コードからの読み取り・推定）: 2GB を超えるモデルは形状推論そのものが失敗し、
    #   中間テンソルの形が分からないので、Expand は「形が分からない」として書き換えずに残る。
    #   戻り値（expand_rewrites）の件数で、書き換えが本当に起きたかを確かめること
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model

    shapes: dict[str, list[int]] = {}
    for coll in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info):
        for vi in coll:
            dims = [d.dim_value if d.HasField("dim_value") else -1
                    for d in vi.type.tensor_type.shape.dim]
            shapes[vi.name] = dims
    for init in inferred.graph.initializer:
        shapes[init.name] = list(init.dims)

    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    g = model.graph
    new_nodes: list[onnx.NodeProto] = []
    changed: list[dict] = []

    # ② Expand を 1 つずつ見て、「サイズ 1 の軸を n 倍に広げる」形なら Concat に置き換える。
    #   target（広げた後の形）は初期化子で与えられている場合だけ扱う
    for node in g.node:
        if node.op_type != "Expand" or len(node.input) < 2:
            new_nodes.append(node)
            continue
        src, shape_name = node.input[0], node.input[1]
        if shape_name not in inits or src not in shapes:
            new_nodes.append(node)   # 形が静的に分からない。触らない
            continue

        in_shape = shapes[src]
        target = [int(v) for v in inits[shape_name].tolist()]
        if len(in_shape) != len(target) or any(d < 0 for d in in_shape):
            new_nodes.append(node)
            continue

        # ★target は**ブロードキャスト形式**でありうる。
        #   実際 dynamo の export は `[1,2,1,16,64]` を `[1,1,7,1,1]` で広げる形を出した
        #   （結果は `[1,2,7,16,64]`）。target をそのまま出力形状と見なすと
        #   「軸1が 2 vs 1 で不一致」と誤判定して**書き換えを取りこぼす**
        #   （2026-09-13 に踏んだ。54分の context 生成を無駄にしかけた）。
        #   ブロードキャスト結果を自分で計算する。
        out_shape = [max(a, t) for a, t in zip(in_shape, target)]
        axes = [(i, o) for i, (a, o) in enumerate(zip(in_shape, out_shape))
                if a == 1 and o > 1]
        if not axes or any(a != o and a != 1 for a, o in zip(in_shape, out_shape)):
            new_nodes.append(node)
            continue

        cur = src
        for k, (axis, rep) in enumerate(axes):
            out = node.output[0] if k == len(axes) - 1 else f"{node.output[0]}_cat{k}"
            new_nodes.append(helper.make_node(
                "Concat", [cur] * rep, [out],
                f"{node.name or 'expand'}_to_concat{k}", axis=axis))
            cur = out
        changed.append({
            "node": node.name or "(no name)",
            "input_shape": in_shape,
            "target_shape": target,
            "broadcast_to": out_shape,
            "concat_axes": [{"axis": a, "repeat": r} for a, r in axes],
        })

    # ③ ノードの並びを入れ替え、2GB 以下なら ONNX の検査器で形式を確かめる
    if changed:
        del g.node[:]
        g.node.extend(new_nodes)
        if _fits_protobuf(model):
            onnx.checker.check_model(model)
    return changed


def fold_scale_into_matmul(model: onnx.ModelProto) -> list[dict]:
    """`Mul(MatMul(a, b), s)` を `MatMul(Mul(a, s), b)` へ移す（s はスカラー定数）。

    **なぜ効くか**（2026-09-12 実測）: attention は QK^T の**後**に 1/sqrt(head_dim)
    を掛ける形で export される。ところが C-8 の回避で QK^T の出力は 8bit に
    落とさざるを得ず、**スケール前の広いレンジのまま量子化される**:

        matmul_1（QK^T 生値）  実測 [-183, +12515]  → 8bit で 1段階 50.9
        mul_8  （スケール後）  実測 [ -23,  +1564]

    スケールを MatMul の**前**へ移せば、量子化されるのはスケール後の値になり、
    同じ 8bit でも**分解能が 8倍**になる。`(a·s)×b = (a×b)·s` なので**厳密に等価**で、
    近似は一切していない。

    条件を満たさない形（s がテンソル、Mul が MatMul 以外を受ける等）は触らない。

    ★2026-10-18 注: 上の「なぜ効くか」は当時の見込みで、実測では効果が無かった
    （層 1 枚の相対 L2 30.3819% → 30.3904%。StudyMemo 5.6 節）。量子化されるのはスケール後の値
    （mul_8）で、Mul を前へ移してもスケール後のレンジは変わらないため。厳密に等価で害も無いので、
    run() の既定（fold_scale=True）で今も本番にかかっている（StudyMemo 5.0 節）。
    引数: model（その場で書き換える）。戻り値: 移した組ごとの記録（MatMul・Mul の名前とスカラーの値）。
    """
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    changed: list[dict] = []
    for node in list(g.node):
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        scalar = [i for i in node.input if i in inits and inits[i].size == 1]
        other = [i for i in node.input if i not in inits]
        if len(scalar) != 1 or len(other) != 1:
            continue
        mm = producer.get(other[0])
        if mm is None or mm.op_type != "MatMul":
            continue
        # MatMul の出力がこの Mul 以外にも使われていたら触らない（意味が変わる）
        if len(consumers.get(other[0], [])) != 1:
            continue
        lhs = mm.input[0]
        if lhs in inits:
            continue  # 重み側に掛けるのは別の話なので対象外

        # Mul を MatMul の前へ移す。
        #   変更前: MatMul(q,k) -> matmul_1 ; Mul(matmul_1,s) -> mul_8
        #   変更後: Mul(q,s) -> q_prescaled ; MatMul(q_prescaled,k) -> mul_8
        # **下流が参照している名前（mul_8）は MatMul 側へ引き継ぐ**。
        # 先に node.output を書き換えてしまうと元の名前を失う（一度これで壊した）。
        old_mul_out = node.output[0]
        pre = f"{lhs}_prescaled"
        node.input[0], node.input[1] = lhs, scalar[0]
        node.output[0] = pre
        mm.input[0] = pre
        mm.output[0] = old_mul_out
        changed.append({"matmul": mm.name or "(no name)",
                        "mul": node.name or "(no name)",
                        "scalar": float(inits[scalar[0]].reshape(-1)[0])})

    # ノードの付け替えで「使う前に作られている」順（トポロジ順）が崩れるので並べ直す。
    # 入力が全部そろったノードから順に取り出す（カーンの方法の素朴な形）
    if changed:
        # Mul が MatMul より前に来るようトポロジ順を直す
        order, seen = [], set()
        avail = {i.name for i in g.initializer} | {v.name for v in g.input}
        pending = list(g.node)
        while pending:
            progress = False
            for n in list(pending):
                if all(i in avail or i == "" for i in n.input):
                    order.append(n)
                    avail.update(n.output)
                    pending.remove(n)
                    progress = True
            if not progress:
                order.extend(pending)   # 解けない場合は諦めて元の順を保つ
                break
        del g.node[:]
        g.node.extend(order)
        if _fits_protobuf(model):
            onnx.checker.check_model(model)
    return changed


def stabilize_softmax(model: onnx.ModelProto) -> list[dict]:
    """`Softmax(x)` を `Softmax(x - ReduceMax(x))` へ書き換える（診断用。既定では使わない）。

    ★2026-10-18 訂正: 旧見出しは「（C-11）」。要件定義書の C-11 は RMSNorm の書き換え
    （`normalize_rmsnorm_input`）で、この softmax の書き換えに C の番号は無い。
    `run(..., stabilize=True)` のときだけかかる（今のパイプラインに True で呼ぶ所は無い）。

    **softmax は行ごとに定数を引いても値が変わらない**（`softmax(x) = softmax(x-c)`）。
    fp32 では数値安定化のための定石だが、**量子化ではもっと効く**。

    実測（Qwen2.5-0.5B 層0、2026-09-12）:

        attention スコアのレンジ            [-26.70, +1600.39]  幅 1627
        行ごとの最大値を引いた後             [-65.72,     0.00]  幅   66
                                                        ↑ 25分の1

    C-8 の回避でスコアは 8bit にせざるを得ない。1段階の幅が

        そのまま        6.38  → softmax 後の誤差 62.8% / argmax 一致 29.2%
        行最大を引く    0.26  → softmax 後の誤差  3.2% / argmax 一致 90.5%

    **スコアそのものの L2 誤差は 0.49% しかない**（大きい値がノルムを支配するため）。
    テンソル単体の誤差だけ見ていると、attention が壊れていることに気付けない。
    **量子化の良し悪しは「その先で何に使われるか」まで見て判断する。**

    （2026-10-18 追記）ただしパイプラインに入れると層全体の相対 L2 は 30.4% → 41.6% に悪化した。
    スコアは MatMul の出力の時点で既に 8bit になっており、後で行最大を引いても手遅れで、足した
    ReduceMax / Sub の量子化誤差だけが乗るため（StudyMemo 5.6 節、run() の ④ の注記）。
    引数: model（その場で書き換える）。戻り値: 書き換えた Softmax ごとの記録（名前と軸）。
    """
    g = model.graph
    new_nodes: list[onnx.NodeProto] = []
    changed: list[dict] = []
    existing = {n.name for n in g.node}

    for node in g.node:
        if node.op_type != "Softmax":
            new_nodes.append(node)
            continue
        axis = -1
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
        src = node.input[0]
        base = node.name or f"softmax_{len(changed)}"
        ax_name = _unique(f"{base}_axes", {}, g)
        g.initializer.append(numpy_helper.from_array(
            np.array([axis], dtype=np.int64), ax_name))
        rmax = f"{base}_rowmax"
        stab = f"{base}_stab"
        # opset 18 では axes は入力。既存ノードと同じ書き方に揃えないと
        # `quant_pre_process` の symbolic shape inference が落ちる
        # （`noop_with_empty_axes` を省くと len(None) で死ぬ）。
        new_nodes.append(helper.make_node(
            "ReduceMax", [src, ax_name], [rmax], f"{base}_reducemax",
            keepdims=1, noop_with_empty_axes=0))
        new_nodes.append(helper.make_node(
            "Sub", [src, rmax], [stab], f"{base}_sub"))
        node.input[0] = stab
        new_nodes.append(node)
        changed.append({"softmax": base, "axis": axis})
        existing.add(base)

    if changed:
        del g.node[:]
        g.node.extend(new_nodes)
        if _fits_protobuf(model):
            onnx.checker.check_model(model)
    return changed


# ---- 保存と健全性の検査の小道具（2GB を超えるモデルの扱い）----
# protobuf の 2GB 制限。これを超えるモデルは外部データ形式でしか保存できない。
PROTOBUF_LIMIT = 2_000_000_000


def _fits_protobuf(model: onnx.ModelProto) -> bool:
    """protobuf の 2GB 上限に収まるか。

    ★`ByteSize()` は 2GB を超えると**値を返さず `EncodeError` を投げる**
    （2026-09-14: Qwen3-0.6B は重み 2.38GB で、判定そのもので落ちた）。
    上限チェックのつもりが例外になるので、包んで False を返す。
    """
    try:
        return model.ByteSize() <= PROTOBUF_LIMIT
    except Exception:       # noqa: BLE001 - 大きすぎる = 収まらない
        return False


def _save(model: onnx.ModelProto, path: Path) -> None:
    """2GB を超えるモデルも安全に保存する。

    **onnx / onnxruntime の保存系は 2GB を超えると黙って壊れた成果物を残す**
    ことがある（2026-09-12: `quant_pre_process` が 47 バイトのモデルを出力し、
    例外も出さなかった。後段の `get_qnn_qdq_config` が
    `StopIteration`（opset_import が空）で落ちて初めて気付いた）。
    大きいモデルは最初から外部データ形式で書く。
    """
    # 外部データ形式: 1024 バイト以上の初期化子を <path>.data の 1 ファイルにまとめ、.onnx には場所だけ書く
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _fits_protobuf(model):
        onnx.save(model, str(path), save_as_external_data=True,
                  all_tensors_to_one_file=True, location=path.name + ".data",
                  size_threshold=1024)
    else:
        onnx.save(model, str(path))


def _assert_sane(path: Path, where: str) -> onnx.ModelProto:
    """成果物が壊れていないことを**必ず**確かめる。

    「ファイルができた＝成功」と読まない。2GB 超で黙って空モデルが残る
    ケースを実際に踏んだため、ノード数と opset を門にする。
    """
    m = onnx.load(str(path), load_external_data=False)
    if len(m.graph.node) == 0 or not m.opset_import:
        raise RuntimeError(
            f"{where}: 成果物が壊れている（nodes={len(m.graph.node)} "
            f"opset={len(m.opset_import)}）: {path} ({path.stat().st_size} bytes)。"
            " 2GB 超のモデルを単一ファイルで保存しようとした可能性が高い")
    return m


def swap_actact_matmul_operands(model: onnx.ModelProto,
                                only: set[str] | None = None) -> list[dict]:
    """`A × B` を `(B^T × A^T)^T` へ書き換える（C-8 の 8bit 側を選ぶため）。

    **なぜ必要か**: v68 の MatMul は非対称で **`u16 × u8` しか通らない**（C-8）。
    つまり **8bit にせざるを得ないのは常に右辺**である。ところが attention の
    QK^T では右辺が K で、K のレンジは実測 ±130（1段階 0.99）と広い。
    一方 左辺の Q は [-10, 6]（1段階 0.064）で**15倍細かい**。

    `A × B = (B^T × A^T)^T` は厳密に等価なので、これで**左右を入れ替えれば
    8bit 側を K から Q にできる**。代償は Transpose が3つ増えること。
    （2026-10-18 追記）効果は実測済み: end-to-end top-1 66.5% → 80.7%、実機 17.42ms → 14.29ms
    （Transpose を足したのに速い。要件定義書 C-8、StudyMemo 5.3 節）。

    `only` にノード名を渡すと、その MatMul だけを書き換える
    （PV の MatMul は右辺 V のレンジが元々狭いので、触らない方がよい）。
    名前は export ごとに変わるので、find_swap_matmuls.py でグラフから選んで spec に書く。

    引数: model（その場で書き換える）、only（対象の MatMul のノード名の集合。None なら act×act の全部）。
    戻り値: 入れ替えた MatMul ごとの記録（元の左辺・右辺・テンソルの階数）。
    """
    # ① 各テンソルの階数（次元の数）を集める。最後の 2 軸を入れ替える Transpose の perm を作るのに使う。
    #   ★注意（2026-10-18 追記。推定）: 形状推論が失敗すると（2GB 超など）中間テンソルの階数が分からず、
    #   下の「r < 2 なら触らない」で入替が黙って飛ばされうる。戻り値の件数で確かめること
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    rank: dict[str, int] = {}
    for coll in (inferred.graph.input, inferred.graph.output, inferred.graph.value_info):
        for vi in coll:
            rank[vi.name] = len(vi.type.tensor_type.shape.dim)
    for init in inferred.graph.initializer:
        rank[init.name] = len(init.dims)

    g = model.graph
    inits = {i.name for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def is_weight(t: str) -> bool:
        """**形だけ変える op を辿って**大元が initializer かを見る。

        ★「入力名が initializer か」だけでは足りない（2026-09-13 に踏んだ）。
          lm_head の重みは `Transpose` を経て MatMul に入るため、
          素朴な判定をすり抜けて**重み側の MatMul まで左右入替してしまい**、
          `Invalid attribute perm {0,2,1}, input shape = {896, 151936}` で落ちた。
        """
        seen = 0
        while t in prod and seen < 6:
            n2 = prod[t]
            if n2.op_type not in ("Transpose", "Reshape", "Identity", "Cast",
                                  "Unsqueeze", "Squeeze"):
                return False
            t = n2.input[0]
            seen += 1
        return t in inits

    # ② 対象の MatMul（重みを含まない・only に名前がある）を
    #   Transpose(B) と Transpose(A) → MatMul(B^T, A^T) → Transpose(結果) の 4 ノードに置き換える。
    #   最後の Transpose の出力名は元の MatMul の出力名にするので、下流は付け替え不要
    new_nodes: list[onnx.NodeProto] = []
    changed: list[dict] = []

    for node in g.node:
        if (node.op_type != "MatMul" or any(is_weight(i) for i in node.input)
                or (only is not None and (node.name or "") not in only)):
            new_nodes.append(node)
            continue
        a, b = node.input[0], node.input[1]
        r = rank.get(a) or rank.get(b) or 0
        if r < 2:
            new_nodes.append(node)
            continue
        perm = list(range(r))
        perm[-1], perm[-2] = perm[-2], perm[-1]

        base = node.name or f"mm{len(changed)}"
        at, bt, st = f"{base}_At", f"{base}_Bt", f"{base}_scoresT"
        new_nodes.append(helper.make_node("Transpose", [b], [bt], f"{base}_tr_B", perm=perm))
        new_nodes.append(helper.make_node("Transpose", [a], [at], f"{base}_tr_A", perm=perm))
        # B^T × A^T = (A × B)^T
        new_nodes.append(helper.make_node("MatMul", [bt, at], [st], f"{base}_swapped"))
        new_nodes.append(helper.make_node("Transpose", [st], [node.output[0]],
                                          f"{base}_tr_out", perm=perm))
        changed.append({"matmul": base, "lhs": a, "rhs": b,
                        "new_rhs": f"{a} (転置)", "rank": r})

    if changed:
        del g.node[:]
        g.node.extend(new_nodes)
        if _fits_protobuf(model):
            onnx.checker.check_model(model)
    return changed


def normalize_rmsnorm_input(model: onnx.ModelProto, eps_new: float = 1e-12) -> list[dict]:
    """RMSNorm を「トークンごとの最大値で割ってから」計算する形へ書き換える（C-11）。

    **なぜ必要か**（2026-09-13 の oracle 実験で確定）:
    24層が壊れる原因は残差ではなく **RMSNorm の内部**だった。
    量子化を外す箇所を変えて精度を測ると:

        全部量子化            top-1  6.2%
        残差だけ除外          top-1  6.2%   ← 全く戻らない
        **RMSNorm だけ除外**  top-1 **74.4%**  ← ここが主因

    理由は `x²` のレンジである。残差には少数の巨大チャンネル（±1677）があるため

        x²  のレンジ = [0, 約280万]  →  uint16 で 1段階 **42.7**

    となり、**通常チャンネルの x²（0.04 程度）が丸ごと 0 に潰れる**。
    トークンによって巨大値の有無が違うので、巨大値を持たないトークンでは
    `mean(x²)` が壊れ、正規化の分母そのものが狂う。

    **RMSNorm はスケール不変**である:

        x / sqrt(mean(x²))  =  (x/m) / sqrt(mean((x/m)²))

    そこで **m = トークンごとの max|x|** で割ってから二乗する:

        変更前: Pow(x,2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul(x, ·)
        変更後: m  = ReduceMax(Abs(x))
                xs = x * (1/m)                     ← xs ∈ [-1, 1]
                Pow(xs,2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul(xs, ·)

    `xs²` は [0,1] に収まるので 1段階 1.5e-05 になる（**280万倍の改善**）。
    さらに `mean(xs²) >= 1/C`（最大要素が必ず 1 を取るため）なので
    **分母がゼロに潰れる心配も無い**。eps は元の 1e-6 のままだと m² 倍されて
    効きすぎるので、小さい値（既定 1e-12）に置き換える。

    数学的には厳密に等価だが、**m が 0 のトークン**（全要素ゼロ）では
    元の式と挙動が変わる。実モデルでは起きないが、起きれば NaN になるので
    `_base` との数値比較で必ず確認すること。
    ★2026-10-18 訂正: 旧「数学的には厳密に等価」「起きれば NaN になる」。
      ・eps を固定の 1e-12 に置き換えているので厳密な等価ではない（厳密には eps/m² が要る）。
        fp32 で書き換え前後を比べると相対 L2 0.22%・top-1 100%（StudyMemo 5.2 節、要件定義書 C-11）。
      ・m には下限 1e-6 を足している（下の「★max に小さい下限を足す」。要件定義書 改訂 3.2 の ④）ので、
        全ゼロのトークンでも 1/m は有限で、出力は 0 になる（NaN にはならない）。

    引数: model（その場で書き換える）、eps_new（置き換える eps。既定 1e-12）。
    戻り値: 書き換えた RMSNorm ごとの記録（Pow の名前・入力 x・差し替えた Mul の数）。
    export された RMSNorm は 1 つの op ではなく Pow → ReduceMean → Add → Sqrt → Reciprocal → Mul
    の並びなので、「指数 2 の Pow の出力が ReduceMean に入る」形を目印に探す（StudyMemo 1.3 節）。
    """
    g = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    changed: list[dict] = []
    extra: list[onnx.NodeProto] = []

    for pw in list(g.node):
        if pw.op_type != "Pow":
            continue
        # 指数が 2 か
        exp = pw.input[1] if len(pw.input) > 1 else None
        if exp not in inits or float(np.asarray(inits[exp]).reshape(-1)[0]) != 2.0:
            continue
        # Pow の出力が ReduceMean に入るか（＝RMSNorm の形）
        rm = [c for c in consumers.get(pw.output[0], []) if c.op_type == "ReduceMean"]
        if not rm:
            continue
        x = pw.input[0]

        # 足すノードの名前を決める（abs・max・1/max・xs）。ReduceMax の軸は最後の軸（-1、隠れ次元）
        # 同じ x を使う Mul（x * rsqrt(...)）を探す。これも xs へ差し替える
        base = pw.name or f"rms{len(changed)}"
        ab, mx, rmx, xs = (f"{base}_abs", f"{base}_max", f"{base}_rmax", f"{base}_xs")
        ax = _unique(f"{base}_axes", {}, g)
        g.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), ax))

        # ★max に小さい下限を足す。
        #   足さないと **全ゼロのテンソルで 1/0 = inf** になり、
        #   qairt-converter が
        #     `"quantization max"=inf which exceeds the size of type: float`
        #   で落ちる（2026-09-13。較正の合成パターンに全ゼロが含まれていた）。
        #   実データの max は 1000 オーダーなので 1e-6 の影響は無視できる。
        flo = _unique(f"{base}_floor", {}, g)
        g.initializer.append(numpy_helper.from_array(np.float32(1e-6), flo))
        mxr = f"{base}_maxr"
        extra.append(helper.make_node("Abs", [x], [ab], f"{base}_abs_n"))
        extra.append(helper.make_node("ReduceMax", [ab, ax], [mxr], f"{base}_max_n",
                                      keepdims=1, noop_with_empty_axes=0))
        extra.append(helper.make_node("Add", [mxr, flo], [mx], f"{base}_maxfloor_n"))
        extra.append(helper.make_node("Reciprocal", [mx], [rmx], f"{base}_rmax_n"))
        extra.append(helper.make_node("Mul", [x, rmx], [xs], f"{base}_xs_n"))

        pw.input[0] = xs
        # eps を小さくする（m² 倍されるため元の値では効きすぎる）
        for add in consumers.get(rm[0].output[0], []):
            if add.op_type != "Add":
                continue
            for i, inp in enumerate(add.input):
                if inp in inits and np.asarray(inits[inp]).size == 1:
                    e2 = _unique(f"{base}_eps", {}, g)
                    g.initializer.append(numpy_helper.from_array(
                        np.array(eps_new, dtype=np.asarray(inits[inp]).dtype), e2))
                    add.input[i] = e2

        # 分子側の x も xs に替えないと、分母（xs で計算）と食い違って m 倍ずれる。
        # 相手の入力が Reciprocal / Sqrt / Div から来る Mul だけを「正規化の掛け算」とみなす
        # x * rsqrt(...) の x を xs に差し替える
        swapped = 0
        for c in consumers.get(x, []):
            if c.op_type != "Mul" or c is pw:
                continue
            other = [i for i in c.input if i != x]
            p2 = producer.get(other[0]) if other else None
            if p2 is not None and p2.op_type in ("Reciprocal", "Sqrt", "Div"):
                for i, inp in enumerate(c.input):
                    if inp == x:
                        c.input[i] = xs
                        swapped += 1
        changed.append({"pow": base, "x": x, "mul_swapped": swapped})

    if changed:
        # 追加ノードを入れてトポロジ順に直す
        nodes = list(g.node) + extra
        order, avail, pending = [], {i.name for i in g.initializer} | {v.name for v in g.input}, list(nodes)
        while pending:
            progress = False
            for n in list(pending):
                if all(i in avail or i == "" for i in n.input):
                    order.append(n)
                    avail.update(n.output)
                    pending.remove(n)
                    progress = True
            if not progress:
                order.extend(pending)
                break
        del g.node[:]
        g.node.extend(order)
        if _fits_protobuf(model):
            onnx.checker.check_model(model)
    return changed


def count_batchnorm(model: onnx.ModelProto) -> int:
    """モデルに残っている BatchNormalization の数（C-5 の検査用）。"""
    return sum(1 for n in model.graph.node if n.op_type == "BatchNormalization")


def fold_batchnorm(src: Path, dst: Path) -> int:
    """独立した BatchNormalization を Mul+Add へ畳み込む。畳み込んだ件数を返す。

    引数: src（読む ONNX）、dst（書き出し先）。戻り値: 畳み込んだ件数。
    係数はモジュールの説明の式（scale = gamma / sqrt(var + eps)、shift = beta - mean * scale）。
    形は NCHW（チャネルが軸 1）を前提に [1, C, 1, 1] で掛ける。LLM には BatchNorm が無いので
    主に P1・P2 の CNN 向け（LLM では 0 件のまま通り抜ける）。
    """
    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    new_nodes: list[onnx.NodeProto] = []
    folded = 0
    for node in graph.node:
        if node.op_type != "BatchNormalization":
            new_nodes.append(node)
            continue

        # 5入力（X, scale, B, mean, var）がすべて initializer である場合のみ畳み込める。
        # そうでなければ学習時グラフなどの可能性があるため、触らずに残して検出させる。
        if len(node.input) < 5 or not all(n in inits for n in node.input[1:5]):
            print(f"  WARN: {node.name}: 入力が定数でないため畳み込めない。そのまま残す")
            new_nodes.append(node)
            continue

        gamma = inits[node.input[1]].astype(np.float64)
        beta = inits[node.input[2]].astype(np.float64)
        mean = inits[node.input[3]].astype(np.float64)
        var = inits[node.input[4]].astype(np.float64)
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)

        # 精度のため float64 で計算してから float32 に落とす
        scale = (gamma / np.sqrt(var + eps)).astype(np.float32)
        shift = (beta - mean * scale).astype(np.float32)

        ch = scale.shape[0]
        # 元の BN は bn1_scale のような initializer を既に持っているため、
        # 素直に f"{node.name}_scale" とすると名前が衝突する（実際に踏んだ）。
        s_name = _unique(f"{node.name}_foldmul_w", inits, graph)
        b_name = _unique(f"{node.name}_foldadd_b", inits, graph)
        # NCHW のチャネル方向ブロードキャスト
        graph.initializer.append(numpy_helper.from_array(scale.reshape(1, ch, 1, 1), s_name))
        graph.initializer.append(numpy_helper.from_array(shift.reshape(1, ch, 1, 1), b_name))

        mul_out = f"{node.name}_mul_out"
        new_nodes.append(helper.make_node("Mul", [node.input[0], s_name], [mul_out],
                                          f"{node.name}_mul"))
        new_nodes.append(helper.make_node("Add", [mul_out, b_name], [node.output[0]],
                                          f"{node.name}_add"))
        folded += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    # 2GB 超では checker が落ちるので、その場合は飛ばす（形は上で保証している）
    if _fits_protobuf(model):
        onnx.checker.check_model(model)
    _save(model, dst)
    return folded


def run(src: Path, folded_path: Path, prepped_path: Path,
        mask_replace: float = MASK_REPLACE, fold_scale: bool = True,
        stabilize: bool = False, swap_matmuls: set[str] | None = None,
        norm_rmsnorm: bool = False) -> dict:
    """S2 全体。結果の要約を返す。

    引数: src（export した ONNX）、folded_path（BN 畳み込み後の書き出し先）、prepped_path（前処理後の
          書き出し先＝S4 の入力）、mask_replace（C-9 の置き換え先）、fold_scale（③）、stabilize（④）、
          swap_matmuls（⑤ の対象の名前）、norm_rmsnorm（⑥）。
    戻り値: 書き換えごとの記録・BatchNorm の数の推移・形状推論の方式・出力パスの辞書。
    書き換えが 1 つでもあれば、書き換え後を <folded の名前>_clamped.onnx に保存し、以降はそれを元にする。
    最後に BatchNorm が残っていれば C-5 違反として止める。
    """
    # ① 極端な定数の置き換えを**最初に**やる。BN 畳み込みや定数畳み込みが
    #    これらの値を別の initializer へ伝播させる前に潰しておく必要がある。
    m0 = onnx.load(str(src))
    clamped = clamp_extreme_constants(m0, replace_with=mask_replace)
    # ② Expand → Concat の書き換え（C-10）。量子化より前にやる:
    #    量子化後だと Q/DQ を挟んだ形を壊さずに書き換えるのが難しくなる。
    expanded = rewrite_expand_to_concat(m0)
    # ③ スケールを MatMul の前へ移す。
    #    ★2026-10-18 訂正: 旧「C-8 の回避で 8bit にせざるを得ない attention スコアの分解能を、
    #    等価変形だけで 8倍にできる。」。実測では効果なし（30.3819% → 30.3904%、StudyMemo 5.6 節）。
    #    量子化されるのはスケール後の値で、移してもそのレンジは変わらないため。等価で害も無いので既定で残している。
    scaled = fold_scale_into_matmul(m0) if fold_scale else []
    # ④ softmax の前で行ごとの最大値を引く。**既定では無効**。
    #    数学的には等価で、numpy 上の単独試算では劇的に効く（softmax 誤差
    #    62.8% → 3.2%）のだが、**実際のパイプラインでは逆に悪化した**
    #    （層全体の相対L2 30.4% → 41.6%、2026-09-12 実測）。
    #    理由: スコアは **MatMul の出力時点で既に 8bit 化されている**ため、
    #    その後に行最大を引いても手遅れで、ReduceMax/Sub の量子化誤差だけが乗る。
    #    **対策を打つ位置が間違っていた。** 記録のため実装は残すが既定は off。
    softmaxes = stabilize_softmax(m0) if stabilize else []
    # ⑤ QK^T の左右を入れ替える。C-8 で 8bit にせざるを得ないのは常に右辺なので、
    #    レンジの狭い方（Q）を右辺へ持ってくる。既定は off（効果を測ってから判断）。
    #    （2026-10-18 追記: 効果は測定済み（top-1 66.5% → 80.7%）。本番は p2b_ops が方式名に swap が
    #    あるとき spec の swap_matmuls を渡して入れる）
    swapped = swap_actact_matmul_operands(m0, only=swap_matmuls) if swap_matmuls else []
    # ⑥ RMSNorm をトークンごとの max で正規化してから計算する（C-11）。
    #    x² のレンジが 280万 → 1 になり、通常チャンネルが潰れなくなる。
    rms = normalize_rmsnorm_input(m0) if norm_rmsnorm else []
    if clamped or expanded or scaled or softmaxes or swapped or rms:
        src = folded_path.parent / (folded_path.stem.replace("02_folded", "01_clamped")
                                    + "_clamped.onnx")
        _save(m0, src)

    before = count_batchnorm(onnx.load(str(src)))
    # BatchNorm を畳み込む（C-5）。前後の個数を数えて記録する
    n = fold_batchnorm(src, folded_path)
    after_fold = count_batchnorm(onnx.load(str(folded_path)))

    # symbolic shape inference は形状に記号が混じるモデルのためのもので、
    # 本パイプラインは**全て静的形状**なので本来は要らない。ただし既定で有効に
    # しているのは、形状が埋まらないと per-channel 量子化が成立しないため。
    #
    # ★ただし onnxruntime の SymbolicShapeInference は opset 18 の `ReduceMax`
    #   （axes が入力）を扱えず `len(None)` で落ちる（2026-09-12 に遭遇）。
    #   静的形状なら通常の shape inference で足りるので、その場合だけ切り替える。
    #   **黙って切り替えず、切り替えたことを記録する。**
    # 2GB を超えるなら最初から外部データ形式で書かせる。
    # 指定しないと**例外を出さずに空のモデルを残す**（上の _save の注記参照）。
    # 入力が 2GB 未満でも、最適化・形状推論を経た**出力**が超えることがある。
    # 実際 1.98GB の入力で出力が壊れた（2026-09-12）。余裕を持って 1GB で切り替える。
    # ★**外部データ（隣の .data）も数える。**
    #   `.onnx` だけ見ると、重みを外部に出したモデルで「小さい」と誤判定する。
    #   Qwen2.5 は inline（1.98GB の .onnx）だったので気付かなかったが、
    #   Qwen3-0.6B は .onnx 1.9MB + .data 2.38GB で **big=False と誤判定**し、
    #   quant_pre_process が inline 保存を試みて `EncodeError` になった（2026-09-14）。
    folded_bytes = folded_path.stat().st_size + sum(
        q.stat().st_size for q in folded_path.parent.glob(folded_path.name + "*.data"))
    big = folded_bytes > 1_000_000_000
    ext = dict(save_as_external_data=True, all_tensors_to_one_file=True,
               external_data_location=prepped_path.name + ".data") if big else {}

    # quant_pre_process（ORT の前処理: グラフの最適化と形状推論）。まず symbolic（記号を含む形状推論）で試し、
    # 例外が出たら通常の形状推論に切り替えてやり直す。どちらで通ったかを symbolic として記録する
    symbolic = True
    try:
        quant_pre_process(
            str(folded_path), str(prepped_path),
            skip_optimization=False,
            skip_symbolic_shape=False,
            **ext,
        )
    except Exception:
        # TypeError だけ捕えていたが、decode モデルでは
        # `Exception: Incomplete symbolic shape inference` が出た（2026-09-13）。
        # 形状が全て静的なら symbolic 推論は不要なので、例外の種類を問わず切り替える。
        symbolic = False
        quant_pre_process(
            str(folded_path), str(prepped_path),
            skip_optimization=False,
            skip_symbolic_shape=True,
            **ext,
        )
    _assert_sane(prepped_path, "S2 quant_pre_process")
    after_prep = count_batchnorm(onnx.load(str(prepped_path)))

    # 何をしたかを全部記録に残す（黙って書き換えないため）
    result = {
        "clamped_constants": clamped,
        "expand_rewrites": expanded,
        "scale_folds": scaled,
        "softmax_stabilized": softmaxes,
        "swapped_matmuls": swapped,
        "rmsnorm_normalized": rms,
        "symbolic_shape_inference": symbolic,
        "batchnorm_before": before,
        "batchnorm_folded": n,
        "batchnorm_after_fold": after_fold,
        "batchnorm_after_preprocess": after_prep,
        "folded_path": str(folded_path),
        "prepped_path": str(prepped_path),
    }
    # C-5 の出口検証。ここで止めれば、後段で error 3110 を食らわずに済む。
    if after_prep != 0:
        raise RuntimeError(
            f"C-5 違反: BatchNormalization が {after_prep} 個残っている。"
            " QNN HTP は signed per-channel Conv QDQ との併存を拒否する（error 3110）"
        )
    return result


def main() -> None:
    """コマンドの入口。--src と --workdir を受けて run() を（①〜③ と BN 畳み込みだけで）流し、要約を表示する。
    出力は <workdir>/02_folded.onnx・02_prepped.onnx（と書き換えがあれば 01_clamped_clamped.onnx）。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()

    r = run(args.src,
            args.workdir / "02_folded.onnx",
            args.workdir / "02_prepped.onnx")
    for c in r["clamped_constants"]:
        print(f"極端な定数を置換: {c['initializer']} {c['shape']} "
              f"min={c['original_min']:.3e} -> {c['replaced_with']} "
              f"({c['n_negative']}要素) [C-9]")
    for e in r["expand_rewrites"]:
        ax = ", ".join(f"axis{a['axis']}x{a['repeat']}" for a in e["concat_axes"])
        print(f"Expand を Concat へ書換: {e['node']} {e['input_shape']} -> "
              f"{e['target_shape']} ({ax}) [C-10]")
    print(f"BatchNorm: {r['batchnorm_before']} -> 畳み込み {r['batchnorm_folded']} 件 -> "
          f"前処理後 {r['batchnorm_after_preprocess']} 件 [C-5 OK]")
    print(f"  {r['prepped_path']}")


if __name__ == "__main__":
    main()
