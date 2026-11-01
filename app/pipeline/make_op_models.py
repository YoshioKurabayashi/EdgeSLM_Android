"""P2b: Transformer 構成要素の単体 ONNX を生成する（基本設計② 6.1節）

なぜ部品単位でやるのか:
  Transformer を丸ごと変換して失敗すると、数百〜数千ノードから原因を探すことになる。
  **どの op が通らないのか**を先に確定させてから組み上げる。

各モデルは「その op を通すために最小限の周辺」だけを持つ。ただし
**op を単独で浮かせない**——量子化は QDQ の前後関係で成立するため、裸の op ひとつ
だけだと現実と違う判定になる。そこで各モデルは「入力 → 対象 op → 出力」の形を
保ちつつ、**LLM で実際に現れる形状と接続**にしてある。

形状は Qwen2.5-0.5B を縮小したもの（構造は同じ、サイズだけ小さい）:
  hidden=128 / heads=4 / kv_heads=2（GQA） / head_dim=32 / seq=16 / past=16
小さくしているのは**反復を速くするため**であって、P2b が問うのは容量ではなく
**op の対応可否**だからである。容量は P4/P5 で別に測る。

生成物:
  work/ops/<name>/<name>_fp32.onnx  … モデル本体
  work/ops/<name>/<name>.spec.json  … 入力仕様・狙い・期待（パイプラインが読む）
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足はこのコメントに書く。
# どの段か: P2b「Transformer の部品ごとに v68 の HTP で通るか」（基本設計② 6.1 節・StudyMemo 1.5 節「どの層の制約か」）。
#   記録: results/P2b_20260912_op_matrix_host.md・op_matrix_device.md・real_form.md（実機は 15/15 が全ノード NPU）。
#   ここで作った部品の結果が C-1〜C-10 の制約（StudyMemo 5 章の Graph Rewrite）の出どころになった。
# 出力:  --out（既定 work/ops）/<名前>/<名前>_fp32.onnx と <名前>.spec.json、--out/manifest.json。
# 動かし方: & $PY2 -m app.pipeline.make_op_models --list（一覧）／--only block_real（1 つだけ）／引数なし（全部）
#   その後 p2b_ops prepare（S2〜S4）→ build_context.sh（S5〜S7）→ 端末の op_check。
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# --- 縮小 Qwen2.5-0.5B の形状 ---
H = 128       # hidden
NH = 4        # query heads
NKV = 2       # kv heads（GQA）
HD = 32       # head_dim (= H / NH)
S = 16        # sequence
P = 16        # past length（固定長KVキャッシュ）
FF = 256      # intermediate（SwiGLU）
VOCAB = 512

# 因果マスクの埋め値。**-inf は使わない**。
# 量子化は inf を表現できず、活性化の値域が壊れて全体の分解能を失う。
MASK_NEG = -60.0

# 重みの乱数は seed 0 で固定（毎回同じモデルを作る。比較をやり直せるように）。
RNG = np.random.default_rng(0)


def _w(name: str, shape: tuple[int, ...], scale: float = 0.05) -> onnx.TensorProto:
    """平均 0・標準偏差 scale の正規乱数の重み（float32）の初期化子を作る。"""
    return numpy_helper.from_array(
        RNG.normal(0.0, scale, shape).astype(np.float32), name)


def _c(name: str, arr) -> onnx.TensorProto:
    """定数（形・型は arr のまま）の初期化子を作る。"""
    return numpy_helper.from_array(np.asarray(arr), name)


def _vi(name: str, shape, dtype=TensorProto.FLOAT):
    """グラフの入出力の型と形の宣言（固定形状だけ。QNN EP は動的形状を扱えない）を作る。"""
    return helper.make_tensor_value_info(name, dtype, list(shape))


class Builder:
    """ノードと initializer を貯めていくだけの小さな器。"""

    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def add(self, op, ins, outs, name=None, domain="", **attrs):
        """ノードを 1 つ足し、最初の出力の名前を返す（次の部品に渡しやすくするため）。名前が無ければ通し番号で付ける。"""
        self._n += 1
        node = helper.make_node(
            op, list(ins), list(outs), name or f"{op.lower()}_{self._n}",
            domain=domain, **attrs)
        self.nodes.append(node)
        return outs[0]

    def init(self, t: onnx.TensorProto) -> str:
        """初期化子を足し、その名前を返す。"""
        self.inits.append(t)
        return t.name


# ------------------------------------------------------------------
# 部品（複数のモデルから使い回す）
# ------------------------------------------------------------------

def _rmsnorm_decomp(b: Builder, x: str, tag: str, hidden: int = H) -> str:
    """x / sqrt(mean(x^2) + eps) * w を素の op で組む。

    ここが 16a16w に落ちると v68 では通らない（V73以上）。
    Pow / ReduceMean / Sqrt / Div がどう量子化されるかが焦点。
    """
    b.init(_c(f"{tag}_two", np.float32(2.0)))
    b.init(_c(f"{tag}_eps", np.float32(1e-6)))
    b.init(_w(f"{tag}_w", (hidden,), 0.02))
    b.add("Pow", [x, f"{tag}_two"], [f"{tag}_sq"], f"{tag}_pow")
    b.add("ReduceMean", [f"{tag}_sq"], [f"{tag}_mean"], f"{tag}_rmean",
          axes=[-1], keepdims=1)
    b.add("Add", [f"{tag}_mean", f"{tag}_eps"], [f"{tag}_ad"], f"{tag}_add_eps")
    b.add("Sqrt", [f"{tag}_ad"], [f"{tag}_rt"], f"{tag}_sqrt")
    b.add("Div", [x, f"{tag}_rt"], [f"{tag}_dv"], f"{tag}_div")
    return b.add("Mul", [f"{tag}_dv", f"{tag}_w"], [f"{tag}_out"], f"{tag}_scale")


def _silu(b: Builder, x: str, tag: str) -> str:
    """SiLU = x * sigmoid(x)。

    量子化では扱いが難しい部類。負側に浅い谷を持つため MinMax だと
    分解能を取られやすく、A8 では潰れる可能性がある（A16 との差が出るはず）。
    """
    b.add("Sigmoid", [x], [f"{tag}_sg"], f"{tag}_sigmoid")
    return b.add("Mul", [x, f"{tag}_sg"], [f"{tag}_out"], f"{tag}_mul")


def _rmsnorm_real(b: Builder, x: str, tag: str, hidden: int = H) -> str:
    """RMSNorm を**本物の export と同じ形**で組む。

    2026-09-12 に Qwen2.5-0.5B を export して判明: HF の RMSNorm は
    `Sqrt` → `Div` ではなく **`Sqrt` → `Reciprocal` → `Mul`**（= rsqrt）になる。
    `_rmsnorm_decomp` は私の推測形で、実物はこちら。
    """
    b.init(_c(f"{tag}_two", np.float32(2.0)))
    b.init(_c(f"{tag}_eps", np.float32(1e-6)))
    b.init(_w(f"{tag}_w", (hidden,), 0.02))
    b.add("Pow", [x, f"{tag}_two"], [f"{tag}_sq"], f"{tag}_pow")
    b.add("ReduceMean", [f"{tag}_sq"], [f"{tag}_mean"], f"{tag}_rmean",
          axes=[-1], keepdims=1)
    b.add("Add", [f"{tag}_mean", f"{tag}_eps"], [f"{tag}_ad"], f"{tag}_add_eps")
    b.add("Sqrt", [f"{tag}_ad"], [f"{tag}_rt"], f"{tag}_sqrt")
    b.add("Reciprocal", [f"{tag}_rt"], [f"{tag}_rc"], f"{tag}_recip")
    b.add("Mul", [x, f"{tag}_rc"], [f"{tag}_nm"], f"{tag}_norm")
    return b.add("Mul", [f"{tag}_nm", f"{tag}_w"], [f"{tag}_out"], f"{tag}_scale")


def _gqa_repeat_real(b: Builder, x: str, tag: str, kv: int, rep: int,
                     seq: int, hd: int) -> str:
    """GQA のヘッド複製を**本物の export と同じ形**で組む。

    私は `Concat` ×rep で書いたが、実物は **`Unsqueeze` → `Expand` → `Reshape`**。
    Expand / Unsqueeze はいずれも P2b で未検証だったため、ここで踏む。
    """
    b.init(_c(f"{tag}_ax", np.array([2], dtype=np.int64)))
    b.init(_c(f"{tag}_ex", np.array([1, kv, rep, seq, hd], dtype=np.int64)))
    b.init(_c(f"{tag}_rs", np.array([1, kv * rep, seq, hd], dtype=np.int64)))
    b.add("Unsqueeze", [x, f"{tag}_ax"], [f"{tag}_u"], f"{tag}_unsq")
    b.add("Expand", [f"{tag}_u", f"{tag}_ex"], [f"{tag}_e"], f"{tag}_expand")
    return b.add("Reshape", [f"{tag}_e", f"{tag}_rs"], [f"{tag}_out"], f"{tag}_reshape")


def _rope_slice(b: Builder, x: str, tag: str, seq: int, hd: int,
                heads: int) -> str:
    """RoPE を**本物の export と同じ形**で組む。

    rotate_half を私は `Split` で書いたが、実物は **`Slice` ×2**。
    """
    half = hd // 2
    # cos/sin の値は op が通るかを見るためのもので、本物の RoPE の周波数の並びとは違う（推定。精度は見ていない）。
    cos = np.cos(np.arange(seq)[:, None] / (10000 ** (np.arange(hd) / hd)))
    sin = np.sin(np.arange(seq)[:, None] / (10000 ** (np.arange(hd) / hd)))
    b.init(_c(f"{tag}_cos", cos.reshape(1, 1, seq, hd).astype(np.float32)))
    b.init(_c(f"{tag}_sin", sin.reshape(1, 1, seq, hd).astype(np.float32)))
    b.init(_c(f"{tag}_s0", np.array([0], dtype=np.int64)))
    b.init(_c(f"{tag}_s1", np.array([half], dtype=np.int64)))
    b.init(_c(f"{tag}_s2", np.array([hd], dtype=np.int64)))
    b.init(_c(f"{tag}_ax", np.array([-1], dtype=np.int64)))
    b.add("Slice", [x, f"{tag}_s0", f"{tag}_s1", f"{tag}_ax"], [f"{tag}_x1"], f"{tag}_sl1")
    b.add("Slice", [x, f"{tag}_s1", f"{tag}_s2", f"{tag}_ax"], [f"{tag}_x2"], f"{tag}_sl2")
    b.add("Neg", [f"{tag}_x2"], [f"{tag}_nx2"], f"{tag}_neg")
    b.add("Concat", [f"{tag}_nx2", f"{tag}_x1"], [f"{tag}_rot"], f"{tag}_cat", axis=-1)
    b.add("Mul", [x, f"{tag}_cos"], [f"{tag}_xc"], f"{tag}_mul_cos")
    b.add("Mul", [f"{tag}_rot", f"{tag}_sin"], [f"{tag}_rs"], f"{tag}_mul_sin")
    return b.add("Add", [f"{tag}_xc", f"{tag}_rs"], [f"{tag}_out"], f"{tag}_add")


def _causal_mask(b: Builder, name: str, q_len: int, k_len: int) -> str:
    """因果マスク [1, 1, q_len, k_len] を定数で作る（見てはいけない未来の位置に MASK_NEG、それ以外 0）。

    k_len が q_len より長い時は、右寄せ（最後の query が最後の key に揃う）で並べる。
    """
    m = np.zeros((1, 1, q_len, k_len), dtype=np.float32)
    off = k_len - q_len
    for i in range(q_len):
        m[0, 0, i, off + i + 1:] = MASK_NEG
    return b.init(_c(name, m))


def _attention_manual(b: Builder, x: str, tag: str, q_len: int = S, k_len: int = S,
                      mask_input: str | None = None) -> str:
    """fused GQA を使わず MatMul / Softmax / MatMul へ手分解した attention。

    GroupQueryAttention は HTP 非対応（GPUのみ / 要件定義書 3.3.1節）。
    したがって**この形が本線**であり、ここが通らなければ経路そのものを変える必要がある。
    """
    b.init(_w(f"{tag}_wq", (H, NH * HD)))
    b.init(_w(f"{tag}_wk", (H, NKV * HD)))
    b.init(_w(f"{tag}_wv", (H, NKV * HD)))
    b.init(_w(f"{tag}_wo", (NH * HD, H)))
    b.init(_c(f"{tag}_qshape", np.array([1, q_len, NH, HD], dtype=np.int64)))
    b.init(_c(f"{tag}_kvshape", np.array([1, k_len, NKV, HD], dtype=np.int64)))
    b.init(_c(f"{tag}_oshape", np.array([1, q_len, NH * HD], dtype=np.int64)))
    b.init(_c(f"{tag}_scale", np.float32(1.0 / np.sqrt(HD))))

    b.add("MatMul", [x, f"{tag}_wq"], [f"{tag}_q"], f"{tag}_proj_q")
    b.add("MatMul", [x, f"{tag}_wk"], [f"{tag}_k"], f"{tag}_proj_k")
    b.add("MatMul", [x, f"{tag}_wv"], [f"{tag}_v"], f"{tag}_proj_v")

    b.add("Reshape", [f"{tag}_q", f"{tag}_qshape"], [f"{tag}_qr"], f"{tag}_rs_q")
    b.add("Reshape", [f"{tag}_k", f"{tag}_kvshape"], [f"{tag}_kr"], f"{tag}_rs_k")
    b.add("Reshape", [f"{tag}_v", f"{tag}_kvshape"], [f"{tag}_vr"], f"{tag}_rs_v")

    b.add("Transpose", [f"{tag}_qr"], [f"{tag}_qt"], f"{tag}_tr_q", perm=[0, 2, 1, 3])
    b.add("Transpose", [f"{tag}_kr"], [f"{tag}_kt"], f"{tag}_tr_k", perm=[0, 2, 3, 1])
    b.add("Transpose", [f"{tag}_vr"], [f"{tag}_vt"], f"{tag}_tr_v", perm=[0, 2, 1, 3])

    # GQA: kv ヘッドを query ヘッド数へ複製する。
    # Expand/Tile ではなく Concat にしているのは、形状 op を増やさず
    # 「複製そのもの」が通るかだけを見たいため。
    rep = NH // NKV
    b.add("Concat", [f"{tag}_kt"] * rep, [f"{tag}_kk"], f"{tag}_rep_k", axis=1)
    b.add("Concat", [f"{tag}_vt"] * rep, [f"{tag}_vv"], f"{tag}_rep_v", axis=1)

    b.add("MatMul", [f"{tag}_qt", f"{tag}_kk"], [f"{tag}_sc"], f"{tag}_qk")
    b.add("Mul", [f"{tag}_sc", f"{tag}_scale"], [f"{tag}_scs"], f"{tag}_scale_mul")
    # mask_input を渡すと**マスクを定数ではなくグラフ入力**にする。
    # パディング長が実行時に変わるモデル（prefill）はこの形になる。
    mask = mask_input if mask_input else _causal_mask(b, f"{tag}_mask", q_len, k_len)
    b.add("Add", [f"{tag}_scs", mask], [f"{tag}_scm"], f"{tag}_mask_add")
    b.add("Softmax", [f"{tag}_scm"], [f"{tag}_pr"], f"{tag}_softmax", axis=-1)
    b.add("MatMul", [f"{tag}_pr", f"{tag}_vv"], [f"{tag}_ct"], f"{tag}_av")
    b.add("Transpose", [f"{tag}_ct"], [f"{tag}_ctt"], f"{tag}_tr_o", perm=[0, 2, 1, 3])
    b.add("Reshape", [f"{tag}_ctt", f"{tag}_oshape"], [f"{tag}_ctr"], f"{tag}_rs_o")
    return b.add("MatMul", [f"{tag}_ctr", f"{tag}_wo"], [f"{tag}_out"], f"{tag}_proj_o")


# ------------------------------------------------------------------
# モデル定義
# ------------------------------------------------------------------

def m_matmul_weight(b: Builder):
    """#1a 重み固定の MatMul。FFN / projection の形。本線 A16W8 の成立確認。"""
    b.init(_w("w", (H, H)))
    b.add("MatMul", ["x", "w"], ["y"], "matmul")
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_matmul_act(b: Builder):
    """#1b **両辺が活性化**の MatMul。attention の QK^T / PV がこれ。

    重み固定版が通っても、こちらは (uint16, uint16) の組合せになり
    v68 で通らない可能性がある——**分けて測らないと判別できない**。
    """
    b.add("MatMul", ["a", "bb"], ["y"], "matmul")
    return ([_vi("a", (1, NH, S, HD)), _vi("bb", (1, NH, HD, S))],
            [_vi("y", (1, NH, S, S))])


def m_rmsnorm(b: Builder):
    """#2 RMSNorm（素の op へ分解）。"""
    _rmsnorm_decomp(b, "x", "rms")
    b.add("Identity", ["rms_out"], ["y"], "out")
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_rmsnorm_fused(b: Builder):
    """#2b SimplifiedLayerNormalization（ORT の融合形）。

    **標準 opset の op ではない**ため checker を通さない。変換のどの段で
    落ちるか（量子化器 / qairt-converter / context 生成）を見るのが目的。
    """
    b.init(_w("w", (H,), 0.02))
    b.add("SimplifiedLayerNormalization", ["x", "w"], ["y"], "rms",
          axis=-1, epsilon=1e-6, stash_type=1)
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_layernorm(b: Builder):
    """#3 LayerNormalization（opset 17 の標準 op）。"""
    b.init(_w("scale", (H,), 0.02))
    b.init(_w("bias", (H,), 0.01))
    b.add("LayerNormalization", ["x", "scale", "bias"], ["y"], "ln",
          axis=-1, epsilon=1e-5)
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_rope(b: Builder):
    """#4 RoPE（素の op へ分解）。

    x*cos + rotate_half(x)*sin。cos/sin は固定長なので initializer にできる。
    Split / Neg / Concat が整数量子化で通るかが焦点。
    """
    half = HD // 2
    cos = np.cos(np.arange(S)[:, None] / (10000 ** (np.arange(HD) / HD)))
    sin = np.sin(np.arange(S)[:, None] / (10000 ** (np.arange(HD) / HD)))
    b.init(_c("cos", cos.reshape(1, 1, S, HD).astype(np.float32)))
    b.init(_c("sin", sin.reshape(1, 1, S, HD).astype(np.float32)))
    b.init(_c("halves", np.array([half, half], dtype=np.int64)))

    b.add("Split", ["x", "halves"], ["x1", "x2"], "split", axis=-1)
    b.add("Neg", ["x2"], ["nx2"], "neg")
    b.add("Concat", ["nx2", "x1"], ["rot"], "concat", axis=-1)
    b.add("Mul", ["x", "cos"], ["xc"], "mul_cos")
    b.add("Mul", ["rot", "sin"], ["rs"], "mul_sin")
    b.add("Add", ["xc", "rs"], ["y"], "add")
    return [_vi("x", (1, NH, S, HD))], [_vi("y", (1, NH, S, HD))]


def m_rope_fused(b: Builder):
    """#4b com.microsoft.RotaryEmbedding。QNN_OP_ROPE に落ちるかを見る。"""
    half = HD // 2
    cos = np.cos(np.arange(S)[:, None] / (10000 ** (np.arange(half) / half)))
    sin = np.sin(np.arange(S)[:, None] / (10000 ** (np.arange(half) / half)))
    b.init(_c("cos_cache", cos.astype(np.float32)))
    b.init(_c("sin_cache", sin.astype(np.float32)))
    b.add("RotaryEmbedding", ["x", "position_ids", "cos_cache", "sin_cache"], ["y"],
          "rope", domain="com.microsoft", interleaved=0, num_heads=NH)
    return ([_vi("x", (1, NH, S, HD)),
             _vi("position_ids", (1, S), TensorProto.INT64)],
            [_vi("y", (1, NH, S, HD))])


def m_softmax(b: Builder):
    """#5 Softmax。出力 encoding の扱いが版で変わった箇所（基本設計② 6.1節#5）。"""
    b.add("Softmax", ["x"], ["y"], "softmax", axis=-1)
    return [_vi("x", (1, NH, S, S))], [_vi("y", (1, NH, S, S))]


def m_silu(b: Builder):
    """#5b SiLU。SwiGLU の活性化。A8 と A16 で差が出るはずの箇所。"""
    _silu(b, "x", "silu")
    b.add("Identity", ["silu_out"], ["y"], "out")
    return [_vi("x", (1, S, FF))], [_vi("y", (1, S, FF))]


def m_attention(b: Builder):
    """#6 attention 全体（手分解）。fused GQA が使えないための本線。"""
    _attention_manual(b, "x", "att")
    b.add("Identity", ["att_out"], ["y"], "out")
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_attn_mask_input(b: Builder):
    """attention のマスクを**グラフ入力**にした形。`attention` との差はそこだけ。

    2026-09-13: 24層 prefill のマスク入力版が端末で err 1002 になったため、
    「float32 のマスクをグラフ入力にすること」自体が v68 で通るかを切り分ける。
    定数マスク版（`attention`）は実機で通っている（P2b）ので、
    ここが落ちれば**原因はマスクの入力化**だと言える。
    """
    _attention_manual(b, "x", "att", mask_input="attn_mask")
    b.add("Identity", ["att_out"], ["y"], "out")
    return ([_vi("x", (1, S, H)), _vi("attn_mask", (1, 1, S, S))],
            [_vi("y", (1, S, H))])


def m_kvcache(b: Builder):
    """#7 固定長KVキャッシュの decode 1ステップ。

    past（長さ P）に新しい1トークン分を Concat し、それに対して attention する。
    **多入力・多出力**になるため、runner 側の多入力対応も同時に試される。
    """
    b.init(_c("kvshape", np.array([1, 1, NKV, HD], dtype=np.int64)))
    b.init(_c("qshape", np.array([1, 1, NH, HD], dtype=np.int64)))
    b.init(_c("oshape", np.array([1, 1, NH * HD], dtype=np.int64)))
    b.init(_c("scale", np.float32(1.0 / np.sqrt(HD))))
    b.init(_w("wq", (H, NH * HD)))
    b.init(_w("wk", (H, NKV * HD)))
    b.init(_w("wv", (H, NKV * HD)))
    b.init(_w("wo", (NH * HD, H)))

    b.add("MatMul", ["x", "wq"], ["q"], "proj_q")
    b.add("MatMul", ["x", "wk"], ["k"], "proj_k")
    b.add("MatMul", ["x", "wv"], ["v"], "proj_v")
    b.add("Reshape", ["q", "qshape"], ["qr"], "rs_q")
    b.add("Reshape", ["k", "kvshape"], ["kr"], "rs_k")
    b.add("Reshape", ["v", "kvshape"], ["vr"], "rs_v")
    b.add("Transpose", ["qr"], ["qt"], "tr_q", perm=[0, 2, 1, 3])
    b.add("Transpose", ["kr"], ["kt"], "tr_k", perm=[0, 2, 1, 3])
    b.add("Transpose", ["vr"], ["vt"], "tr_v", perm=[0, 2, 1, 3])

    # ここが KVキャッシュの本体。past は入力、present は出力として外に出す。
    b.add("Concat", ["past_k", "kt"], ["present_k"], "cat_k", axis=2)
    b.add("Concat", ["past_v", "vt"], ["present_v"], "cat_v", axis=2)

    rep = NH // NKV
    b.add("Concat", ["present_k"] * rep, ["kk"], "rep_k", axis=1)
    b.add("Concat", ["present_v"] * rep, ["vv"], "rep_v", axis=1)
    b.add("Transpose", ["kk"], ["kkt"], "tr_kk", perm=[0, 1, 3, 2])

    b.add("MatMul", ["qt", "kkt"], ["sc"], "qk")
    b.add("Mul", ["sc", "scale"], ["scs"], "scale_mul")
    b.add("Softmax", ["scs"], ["pr"], "softmax", axis=-1)
    b.add("MatMul", ["pr", "vv"], ["ct"], "av")
    b.add("Transpose", ["ct"], ["ctt"], "tr_o", perm=[0, 2, 1, 3])
    b.add("Reshape", ["ctt", "oshape"], ["ctr"], "rs_o")
    b.add("MatMul", ["ctr", "wo"], ["y"], "proj_o")

    return ([_vi("x", (1, 1, H)),
             _vi("past_k", (1, NKV, P, HD)),
             _vi("past_v", (1, NKV, P, HD))],
            [_vi("y", (1, 1, H)),
             _vi("present_k", (1, NKV, P + 1, HD)),
             _vi("present_v", (1, NKV, P + 1, HD))])


def m_embed(b: Builder):
    """#8 Gather（埋め込み表引き）+ 形状操作。

    int64 の索引が整数量子化グラフで通るか。埋め込み表は LLM で最大級の
    initializer になるため、量子化のされ方も見る。
    """
    b.init(_w("table", (VOCAB, H), 0.02))
    b.init(_c("shape", np.array([1, S, H], dtype=np.int64)))
    b.init(_w("w", (H, H)))
    b.add("Gather", ["table", "ids"], ["g"], "gather", axis=0)
    b.add("Reshape", ["g", "shape"], ["r"], "reshape")
    b.add("MatMul", ["r", "w"], ["y"], "matmul")
    return ([_vi("ids", (1, S), TensorProto.INT64)], [_vi("y", (1, S, H))])


def m_block(b: Builder):
    """#9 Transformer 1ブロック（**P2b Gate**）。

    RMSNorm → attention → 残差 → RMSNorm → SwiGLU FFN → 残差。
    これが HTP に完全 delegate できれば P2b 合格。
    """
    n1 = _rmsnorm_decomp(b, "x", "n1")
    at = _attention_manual(b, n1, "att")
    b.add("Add", ["x", at], ["r1"], "resid1")

    n2 = _rmsnorm_decomp(b, "r1", "n2")
    b.init(_w("w_gate", (H, FF)))
    b.init(_w("w_up", (H, FF)))
    b.init(_w("w_down", (FF, H)))
    b.add("MatMul", [n2, "w_gate"], ["ffn_g"], "ffn_gate")
    b.add("MatMul", [n2, "w_up"], ["ffn_u"], "ffn_up")
    a = _silu(b, "ffn_g", "ffn_act")
    b.add("Mul", [a, "ffn_u"], ["ffn_m"], "ffn_mul")
    b.add("MatMul", ["ffn_m", "w_down"], ["ffn_d"], "ffn_down")
    b.add("Add", ["r1", "ffn_d"], ["y"], "resid2")
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_gqa_repeat(b: Builder):
    """#10 GQA のヘッド複製（**実 export 形**）。Unsqueeze / Expand が未検証だった。

    ★先頭に MatMul を置いてあるのは飾りではない。
      当初は入力を直接 Unsqueeze へ入れていたが、量子化器は形状 op を
      「そのまま通す」扱いにするため**グラフ入力が量子化されないまま残り**、
      C-1 違反（float セグメント）になった。実際の Transformer では
      この複製は projection の後ろに来るので、**その文脈を再現する**のが正しい。
      単体化のために現実と違う形にすると、判定そのものが無意味になる。
    """
    b.init(_w("wk", (H, NKV * HD)))
    b.init(_c("kvshape", np.array([1, S, NKV, HD], dtype=np.int64)))
    b.add("MatMul", ["x", "wk"], ["k"], "proj_k")
    b.add("Reshape", ["k", "kvshape"], ["kr"], "rs_k")
    b.add("Transpose", ["kr"], ["kt"], "tr_k", perm=[0, 2, 1, 3])
    out = _gqa_repeat_real(b, "kt", "gqa", NKV, NH // NKV, S, HD)
    b.init(_w("w", (HD, HD)))
    b.add("MatMul", [out, "w"], ["y"], "matmul")
    return [_vi("x", (1, S, H))], [_vi("y", (1, NH, S, HD))]


def m_rope_slice(b: Builder):
    """#11 RoPE（**実 export 形**）。rotate_half が Split ではなく Slice。"""
    _rope_slice(b, "x", "rope", S, HD, NH)
    b.add("Identity", ["rope_out"], ["y"], "out")
    return [_vi("x", (1, NH, S, HD))], [_vi("y", (1, NH, S, HD))]


def m_rmsnorm_real(b: Builder):
    """#12 RMSNorm（**実 export 形**）。Div ではなく Reciprocal。"""
    _rmsnorm_real(b, "x", "rms")
    b.add("Identity", ["rms_out"], ["y"], "out")
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_block_real(b: Builder):
    """#13 Transformer 1ブロック（**実 export 形**）。**本命**。

    2026-09-12 に Qwen2.5-0.5B を export して判明した形に合わせてある:
      RMSNorm  : Sqrt → Reciprocal → Mul
      GQA複製  : Unsqueeze → Expand → Reshape
      RoPE     : Slice ×2
      マスク   : Add（定数）※値は量子化可能な範囲に置き換え済み
      FFN      : SwiGLU（Sigmoid + Mul）

    `m_block` は私の推測形で、こちらが実物に対応する。**両方を残す**のは、
    推測がどこで外れたかを後から追えるようにするため。
    """
    rep = NH // NKV
    n1 = _rmsnorm_real(b, "x", "n1")

    b.init(_w("wq", (H, NH * HD)))
    b.init(_w("wk", (H, NKV * HD)))
    b.init(_w("wv", (H, NKV * HD)))
    b.init(_w("wo", (NH * HD, H)))
    b.init(_c("qshape", np.array([1, S, NH, HD], dtype=np.int64)))
    b.init(_c("kvshape", np.array([1, S, NKV, HD], dtype=np.int64)))
    b.init(_c("oshape", np.array([1, S, NH * HD], dtype=np.int64)))
    b.init(_c("scale", np.float32(1.0 / np.sqrt(HD))))

    b.add("MatMul", [n1, "wq"], ["q"], "proj_q")
    b.add("MatMul", [n1, "wk"], ["k"], "proj_k")
    b.add("MatMul", [n1, "wv"], ["v"], "proj_v")
    b.add("Reshape", ["q", "qshape"], ["qr"], "rs_q")
    b.add("Reshape", ["k", "kvshape"], ["kr"], "rs_k")
    b.add("Reshape", ["v", "kvshape"], ["vr"], "rs_v")
    b.add("Transpose", ["qr"], ["qt"], "tr_q", perm=[0, 2, 1, 3])
    b.add("Transpose", ["kr"], ["kt"], "tr_k", perm=[0, 2, 1, 3])
    b.add("Transpose", ["vr"], ["vt"], "tr_v", perm=[0, 2, 1, 3])

    # RoPE は q と k に掛かる（実物と同じ）
    qrope = _rope_slice(b, "qt", "ropeq", S, HD, NH)
    krope = _rope_slice(b, "kt", "ropek", S, HD, NKV)

    kk = _gqa_repeat_real(b, krope, "gqak", NKV, rep, S, HD)
    vv = _gqa_repeat_real(b, "vt", "gqav", NKV, rep, S, HD)

    b.add("Transpose", [kk], ["kkt"], "tr_kk", perm=[0, 1, 3, 2])
    b.add("MatMul", [qrope, "kkt"], ["sc"], "qk")
    b.add("Mul", ["sc", "scale"], ["scs"], "scale_mul")
    mask = _causal_mask(b, "mask", S, S)
    b.add("Add", ["scs", mask], ["scm"], "mask_add")
    b.add("Softmax", ["scm"], ["pr"], "softmax", axis=-1)
    b.add("MatMul", ["pr", vv], ["ct"], "av")
    b.add("Transpose", ["ct"], ["ctt"], "tr_o", perm=[0, 2, 1, 3])
    b.add("Reshape", ["ctt", "oshape"], ["ctr"], "rs_o")
    b.add("MatMul", ["ctr", "wo"], ["att"], "proj_o")
    b.add("Add", ["x", "att"], ["r1"], "resid1")

    n2 = _rmsnorm_real(b, "r1", "n2")
    b.init(_w("w_gate", (H, FF)))
    b.init(_w("w_up", (H, FF)))
    b.init(_w("w_down", (FF, H)))
    b.add("MatMul", [n2, "w_gate"], ["ffn_g"], "ffn_gate")
    b.add("MatMul", [n2, "w_up"], ["ffn_u"], "ffn_up")
    a = _silu(b, "ffn_g", "ffn_act")
    b.add("Mul", [a, "ffn_u"], ["ffn_m"], "ffn_mul")
    b.add("MatMul", ["ffn_m", "w_down"], ["ffn_d"], "ffn_down")
    b.add("Add", ["r1", "ffn_d"], ["y"], "resid2")
    return [_vi("x", (1, S, H))], [_vi("y", (1, S, H))]


def m_unsqueeze_only(b: Builder):
    """#10a `Unsqueeze` だけ（Expand なし）。

    `gqa_repeat` が端末で err 1002 になったため、Unsqueeze と Expand の
    **どちらが原因かを切り分ける**ための対照モデル。
    形状は同じ経路を通し、Expand を挟まない点だけが違う。
    """
    b.init(_w("wk", (H, NKV * HD)))
    b.init(_c("kvshape", np.array([1, S, NKV, HD], dtype=np.int64)))
    b.init(_c("ax", np.array([2], dtype=np.int64)))
    b.init(_c("rs", np.array([1, NKV, S, HD], dtype=np.int64)))
    b.init(_w("w", (HD, HD)))
    b.add("MatMul", ["x", "wk"], ["k"], "proj_k")
    b.add("Reshape", ["k", "kvshape"], ["kr"], "rs_k")
    b.add("Transpose", ["kr"], ["kt"], "tr_k", perm=[0, 2, 1, 3])
    b.add("Unsqueeze", ["kt", "ax"], ["u"], "unsq")
    b.add("Reshape", ["u", "rs"], ["r"], "reshape")
    b.add("MatMul", ["r", "w"], ["y"], "matmul")
    return [_vi("x", (1, S, H))], [_vi("y", (1, NKV, S, HD))]


def m_expand_only(b: Builder):
    """#10b `Expand` だけ（Unsqueeze なし）。上の対照。

    既に 5次元になっている必要がないよう、ヘッド軸が 1 のテンソルを
    Expand で NH まで広げる形にしてある（実質 GQA 複製と同じ動作）。
    """
    b.init(_w("wk", (H, HD)))
    b.init(_c("kvshape", np.array([1, S, 1, HD], dtype=np.int64)))
    b.init(_c("ex", np.array([1, NH, S, HD], dtype=np.int64)))
    b.init(_w("w", (HD, HD)))
    b.add("MatMul", ["x", "wk"], ["k"], "proj_k")
    b.add("Reshape", ["k", "kvshape"], ["kr"], "rs_k")
    b.add("Transpose", ["kr"], ["kt"], "tr_k", perm=[0, 2, 1, 3])
    b.add("Expand", ["kt", "ex"], ["e"], "expand")
    b.add("MatMul", ["e", "w"], ["y"], "matmul")
    return [_vi("x", (1, S, H))], [_vi("y", (1, NH, S, HD))]


def m_gqa_concat(b: Builder):
    """#10c GQA 複製を `Concat` で行う版（**回避策の候補**）。

    `block`（推測形）が端末で動いているので Concat は使えることが分かっている。
    Expand が原因なら、実 export の Expand をこの形へ書き換えれば通るはず。
    ここではその置き換え後の形を**単体で**確かめる。
    """
    b.init(_w("wk", (H, NKV * HD)))
    b.init(_c("kvshape", np.array([1, S, NKV, HD], dtype=np.int64)))
    b.init(_w("w", (HD, HD)))
    b.add("MatMul", ["x", "wk"], ["k"], "proj_k")
    b.add("Reshape", ["k", "kvshape"], ["kr"], "rs_k")
    b.add("Transpose", ["kr"], ["kt"], "tr_k", perm=[0, 2, 1, 3])
    b.add("Concat", ["kt"] * (NH // NKV), ["c"], "rep", axis=1)
    b.add("MatMul", ["c", "w"], ["y"], "matmul")
    return [_vi("x", (1, S, H))], [_vi("y", (1, NH, S, HD))]


# name -> (builder, 狙い, 優先度, 標準opsetか)
# 優先度の「Gate」は、それが通れば P2b を合格とする関門のモデル。標準 opset 外（False）は checker を通さない。
MODELS = {
    "matmul_weight": (m_matmul_weight, "重み固定 MatMul。本線 A16W8 の成立確認", "最高", True),
    "matmul_act":    (m_matmul_act, "両辺が活性化の MatMul（QK^T / PV）。(u16,u16) の可否", "最高", True),
    "rmsnorm":       (m_rmsnorm, "RMSNorm を素の op へ分解。16a16w に落ちないか", "最高", True),
    "rmsnorm_fused": (m_rmsnorm_fused, "SimplifiedLayerNormalization（融合形）", "中", False),
    "layernorm":     (m_layernorm, "LayerNormalization（opset 17 標準）", "高", True),
    "rope":          (m_rope, "RoPE を素の op へ分解。Split/Neg/Concat の可否", "高", True),
    "rope_fused":    (m_rope_fused, "com.microsoft.RotaryEmbedding。QNN_OP_ROPE に落ちるか", "高", False),
    "softmax":       (m_softmax, "Softmax。出力 encoding の扱い", "中", True),
    "silu":          (m_silu, "SiLU。SwiGLU の活性化。A8/A16 で差が出るはず", "中", True),
    "attention":     (m_attention, "attention 全体を手分解（fused GQA は HTP 非対応）", "高", True),
    "kvcache":       (m_kvcache, "固定長KVキャッシュの decode 1ステップ。多入出力", "高", True),
    "embed":         (m_embed, "Gather 埋め込み表引き + 形状操作。int64 索引", "中", True),
    "block":         (m_block, "Transformer 1ブロック（推測形）", "Gate", True),
    # --- ここから下は 2026-09-12 に本物の Qwen2.5-0.5B を export して判明した実形 ---
    "gqa_repeat":    (m_gqa_repeat, "GQAヘッド複製（実形）。**Unsqueeze/Expand が未検証**", "最高", True),
    "unsqueeze_only": (m_unsqueeze_only, "Unsqueeze 単独（Expand なし）。切り分け用", "最高", True),
    "expand_only":   (m_expand_only, "Expand 単独（Unsqueeze なし）。切り分け用", "最高", True),
    "gqa_concat":    (m_gqa_concat, "GQA複製を Concat で。**回避策の候補**", "最高", True),
    "rope_slice":    (m_rope_slice, "RoPE（実形）。rotate_half が **Slice**", "最高", True),
    "rmsnorm_real":  (m_rmsnorm_real, "RMSNorm（実形）。Div ではなく **Reciprocal**", "最高", True),
    "block_real":    (m_block_real, "Transformer 1ブロック（**実形**）。**本命の Gate**", "Gate", True),
    # --- 2026-09-13 追加: マスク入力版 prefill の err 1002 切り分け ---
    "attn_mask_input": (m_attn_mask_input, "attention のマスクを**グラフ入力**に。定数版との差はそこだけ", "最高", True),
}


def build_one(name: str, out_dir: Path) -> dict:
    """名前の部品モデルを組み立てて保存し、パイプラインが読む spec（入力の形・型・狙い）を書いて返す。"""
    fn, note, prio, standard = MODELS[name]
    b = Builder()
    inputs, outputs = fn(b)

    # ① 標準 opset 17 と、使った独自ドメイン（com.microsoft など）を宣言する。IR は 9 に固定。
    opsets = [helper.make_opsetid("", 17)]
    domains = sorted({n.domain for n in b.nodes if n.domain})
    for d in domains:
        opsets.append(helper.make_opsetid(d, 1))

    graph = helper.make_graph(b.nodes, f"p2b_{name}", inputs, outputs, initializer=b.inits)
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 9

    if standard:
        onnx.checker.check_model(model)
    # 標準 opset 外は checker を通さない。通さなかったことを spec に残す
    # （黙って飛ばすと「検査済み」と誤解される）。

    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_fp32.onnx"
    onnx.save(model, str(path))

    def io_spec(vis, is_input: bool):
        out = []
        for vi in vis:
            tt = vi.type.tensor_type
            shape = [dim.dim_value for dim in tt.shape.dim]
            e = {"name": vi.name, "shape": shape,
                 "dtype": onnx.TensorProto.DataType.Name(tt.elem_type).lower()}
            if is_input and tt.elem_type == TensorProto.INT64:
                # 索引として使われる入力。値域を外すと Gather が範囲外になる。
                e["int_max"] = (VOCAB - 1) if vi.name == "ids" else (max(shape) - 1)
            out.append(e)
        return out

    # ② spec: p2b_ops が読む。shape_config は縮小した形の値（どの形で試したかの記録）。
    spec = {
        "name": name,
        "note": note,
        "priority": prio,
        "standard_opset": standard,
        "checker_passed": bool(standard),
        "model": str(path).replace("\\", "/"),
        "nodes": len(b.nodes),
        "op_types": sorted({n.op_type for n in b.nodes}),
        "domains": domains,
        "inputs": io_spec(inputs, True),
        "outputs": io_spec(outputs, False),
        "shape_config": {"hidden": H, "heads": NH, "kv_heads": NKV,
                         "head_dim": HD, "seq": S, "past": P,
                         "intermediate": FF, "vocab": VOCAB},
    }
    (d / f"{name}.spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec


def main() -> None:
    """--list なら一覧を出すだけ。そうでなければ指定（既定は全部）のモデルを作って manifest.json を書く。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("work/ops"))
    ap.add_argument("--only", nargs="*", help="生成する名前（既定: 全部）")
    ap.add_argument("--list", action="store_true", help="一覧だけ出す")
    args = ap.parse_args()

    if args.list:
        for n, (_, note, prio, std) in MODELS.items():
            kind = "std" if std else "ext"
            print(f"{n:16s} [{prio:4s}] {kind} {note}")
        return

    names = args.only or list(MODELS)
    manifest = []
    for n in names:
        if n not in MODELS:
            raise SystemExit(f"未知のモデル: {n}（--list で一覧）")
        spec = build_one(n, args.out)
        manifest.append(spec)
        flag = "" if spec["standard_opset"] else "  ※標準opset外（checker未実施）"
        print(f"WROTE {spec['model']}  nodes={spec['nodes']} "
              f"in={len(spec['inputs'])} out={len(spec['outputs'])}{flag}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} models -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
