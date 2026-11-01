"""P3 準備: 本物の Qwen2.5-0.5B を ONNX に export し、**実際に出てくる op を数える**

## なぜこれを最初にやるのか

P2b では Transformer の構成要素13種を**手で書いて**検証した。しかしあれは
「HuggingFace の export はこういう op を出すだろう」という**私の推測**である。
実際の export が違う形を出すなら、P2b の結論はそのままは転用できない。

推測した箇所と、違いうる形:

| 部位 | P2b で書いた形 | 実際はこうかもしれない |
|---|---|---|
| GQA のヘッド複製 | `Concat` ×rep | `Expand` / `Tile` / `Unsqueeze`+`Expand`+`Reshape` |
| 因果マスク | `Add`（定数 -60.0） | `Where` / `Trilu` / `Cast`+`Mul` |
| RMSNorm | `Pow`/`ReduceMean`/`Sqrt`/`Div`/`Mul` | 同じか、`ReduceL2` など別の形 |
| RoPE | `Split`/`Neg`/`Concat`/`Mul`/`Add` | `Slice` ベース |
| SwiGLU | `Sigmoid`+`Mul` | 同じか、`Erf` ベースの GELU 相当 |

**C-8（活性化×活性化の MatMul が 16bit で通らない）だけは実装に依存しない。**
QK^T と PV はどう書いても両辺が活性化だからである。

## 注意: `attn_implementation` で graph が変わる

`sdpa`（既定）だと `scaled_dot_product_attention` が融合されて出るため、
**HTP に載せたい形（明示的な MatMul / Softmax / MatMul）にならない**。
本 export は `eager` を強制する。これは「HTP 向けにこう export すべき」という
方針そのものなので、既定に任せずここで明示する。

## 層数について

op の**種類**を数えるだけなら 24層すべては要らない（同じブロックの繰り返し）。
`--layers 2` で構造を保ったまま小さく出せる。重みは本物のまま先頭 N 層を使う。
精度と容量を測る段になったら `--layers 0`（全層）にする。

## 学習用の補足（2026-10-18 追記）

- 段: P2b の続き〜P3 の準備（results/P2b_20260912_real_form.md の「1. 発端」）。本物の export を数えたら、
  手書きの推測が 4 箇所外れていた（GQA の複製が Unsqueeze→Expand→Reshape、RoPE が Slice ×2、
  RMSNorm の除算が Sqrt→Reciprocal→Mul、マスク定数が -3.4028235e+38）。これが C-9・C-10 の発見につながった
- 後継: KV キャッシュ付きの decode と prefill は export_qwen_decode.py が書き出す（重みは本物・配線は自前）。
  この道具は「素の forward」と「層 1 枚」だけ
- 入力: HF のモデル ID（既定 Qwen/Qwen2.5-0.5B-Instruct。torch・transformers が要る）
- 出力: --out（既定 work/qwen）の qwen05b_<tag>.onnx と <tag>_inventory.json（op の数・P2b との差分）。
  tag は prefill なら prefill_L<層数>_S<長さ>、layer なら layer1_S<長さ>
- 動かし方（プロジェクトの直下）:
    python -m app.pipeline.export_qwen --layers 2 --seq 16                # 2 層の prefill（op の種類を数える）
    python -m app.pipeline.export_qwen --mode layer --seq 16 --opset 21   # 層 1 枚（P3 の精度の測定用。opset は例）
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

# 既定のモデル（Qwen2.5-0.5B-Instruct: 24 層・hidden 896・14 ヘッド・KV 2 ヘッド。results/P3_20260912_accuracy_layer1.md）
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load(model_id: str, layers: int):
    """HF のモデルを fp32・eager の attention で読み、layers > 0 なら先頭の layers 層だけを残す。

    戻り値: (モデル, 元の層の数)。layers が 0 か、元の層の数以上なら全層のまま
    """
    from transformers import AutoModelForCausalLM

    kwargs = dict(attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kwargs)
    except TypeError:
        # transformers 4.x は torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, **kwargs)
    model.eval()

    n_full = len(model.model.layers)
    if layers and 0 < layers < n_full:
        # **重みは本物のまま**先頭 N 層だけ残す。op の種類は層数に依らない。
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:layers]))
        model.config.num_hidden_layers = layers
    return model, n_full


class Prefill(torch.nn.Module):
    """prefill（AR-N）。KVキャッシュを使わない素の forward。

    まず op の一覧を知るのが目的なので、キャッシュの配線は入れない。
    入れると 24層×2 の入出力が増え、**知りたい op が埋もれる**。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        """input_ids [1, S]（int64）→ logits [1, S, 語彙数]。"""
        return self.model(input_ids=input_ids, use_cache=False).logits


class OneLayer(torch.nn.Module):
    """Qwen2 の**デコーダ層1枚だけ**を export するラッパー。

    埋め込み表（151936 x 896 = 1.4億パラメータ）と lm_head を外すので、
    「**本物の重み・本物の形状でブロックが通るか**」だけを切り離して見られる。
    埋め込みと出力射影は規模の話なので別に測る（P4/P5）。

    因果マスクは**モデルが内部で作る形（`finfo.min`）をそのまま**与える。
    C-9（マスク定数の置き換え）がパイプラインで効くことも同時に確かめたいため、
    ここで手加減しない。
    """

    def __init__(self, model, seq: int):
        # 使うのはいつも 0 番目の層（--layers に関係なく）。マスクは上三角を float32 の最小値にした [1, 1, S, S]、
        # 位置は 0〜S−1 で固定（形を静的にするため、forward の外で作っておく）
        super().__init__()
        self.layer = model.model.layers[0]
        self.rotary = model.model.rotary_emb
        self.seq = seq
        neg = torch.finfo(torch.float32).min
        mask = torch.full((seq, seq), neg, dtype=torch.float32).triu(1)
        self.register_buffer("mask", mask.view(1, 1, seq, seq), persistent=False)
        self.register_buffer("pos", torch.arange(seq).unsqueeze(0), persistent=False)

    def forward(self, hidden_states):
        """hidden_states [1, S, hidden] → 層 1 枚の出力 [1, S, hidden]（RoPE の cos/sin はモデルの rotary_emb で作る）。"""
        cos, sin = self.rotary(hidden_states, self.pos)
        return self.layer(hidden_states,
                          attention_mask=self.mask,
                          position_ids=self.pos,
                          position_embeddings=(cos, sin),
                          use_cache=False)


def op_inventory(path: Path) -> dict:
    """ONNX の op を種類ごとに数える（重みは読まない）。戻り値: パス・ノード数・種類数・{op: 個数}（多い順）・opset・初期化子の数。"""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts = collections.Counter(n.op_type for n in model.graph.node)
    return {
        "path": str(path).replace("\\", "/"),
        "nodes": len(model.graph.node),
        "op_types": len(counts),
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "opset": {o.domain or "": o.version for o in model.opset_import},
        "initializers": len(model.graph.initializer),
    }


# P2b で手書きした13モデルが使った op（make_op_models.py より）
P2B_OPS = {
    "MatMul", "Pow", "ReduceMean", "Add", "Sqrt", "Div", "Mul", "Identity",
    "SimplifiedLayerNormalization", "LayerNormalization", "Split", "Neg",
    "Concat", "RotaryEmbedding", "Softmax", "Sigmoid", "Reshape", "Transpose",
    "Gather",
}


def compare(inv: dict) -> dict:
    """実 export の op の集合と、P2b で手書きした op（P2B_OPS）を比べ、両方・実 export だけ・P2b だけ、に分ける。"""
    real = set(inv["counts"])
    return {
        "in_both": sorted(real & P2B_OPS),
        "only_in_real_export": sorted(real - P2B_OPS),
        "only_in_p2b": sorted(P2B_OPS - real),
    }


def main() -> None:
    """① 読む → ② ラッパーを選ぶ（prefill か層 1 枚）→ ③ 静的形状で export → ④ op を数えて P2b と比べ、JSON に保存する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--layers", type=int, default=2,
                    help="先頭N層だけ使う（0 で全層）。op の種類は層数に依らない")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("work/qwen"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--mode", choices=["prefill", "layer"], default="prefill",
                    help="layer: デコーダ層1枚だけ（埋め込み・lm_head を外す）")
    ap.add_argument("--dynamo", action="store_true",
                    help="新しい exporter を使う。transformers 5.x のマスク生成は "
                         "aten::diff を含み、旧 TorchScript exporter では出せない")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"=== load {args.model_id} (attn_implementation=eager, fp32, CPU) ===")
    model, n_full = load(args.model_id, args.layers)
    n_used = len(model.model.layers)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  layers {n_used}/{n_full}   params {n_param/1e6:.1f}M")

    # ② 例の入力（中身は 0）。形だけが export に効く
    if args.mode == "layer":
        wrapper = OneLayer(model, args.seq)
        example = torch.zeros((1, args.seq, model.config.hidden_size), dtype=torch.float32)
        in_names, out_names = ["hidden_states"], ["out"]
        tag = f"layer1_S{args.seq}"
    else:
        wrapper = Prefill(model)
        example = torch.zeros((1, args.seq), dtype=torch.long)
        in_names, out_names = ["input_ids"], ["logits"]
        tag = f"prefill_L{n_used}_S{args.seq}"

    onnx_path = args.out / f"qwen05b_{tag}.onnx"
    print(f"=== export -> {onnx_path} ===")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (example,), str(onnx_path),
            input_names=in_names, output_names=out_names,
            # **動的軸を一切与えない**。QNN EP は動的形状を扱えないため、
            # ここで dynamic_axes を渡すと後段で必ず作り直しになる。
            dynamic_axes=None,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=args.dynamo,
        )

    # ④ 数えて比べる。★の付いた op は P2b で検証していないもの（端末で通るか確かめる必要がある）
    inv = op_inventory(onnx_path)
    cmp = compare(inv)
    report = {"model_id": args.model_id, "mode": args.mode,
              "layers_used": n_used, "layers_full": n_full,
              "seq": args.seq, "params": n_param, "inventory": inv, "compare_with_p2b": cmp}
    (args.out / f"{tag}_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== op 一覧（{inv['nodes']} ノード / {inv['op_types']} 種）===")
    for op, c in inv["counts"].items():
        mark = "  " if op in P2B_OPS else "★"
        print(f"  {mark} {op:26s} {c}")

    print("\n=== P2b の手書きモデルとの差分 ===")
    print(f"  両方にある     : {', '.join(cmp['in_both'])}")
    print(f"  ★実 export だけ: {', '.join(cmp['only_in_real_export']) or '(なし)'}")
    print(f"  P2b だけ       : {', '.join(cmp['only_in_p2b']) or '(なし)'}")
    print(f"\n  -> {args.out / (tag + '_inventory.json')}")


if __name__ == "__main__":
    main()
