"""S8: EPContext ONNX ラッパー生成

基本設計② S8 の実装。

QNN が生成した context binary を、ORT の QNN EP が読める ONNX として包む。
**既知正解のスキーマ**（ZipDepth の出荷済み動作版と同一の属性構成）を使う。

第一候補は ORT 公式の `gen_qnn_ctx_onnx_model.py` だが、そちらは QNN 生成 binary の
channel-last レイアウトや 8/16bit I/O を吸収するためのラッパーを作る。
本パイプラインは `--preserve_io_datatype` でグラフ I/O を float32 に保っているため、
**変換ノードを挟まない最小のラッパーで足りる**。合わない場合に公式版へ切り替える。

ただし **float32 決め打ちにはできない**。埋め込み表引きの索引のような整数入力は
量子化されず int64 のまま残るため、型は QDQ モデルから引き継ぐ（`_io_from` の注記）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# ZipDepth の出荷済み動作版と同じ属性構成（要件定義書 3.3.2節 / 基本設計② S8）
PARTITION_NAME = "QNNExecutionProvider_QNN_0"
SOURCE = "QNNExecutionProvider"


def _io_from(model: onnx.ModelProto):
    g = model.graph
    inits = {i.name for i in g.initializer}
    ins = [v for v in g.input if v.name not in inits]
    outs = list(g.output)
    if not ins or not outs:
        raise RuntimeError("QDQ モデルに入力または出力が無い")

    def spec(vi):
        shape = []
        for d in vi.type.tensor_type.shape.dim:
            # バッチは 1 に固定する（ZipDepth の既知正解と同じ）。
            # QNN EP は動的形状を扱えないため、ここで残すと端末で失敗する。
            shape.append(d.dim_value if d.dim_value > 0 else 1)
        # ★型は QDQ モデルから**そのまま引き継ぐ**。float32 決め打ちにしてはいけない。
        #   2026-09-12 に踏んだ: 埋め込み表引きの `ids`(int64) を float32 と宣言したため、
        #   ロードは成功するのに Run で落ちた:
        #     ExecuteGraph ORT Tensor data size does not match QNN tensor data size
        #   ONNX 側 4byte × N に対し QNN 側 8byte × N を期待するので**バイト数が合わない**。
        #   「ロードは通るのに実行で落ちる」ため、C-1〜C-8 とは別の症状として現れる。
        return vi.name, shape, vi.type.tensor_type.elem_type
    return [spec(v) for v in ins], [spec(v) for v in outs]


def run(ctx_bin: Path, qdq_src: Path, out_onnx: Path, tag: str,
        sdk_version: str = "2.48.0", embed: bool = True) -> dict:
    """embed=False にすると context binary を**別ファイル**として参照する。

    ★大きなモデルで必須（2026-09-13 に実機で踏んだ）:
      embed=True だと ONNX が context binary を丸ごと抱えるため、
      ロード時に **ONNX 本体 + QNN 側のコピー**で約2倍のメモリを使う。
      24層（binary 736MB / ONNX 772MB）では RSS が **1.47GB** まで伸び、
      Android に SIGKILL された。embed=False なら二重持ちを避けられる。
      `ep_cache_context` にはファイル名を入れ、**ONNX と同じディレクトリ**に置く。
    """
    ctx = ctx_bin.read_bytes()
    src = onnx.load(str(qdq_src))
    ins, outs = _io_from(src)

    node = helper.make_node(
        "EPContext",
        name=f"{tag}_QNN_ctx",
        inputs=[n for n, _, _ in ins],
        outputs=[n for n, _, _ in outs],
        domain="com.microsoft",
        embed_mode=1 if embed else 0,
        ep_cache_context=(ctx if embed else ctx_bin.name),
        ep_sdk_version=sdk_version,
        main_context=1,
        notes=f"EdgeSLM AoT context {tag}, SM8350/V68, QAIRT {sdk_version}",
        partition_name=PARTITION_NAME,
        source=SOURCE,
    )
    graph = helper.make_graph(
        [node],
        f"edgeslm_epctx_{tag}",
        [helper.make_tensor_value_info(n, t, sh) for n, sh, t in ins],
        [helper.make_tensor_value_info(n, t, sh) for n, sh, t in outs],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 9
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    if not embed:
        # binary を ONNX と同じ場所へ置く（相対パスで解決されるため）
        side = out_onnx.parent / ctx_bin.name
        if side.resolve() != ctx_bin.resolve():
            side.write_bytes(ctx)
    onnx.save(model, str(out_onnx))
    onnx.checker.check_model(str(out_onnx))

    return {
        "epctx_path": str(out_onnx),
        "bytes": out_onnx.stat().st_size,
        "context_bytes": len(ctx),
        "embed_mode": 1 if embed else 0,
        "context_file": None if embed else ctx_bin.name,
        "inputs": [{"name": n, "shape": sh,
                    "dtype": TensorProto.DataType.Name(t).lower()} for n, sh, t in ins],
        "outputs": [{"name": n, "shape": sh,
                     "dtype": TensorProto.DataType.Name(t).lower()} for n, sh, t in outs],
        "ep_sdk_version": sdk_version,
        "partition_name": PARTITION_NAME,
        "source": SOURCE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=Path, required=True, help="qnn-context-binary-generator の .bin")
    ap.add_argument("--qdq", type=Path, required=True, help="I/O 名と形状の参照元（QDQ ONNX）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sdk-version", default="2.48.0")
    args = ap.parse_args()

    r = run(args.ctx, args.qdq, args.out, args.tag, args.sdk_version)
    print(f"WROTE {r['epctx_path']}  {r['bytes']/1e6:.2f} MB "
          f"(context {r['context_bytes']/1e6:.2f} MB)")
    for i in r["inputs"]:
        print(f"  in  {i['name']} {i['shape']} {i['dtype']}")
    for o in r["outputs"]:
        print(f"  out {o['name']} {o['shape']} {o['dtype']}")


if __name__ == "__main__":
    main()
