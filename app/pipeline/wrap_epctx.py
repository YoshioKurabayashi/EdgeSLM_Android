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

（2026-10-18 追記: 学習用の説明）
■ 何をするか
  端末の ORT（QNN EP）は context binary（HTP 向けにホストで先にコンパイルしたもの）を直接は
  読めないので、「EPContext」というノード 1 個だけの ONNX を作り、その属性に binary そのもの
  （embed_mode=1）か binary のファイル名（embed_mode=0）を入れる。ORT はこの ONNX を開くと、
  中のノードを QNN EP にそのまま渡す（StudyMemo 1.4 節 ⑧）。
■ どの段で使うか
  S8（パイプラインの最後）。context 生成（S6、scripts/build_context.sh）の後。
  - 本番の配備: deploy_pm64.py が run(..., embed=False) で呼ぶ（embed_mode=0、C-12）。
    deploy_pm64.py の無いプロジェクト（Gemma4・VLM など）では、組み立て用のスクリプト
    （build_full.py・build_vit_ctx.py など）が呼ぶ
  - 単体 op の検証: p2b_ops.finalize() が run() を既定の embed=True で呼ぶ
■ 入力と出力
  入力: context binary（.bin）と、入出力の名前・形・型を写すための QDQ の ONNX。
  出力: EPContext の ONNX（embed_mode=0 なら同じフォルダに .bin も置く）と、その情報の辞書。
■ 動かし方（単体。embed_mode=1 になる。main() の注意を参照）
  python -m app.pipeline.wrap_epctx --ctx <ctx.bin> --qdq <qdq.onnx> --out <epctx.onnx> --tag <名前>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# ZipDepth の出荷済み動作版と同じ属性構成（要件定義書 3.3.2節 / 基本設計② S8）
# partition_name と source は、ORT が「この EPContext は QNN EP のもの」と判断するのに使う値
PARTITION_NAME = "QNNExecutionProvider_QNN_0"
SOURCE = "QNNExecutionProvider"


def _io_from(model: onnx.ModelProto):
    """QDQ モデルのグラフ入力（初期化子を除く）と出力から、(名前, 形, 型) のリストを 2 つ作る。

    戻り値: ([(入力名, [次元…], onnx の型番号), …], [(出力名, …), …])。
    EPContext の ONNX の入出力は、量子化前と同じ名前・形・型でなければ端末の呼び出し側
    （job の入力ファイル）と合わない。--preserve_io_datatype で I/O は float32 のまま残るが、
    整数の入力（埋め込みの索引など）は int64 のまま残るので、型も写す。
    """
    g = model.graph
    inits = {i.name for i in g.initializer}
    ins = [v for v in g.input if v.name not in inits]
    outs = list(g.output)
    if not ins or not outs:
        raise RuntimeError("QDQ モデルに入力または出力が無い")

    def spec(vi):
        """1 本の入出力の (名前, 形, 型) を返す。次元が決まっていなければ 1 にする。"""
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

    引数: ctx_bin（context binary）、qdq_src（入出力を写す QDQ）、out_onnx（書き出し先）、
          tag（ノード名・グラフ名に付ける名前）、sdk_version（記録用の QAIRT の版）、embed。
    戻り値: 出力パス・バイト数・context のバイト数・embed_mode・入出力の名前と形と型などの辞書。
    （C-12。要件定義書の表では、embed_mode=0 で ONNX が 772MB → 442 バイトになった）
    """
    # ① context binary を読み、QDQ モデルから入出力の名前・形・型を写す（QDQ の重みも読み込まれる）
    ctx = ctx_bin.read_bytes()
    src = onnx.load(str(qdq_src))
    ins, outs = _io_from(src)

    # ② EPContext ノードを 1 個作る（ドメインは com.microsoft）。
    #   embed_mode=1 なら ep_cache_context に binary のバイト列そのもの、0 ならファイル名を入れる。
    #   main_context=1 は「この binary を持つ主のノード」の印（ZipDepth の既知正解と同じ構成）
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
    # ③ そのノードだけのグラフとモデルを作る。opset は標準 17 と com.microsoft 1、IR は 9（既知正解と同じ）
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
    # ④ 保存して、ONNX の検査器で形式が正しいかを確かめる（端末に持っていく前に壊れた ONNX を止める）
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
    """コマンドの入口。--ctx / --qdq / --out / --tag / --sdk-version を受けて run() を呼び、入出力を表示する。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=Path, required=True, help="qnn-context-binary-generator の .bin")
    ap.add_argument("--qdq", type=Path, required=True, help="I/O 名と形状の参照元（QDQ ONNX）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sdk-version", default="2.48.0")
    args = ap.parse_args()

    # ★注意（2026-10-18 追記）: このコマンドには embed の切り替えが無く、run() の既定の embed=True
    # （embed_mode=1。ONNX が context binary を丸ごと抱える）で包む。大きいモデルは端末で二重持ちに
    # なって SIGKILL される（C-12）。本番の embed_mode=0 は deploy_pm64 が run(..., embed=False) で呼んで作る。
    r = run(args.ctx, args.qdq, args.out, args.tag, args.sdk_version)
    print(f"WROTE {r['epctx_path']}  {r['bytes']/1e6:.2f} MB "
          f"(context {r['context_bytes']/1e6:.2f} MB)")
    for i in r["inputs"]:
        print(f"  in  {i['name']} {i['shape']} {i['dtype']}")
    for o in r["outputs"]:
        print(f"  out {o['name']} {o['shape']} {o['dtype']}")


if __name__ == "__main__":
    main()
