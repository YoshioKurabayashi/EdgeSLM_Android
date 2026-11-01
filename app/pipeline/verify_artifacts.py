"""C-1〜C-5 の機械的検証（基本設計② 5章）

要件定義書 3.4節の禁止事項は、違反すると**端末でのロード時に error 1002 で落ちる**。
しかもエラーは変換時ではなく**端末に持っていった後**に出るため、気付くのが遅い。
したがって各工程の出口でここを通し、**失敗したら次工程へ進ませない**。

人間のレビューに委ねない。件数だけでなく**どのノードが違反しているか**を列挙する
（「どこが float のままか」が分からないと直せないため）。

（2026-10-18 追記: 学習用の説明）
■ 何のための道具か
  要件定義書 3.4 節の禁止事項（C-1〜C-5 は最初に ZipDepth の CNN で実証された 5 つ）のうち、ファイルとコマンドから
  機械的に判定できるものを調べる。どれも違反すると端末でのロードが err 1002 / 3110 で落ちる。
    C-1 float の区間が残っていない（QDQ を見る）      C-5 BatchNormalization が残っていない（QDQ を見る）
    C-2 context binary の socModel が 0（メタ JSON を見る）
    C-3 生成コマンドに backend_extensions の config が無い（コマンド文字列を見る）
    C-4 --preserve_io_datatype を付けるのは C-1 を満たすモデルだけ（コマンドと C-1 の結果）
■ どの段で使うか
  p2b_ops.prepare() の最後（S4 の出口）で C-1・C-5、p2b_ops.finalize()（S8 の後）で C-2〜C-4。
  C-1 を割ったら prepare は fail_C1_float_segment で止まり、約 1 時間の context 生成へ進ませない。
  解説は StudyMemo 2.8 節（QDQ と C-1）。
■ 入力と出力
  入力: QDQ の ONNX、context 生成のメタ JSON（*_meta.json）、生成に使ったコマンド（*_commands.txt の各行）。
  出力: CheckResult（dict。"pass"・"violations"・"detail" を持ち、.ok で合否）。
■ 動かし方（単体）
  python -m app.pipeline.verify_artifacts --qdq <qdq.onnx> [--meta <meta.json>] [--command "<コマンド>" ...]
  全部 PASS なら終了コード 0、どれか FAIL なら 1。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto

FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE, TensorProto.BFLOAT16}
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}


class CheckResult(dict):
    """検査 1 件の結果。中身は dict（"check" 名・"pass"・"violations"・"detail" など）で、
    そのまま JSON に書ける。合否は .ok（"pass" を bool にしたもの）で読む。"""
    @property
    def ok(self) -> bool:
        """合格なら True（"pass" が無ければ False）。"""
        return bool(self.get("pass"))


def _tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    """全テンソルの elem_type を集める（shape inference 済みモデル前提）。

    戻り値: {テンソル名: onnx の型番号}。グラフの入出力・value_info（形状推論で中間テンソルの
    型が入る所）・初期化子から集める。形状推論に失敗しても止めず、分かる分だけで判定する
    （その場合、型の分からない中間テンソルは検査から漏れる）。
    """
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass  # 推論できなくても value_info にある分だけで判定する
    types: dict[str, int] = {}
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    return types


def check_c1_no_float_segments(qdq_path: Path) -> CheckResult:
    """C-1: float セグメントの残存ゼロ。

    判定: **float テンソルが「計算ノード → 計算ノード」で直接流れてはならない。**
    QDQ 表現では DequantizeLinear → 計算 → QuantizeLinear の間の float は正常。
    違反は「DQ を経ずに計算ノードへ入る float」または
    「Q を経ずに計算ノードへ渡る float」である。

    float が残ると fp16 HTP op としてコンパイルされ、**v68 の skel は fp16 非対応**
    のため createFromBinary が err 1002 になる。

    ★グラフ I/O も検査する（2026-09-12 に穴が露見して追加）:
      当初「`--preserve_io_datatype` で I/O は float32 のままだから対象外」として
      グラフ I/O を丸ごと除外していた。しかしそれで正当なのは**境界そのもの**だけで、
      「float の入力が QuantizeLinear を経ずに直接計算ノードへ入る」のは立派な違反である。
      実際 `gqa_repeat`（先頭が Unsqueeze）で量子化器が入力を量子化せず、
      本検査は PASS を返したのに context 生成が落ちた:
        "QNN_DATATYPE_FLOAT_32_converted_k_QNN_DATATYPE_FLOAT_16" generated: could not create op
      P2 のモデルは先頭が Conv（必ず量子化される）だったため穴が見えていなかった。
    """
    # ① 型の表・初期化子・グラフの入出力・「誰が出して誰が使うか」の表を作る
    model = onnx.load(str(qdq_path))
    g = model.graph
    types = _tensor_types(model)
    inits = {i.name for i in g.initializer}
    graph_io = {v.name for v in list(g.input) + list(g.output)}

    producer: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in g.node:
        for o in n.output:
            if o:
                producer[o] = n
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)

    violations = []

    # --- グラフ I/O 境界の検査 ---
    # 入力: float なら全消費先が QuantizeLinear であること
    # 出力: float なら生産元が DequantizeLinear であること
    for vi in g.input:
        if vi.name in inits:
            continue
        if types.get(vi.name) not in FLOAT_TYPES:
            continue  # int64 の索引などは量子化されないので対象外
        for c in consumers.get(vi.name, []):
            if c.op_type != "QuantizeLinear":
                violations.append({
                    "tensor": vi.name,
                    "producer": "(graph input)",
                    "consumer": f"{c.op_type}({c.name or '-'})",
                    "note": "float のグラフ入力が QuantizeLinear を経ずに計算ノードへ入っている",
                })
    for vo in g.output:
        if types.get(vo.name) not in FLOAT_TYPES:
            continue
        p_out = producer.get(vo.name)
        if p_out is not None and p_out.op_type != "DequantizeLinear":
            violations.append({
                "tensor": vo.name,
                "producer": f"{p_out.op_type}({p_out.name or '-'})",
                "consumer": "(graph output)",
                "note": "float のグラフ出力が DequantizeLinear 以外から出ている",
            })

    # ② 中間テンソルの検査: float のテンソルで、出したのが Q/DQ 以外の計算ノード（＝ float の計算結果）
    #   なのに、使う側にも Q/DQ 以外の計算ノードがある → 「計算ノード → 計算ノード」を float が流れている
    for tensor, t_elem in types.items():
        if t_elem not in FLOAT_TYPES:
            continue
        if tensor in inits or tensor in graph_io:
            continue  # scale / bias / グラフI/O境界は対象外
        p = producer.get(tensor)
        if p is None or p.op_type in QDQ_OPS:
            continue  # DQ 出力の float は正常
        for c in consumers.get(tensor, []):
            if c.op_type in QDQ_OPS:
                continue  # Q への入力は正常
            violations.append({
                "tensor": tensor,
                "producer": f"{p.op_type}({p.name or '-'})",
                "consumer": f"{c.op_type}({c.name or '-'})",
            })

    # ③ 違反を 1 件ずつ（テンソル名・出したノード・使うノード）並べて返す。直す場所が分かるように
    return CheckResult({
        "check": "C-1 float セグメント残存ゼロ",
        "pass": not violations,
        "violations": violations,
        "detail": ("計算ノード間・グラフI/O境界ともに float の残存はない"
                   if not violations else
                   f"{len(violations)} 箇所で float が量子化を経ずに流れている"),
    })


def check_c5_no_batchnorm(model_path: Path) -> CheckResult:
    """C-5: BatchNormalization が残っていないこと。

    HTP は signed per-channel の Conv QDQ と BatchNorm の併存を拒否する（error 3110）。
    preprocess.run() が量子化の前に Mul + Add へ畳むので、ここでは残っていないことを確かめる。
    LLM には BatchNorm が無いので、主に P1・P2 の CNN 向けの検査。
    """
    model = onnx.load(str(model_path))
    bad = [n.name or "(no name)" for n in model.graph.node if n.op_type == "BatchNormalization"]
    return CheckResult({
        "check": "C-5 BatchNormalization 残存ゼロ",
        "pass": not bad,
        "violations": bad,
        "detail": "BatchNormalization なし" if not bad else
                  f"{len(bad)} 個残存（HTP は signed per-channel Conv QDQ との併存を拒否: error 3110）",
    })


def check_c2_socmodel_zero(meta_json: Path) -> CheckResult:
    """C-2: 生成した context binary の socModel が 0 であること。

    soc_id を刻むと unsigned PD は DSP の SoC id を照会できず、
    createFromBinary が err 1002 で拒否する。
    """
    if not meta_json.is_file():
        return CheckResult({"check": "C-2 socModel==0", "pass": False,
                            "detail": f"メタデータが無い: {meta_json}"})
    meta = json.loads(meta_json.read_text(encoding="utf-8", errors="replace"))
    found = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() in ("socmodel", "soc_model"):
                    found.append(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    # メタ JSON のどの階層にあっても拾えるように、キー名 socModel / soc_model を再帰的に探す。
    # 1 つも見つからない場合も不合格にする（「確かめられなかった」を合格にしない）
    walk(meta)
    bad = [v for v in found if str(v) not in ("0",)]
    return CheckResult({
        "check": "C-2 socModel==0",
        "pass": bool(found) and not bad,
        "found": found,
        "violations": bad,
        "detail": ("socModel=0" if found and not bad else
                   ("socModel が見つからない" if not found else f"socModel={bad} （0 でなければならない）")),
    })


def check_c3_no_backend_config(commands: list[str]) -> CheckResult:
    """C-3: 生成時に backend_extensions の config を渡していないこと。

    devices/cores セクション（perf_profile: burst など）を埋め込むと
    unsigned PD は power config を適用できずロードが失敗する。
    """
    # 文字列の部分一致で見る。"--config" を含む引数（--config_file など）があれば違反とみなす
    # （本プロジェクトの生成コマンドは --htp_socs sm8350 だけを付ける。scripts/build_context.sh）
    bad = [c for c in commands
           if "--config" in c or "backend_extensions" in c or "--backend_extension" in c]
    return CheckResult({
        "check": "C-3 backend_extensions config 未使用",
        "pass": not bad,
        "violations": bad,
        "detail": "config 指定なし" if not bad else "生成コマンドに config 指定がある",
    })


def check_c4_preserve_io(commands: list[str], c1: CheckResult) -> CheckResult:
    """C-4: --preserve_io_datatype は完全QDQに対してのみ付ける。

    引数: commands（変換・生成に使ったコマンドの行）、c1（同じモデルの C-1 の結果）。
    フラグを使っていなければ合格。使っているなら C-1 に合格していることが条件。
    float が残ったモデルにこのフラグを付けると、変換器が準備できない fp32 op を出して err 1002 になる。
    """
    used = any("--preserve_io_datatype" in c for c in commands)
    ok = (not used) or c1.ok
    return CheckResult({
        "check": "C-4 --preserve_io_datatype の前提充足",
        "pass": ok,
        "used": used,
        "detail": ("未使用" if not used else
                   ("C-1 pass 済みのモデルに付与" if ok else
                    "C-1 未達のモデルに付与している（unpreparable な fp32 op が出る）")),
    })


def verify_all(qdq_path: Path, meta_json: Path | None = None,
               commands: list[str] | None = None) -> dict:
    """C-1・C-5（と meta があれば C-2）・C-3・C-4 をまとめて流す。
    戻り値: {"all_pass": 全部合格か, "checks": [CheckResult, ...]}。main() から使う。"""
    commands = commands or []
    c1 = check_c1_no_float_segments(qdq_path)
    checks = [c1, check_c5_no_batchnorm(qdq_path)]
    if meta_json is not None:
        checks.append(check_c2_socmodel_zero(meta_json))
    checks.append(check_c3_no_backend_config(commands))
    checks.append(check_c4_preserve_io(commands, c1))
    return {"all_pass": all(c.ok for c in checks), "checks": checks}


def main() -> None:
    """コマンドの入口。verify_all() の結果を 1 行ずつ（違反は先頭 10 件まで）表示し、合否を終了コードで返す。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--command", action="append", default=[])
    args = ap.parse_args()

    r = verify_all(args.qdq, args.meta, args.command)
    for c in r["checks"]:
        mark = "OK  " if c.ok else "NG  "
        print(f"  {mark}{c['check']}: {c['detail']}")
        for v in (c.get("violations") or [])[:10]:
            print(f"        - {v}")
    print()
    print("  総合: " + ("PASS" if r["all_pass"] else "FAIL"))
    raise SystemExit(0 if r["all_pass"] else 1)


if __name__ == "__main__":
    main()
