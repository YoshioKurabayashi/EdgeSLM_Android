"""C-1〜C-5 の機械的検証（基本設計② 5章）

要件定義書 3.4節の禁止事項は、違反すると**端末でのロード時に error 1002 で落ちる**。
しかもエラーは変換時ではなく**端末に持っていった後**に出るため、気付くのが遅い。
したがって各工程の出口でここを通し、**失敗したら次工程へ進ませない**。

人間のレビューに委ねない。件数だけでなく**どのノードが違反しているか**を列挙する
（「どこが float のままか」が分からないと直せないため）。
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
    @property
    def ok(self) -> bool:
        return bool(self.get("pass"))


def _tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    """全テンソルの elem_type を集める（shape inference 済みモデル前提）。"""
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

    return CheckResult({
        "check": "C-1 float セグメント残存ゼロ",
        "pass": not violations,
        "violations": violations,
        "detail": ("計算ノード間・グラフI/O境界ともに float の残存はない"
                   if not violations else
                   f"{len(violations)} 箇所で float が量子化を経ずに流れている"),
    })


def check_c5_no_batchnorm(model_path: Path) -> CheckResult:
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
    bad = [c for c in commands
           if "--config" in c or "backend_extensions" in c or "--backend_extension" in c]
    return CheckResult({
        "check": "C-3 backend_extensions config 未使用",
        "pass": not bad,
        "violations": bad,
        "detail": "config 指定なし" if not bad else "生成コマンドに config 指定がある",
    })


def check_c4_preserve_io(commands: list[str], c1: CheckResult) -> CheckResult:
    """C-4: --preserve_io_datatype は完全QDQに対してのみ付ける。"""
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
    commands = commands or []
    c1 = check_c1_no_float_segments(qdq_path)
    checks = [c1, check_c5_no_batchnorm(qdq_path)]
    if meta_json is not None:
        checks.append(check_c2_socmodel_zero(meta_json))
    checks.append(check_c3_no_backend_config(commands))
    checks.append(check_c4_preserve_io(commands, c1))
    return {"all_pass": all(c.ok for c in checks), "checks": checks}


def main() -> None:
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
