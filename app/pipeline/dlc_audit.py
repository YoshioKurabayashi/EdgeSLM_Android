r"""**DLC（HTP が実際に実行する表現）を検査する。** ONNX/QDQ の設定は見ない。

## なぜ必要か（2026-09-14 に踏んだ）

ONNX 側で RMSNorm の gamma を **uint16**（zero_point=0、scale で [0, 1.99] を表現）に
量子化していた。ところが converter は RMSNorm を融合 op `RmsNorm` にし、
gamma を **`sFxp_16`（符号付き）** として渡していた。

    DLC: data type: sFxp_16
         encoding : bitwidth 16, min 0.0, max 1.9921875, scale 3.0399e-05, offset 0
                    ↑ 型は符号付き、エンコーディングは符号なし前提

**量子化値が 32767 を超える要素が黙って頭打ちになる。** 警告もエラーも出ない。
RMSNorm 単体で相対L2 13.54%、24層では実機の誤差の主因になっていた。

`convert.log` には手がかりが無い（`RMSNormalization` という文字列は
「この op は登録できなかった」という**起動時の警告**であって中身とは無関係）。
**`qairt-dlc-info` で DLC を開くのが唯一の確認方法である。**

## パイプラインのどこで使うか

    S4 量子化 → S5 DLC変換 → **ここで検査** → S6 context生成 → 配備 → 実機

`deploy_pm64.py` の push 前ゲートとして呼ばれる。不整合があれば**端末へ送る前に止まる**。
ONNX 側の検査（`weight_percg` / `fix_norm_gamma --audit`）と**両方**要る。
融合されると ONNX のノードは消えるので、ONNX だけ見ていても気付けない。

## 検査すること

型（signed/unsigned）・bitwidth・scale・offset・min/max を突き合わせ、
**符号付きの型にエンコーディングの範囲が収まっているか**を確かめる。

## 使い方

    python -m app.pipeline.dlc_audit --dlc work/ops/<name>/ctx/<tag>.dlc
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

QAIRT = Path(os.environ.get("QAIRT_ROOT", r"D:\QAIRT\v2.48.0.260626\qairt\2.48.0.260626"))
QAIRT_ENV = os.environ.get("QAIRT_PYENV", r"D:\QAIRT\qairtenv")

# 型ごとの表現できる量子化値の範囲
LIMITS = {"sFxp_8": (-128, 127), "uFxp_8": (0, 255),
          "sFxp_16": (-32768, 32767), "uFxp_16": (0, 65535),
          "sFxp_32": (-2**31, 2**31 - 1), "uFxp_32": (0, 2**32 - 1)}

RE_TYPE = re.compile(r"([A-Za-z0-9_.]+) \(data type: ([a-zA-Z_0-9]+)")
RE_ENC = re.compile(r"([A-Za-z0-9_.]+) encoding : bitwidth (\d+), min ([-0-9.eE]+), "
                    r"max ([-0-9.eE]+), scale ([0-9.eE-]+), offset ([-0-9.eE]+)")


def dump(dlc: Path) -> str:
    """`qairt-dlc-info` を呼んで DLC の中身をテキストで得る。"""
    qw = str(QAIRT)
    bin_dir = QAIRT / 'bin/x86_64-windows-msvc'
    lib_dir = QAIRT / 'lib/x86_64-windows-msvc'
    env = dict(os.environ)
    env['VIRTUAL_ENV'] = QAIRT_ENV
    env['PYTHONPATH'] = str(QAIRT / 'lib/python')
    env['PYTHONUTF8'] = '1'
    env['PATH'] = os.pathsep.join([str(bin_dir), str(lib_dir), env.get('PATH', '')])
    r = subprocess.run([str(Path(QAIRT_ENV) / 'Scripts/python.exe'),
                        str(bin_dir / 'qairt-dlc-info'),
                        '-i', str(dlc.resolve())],
                       capture_output=True, text=True, env=env, errors='replace')
    if r.returncode != 0 and not r.stdout:
        raise SystemExit('qairt-dlc-info に失敗: ' + r.stderr[-2000:])
    return r.stdout


def audit(dlc: Path, pattern: str | None = None) -> dict:
    txt = dump(dlc)
    types = {m.group(1): m.group(2) for m in RE_TYPE.finditer(txt)}
    encs = {}
    for m in RE_ENC.finditer(txt):
        encs[m.group(1)] = {"bitwidth": int(m.group(2)), "min": float(m.group(3)),
                            "max": float(m.group(4)), "scale": float(m.group(5)),
                            "offset": float(m.group(6))}
    rx = re.compile(pattern) if pattern else None
    problems, rows = [], []
    for name, e in encs.items():
        if rx and not rx.search(name):
            continue
        t = types.get(name)
        row = {"name": name, "dtype": t, **e}
        if t in LIMITS and e["scale"] > 0:
            lo, hi = LIMITS[t]
            # ★QNN の約束は `実数 = (量子化値 + offset) × scale`。
            #   したがって必要な量子化値は `実数/scale - offset` である
            #   （符号を逆にすると全テンソルが「不整合」に見える。2026-09-14 に一度間違えた）
            q_max = e["max"] / e["scale"] - e["offset"]
            q_min = e["min"] / e["scale"] - e["offset"]
            row["q_max_needed"] = q_max
            row["q_min_needed"] = q_min
            if q_max > hi + 1.5 or q_min < lo - 1.5:
                row["problem"] = (f"{t} は [{lo}, {hi}] しか表せないのに "
                                  f"エンコーディングは [{q_min:.0f}, {q_max:.0f}] を要求している")
                problems.append(row)
        rows.append(row)
    ops = re.search(r"Ops used by Graph: (.+)", txt)
    return {"dlc": str(dlc), "tensors": len(rows), "rows": rows,
            "problems": problems, "ops": ops.group(1).strip() if ops else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dlc", type=Path, required=True)
    ap.add_argument("--pattern", default=None, help="テンソル名の絞り込み（正規表現）")
    ap.add_argument("--strict", action="store_true", help="問題があれば異常終了する")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    a = audit(args.dlc, args.pattern)
    print(f"=== DLC 検査: {args.dlc.name} ===")
    print(f"  使われている op : {a['ops']}")
    print(f"  エンコーディングを持つテンソル: {a['tensors']}")
    by_type: dict[str, int] = {}
    for r in a["rows"]:
        by_type[str(r["dtype"])] = by_type.get(str(r["dtype"]), 0) + 1
    print(f"  型の内訳        : {by_type}")
    if a["problems"]:
        print(f"\n  ★型とエンコーディングが不整合: {len(a['problems'])} 本")
        for r in a["problems"][:6]:
            print(f"    {r['name']}")
            print(f"      {r['problem']}")
    else:
        print("\n  OK  型とエンコーディングの不整合なし")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.json}")
    if args.strict and a["problems"]:
        raise SystemExit("DLC の検査に失敗した")


if __name__ == "__main__":
    main()
