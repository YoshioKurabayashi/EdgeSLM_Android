# -*- coding: utf-8 -*-
"""演習用: 本物のソースから**関数の定義だけ**を取り出して使う（2026-10-14）。

道具のファイルは先頭で onnx・onnxruntime・torch などを import しているので、そのまま import すると
それらが要る。演習で使いたいのは numpy だけで動く小さな関数（指標の式・量子化の式など）なので、
ファイルを構文木として読み、指定した名前の関数（と定数）の定義だけを numpy と一緒に実行する。
**書き写さないので、ソースを直せば演習もそのまま追従する**（写し間違いが起きない）。
"""
from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "README.md").exists())


def load(rel: str, names: list[str], extra_globals: dict | None = None) -> dict:
    """rel（プロジェクトからの相対パス）の中の、names に挙げた関数・クラス・定数の定義だけを実行して返す"""
    import numpy as np
    src = (ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                keep.append(node)
    found = {getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) else n.target.id) for n in keep}
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"{rel} に {missing} が無い（ソースが変わった？）")
    g = {"np": np, "__name__": "study_src", **(extra_globals or {})}
    mod = ast.Module(body=keep, type_ignores=[])
    import __future__
    # 型注釈は評価しない（onnx.ModelProto などの注釈があっても、onnx を入れずに動くように）
    exec(compile(mod, str(ROOT / rel), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), g)
    return {n: g[n] for n in names}
