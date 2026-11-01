# -*- coding: utf-8 -*-
"""StudyMemo の「ソースと動かし方」「コードで読む検証」の章を、テンプレートと実際のソースから作る。

    python -B -m app.study.make_code_tour            # 作る（テンプレートごとに DoC/StudyMemo/ の章を書き直す）
    python -B -m app.study.make_code_tour --check    # 書かずに、テンプレートの印が全部ソースに見つかるかだけ確かめる

どの venv でもよい（標準ライブラリだけ）。EdgeSLM_AndroidVLMSpec の同名のスクリプト（2026-10-13）を、
テンプレートを複数持てるようにして各プロジェクトへ写したもの（2026-10-14）。

## なぜ生成にするのか

StudyMemo の「コードで読む」章は、検証のときに実際に動いたコードを**ソースの抜き出しのまま**載せて解説する。
手で写すと、ソースを直したときに抜き出し・行番号がずれ、気づかないまま古い説明が残る。
そこで解説だけをテンプレート（`<このファイルの場所>/tour/*.md`）に書き、抜き出し・行番号・結果の値は毎回ソースから取る。
抜き出しの始まりの目印（正規表現）がソースに **ちょうど 1 か所**見つからなければ止まる（ずれたら気づける）。

## テンプレート

1 行目に出力先を書く: `<<OUT DoC/StudyMemo/13_ソースと動かし方.md>>`（プロジェクトからの相対パス）

ブロックの印（その行が丸ごと置き換わる）:
  <<CODE パス | 始まり | 終わり [| from=目印] [| omit=A..B] …>>
      ソースの「始まり」に合う行から「終わり」に合う行まで（両端を含む）を、行番号つきで載せる。
      - 始まり: 正規表現。ファイルの中（from= があれば、その目印の行より後）に**ちょうど 1 か所**合うこと
      - 終わり: 正規表現（始まりの行の次の行から探して最初に合う行）、または `+N`（始まりから N 行）
      - omit=A..B: 抜き出しの中の、A に合う行の次から B に合う行の手前までを「（中略: N 行）」にする（いくつでも）
  <<DIFF 元のパス | 新しいパス>>     2 つのファイルの差分（unified 形式、前後 2 行）
  <<JSON パス | 式>>                  JSON ファイルを d として読み、式の値を整形して載せる（長いときは 60 行で切る）
  <<MD5_TABLE>>                       この章で使ったファイルの行数と md5 の表

文の途中に書ける印:
  <<L パス | 正規表現>>               その正規表現に合う行の番号（ちょうど 1 か所）を「L123」の形で
  <<VAL パス | 式>>                   JSON ファイルを d として読み、式の値を 1 行で

パスはプロジェクトからの相対パス（`../EdgeSLM_Android/…` のように、読むだけのほかのプロジェクトも指せる）。
テンプレートの中の ``` で囲んだ所では印を置き換えない（印の書き方の例を載せられるように）。
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "README.md").exists())   # プロジェクトの根（README.md のある所）
TOUR = HERE / "tour"
JSON_MAX_LINES = 60

LANG = {".py": "python", ".cpp": "cpp", ".h": "cpp", ".hpp": "cpp", ".kt": "kotlin", ".json": "json",
        ".sh": "bash", ".ps1": "powershell", ".md": "markdown", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
        ".txt": "text", ".cmake": "cmake"}
BLOCK = re.compile(r"^\s*<<(CODE|DIFF|JSON)\s+(.*)>>\s*$")
INLINE = re.compile(r"<<(L|VAL)\s+(.*?)>>")
OUTLINE = re.compile(r"^<<OUT\s+(.+?)>>\s*$")


class TourError(Exception):
    pass


_cache: dict[str, list[str]] = {}
_used: dict[str, Path] = {}                                    # この章で使ったファイル（最後の md5 の表）


def lines_of(rel: str) -> list[str]:
    if rel not in _cache:
        p = (ROOT / rel).resolve()
        if not p.exists():
            raise TourError(f"ファイルが無い: {rel}")
        _cache[rel] = p.read_text(encoding="utf-8", errors="replace").splitlines()
    _used[rel] = (ROOT / rel).resolve()
    return _cache[rel]


def find_one(rel: str, pattern: str, after: int = -1) -> int:
    """pattern に合う行（0 始まり）を、after より後から探す。ちょうど 1 か所でなければ止める"""
    lines = lines_of(rel)
    rx = re.compile(pattern)
    hits = [i for i, s in enumerate(lines) if i > after and rx.search(s)]
    if len(hits) != 1:
        where = "、".join(f"L{i + 1}" for i in hits[:6])
        raise TourError(f"{rel}: 目印 /{pattern}/ が {len(hits)} か所に合う（ちょうど 1 か所のこと）{(': ' + where) if hits else ''}")
    return hits[0]


def split_fields(body: str) -> list[str]:
    # 区切りは「 | 」（前後に空白）。正規表現の中の | （空白なし）はそのまま使える
    return [x.strip() for x in re.split(r"\s\|\s", body)]


def do_code(body: str) -> str:
    f = split_fields(body)
    if len(f) < 3:
        raise TourError(f"CODE の項目が足りない: {body}")
    rel, start, end, opts = f[0], f[1], f[2], f[3:]
    after, omits = -1, []
    for o in opts:
        if o.startswith("from="):
            after = find_one(rel, o[5:])
        elif o.startswith("omit="):
            a, _, b = o[5:].partition("..")
            omits.append((a, b))
        else:
            raise TourError(f"CODE の知らない指定: {o}")
    lines = lines_of(rel)
    s = find_one(rel, start, after)
    if re.fullmatch(r"\+\d+", end):
        e = min(len(lines) - 1, s + int(end[1:]) - 1)
    else:
        rx = re.compile(end)
        e = next((i for i in range(s + 1, len(lines)) if rx.search(lines[i])), None)
        if e is None:
            raise TourError(f"{rel}: 終わりの目印 /{end}/ が L{s + 1} より後に無い")
    keep = list(range(s, e + 1))
    out_rows: list[str] = []
    i = 0
    while i < len(keep):
        ln = keep[i]
        text = lines[ln]
        out_rows.append(f"{ln + 1:>5}| {text}")
        for a, b in omits:
            if re.search(a, text):
                rb = re.compile(b)
                j = next((k for k in range(i + 1, len(keep)) if rb.search(lines[keep[k]])), None)
                if j is None:
                    raise TourError(f"{rel}: omit の終わり /{b}/ が L{ln + 1} より後（抜き出しの中）に無い")
                n = j - i - 1
                if n > 0:
                    first = next((lines[keep[k]] for k in range(i + 1, j) if lines[keep[k]].strip()), "")
                    indent = re.match(r"\s*", first).group(0)          # 省いた行のうち、最初の空でない行の字下げに合わせる
                    out_rows.append(f"     | {indent}…（中略: {n} 行。L{keep[i + 1] + 1}〜L{keep[j - 1] + 1}）…")
                i = j - 1
                break
        i += 1
    lang = LANG.get(Path(rel).suffix, "")
    head = f"`{rel}` の L{s + 1}〜L{e + 1}"
    return f"{head}\n\n```{lang}\n" + "\n".join(out_rows) + "\n```"


def do_diff(body: str) -> str:
    f = split_fields(body)
    if len(f) != 2:
        raise TourError(f"DIFF は 2 つのパス: {body}")
    a, b = lines_of(f[0]), lines_of(f[1])
    d = list(difflib.unified_diff(a, b, fromfile=f[0], tofile=f[1], n=2, lineterm=""))
    if not d:
        return f"（`{f[0]}` と `{f[1]}` は同じ）"
    return "```diff\n" + "\n".join(d) + "\n```"


SAFE = {"len": len, "round": round, "sorted": sorted, "list": list, "dict": dict, "sum": sum, "min": min, "max": max,
        "next": next, "str": str, "tuple": tuple, "float": float, "int": int, "abs": abs, "any": any, "all": all,
        "enumerate": enumerate, "zip": zip, "range": range}


def eval_json(rel: str, expr: str):
    p = (ROOT / rel).resolve()
    if not p.exists():
        raise TourError(f"ファイルが無い: {rel}")
    _used[rel] = p
    d = json.loads(p.read_text(encoding="utf-8"))
    try:
        return eval(expr, {"__builtins__": SAFE, "d": d})       # d は globals に置く（内包表記の中からも見えるように）
    except Exception as e:                                       # 式の誤り・キーが無い
        raise TourError(f"{rel}: 式 {expr} が評価できない: {type(e).__name__}: {e}")


def do_json(body: str) -> str:
    f = split_fields(body)
    if len(f) != 2:
        raise TourError(f"JSON はパスと式: {body}")
    v = eval_json(f[0], f[1])
    txt = json.dumps(v, ensure_ascii=False, indent=1).splitlines()
    if len(txt) > JSON_MAX_LINES:
        txt = txt[:JSON_MAX_LINES] + [f"…（あと {len(txt) - JSON_MAX_LINES} 行）"]
    return f"`{f[0]}` から\n\n```json\n" + "\n".join(txt) + "\n```"


def do_inline(m: re.Match) -> str:
    kind, body = m.group(1), m.group(2)
    f = split_fields(body)
    if len(f) != 2:
        raise TourError(f"{kind} はパスと目印（式）: {body}")
    if kind == "L":
        return f"L{find_one(f[0], f[1]) + 1}"
    v = eval_json(f[0], f[1])
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render(template: str, name: str) -> tuple[str | None, str, list[str]]:
    lines = template.splitlines()
    m = OUTLINE.match(lines[0]) if lines else None
    if not m:
        return None, "", [f"{name}: 1 行目に <<OUT 出力先>> が無い"]
    out, errors, in_fence = [], [], False
    for n, line in enumerate(lines[1:], 2):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        try:
            bm = BLOCK.match(line) if not in_fence else None
            if bm:
                out.append({"CODE": do_code, "DIFF": do_diff, "JSON": do_json}[bm.group(1)](bm.group(2)))
            else:
                out.append(line if in_fence else INLINE.sub(do_inline, line))
        except TourError as e:
            errors.append(f"{name} の {n} 行目: {e}")
            out.append(f"**（作れなかった: {e}）**")
    return m.group(1).strip(), "\n".join(out) + "\n", errors


def md5_table() -> str:
    rows = ["| ファイル | 行数 | md5（先頭 8 桁） |", "|---|---:|---|"]
    for rel in sorted(_used):
        p = _used[rel]
        h = hashlib.md5(p.read_bytes()).hexdigest()[:8]
        n = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        rows.append(f"| `{rel}` | {n:,} | {h} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="書かずに、印が全部解決できるかだけ確かめる")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    temps = sorted(TOUR.glob("*.md"))
    if not temps:
        raise SystemExit(f"テンプレートが無い: {TOUR}")
    all_errors, done = [], []
    for t in temps:
        _used.clear()
        out_rel, text, errors = render(t.read_text(encoding="utf-8"), t.name)
        all_errors += errors
        if out_rel is None:
            continue
        text = text.replace("<<MD5_TABLE>>", md5_table())
        n_code = len(re.findall(r"^`[^`]+` の L\d+〜L\d+$", text, flags=re.M))
        done.append((t.name, out_rel, text, n_code, len(_used)))
    if all_errors:
        print(f"★印が {len(all_errors)} 個解決できない:")
        for e in all_errors:
            print("  " + e)
        raise SystemExit(1)
    for name, out_rel, text, n_code, n_src in done:
        print(f"{name} → {out_rel}: 抜き出し {n_code} 個・ソース {n_src} 本・{len(text.splitlines()):,} 行")
        if not a.check:
            p = ROOT / out_rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
    print("（--check なので書いていない）" if a.check else "書いた")


if __name__ == "__main__":
    main()
