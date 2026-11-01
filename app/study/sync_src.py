# -*- coding: utf-8 -*-
"""StudyMemo の学習の章（手書き）に、解説している所の**実際のソース**を埋め込み、ソースから自動で更新する（2026-10-17）。

    python -B -m <この場所>.sync_src            # 学習の章の「枠」を全部ソースから書き直す（変わった枠だけ）
    python -B -m <この場所>.sync_src --check    # 書かずに、目印が全部解決できるか・枠が古くなっていないかを確かめる
    python -B -m <この場所>.sync_src DoC/StudyMemo/03_xx.md …   # 指定した章だけ（目次の一覧 srcindex は書き直さない）

どの venv でもよい（標準ライブラリだけ）。make_code_tour.py（地図・コードの章を丸ごと生成する）とは別の道具で、
こちらは**手書きの章の中の決まった枠だけ**を書き換える。枠の外（解説・表）には触らない。

## なぜ「枠」にするのか

解説の横にソースを載せると、ソースを直したときに抜き出しが古くなり、気づかないまま間違った説明が残る。
解説は手で直したいので章を丸ごと生成にはできない。そこで、解説の中に印で囲んだ枠を置き、**枠の中だけ**を毎回ソースから作り直す。
始まりの目印がソースに**ちょうど 1 か所**見つからなければ止まるので、ソースが変わったら必ず気づける。

## 枠の書き方（学習の章の .md の中。印の行はそのまま残る＝Markdown では見えない）

    <!-- src: パス | 始まり | 終わり [| from=目印] [| omit=A..B] [| note=一言] -->
    （ここは sync_src が書く。手で書いても次の実行で上書きされる）
    <!-- /src -->

  - パス: プロジェクトの根（README.md のある所）からの相対パス。`../EdgeSLM_Android/…` のように、読むだけのほかのプロジェクトも指せる
  - 始まり: 正規表現。ファイルの中（from= があれば、その目印の行より後）に**ちょうど 1 か所**合うこと
  - 終わり: 正規表現（始まりの次の行から探して最初に合う行）、または `+N`（始まりから N 行）
  - omit=A..B: 抜き出しの中の、A に合う行の次から B に合う行の手前までを「（中略）」にする（いくつでも）
  - note=…: 見出しの行に添える一言（どの関数か、など）
  - 区切りは「空白 + | + 空白」。正規表現の中の | （空白なし）はそのまま使える

    <!-- src: なし | 理由 -->
    <!-- /src -->

  - その節に対応するソースが無いことをはっきり書く（考え方の説明・まとめ・他人のライブラリの中の話・端末の外の測定など）

    <!-- srcindex -->
    <!-- /srcindex -->

  - 目次（StudyMemo.md など）に置くと、章・節ごとに「どの枠がどのソースのどこを載せているか」の一覧を書く。
    枠も「なし」も無い節は「（未記入）」と出る（--check でも数を出す）

## 対象のファイル

根の下の `DoC/StudyMemo.md`・`DoC/StudyMemo/*.md`・`docs/StudyMemo.md`・`docs/StudyMemo/*.md`・`StudyMemo*.md`。
先頭 8 行に「このファイルは生成物」とある章（make_code_tour が作る地図・コードの章）は対象にしない。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "README.md").exists())   # プロジェクトの根（README.md のある所）
GLOBS = ["DoC/StudyMemo.md", "DoC/StudyMemo/*.md", "docs/StudyMemo.md", "docs/StudyMemo/*.md", "StudyMemo*.md"]

LANG = {".py": "python", ".cpp": "cpp", ".h": "cpp", ".hpp": "cpp", ".kt": "kotlin", ".json": "json",
        ".sh": "bash", ".ps1": "powershell", ".md": "markdown", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
        ".txt": "text", ".cmake": "cmake", ".java": "java", ".gradle": "groovy", ".kts": "kotlin", ".xml": "xml",
        ".bat": "bat", ".c": "c"}
OPEN = re.compile(r"^<!--\s*src:\s*(.*?)\s*-->\s*$")
CLOSE = re.compile(r"^<!--\s*/src\s*-->\s*$")
IOPEN = re.compile(r"^<!--\s*srcindex\s*-->\s*$")
ICLOSE = re.compile(r"^<!--\s*/srcindex\s*-->\s*$")
HEAD = re.compile(r"^(#{1,2})\s+(.*\S)\s*$")
SKIP_COUNT = re.compile(r"用語集|ソースと動かし方|コードで読む")   # 未記入・一覧に数えない章（用語集と、ソースそのものを扱う地図・コードの章）


class SrcError(Exception):
    pass


_cache: dict[str, list[str]] = {}


def lines_of(rel: str) -> list[str]:
    if rel not in _cache:
        p = (ROOT / rel).resolve()
        if not p.is_file():
            raise SrcError(f"ファイルが無い: {rel}")
        _cache[rel] = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return _cache[rel]


def md5_of(rel: str) -> str:
    return hashlib.md5((ROOT / rel).resolve().read_bytes()).hexdigest()[:8]


def find_one(rel: str, pattern: str, after: int = -1) -> int:
    lines = lines_of(rel)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise SrcError(f"正規表現の誤り /{pattern}/: {e}")
    hits = [i for i, s in enumerate(lines) if i > after and rx.search(s)]
    if len(hits) != 1:
        where = "、".join(f"L{i + 1}" for i in hits[:6])
        raise SrcError(f"{rel}: 目印 /{pattern}/ が {len(hits)} か所に合う（ちょうど 1 か所のこと）{(': ' + where) if hits else ''}")
    return hits[0]


def split_fields(body: str) -> list[str]:
    return [x.strip() for x in re.split(r"\s\|\s", body)]


def excerpt(body: str) -> tuple[str, dict]:
    """枠の中身（Markdown）と、一覧用の情報を返す"""
    f = split_fields(body)
    if f and f[0] == "なし":
        reason = " | ".join(f[1:]).strip() or "（理由の記入が無い）"
        return f"> **ソース**: なし — {reason}", {"kind": "none", "reason": reason}
    if len(f) < 3:
        raise SrcError(f"src の項目が足りない（パス | 始まり | 終わり）: {body}")
    rel, start, end, opts = f[0], f[1], f[2], f[3:]
    after, omits, note = -1, [], ""
    for o in opts:
        if o.startswith("from="):
            after = find_one(rel, o[5:])
        elif o.startswith("omit="):
            a, _, b = o[5:].partition("..")
            omits.append((a, b))
        elif o.startswith("note="):
            note = o[5:]
        else:
            raise SrcError(f"src の知らない指定: {o}")
    lines = lines_of(rel)
    s = find_one(rel, start, after)
    if re.fullmatch(r"\+\d+", end):
        e = min(len(lines) - 1, s + int(end[1:]) - 1)
    else:
        rx = re.compile(end)
        e = next((i for i in range(s + 1, len(lines)) if rx.search(lines[i])), None)
        if e is None:
            raise SrcError(f"{rel}: 終わりの目印 /{end}/ が L{s + 1} より後に無い")
    keep = list(range(s, e + 1))
    rows: list[str] = []
    i = 0
    while i < len(keep):
        ln = keep[i]
        text = lines[ln]
        rows.append(f"{ln + 1:>5}| {text}")
        for a, b in omits:
            if re.search(a, text):
                rb = re.compile(b)
                j = next((k for k in range(i + 1, len(keep)) if rb.search(lines[keep[k]])), None)
                if j is None:
                    raise SrcError(f"{rel}: omit の終わり /{b}/ が L{ln + 1} より後（抜き出しの中）に無い")
                n = j - i - 1
                if n > 0:
                    first = next((lines[keep[k]] for k in range(i + 1, j) if lines[keep[k]].strip()), "")
                    indent = re.match(r"\s*", first).group(0)
                    rows.append(f"     | {indent}…（中略: {n} 行。L{keep[i + 1] + 1}〜L{keep[j - 1] + 1}）…")
                i = j - 1
                break
        i += 1
    lang = LANG.get(Path(rel).suffix.lower(), "")
    fence = "````" if any("```" in r for r in rows) else "```"
    head = f"**ソース** `{rel}` L{s + 1}〜L{e + 1}" + (f"（{note}）" if note else "") + f" — md5 {md5_of(rel)}・`sync_src` が自動で更新"
    md = f"{head}\n\n{fence}{lang}\n" + "\n".join(rows) + f"\n{fence}"
    return md, {"kind": "src", "rel": rel, "range": f"L{s + 1}〜L{e + 1}", "note": note}


def targets() -> list[Path]:
    seen, out = set(), []
    for g in GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            head = "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[:8])
            if "このファイルは生成物" in head:
                continue
            out.append(p)
    return out


def process(path: Path, errors: list[str]):
    """枠を書き直した本文・枠の一覧（節つき）・枠の数・章の題・節の一覧を返す"""
    src = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in src else "\n"
    lines = src.splitlines()
    out, items, n_box = [], [], 0
    chapter, section, in_fence, i = "", "", False, 0
    sections: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```") and not OPEN.match(line):
            in_fence = not in_fence
        hm = HEAD.match(line) if not in_fence else None
        if hm:
            if len(hm.group(1)) == 1 and not chapter:
                chapter = hm.group(2)
            elif len(hm.group(1)) == 2:
                section = hm.group(2)
                sections.append(section)
        om = OPEN.match(line) if not in_fence else None
        if om:
            j = next((k for k in range(i + 1, len(lines)) if CLOSE.match(lines[k]) or OPEN.match(lines[k])), None)
            if j is None or OPEN.match(lines[j]):
                errors.append(f"{path.relative_to(ROOT)} の {i + 1} 行目: <!-- src --> に対応する <!-- /src --> が無い")
                out.append(line)
                i += 1
                continue
            n_box += 1
            try:
                md, info = excerpt(om.group(1))
            except SrcError as e:
                errors.append(f"{path.relative_to(ROOT)} の {i + 1} 行目: {e}")
                md, info = "\n".join(lines[i + 1:j]), None                 # 解決できなければ前の中身を残す
            out.append(line)
            out.extend(md.splitlines() if md else [])
            out.append(lines[j])
            if info:
                items.append({**info, "section": section, "line": i + 1})
            i = j + 1
            continue
        out.append(line)
        i += 1
    text = nl.join(out) + (nl if src.endswith(("\n", "\r\n")) else "")
    return text, items, n_box, chapter, sections


def build_index(index_path: Path, catalog: list[tuple[Path, str, list[str], list[dict]]]) -> tuple[str, int]:
    rows = ["| 章 | 節 | 載せているソース |", "|---|---|---|"]
    missing = 0
    for path, chapter, sections, items in catalog:
        if (path == index_path and path.name == "StudyMemo.md") or (SKIP_COUNT.search(path.name) and not items):
            continue
        link = os.path.relpath(path, index_path.parent).replace(os.sep, "/")
        by_sec: dict[str, list[dict]] = {}
        for it in items:
            by_sec.setdefault(it["section"], []).append(it)
        first = True
        for sec in sections:
            its = by_sec.get(sec, [])
            if not its:
                cell = "（未記入）"
                missing += 1
            else:
                parts = []
                for it in its:
                    if it["kind"] == "none":
                        parts.append(f"なし（{it['reason']}）")
                    else:
                        parts.append(f"`{it['rel']}` {it['range']}" + (f"（{it['note']}）" if it["note"] else ""))
                cell = "<br>".join(parts)
            chap = f"[{chapter}]({link})" if first else ""
            rows.append(f"| {chap} | {sec.replace('|', '｜')} | {cell} |")
            first = False
    return "\n".join(rows), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="書かずに、目印が全部解決できるか・枠が古くないかだけ確かめる")
    ap.add_argument("paths", nargs="*", help="この章だけを扱う（根からの相対パス）。省くと対象のファイル全部")
    a = ap.parse_args()
    only = {(ROOT / x).resolve() for x in a.paths}
    for x in only:
        if not x.is_file():
            raise SystemExit(f"ファイルが無い: {x}")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    errors: list[str] = []
    results, catalog = [], []
    for p in targets():
        if only and p.resolve() not in only:
            continue
        text, items, n_box, chapter, sections = process(p, errors)
        results.append((p, text, n_box, items))
        catalog.append((p, chapter or p.stem, sections, items))
    total_missing = 0
    for p, chapter, sections, items in catalog:
        if p.name == "StudyMemo.md" or SKIP_COUNT.search(p.name):
            continue
        have = {it["section"] for it in items}
        miss = [s for s in sections if s not in have]
        total_missing += len(miss)
        if miss and a.check:
            print(f"  未記入 {p.relative_to(ROOT)}: " + "・".join(m.split(" ")[0] for m in miss[:30]))
    final = []
    for p, text, n_box, items in results:
        lines = text.splitlines()
        if not only and any(IOPEN.match(x) for x in lines):
            idx_md, _ = build_index(p, catalog)
            out, i = [], 0
            while i < len(lines):
                if IOPEN.match(lines[i]):
                    j = next((k for k in range(i + 1, len(lines)) if ICLOSE.match(lines[k])), None)
                    if j is None:
                        errors.append(f"{p.relative_to(ROOT)} の {i + 1} 行目: <!-- srcindex --> に対応する <!-- /srcindex --> が無い")
                        out.append(lines[i]); i += 1; continue
                    out += [lines[i]] + idx_md.splitlines() + [lines[j]]
                    i = j + 1
                    continue
                out.append(lines[i]); i += 1
            nl = "\r\n" if "\r\n" in text else "\n"
            text = nl.join(out) + (nl if text.endswith(("\n", "\r\n")) else "")
        final.append((p, text, n_box, items))
    stale = []
    for p, text, n_box, items in final:
        old = p.read_bytes().decode("utf-8")
        changed = old != text
        n_src = sum(1 for it in items if it["kind"] == "src")
        n_none = sum(1 for it in items if it["kind"] == "none")
        mark = ("古い→書き直す" if not a.check else "★古い（sync_src を流すこと）") if changed else "そのまま"
        if n_box or changed:
            print(f"{p.relative_to(ROOT)}: 枠 {n_box} 個（ソース {n_src}・なし {n_none}）・{mark}")
        if changed:
            stale.append(p)
    print(f"枠も「なし」も無い節（未記入）: {total_missing} 個" + ("" if a.check else "（どの節かは --check で出る）"))
    if errors:
        print(f"★解決できない印が {len(errors)} 個:")
        for e in errors:
            print("  " + e)
        raise SystemExit(1)
    if a.check:
        if stale:
            print(f"★古い枠のあるファイルが {len(stale)} 本（--check なので書いていない）")
            raise SystemExit(1)
        print("（--check: すべて最新）")
        return
    for p, text, n_box, items in final:
        if p in stale:
            p.write_bytes(text.encode("utf-8"))
    print(f"書いた: {len(stale)} 本")


if __name__ == "__main__":
    main()
