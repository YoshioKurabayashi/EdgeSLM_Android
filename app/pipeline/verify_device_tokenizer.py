r"""端末側トークナイザ（Qwen2Tokenizer.kt）が HF と同じ ID 列を出すかを測る。

対象は `EdgeSLM_AndroidDemo/demo_app/.../Qwen2Tokenizer.kt`（別セッションが実装したもの）。
**実装を動かす代わりに、同じ資材と同じ手順を Python で再現して突き合わせる。**
Kotlin をホストで動かす環境が無くてもアルゴリズムの同値性は確かめられる。

使い方:
    python -m app.pipeline.verify_device_tokenizer [資材ディレクトリ]
    既定: D:/AI/source/n_chap5/EdgeSLM_AndroidDemo/host/out

2026-09-13 の実測: **NFC なし 27/31 / NFC あり 31/31**。
外れる4件はすべて**入力が分解済み（NFD）の場合**で、HF の normalizer: NFC が効く分である。
Kotlin 側は `java.text.Normalizer.normalize(text, Form.NFC)` を1行入れれば閉じる
（Android に標準で入っているので追加依存は無い）。

目的は「端末側トークナイザが HF と同じ列を出すか」を**測る**こと。
Kotlin を動かす代わりに、同じ資材（vocab.txt / merges.txt / specials.tsv）と
同じ手順を Python で再現する。差が出たらそれは実装の差である。

Kotlin 側との対応:
  PATTERN        → regex モジュール（\p{L} 等が必要。Python 標準 re では書けない）
  BYTE_TO_CHAR   → 同じ GPT-2 写像
  bpe()          → 「最小 rank のペアを選び、その出現箇所を全部まとめて結合」
  encodeInto()   → 特殊トークンを最左一致で切る再帰
  ★NFC 正規化は Kotlin 側に無い（HF は normalizer: NFC を持つ）
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】このファイルは関数を定義するだけでなく、import した時点で上から順に
#   資材の読み込み → 31 ケースの照合 → 表示 を実行するスクリプト（main は無い）。
# 入力:  資材ディレクトリ（第 1 引数。既定は EdgeSLM_AndroidDemo/host/out）の vocab.txt・merges.txt・specials.tsv、
#        HF の Qwen/Qwen2.5-0.5B-Instruct のトークナイザ（基準）。regex モジュールが要る（標準の re では \p{L} が書けない）。
# 出力:  画面の表（ケースごとに NFC なし／あり で HF と一致したか）。ファイルは書かない。
# ---------------------------------------------------------------------------
import sys, unicodedata
from pathlib import Path
import regex as re

ASSETS = Path(sys.argv[1] if len(sys.argv) > 1 else
              "D:/AI/source/n_chap5/EdgeSLM_AndroidDemo/host/out")

PATTERN = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def byte_to_char():
    """GPT-2 のバイト → 文字の写像（256 個）を作る。

    表示できるバイト（0x21〜0x7E・0xA1〜0xAC・0xAE〜0xFF）はそのままの文字、それ以外（空白・制御文字など）は
    U+0100 以降の文字に順に割り当てる。BPE の語彙はこの文字で書かれている。
    """
    direct = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    out, n = {}, 0
    for b in range(256):
        if b in direct:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + n); n += 1
    return out

B2C = byte_to_char()

# 資材を読む: vocab.txt は 1 行 1 語（行番号 = ID）、merges.txt は「左 右」の結合規則（行番号 = 優先順位）、
# specials.tsv は「ID<TAB>特殊トークン文字列」。特殊トークンは長いものから試すよう並べ替える。
vocab = ASSETS.joinpath("vocab.txt").read_text(encoding="utf-8").split("\n")
if vocab and vocab[-1] == "": vocab.pop()
tok2id = {t: i for i, t in enumerate(vocab)}
ranks = {}
for i, line in enumerate(ASSETS.joinpath("merges.txt").read_text(encoding="utf-8").split("\n")):
    if line: ranks[line] = i
specials = []
for line in ASSETS.joinpath("specials.tsv").read_text(encoding="utf-8").split("\n"):
    if "\t" in line:
        a, b = line.split("\t", 1)
        specials.append((b, int(a)))
specials.sort(key=lambda x: -len(x[0]))
print(f"assets: vocab={len(vocab)} merges={len(ranks)} specials={len(specials)}")

_cache = {}
def bpe(token: str):
    """BPE で 1 つの単語（バイト写像後の文字列）を語彙の記号の列に分ける。結果はキャッシュする。

    手順: 隣り合う 2 記号の組のうち merges の順位が最小の組を選び、その組の出現箇所を左から全部まとめて結合する。
    結合できる組が無くなるまでくり返す（Kotlin 側の bpe() と同じ）。
    """
    if token in _cache: return _cache[token]
    word = list(token)
    while len(word) >= 2:
        best, ba, bb = 1 << 30, None, None
        for i in range(len(word) - 1):
            r = ranks.get(word[i] + " " + word[i + 1])
            if r is not None and r < best:
                best, ba, bb = r, word[i], word[i + 1]
        if ba is None: break
        nxt, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == ba and word[i + 1] == bb:
                nxt.append(ba + bb); i += 2
            else:
                nxt.append(word[i]); i += 1
        word = nxt
    _cache[token] = word
    return word

def emit(piece, out):
    """pre-tokenize で切った 1 片を UTF-8 のバイト → 写像の文字 → BPE → ID にして out に足す。

    語彙に無い記号は 1 文字ずつの ID に分けて足す（Kotlin 側と同じ扱いにしている）。
    """
    if not piece: return
    s = "".join(B2C[b] for b in piece.encode("utf-8"))
    for sym in bpe(s):
        i = tok2id.get(sym)
        if i is not None: out.append(i)
        else:
            for c in sym:
                j = tok2id.get(c)
                if j is not None: out.append(j)

# NFC を入れるかどうかの切り替え（kt_encode() が毎回書き換える。グローバル変数）。
NFC_IN_ORDINARY = True   # HF と同じ順序: 特殊トークン分割 → NFC → pre-tokenize

def encode_ordinary(text, out):
    """特殊トークンを含まない文字列を ID にする。必要なら NFC 正規化し、PATTERN で切った片ごとに emit する。

    PATTERN に当たらない隙間があれば、その部分もそのまま 1 片として emit する。
    """
    if not text: return
    if NFC_IN_ORDINARY: text = unicodedata.normalize("NFC", text)
    cursor = 0
    for m in PATTERN.finditer(text):
        if m.start() > cursor: emit(text[cursor:m.start()], out)
        emit(m.group(), out)
        cursor = m.end()
    if cursor < len(text): emit(text[cursor:], out)

def encode_into(text, out):
    """文字列の中で一番左にある特殊トークンで切り、前を普通に、特殊トークンを 1 つの ID に、後ろを再帰で処理する。"""
    if not text: return
    best_pos, best_sp = -1, None
    for sp in specials:
        p = text.find(sp[0])
        if p >= 0 and (best_pos < 0 or p < best_pos):
            best_pos, best_sp = p, sp
    if best_sp is not None:
        encode_ordinary(text[:best_pos], out)
        out.append(best_sp[1])
        encode_into(text[best_pos + len(best_sp[0]):], out)
        return
    encode_ordinary(text, out)

def kt_encode(text, nfc=False):
    """端末の Kotlin 実装を再現した encode。nfc=True で NFC 正規化を入れた場合を試す。戻り値: ID のリスト。"""
    global NFC_IN_ORDINARY
    NFC_IN_ORDINARY = nfc
    out = []
    encode_into(text, out)
    return out

# ---------------- テスト集合 ----------------
CASES = [
    ("英語・基本", "The capital city of France is"),
    ("英語・長め", "The capital city of Japan is Tokyo. The capital city of France is"),
    ("短縮形", "It's John's car, they're here, I'd go, we've won, you'll see, I'm ok"),
    ("大文字短縮形", "IT'S JOHN'S CAR"),
    ("数字", "Order 12345 costs $6,789.00 in 2026"),
    ("連続空白", "a  b   c    d"),
    ("行末空白", "trailing spaces   "),
    ("改行", "line1\nline2\r\nline3\n\n\nline4"),
    ("タブ", "col1\tcol2\t\tcol3"),
    ("記号", "!!! ??? ... --- *** (){}[]<>"),
    ("日本語", "日本の首都は東京です。"),
    ("日本語・長め", "むかしむかし、あるところにおじいさんとおばあさんが住んでいました。"),
    ("全角スペース", "日本語　全角スペース　入り"),
    ("半角カナ", "ﾊﾝｶｸ ｶﾀｶﾅ ﾃｽﾄ"),
    ("絵文字", "Hello 👋 world 🌍 family 👨‍👩‍👧‍👦"),
    ("中国語", "中华人民共和国"),
    ("韓国語", "대한민국 서울특별시"),
    ("混在", "Pythonで日本語を tokenize する 2026年"),
    ("特殊トークン", "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"),
    ("特殊トークン単体", "<|endoftext|>"),
    ("空文字", ""),
    ("空白のみ", "   "),
    ("NFC: 合成済みが", "が ぎ ぐ げ ご"),
    ("NFC: 分解済みが", "か\u3099 き\u3099 く\u3099 け\u3099 こ\u3099"),
    ("NFC: 合成済み é", "café résumé naïve"),
    ("NFC: 分解済み é", "cafe\u0301 re\u0301sume\u0301 nai\u0308ve"),
    ("NFC: ハングル合成", "한"),
    ("NFC: ハングル分解", "\u1112\u1161\u11ab"),
    ("NFC: 互換文字", "①②③ ㈱ ﬁ"),
    ("ゼロ幅", "a\u200bb\u200cc"),
    ("結合記号", "a\u0301\u0302\u0303b"),
]

# ---------------- 照合 ----------------
# 基準は HF のトークナイザ（add_special_tokens=False で、特殊トークンを自分で足さない）。
from transformers import AutoTokenizer
hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

rows, ng_plain, ng_nfc = [], 0, 0
for name, text in CASES:
    exp = hf(text, add_special_tokens=False)["input_ids"]
    got = kt_encode(text, nfc=False)
    got_nfc = kt_encode(text, nfc=True)
    ok, ok_nfc = got == exp, got_nfc == exp
    ng_plain += (not ok); ng_nfc += (not ok_nfc)
    rows.append((name, text, exp, got, got_nfc, ok, ok_nfc))

print(f"\n{'判定':<6} {'NFC付':<6} ケース")
print("-" * 78)
for name, text, exp, got, got_nfc, ok, ok_nfc in rows:
    print(f"{'OK' if ok else '**NG**':<6} {'OK' if ok_nfc else '**NG**':<6} {name}")
    if not ok:
        print(f"         HF  : {exp}")
        print(f"         端末: {got}")
        if ok_nfc: print(f"         → NFC を入れれば一致する")
        else:      print(f"         端末+NFC: {got_nfc}")
print("-" * 78)
n = len(CASES)
print(f"現状の実装（NFC なし）: {n - ng_plain}/{n} 一致")
print(f"NFC を足した場合      : {n - ng_nfc}/{n} 一致")
