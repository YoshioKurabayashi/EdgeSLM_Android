#!/bin/bash
# S5〜S7: QDQ ONNX → DLC → context binary → メタデータ
#
# 基本設計② S5/S6/S7 の実装。**QAIRT 専用 Python 環境（3.10）で実行する。**
# 本体の .venv（3.12）とは numpy/opencv が共存できないため分けている（NFR-5）。
#
# 使い方:
#   bash scripts/build_context.sh <qdq.onnx> <出力ディレクトリ> <tag>
#
# ★守るべき禁止事項（要件定義書 3.4節）:
#   C-2: soc_id を刻まない。--htp_socs sm8350 のみ
#   C-3: backend_extensions の config を渡さない（perf_profile を埋め込むとロード失敗）
#   C-4: --preserve_io_datatype は完全QDQに対してのみ
#
# （2026-10-18 追記: 学習用の説明）
# 何をするか: 量子化の済んだ QDQ の ONNX を、HTP（NPU）が実行する形まで PC で先に作る。
#   S5 qairt-converter            : ONNX → DLC（QAIRT の中間形式。ここで RMSNorm などが融合 op になる）
#   S6 qnn-context-binary-generator: DLC → context binary（HTP v68 向けにコンパイル済みのもの。最も時間がかかる）
#   S7 qnn-context-binary-utility : context binary → メタデータ JSON（socModel を C-2 の検査に使う）
# 流れの中の位置（StudyMemo 13.1 節）:
#   p2b_ops prepare（S2〜S4）→ downgrade_actact_rhs（C-8）→ ★ここ（約 1 時間）→ dlc_audit → wrap_epctx / deploy_pm64
# 入力: <qdq.onnx>（ふつうは <name>_final_qdq.onnx）、<出力ディレクトリ>、<tag>（出力の名前）
# 出力: <出力ディレクトリ>/<tag>.dlc、ctx_<tag>/<tag>_ctx*.bin、<tag>_meta.json、<tag>_commands.txt、
#       各段のログ（<tag>_convert.log・<tag>_gen.log・<tag>_meta.log）
# 動かすのは Git Bash（Windows の exe と QAIRT の Python を呼ぶ）。

# 未定義の変数を使ったら止める。-e は付けていないので、各段の成否は生成物ができたかで判定する
set -u
# Git Bash（MSYS）が "/sdcard/…" などの引数を Windows のパスに書き換えないようにする（adb 用の決まり文句）
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc;/sdcard'

QDQ="${1:?usage: build_context.sh <qdq.onnx> <outdir> <tag>}"
OUTDIR="${2:?usage: build_context.sh <qdq.onnx> <outdir> <tag>}"
TAG="${3:?usage: build_context.sh <qdq.onnx> <outdir> <tag>}"

# QAIRT の場所。_U は bash の書き方（/d/…）、_W は Windows の書き方（D:\…。exe に渡す引数用）。
# 版は環境変数 QAIRT_VER で変えられる（既定 2.48.0.260626）
QAIRT_VER="${QAIRT_VER:-2.48.0.260626}"
QAIRT_U="/d/QAIRT/v${QAIRT_VER}/qairt/${QAIRT_VER}"
QAIRT_W="D:\\QAIRT\\v${QAIRT_VER}\\qairt\\${QAIRT_VER}"
QPY="/d/QAIRT/qairtenv/Scripts/python.exe"

# 前提の確認。足りなければ長い処理に入る前に止める
[ -d "$QAIRT_U" ] || { echo "ERROR: QAIRT が無い: $QAIRT_U" >&2; exit 1; }
[ -x "$QPY" ]     || { echo "ERROR: QAIRT 専用 Python が無い: $QPY" >&2; exit 1; }
[ -f "$QDQ" ]     || { echo "ERROR: QDQ モデルが無い: $QDQ" >&2; exit 1; }

mkdir -p "$OUTDIR"

# QAIRT ツールの実行に必要な環境（環境構築記録 3.6節で確認した組み合わせ）
export VIRTUAL_ENV='D:\QAIRT\qairtenv'
export PYTHONPATH="${QAIRT_W}\\lib\\python"
export PYTHONUTF8=1
export PATH="$QAIRT_U/bin/x86_64-windows-msvc:$QAIRT_U/lib/x86_64-windows-msvc:$PATH"

# win: bash のパス（/d/…）を Windows のパス（D:\…）に直す。cygpath が無ければそのまま返す
win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }

# 出力のファイル名。<tag>_commands.txt には実際に流したコマンドを残し、
# p2b_ops finalize が C-3・C-4 の検査（verify_artifacts）に読む。毎回空にしてから書く
DLC="$OUTDIR/${TAG}.dlc"
CTXDIR="$OUTDIR/ctx_${TAG}"
META="$OUTDIR/${TAG}_meta.json"
CMDLOG="$OUTDIR/${TAG}_commands.txt"
mkdir -p "$CTXDIR"
: > "$CMDLOG"

echo "=== S5: qairt-converter ==="
# --preserve_io_datatype はグラフI/Oを float32 に保ち内部を完全な固定小数点にする。
# **完全QDQに対してのみ**付けること（C-4）。float 混在だと変換不可能な fp32 op が出る。
CMD5="qairt-converter --input_network $(win "$QDQ") --preserve_io_datatype --output_path $(win "$DLC")"
echo "$CMD5" >> "$CMDLOG"
echo "  $CMD5"
# QAIRT の Python で converter を動かす。画面には出さず、ログは <tag>_convert.log に書く
"$QPY" "$QAIRT_U/bin/x86_64-windows-msvc/qairt-converter" \
    --input_network "$(win "$QDQ")" \
    --preserve_io_datatype \
    --output_path "$(win "$DLC")" > "$OUTDIR/${TAG}_convert.log" 2>&1
# 成否は DLC ができたかで判定する（converter は警告が多く、終了コードやログの ERROR 行は当てにしない）。
# ★注意（2026-10-18 追記。コードからの読み取り）: 同じ出力ディレクトリに前回の DLC が残っていると、
#   今回の変換が失敗しても OK に見える（消してから流す処理は無い）
if [ ! -f "$DLC" ]; then
  echo "  FAILED. ログ末尾:"; tail -25 "$OUTDIR/${TAG}_convert.log" | sed 's/^/    /'
  exit 1
fi
echo "  OK  $(ls -l "$DLC" | awk '{printf "%.2f MiB", $5/1048576}')"

# S6: context binary を作る。HTP の世代は --htp_socs sm8350（Snapdragon 888、v68）だけで指定する。
#   --soc_id を刻むと unsigned PD の端末でロードが err 1002（C-2）、backend の config を渡すと
#   perf_profile が埋め込まれて同じく err 1002（C-3）。ホストで通っても端末で落ちる op がある（C-10 の Expand）ので、
#   ここの成功は「端末で動く」ことの保証ではない
echo "=== S6: qnn-context-binary-generator ==="
# ★--htp_socs sm8350 のみ。soc_id を刻まず（C-2）、config も渡さない（C-3）。
CMD6="qnn-context-binary-generator --dlc_path $(win "$DLC") --backend ${QAIRT_W}\\lib\\x86_64-windows-msvc\\QnnHtp.dll --binary_file ${TAG}_ctx --output_dir $(win "$CTXDIR") --htp_socs sm8350"
echo "$CMD6" >> "$CMDLOG"
echo "  $CMD6"
"$QAIRT_U/bin/x86_64-windows-msvc/qnn-context-binary-generator.exe" \
    --dlc_path "$(win "$DLC")" \
    --backend "${QAIRT_W}\\lib\\x86_64-windows-msvc\\QnnHtp.dll" \
    --binary_file "${TAG}_ctx" \
    --output_dir "$(win "$CTXDIR")" \
    --htp_socs sm8350 > "$OUTDIR/${TAG}_gen.log" 2>&1
# できた .bin を探す（生成器はファイル名に SoC 名を足す。例 <tag>_ctx.SM8350.bin）。
# ★注意（2026-10-18 追記。コードからの読み取り）: ctx_<tag>/ に前回の .bin が残っていると、今回の生成が
#   失敗しても OK に見える。2026-09-13 の「書き換え途中の binary を配った」件の後、配る前の確認は
#   deploy_pm64.py の「安定を待つ」と md5 の 3 点照合で行っている
BIN=$(ls "$CTXDIR"/*.bin 2>/dev/null | head -1)
if [ -z "$BIN" ]; then
  echo "  FAILED. ログ末尾:"; tail -25 "$OUTDIR/${TAG}_gen.log" | sed 's/^/    /'
  exit 1
fi
echo "  OK  $(basename "$BIN")  $(ls -l "$BIN" | awk '{printf "%.2f MiB", $5/1048576}')"

# S7: context binary の中身をメタデータ JSON に書き出す。socModel が 0 であること（C-2）を
#   p2b_ops finalize → verify_artifacts.check_c2_socmodel_zero が機械で確かめる。失敗しても止めない
echo "=== S7: qnn-context-binary-utility（socModel の検証用メタデータ） ==="
CMD7="qnn-context-binary-utility --context_binary $(win "$BIN") --json_file $(win "$META")"
echo "$CMD7" >> "$CMDLOG"
"$QAIRT_U/bin/x86_64-windows-msvc/qnn-context-binary-utility.exe" \
    --context_binary "$(win "$BIN")" \
    --json_file "$(win "$META")" > "$OUTDIR/${TAG}_meta.log" 2>&1
if [ -f "$META" ]; then
  echo "  OK  $META"
else
  echo "  WARN メタデータを取得できなかった（C-2 の機械検証ができない）"
  tail -10 "$OUTDIR/${TAG}_meta.log" | sed 's/^/    /'
fi

# 最後に生成物の場所をまとめて表示する
echo
echo "生成物:"
echo "  DLC : $DLC"
echo "  BIN : $BIN"
echo "  META: $META"
echo "  CMD : $CMDLOG"
