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

set -u
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc;/sdcard'

QDQ="${1:?usage: build_context.sh <qdq.onnx> <outdir> <tag>}"
OUTDIR="${2:?usage: build_context.sh <qdq.onnx> <outdir> <tag>}"
TAG="${3:?usage: build_context.sh <qdq.onnx> <outdir> <tag>}"

QAIRT_VER="${QAIRT_VER:-2.48.0.260626}"
QAIRT_U="/d/QAIRT/v${QAIRT_VER}/qairt/${QAIRT_VER}"
QAIRT_W="D:\\QAIRT\\v${QAIRT_VER}\\qairt\\${QAIRT_VER}"
QPY="/d/QAIRT/qairtenv/Scripts/python.exe"

[ -d "$QAIRT_U" ] || { echo "ERROR: QAIRT が無い: $QAIRT_U" >&2; exit 1; }
[ -x "$QPY" ]     || { echo "ERROR: QAIRT 専用 Python が無い: $QPY" >&2; exit 1; }
[ -f "$QDQ" ]     || { echo "ERROR: QDQ モデルが無い: $QDQ" >&2; exit 1; }

mkdir -p "$OUTDIR"

# QAIRT ツールの実行に必要な環境（環境構築記録 3.6節で確認した組み合わせ）
export VIRTUAL_ENV='D:\QAIRT\qairtenv'
export PYTHONPATH="${QAIRT_W}\\lib\\python"
export PYTHONUTF8=1
export PATH="$QAIRT_U/bin/x86_64-windows-msvc:$QAIRT_U/lib/x86_64-windows-msvc:$PATH"

win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }

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
"$QPY" "$QAIRT_U/bin/x86_64-windows-msvc/qairt-converter" \
    --input_network "$(win "$QDQ")" \
    --preserve_io_datatype \
    --output_path "$(win "$DLC")" > "$OUTDIR/${TAG}_convert.log" 2>&1
if [ ! -f "$DLC" ]; then
  echo "  FAILED. ログ末尾:"; tail -25 "$OUTDIR/${TAG}_convert.log" | sed 's/^/    /'
  exit 1
fi
echo "  OK  $(ls -l "$DLC" | awk '{printf "%.2f MiB", $5/1048576}')"

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
BIN=$(ls "$CTXDIR"/*.bin 2>/dev/null | head -1)
if [ -z "$BIN" ]; then
  echo "  FAILED. ログ末尾:"; tail -25 "$OUTDIR/${TAG}_gen.log" | sed 's/^/    /'
  exit 1
fi
echo "  OK  $(basename "$BIN")  $(ls -l "$BIN" | awk '{printf "%.2f MiB", $5/1048576}')"

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

echo
echo "生成物:"
echo "  DLC : $DLC"
echo "  BIN : $BIN"
echo "  META: $META"
echo "  CMD : $CMDLOG"
