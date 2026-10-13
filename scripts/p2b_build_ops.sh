#!/bin/bash
# P2b: Transformer 構成要素の単体 op を S1〜S8 に通し、**どの段で落ちたか**を表にする
#
# 基本設計② 6.1節 / app/pipeline/p2b_ops.py の説明を参照。
#
# ホストの Python 環境が2つに割れている（NFR-5）ため、ここが唯一の通し実行者になる:
#   S1        make_op_models.py        （本体 .venv）
#   S2〜S4    p2b_ops.py prepare       （本体 .venv）
#   S5〜S7    build_context.sh         （QAIRT 環境 / Python 3.10）
#   S8        p2b_ops.py finalize      （本体 .venv）
#   端末      scripts/run_runner.sh    （別途。op_check task）
#
# 使い方:
#   bash scripts/p2b_build_ops.sh                 # 全部
#   bash scripts/p2b_build_ops.sh matmul_act rope # 指定だけ
#
# 落ちたモデルがあっても**止めない**。P2b の目的は「どこまで通るか」の一覧を
# 作ることであって、最初の失敗で終わることではない。

set -u
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc;/sdcard'
export PYTHONIOENCODING=utf-8

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

PY="./.venv/Scripts/python.exe"
WORK="work/ops"
SCHEME="${SCHEME:-a16w8}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/P2b_${TS}"
mkdir -p "$OUT"

[ -x "$PY" ] || { echo "ERROR: 本体 venv が無い: $PY" >&2; exit 1; }

ALL="matmul_weight matmul_act rmsnorm rmsnorm_fused layernorm rope rope_fused \
softmax silu attention kvcache embed block"
NAMES="${*:-$ALL}"

echo "=== S1: 単体 op モデルの生成 ==="
"$PY" -m app.pipeline.make_op_models --out "$WORK" > "$OUT/s1_generate.log" 2>&1
tail -3 "$OUT/s1_generate.log" | sed 's/^/  /'

# QAIRT のログから**真因**を取り出す。
# 無害な行（ホストに端末側 FastRPC ライブラリが無い旨）を除外し、
# QNN 自身の診断行 `<E>` を優先する。
qnn_cause() {
  local f="$1"
  [ -f "$f" ] || { echo "(ログなし)"; return; }
  local line
  line="$(grep -E '<E>' "$f" 2>/dev/null           | grep -viE 'WindowsFileIO|libcdsprpc|libadsprpc' | head -1)"
  if [ -z "$line" ]; then
    line="$(grep -iE 'error|fail' "$f" 2>/dev/null             | grep -viE 'WindowsFileIO|libcdsprpc|libadsprpc' | head -1)"
  fi
  echo "$line" | sed 's/^ *//;s/  */ /g' | cut -c1-160
}

SUMMARY="$OUT/summary.tsv"
printf 'name\tS2_S4\tS5_convert\tS6_context\tS8_epctx\tverdict\tdetail\n' > "$SUMMARY"

for NAME in $NAMES; do
  echo
  echo "================ $NAME ================"
  D="$WORK/$NAME"
  CTX="$D/ctx"
  QDQ="$D/${NAME}_${SCHEME}_qdq.onnx"
  rm -rf "$CTX"

  S24="-"; S5="-"; S6="-"; S8="-"; VERDICT=""; DETAIL=""

  # --- S2〜S4 ---
  if "$PY" -m app.pipeline.p2b_ops prepare --name "$NAME" --scheme "$SCHEME" \
       --n-random 12 > "$OUT/${NAME}_prepare.log" 2>&1; then
    S24="OK"
    sed -n '2,8p' "$OUT/${NAME}_prepare.log" | sed 's/^/  /'
  else
    S24="NG"
    VERDICT="$(grep -o 'verdict=[a-zA-Z0-9_]*' "$OUT/${NAME}_prepare.log" | head -1 | cut -d= -f2)"
    DETAIL="$(grep -m1 'NG ' "$OUT/${NAME}_prepare.log" | sed 's/^ *//' | cut -c1-150)"
    sed -n '1,6p' "$OUT/${NAME}_prepare.log" | sed 's/^/  /'
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$NAME" "$S24" "$S5" "$S6" "$S8" "${VERDICT:-fail}" "$DETAIL" >> "$SUMMARY"
    continue
  fi

  # --- S5〜S7（QAIRT 環境）---
  # ★ログから真因を拾うときの注意（2026-09-12 に一度誤判定した）:
  #   「87.7ms [ ERROR ] WindowsFileIO couldn't open libcdsprpc.dll」は
  #   **成功したときのログにも出る**。ホスト(x86)には端末側の FastRPC ライブラリが
  #   無いというだけで、context 生成の成否とは無関係。素朴に最初の ERROR 行を
  #   拾うとこれを真因として報告してしまう。
  #   真因は QNN 自身が出す `<E>` 行にある（例: 「has incorrect Value 68, expected >= 73」）。
  if bash scripts/build_context.sh "$QDQ" "$CTX" "$NAME" \
       > "$OUT/${NAME}_context.log" 2>&1; then
    S5="OK"; S6="OK"
    grep -E '^  OK' "$OUT/${NAME}_context.log" | sed 's/^/  /'
  else
    # どちらの段で落ちたかを DLC の有無で切り分ける。
    # S5（converter が op を知らない）と S6（HTP が受け付けない）では対処が違う。
    if [ -f "$CTX/${NAME}.dlc" ]; then
      S5="OK"; S6="NG"
      DETAIL="$(qnn_cause "$CTX/${NAME}_gen.log")"
      VERDICT="fail_S6_context"
    else
      S5="NG"; S6="-"
      DETAIL="$(qnn_cause "$CTX/${NAME}_convert.log")"
      VERDICT="fail_S5_convert"
    fi
    echo "  NG $VERDICT: $DETAIL"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$NAME" "$S24" "$S5" "$S6" "$S8" "$VERDICT" "$DETAIL" >> "$SUMMARY"
    continue
  fi

  # --- S8 + C-2/C-3/C-4 ---
  if "$PY" -m app.pipeline.p2b_ops finalize --name "$NAME" --scheme "$SCHEME" \
       --ctx-dir "$CTX" > "$OUT/${NAME}_finalize.log" 2>&1; then
    S8="OK"; VERDICT="ready_for_device"
  else
    S8="NG"
    VERDICT="$(grep -o 'verdict=[a-zA-Z0-9_]*' "$OUT/${NAME}_finalize.log" | head -1 | cut -d= -f2)"
    DETAIL="$(grep -m1 'NG ' "$OUT/${NAME}_finalize.log" | sed 's/^ *//' | cut -c1-150)"
  fi
  sed -n '2,8p' "$OUT/${NAME}_finalize.log" | sed 's/^/  /'

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$NAME" "$S24" "$S5" "$S6" "$S8" "$VERDICT" "$DETAIL" >> "$SUMMARY"
done

echo
echo "================ 集計 ================"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
echo "ログ: $OUT/"
echo "端末に持って行く分:"
ls -1 "$WORK"/*/*_epctx.onnx 2>/dev/null | sed 's/^/  /' || echo "  （なし）"
