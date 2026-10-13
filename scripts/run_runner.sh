#!/bin/bash
# EdgeSLM Runner をホストから駆動する
#
# 使い方:
#   bash scripts/run_runner.sh install                      # APK をインストールし初期設定
#   bash scripts/run_runner.sh push-model <onnxファイル>     # モデルを端末へ（再ビルド不要）
#   bash scripts/run_runner.sh push-ops                     # P2b の単体 op モデルを一括で
#   bash scripts/run_runner.sh run <ジョブ仕様ファイル>       # ジョブ実行 → 結果回収
#
# 設計（基本設計③）:
#   モデルとジョブ仕様は APK の外（外部ファイルディレクトリ）に置く。
#   これにより **モデルやパラメータを変えても APK を作り直さずに済む**。
#   ビルドは JVM 制約によりユーザー実行になるため、その回数を減らすことが最優先。

set -u
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc;/sdcard'

HERE="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$HERE/android_runner"
OUTDIR="$HERE/results"
mkdir -p "$OUTDIR"

ADB="${ADB:-adb}"
PKG="org.edgeslm.runner"
DEV_BASE="/sdcard/Android/data/$PKG/files"
APK="$RUNNER/app/build/outputs/apk/debug/app-debug.apk"

die() { echo "ERROR: $*" >&2; exit 1; }

ensure_device() {
  "$ADB" get-state >/dev/null 2>&1 || die "端末が接続されていない"
}

cmd_install() {
  ensure_device
  [ -f "$APK" ] || die "APK が無い: $APK
  先に PowerShell で次を実行すること（JVM 制約によりビルドは代行できない）:
    \$env:JAVA_HOME=\"D:\\Android\\jdk17\"; cd $RUNNER; .\\gradlew.bat assembleDebug"

  echo "=== APK 情報 ==="
  ls -l "$APK" | awk '{printf "  %.1f MiB\n", $5/1048576}'
  # C-6 の確認: extractNativeLibs が true でなければ FastRPC が skeleton を開けない
  AAPT="${ANDROID_SDK_ROOT:-D:/Android/sdk}/build-tools/34.0.0/aapt2.exe"
  if [ -x "$AAPT" ]; then
    echo "  extractNativeLibs: $("$AAPT" dump xmltree --file AndroidManifest.xml "$APK" 2>/dev/null \
      | grep -o 'extractNativeLibs([^)]*)=[a-z]*' | head -1)"
  fi

  echo "=== インストール ==="
  "$ADB" install -r "$(cygpath -w "$APK" 2>/dev/null || echo "$APK")" || die "install 失敗"

  echo "=== 通知権限を付与（Foreground Service 用。未付与でも動作はする） ==="
  "$ADB" shell pm grant $PKG android.permission.POST_NOTIFICATIONS 2>&1 | sed 's/^/  /'

  echo "=== 外部ファイルディレクトリを作成（★アプリ自身に作らせる） ==="
  # adb shell の mkdir で作ると所有者が shell になり、アプリから入れなくなる。
  # アプリを一度起動してサブディレクトリを作らせるのが正しい手順。
  "$ADB" shell am start -n "$PKG/.RunnerActivity" >/dev/null 2>&1
  "$ADB" shell 'for i in 1 2 3 4 5; do [ -d '"$DEV_BASE"'/jobs ] && break; sleep 1; done'
  "$ADB" shell "ls -la $DEV_BASE" | sed 's/^/  /'

  echo "=== 展開された .so（C-6 の実証） ==="
  "$ADB" shell "p=\$(pm path $PKG | head -1 | cut -d: -f2); ls -l \$(dirname \$p)/lib/arm64 2>/dev/null | grep -iE 'QnnHtp|onnxruntime|edgeslm' " | sed 's/^/  /'
}

cmd_push_model() {
  ensure_device
  local src="${1:-}"
  [ -f "$src" ] || die "モデルが無い: $src"
  local name; name="$(basename "$src")"
  echo "=== push: $name ($(ls -l "$src" | awk '{printf "%.2f MiB", $5/1048576}')) ==="
  "$ADB" push "$(cygpath -w "$src" 2>/dev/null || echo "$src")" "$DEV_BASE/models/$name" | tail -1
}

cmd_push_ops() {
  # P2b: ホスト側で ready_for_device になった単体 op モデルを一括で送る。
  # 1本ずつ push-model するのは現実的でないため専用にした。
  ensure_device
  local n=0
  # models/ops はアプリが作らないので、**アプリのUIDで**作る必要がある。
  # adb shell の mkdir で作ると所有者が shell になり、アプリから書けなくなる
  # （v0.1 で実際に踏んだ。基本設計③ 7章）。run-as も untrusted_app とは別ドメイン
  # なので使えない。ここでは push 先を掘らせるために先に1本送って確認する。
  for f in "$HERE"/work/ops/*/*_epctx.onnx; do
    [ -f "$f" ] || continue
    local name; name="$(basename "$f")"
    "$ADB" push "$(cygpath -w "$f" 2>/dev/null || echo "$f")" \
        "$DEV_BASE/models/ops/$name" >/dev/null || die "push 失敗: $name"
    n=$((n+1))
    printf '  %-40s %s\n' "$name" "$(ls -l "$f" | awk '{printf "%.1f KiB", $5/1024}')"
  done
  [ "$n" -gt 0 ] || die "送るモデルが無い。先に bash scripts/p2b_build_ops.sh を実行すること"
  echo "=== $n 本を $DEV_BASE/models/ops/ へ送った ==="
}

cmd_run() {
  ensure_device
  local job_src="${1:-}"
  [ -f "$job_src" ] || die "ジョブ仕様が無い: $job_src"
  local job_name; job_name="$(basename "$job_src")"
  local stamp; stamp="$(date +%Y%m%d_%H%M%S)"

  echo "=== ジョブ仕様を push: $job_name ==="
  # ディレクトリはアプリが作る。ここでは作らない（所有者が shell になるため）。
  "$ADB" push "$(cygpath -w "$job_src" 2>/dev/null || echo "$job_src")" "$DEV_BASE/jobs/$job_name" | tail -1

  echo "=== 前回の結果を消す ==="
  "$ADB" shell "rm -f $DEV_BASE/out/last_result.json $DEV_BASE/out/last_describe.json $DEV_BASE/out/last_error.json"

  "$ADB" logcat -c 2>/dev/null

  echo "=== 起動 ==="
  # Activity 経由で起動する。Android 12+ はバックグラウンドからの
  # Foreground Service 起動を制限するため（基本設計③ 3章）。
  "$ADB" shell am start -n "$PKG/.RunnerActivity" \
      -a "$PKG.RUN" --es job "jobs/$job_name" 2>&1 | sed 's/^/  /'

  echo "=== 完了待ち（最大180秒） ==="
  "$ADB" shell 'for i in $(seq 1 90); do
    if [ -f '"$DEV_BASE"'/out/last_result.json ] || [ -f '"$DEV_BASE"'/out/last_error.json ] || [ -f '"$DEV_BASE"'/out/last_describe.json ]; then
      echo "  done at $((i*2))s"; exit 0
    fi
    sleep 2
  done; echo "  TIMEOUT"'

  echo "=== 結果 ==="
  for f in last_result last_describe last_error; do
    if "$ADB" shell "[ -f $DEV_BASE/out/$f.json ]" 2>/dev/null; then
      echo "--- $f.json ---"
      "$ADB" shell "cat $DEV_BASE/out/$f.json"
      echo
      "$ADB" pull "$DEV_BASE/out/$f.json" "$OUTDIR/${stamp}_${f}.json" >/dev/null 2>&1
    fi
  done

  echo "=== アプリログ ==="
  "$ADB" logcat -d -s EdgeSLMRunner 2>/dev/null | tail -40

  echo ""
  echo "保存先: $OUTDIR/${stamp}_*.json"
}

case "${1:-}" in
  install)    cmd_install ;;
  push-model) shift; cmd_push_model "$@" ;;
  push-ops)   cmd_push_ops ;;
  run)        shift; cmd_run "$@" ;;
  *) echo "usage: $0 {install|push-model <onnx>|push-ops|run <job.json>}"; exit 1 ;;
esac
