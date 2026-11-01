#!/bin/bash
# P0-a: SOG05 端末素性の一括取得
#
# 目的: 基本設計①（計測ハーネス設計）の入力となる事実を、取りこぼしなく1回で集める。
#       特に ①thermal zone の構成 ②QNN/FastRPC ランタイムの在否 ③/data/local/tmp の実行可否
#       の3点が後続フェーズの前提になる。
#
# 使い方: Git Bash から
#   bash scripts/p0a_collect.sh
#
# 出力: results/P0a_<YYYYmmdd_HHMMSS>.txt （標準出力にも流す）
#
# 注意: 1つの探査が失敗しても全体は止めない（|| true）。端末・firmware により
#       読めない sysfs があるのは想定内で、「読めない」こと自体が設計上の情報である。
#
# 【補足（学習用、2026-10-18 追記）】
# どの段か: P0（環境と端末の確認）。StudyMemo 12 章「新しいモデルを載せるかの判断」で、端末の確認に使う道具として挙がる。
# 結果: results/P0a_20260912_150041.txt（ro.board.platform=lahaina・thermal zone 93 個など）。基本設計①の入力になった。
# 書き方の注意: 各節の adb shell '…' は端末側で動く 1 つのシェルスクリプト（単一引用符の中は端末の sh が解釈する）。

set -u

# Git Bash(MSYS) は引数先頭の / を Windows パスへ変換するため、端末側パス
# (/data/local/tmp/...) が 'C:/Program Files/Git/data/...' に化ける。
# ただし MSYS_NO_PATHCONV=1 や EXCL='*' で全面無効化すると、今度はホスト側パス
# (/d/AI/... → D:/AI/...) の変換も止まり clang がソースを見つけられなくなる。
# そこで「端末側のプレフィックスだけ」を変換対象から除外する。
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc'
# HERE = プロジェクトの根（scripts/ の 1 つ上）。結果は results/ に日時付きで残す。
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="$HERE/results"
mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTDIR/P0a_${STAMP}.txt"

# adb が PATH にない場合はここを書き換える
ADB="${ADB:-adb}"

# sec: 節の見出しを出す小さな関数。
sec() { echo ""; echo "############################################################"; echo "# $*"; echo "############################################################"; }

# ここから最後の } までの出力を、画面とファイルの両方に流す（末尾の tee）。
{
echo "EdgeSLM_Android P0-a 端末素性取得"
echo "取得時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "ホスト: $(uname -s) / adb: $("$ADB" version 2>/dev/null | head -1)"

sec "0. 接続確認"
"$ADB" devices -l || true

sec "1. 端末識別（期待値: ro.board.platform=lahaina / ro.product.cpu.abi=arm64-v8a）"
"$ADB" shell 'for p in ro.board.platform ro.product.cpu.abi ro.product.cpu.abilist \
  ro.build.version.release ro.build.version.sdk ro.build.id ro.build.display.id \
  ro.product.manufacturer ro.product.model ro.product.name ro.hardware \
  ro.soc.manufacturer ro.soc.model; do
  echo "$p = $(getprop $p)"
done' || true

sec "2. CPU 構成（big.LITTLE の確認。スレッド数・親和性の設計に使う）"
"$ADB" shell 'echo "cores: $(grep -c ^processor /proc/cpuinfo)"; grep -m1 "Hardware" /proc/cpuinfo 2>/dev/null
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  n=$(basename $c)
  mx=$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)
  mn=$(cat $c/cpufreq/cpuinfo_min_freq 2>/dev/null)
  gov=$(cat $c/cpufreq/scaling_governor 2>/dev/null)
  echo "$n max=${mx:-NA} min=${mn:-NA} gov=${gov:-NA}"
done' || true

sec "3. メモリ"
"$ADB" shell 'grep -E "^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree)" /proc/meminfo' || true

sec "4. thermal zone（★基本設計①の主入力。type と現在温度、読めないゾーンも記録）"
"$ADB" shell 'for z in /sys/class/thermal/thermal_zone*; do
  t=$(cat $z/type 2>/dev/null); v=$(cat $z/temp 2>/dev/null)
  echo "$(basename $z)  type=${t:-UNREADABLE}  temp=${v:-UNREADABLE}"
done' || true

sec "5. cooling device（スロットリングの観測対象）"
"$ADB" shell 'for d in /sys/class/thermal/cooling_device*; do
  t=$(cat $d/type 2>/dev/null); c=$(cat $d/cur_state 2>/dev/null); m=$(cat $d/max_state 2>/dev/null)
  echo "$(basename $d)  type=${t:-UNREADABLE}  cur=${c:-NA}/max=${m:-NA}"
done' || true

# 6: FastRPC（CPU から DSP/HTP を呼ぶ仕組み）のライブラリとデバイスノードがあるか。
sec "6. ★QNN / FastRPC ランタイムの在否（P1 の成否見通しに直結）"
"$ADB" shell 'for f in /vendor/lib64/libcdsprpc.so /vendor/lib/libcdsprpc.so \
  /vendor/lib64/libadsprpc.so /vendor/lib64/libsdsprpc.so; do
  [ -f "$f" ] && echo "FOUND  $f" || echo "absent $f"
done
echo "--- /vendor/lib64 の QNN/SNPE 関連 ---"
ls /vendor/lib64 2>/dev/null | grep -iE "qnn|snpe|hta|dsp" || echo "(none matched)"
echo "--- DSP 用 rfsa ディレクトリ ---"
ls -d /vendor/lib/rfsa/adsp /vendor/lib/rfsa/dsp /dsp 2>/dev/null || echo "(none)"
echo "--- rpc デバイスノード ---"
ls -l /dev/*rpc* 2>/dev/null || echo "(none visible)"' || true

sec "7. GPU（Adreno 660 / OpenCL の在否）"
"$ADB" shell 'ls /vendor/lib64 2>/dev/null | grep -iE "OpenCL|libgsl|adreno" || echo "(none matched)"
cat /sys/class/kgsl/kgsl-3d0/gpu_model 2>/dev/null || echo "gpu_model UNREADABLE"' || true

# 8: 小さなスクリプトを置いて sh 経由・直接の 2 通りで実行できるかを試し、最後に消す。
sec "8. /data/local/tmp の実行可否（P0-b のネイティブCLI配置先）"
"$ADB" shell 'd=/data/local/tmp/edgeslm_probe; mkdir -p $d 2>/dev/null
printf "#!/system/bin/sh\necho EXEC_OK\n" > $d/t.sh 2>/dev/null
chmod 755 $d/t.sh 2>/dev/null
sh $d/t.sh 2>&1 || echo "sh-exec FAILED"
$d/t.sh 2>&1 || echo "direct-exec FAILED (noexec mount の可能性)"
echo "--- 空き容量 ---"; df /data 2>/dev/null | tail -2
rm -rf $d 2>/dev/null' || true

sec "9. 測定条件として記録すべき項目（電池・充電・SELinux）"
"$ADB" shell 'dumpsys battery 2>/dev/null | grep -E "level|temperature|status|powered|present" ' || true
"$ADB" shell 'echo "SELinux: $(getenforce 2>/dev/null)"; echo "uptime: $(uptime)"' || true

# 10: 前提（SM8350 = lahaina、arm64、minSdk 26 以上）と合っているかを 1 行ずつ判定する。
sec "10. まとめ（期待値との突き合わせ）"
PLAT="$("$ADB" shell getprop ro.board.platform 2>/dev/null | tr -d '\r')"
ABI="$("$ADB" shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r')"
SDKV="$("$ADB" shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')"
echo "ro.board.platform = '$PLAT'  (期待: lahaina)"
echo "ro.product.cpu.abi = '$ABI'  (期待: arm64-v8a)"
echo "Android SDK level  = '$SDKV' (minSdk 26 以上であること)"
[ "$PLAT" = "lahaina" ] && echo "=> PLATFORM OK" || echo "=> PLATFORM MISMATCH: 前提（SM8350）から見直しが必要"
[ "$ABI" = "arm64-v8a" ] && echo "=> ABI OK" || echo "=> ABI MISMATCH"

echo ""
echo "（このファイルの内容をそのまま基本設計①の入力とする）"
} 2>&1 | tee "$OUT"

echo ""
echo "-----------------------------------------------"
echo "保存先: $OUT"
