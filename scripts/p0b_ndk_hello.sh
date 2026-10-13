#!/bin/bash
# P0-b: NDK クロスビルド → 端末での実行確認
#
# 目的: 「PCでビルドした arm64 ネイティブバイナリが SOG05 上で動く」という
#       本プロジェクトの最終成果物（adb で動くネイティブCLI）の土台を検証する。
#
# 使い方: Git Bash から
#   bash scripts/p0b_ndk_hello.sh
#
# 前提: Android NDK 21.4.7075529 が D:\Android\sdk\ndk\ に導入済み（P0 で確認済み）
#       端末が adb で接続済み（P0-a が通っていること）
#
# 注意: NDK のバージョンは ZipDepth（P1）の要求に合わせて 21.4.7075529 に固定している。
#       新しいNDKに上げてはいけない（要件定義書 3.3.3節）。

set -u

# Git Bash(MSYS) は引数先頭の / を Windows パスへ変換するため、端末側パス
# (/data/local/tmp/...) が 'C:/Program Files/Git/data/...' に化ける。
# ただし MSYS_NO_PATHCONV=1 や EXCL='*' で全面無効化すると、今度はホスト側パス
# (/d/AI/... → D:/AI/...) の変換も止まり clang がソースを見つけられなくなる。
# そこで「端末側のプレフィックスだけ」を変換対象から除外する。
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc'
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="$HERE/results"
BUILD="$HERE/build/p0b"
mkdir -p "$OUTDIR" "$BUILD"

SDK="${ANDROID_SDK_ROOT:-D:/Android/sdk}"
NDK_VER="21.4.7075529"
TC="$SDK/ndk/$NDK_VER/toolchains/llvm/prebuilt/windows-x86_64/bin"
# minSdk は ZipDepth と揃えて 26
CC="$TC/aarch64-linux-android26-clang.cmd"
ADB="${ADB:-adb}"
DEVDIR="/data/local/tmp/edgeslm"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTDIR/P0b_${STAMP}.txt"

{
echo "EdgeSLM_Android P0-b NDKクロスビルド検証"
echo "取得時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "NDK: $NDK_VER / driver: $(basename "$CC")"

echo ""
echo "--- 1. コンパイラの存在確認 ---"
if [ ! -f "$CC" ]; then
  echo "ERROR: コンパイラが見つからない: $CC"
  echo "　　　 NDK $NDK_VER が導入されているか確認する"
  exit 1
fi
echo "OK: $CC"

echo ""
echo "--- 2. ソース生成 ---"
cat > "$BUILD/hello_arm64.c" <<'EOF'
#include <stdio.h>
#include <unistd.h>
#include <sys/utsname.h>

int main(void) {
    struct utsname u;
    printf("EdgeSLM_Android P0-b: native arm64 binary running on device\n");
    if (uname(&u) == 0) {
        printf("  sysname=%s release=%s machine=%s\n", u.sysname, u.release, u.machine);
    }
    printf("  pid=%d  page_size=%ld  nproc_onln=%ld\n",
           (int)getpid(), sysconf(_SC_PAGESIZE), sysconf(_SC_NPROCESSORS_ONLN));
    return 0;
}
EOF
echo "OK: $BUILD/hello_arm64.c"

echo ""
echo "--- 3. クロスコンパイル ---"
"$CC" -O2 -o "$BUILD/hello_arm64" "$BUILD/hello_arm64.c" || { echo "ERROR: コンパイル失敗"; exit 1; }
ls -l "$BUILD/hello_arm64"

echo ""
echo "--- 4. ELFヘッダ検査（e_machine=0xB7 が AArch64） ---"
od -An -tx1 -N20 "$BUILD/hello_arm64" | head -2
echo "（offset 18-19 が 'b7 00' であれば AArch64。'3e 00' なら x86-64 で誤り）"

echo ""
echo "--- 5. 端末へ転送して実行 ---"
if ! "$ADB" get-state >/dev/null 2>&1; then
  echo "SKIP: 端末が接続されていない（adb get-state 失敗）。ビルドのみ検証した"
else
  "$ADB" shell "mkdir -p $DEVDIR" || true
  "$ADB" push "$BUILD/hello_arm64" "$DEVDIR/hello_arm64"
  "$ADB" shell "chmod 755 $DEVDIR/hello_arm64"
  echo "--- 端末上での実行結果 ---"
  "$ADB" shell "$DEVDIR/hello_arm64" || echo "ERROR: 端末上で実行できなかった（noexec や権限を確認）"
fi
} 2>&1 | tee "$OUT"

echo ""
echo "-----------------------------------------------"
echo "保存先: $OUT"
