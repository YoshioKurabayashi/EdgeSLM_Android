#!/bin/bash
# P0-c: adb shell から FastRPC(CDSP) にアクセスできるかの実測
#
# 背景: P0-a で /dev/adsprpc-smd の権限が crw-rw-r-- (system:system) と判明した。
#       adb shell は uid=2000(shell) で system グループに属さないため、
#       other の 'r' しか無い = 書き込みオープンできない可能性がある。
#
# なぜ重要か: 要件定義書 4.1 は成果物を「adb 経由で動作する arm64 ネイティブ CLI」と
#       しているが、CLI から FastRPC が開けないなら HTP 推論を CLI 形態で行えず、
#       スコープ（アプリ化を対象外とした判断）の再検討が必要になる。
#       → 本スクリプトはその判断材料を取るためのもの。
#
# 使い方: bash scripts/p0c_fastrpc_probe.sh
#
# 【補足（学習用、2026-10-18 追記）】
# 結果: results/P0c_20260912_150417.txt。adb shell（uid 2000）からは /dev/adsprpc-smd を O_RDONLY でも O_RDWR でも
#   開けなかった（errno=13 Permission denied）。続く results/P0c2_20260912_selinux_domain.md で run-as も駄目、
#   アプリ本体のプロセス（untrusted_app）なら開けると分かり、「推論はアプリの中で動かす」（android_runner）に決まった。
# 出力: build/p0c/fastrpc_probe（と .c）、results/P0c_<日時>.txt。

set -u

# 端末側パスのみ MSYS のパス変換から除外する（p0b と同じ理由）
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc'

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUTDIR="$HERE/results"
BUILD="$HERE/build/p0c"
mkdir -p "$OUTDIR" "$BUILD"

SDK="${ANDROID_SDK_ROOT:-D:/Android/sdk}"
NDK_VER="21.4.7075529"
TC="$SDK/ndk/$NDK_VER/toolchains/llvm/prebuilt/windows-x86_64/bin"
CC="$TC/aarch64-linux-android26-clang.cmd"
ADB="${ADB:-adb}"
DEVDIR="/data/local/tmp/edgeslm"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTDIR/P0c_${STAMP}.txt"

# 端末で動かす小さな C のプローブを書き出す（ヒアドキュメントの中は C のソースなので、説明は C の // で書いてある）。
cat > "$BUILD/fastrpc_probe.c" <<'EOF'
// FastRPC/CDSP へのアクセス可否を調べる最小プローブ。
// 1) DSP のデバイスノードを O_RDONLY / O_RDWR で開けるか（errno を報告）
// 2) FastRPC のユーザ空間ライブラリを dlopen できるか
// 3) dlopen できた場合、CDSP ドメインを開く API シンボルが見えるか
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

static void try_open(const char *path, int flags, const char *label) {
    int fd = open(path, flags);
    if (fd >= 0) {
        printf("  OPEN OK    %-28s (%s)\n", path, label);
        close(fd);
    } else {
        printf("  OPEN FAIL  %-28s (%s) errno=%d %s\n", path, label, errno, strerror(errno));
    }
}

static void try_dlopen(const char *so) {
    void *h = dlopen(so, RTLD_NOW);
    if (!h) {
        printf("  DLOPEN FAIL %-40s %s\n", so, dlerror());
        return;
    }
    printf("  DLOPEN OK   %s\n", so);
    const char *syms[] = {"remote_handle64_open", "remote_handle_open",
                          "remote_session_control", "fastrpc_init_once", NULL};
    for (int i = 0; syms[i]; i++) {
        void *s = dlsym(h, syms[i]);
        printf("    sym %-24s %s\n", syms[i], s ? "found" : "-");
    }
    dlclose(h);
}

int main(void) {
    printf("uid=%d gid=%d\n", (int)getuid(), (int)getgid());

    printf("[1] device nodes\n");
    try_open("/dev/adsprpc-smd", O_RDONLY, "O_RDONLY");
    try_open("/dev/adsprpc-smd", O_RDWR,   "O_RDWR  <-- FastRPC に必要");
    try_open("/dev/adsprpc-smd-secure", O_RDWR, "O_RDWR");

    printf("[2] fastrpc user libs\n");
    try_dlopen("libcdsprpc.so");
    try_dlopen("/vendor/lib64/libcdsprpc.so");
    try_dlopen("libadsprpc.so");

    printf("[3] SNPE (参考: 端末に preinstall されている)\n");
    try_dlopen("/vendor/lib64/libSNPE.so");
    return 0;
}
EOF

# ここから最後の } までの出力を、画面とファイルの両方に流す（末尾の tee）。
{
echo "EdgeSLM_Android P0-c FastRPC アクセス可否の実測"
echo "取得時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "--- ビルド ---"
"$CC" -O2 -o "$BUILD/fastrpc_probe" "$BUILD/fastrpc_probe.c" -ldl || { echo "ERROR: コンパイル失敗"; exit 1; }
ls -l "$BUILD/fastrpc_probe"

echo ""
echo "--- 端末側の権限（再確認） ---"
"$ADB" shell 'ls -l /dev/adsprpc-smd /dev/adsprpc-smd-secure 2>&1; id'

echo ""
echo "--- 端末での実行結果 ---"
"$ADB" shell "mkdir -p $DEVDIR"
"$ADB" push "$BUILD/fastrpc_probe" "$DEVDIR/fastrpc_probe" >/dev/null
"$ADB" shell "chmod 755 $DEVDIR/fastrpc_probe"
"$ADB" shell "$DEVDIR/fastrpc_probe" 2>&1

echo ""
echo "--- 判定の読み方 ---"
echo "  [1] の O_RDWR が OPEN OK      -> CLI 形態で FastRPC を使える見込み。成果物の形は変更不要"
echo "  [1] の O_RDWR が errno=13     -> shell ユーザーでは開けない。HTP 推論を CLI で行えない可能性が高く、"
echo "      (EACCES)                     アプリ化（対象外としていた範囲）の再検討が必要"
echo "  [2] の dlopen が失敗          -> ライブラリ自体の依存不足。linker の namespace 制約の可能性もある"
} 2>&1 | tee "$OUT"

echo ""
echo "-----------------------------------------------"
echo "保存先: $OUT"
