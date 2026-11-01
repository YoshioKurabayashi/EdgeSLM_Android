#!/bin/bash
# P1 準備: ZipDepth リポジトリの Git LFS 実体を検証し、ビルド前提を確認する
#
# 目的:
#   Git LFS を有効にせずクローンすると、巨大バイナリが 130 バイト程度の
#   「ポインタ」テキストのまま取得される。この状態でも **ビルドは成功し、
#   端末で実行した瞬間にネイティブクラッシュする**（原因が分かりにくい典型的な罠）。
#   そこで DEPENDENCIES.md 記載の SHA-256 と実ファイルを照合して実体性を保証する。
#
# 設計: 期待値はスクリプトにハードコードせず、**リポジトリ内の DEPENDENCIES.md から読む**。
#       上流が版を上げても追従できる。
#
# 使い方: bash scripts/p1_verify_zipdepth.sh
#
# 【補足（学習用、2026-10-18 追記）】
# どの段か: P1（参考実装 ZipDepth を端末で動かす。DoC/参考実装_ZipDepth解説.md・results/P1_20260912_zipdepth_run.md）。
# 結果: results/P1verify_20260912_155505.txt・155532.txt。
# 注意: このファイルは改行が CRLF。Git Bash の設定によっては行末の CR が残ってエラーになりうる（推定。実行記録は残っている）。
#       また 2 節の printf の書式（OK の行）は文字列の途中で改行していて、出力にもその改行（CRLF なら CR も）が入る。

set -u
export MSYS2_ARG_CONV_EXCL='/data/local/tmp;/sys;/vendor;/dev;/system;/proc'

HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$HERE/third_party/Edge_ZipDepth-SM8350"
OUTDIR="$HERE/results"
mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$OUTDIR/P1verify_${STAMP}.txt"

SDK="${ANDROID_SDK_ROOT:-D:/Android/sdk}"
JDK="${JAVA_HOME_17:-D:/Android/jdk17}"

{
echo "ZipDepth リポジトリ検証"
echo "取得時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "対象: $REPO"

if [ ! -d "$REPO" ]; then
  echo "ERROR: リポジトリが無い。先に clone すること"
  exit 1
fi

echo ""
echo "=== 1. リポジトリ基本情報 ==="
git -C "$REPO" log -1 --format='commit %H%nauthored %ad%nsubject %s' 2>/dev/null
echo "総サイズ: $(du -sh "$REPO" 2>/dev/null | cut -f1)"

echo ""
echo "=== 2. LFS 実体の SHA-256 照合（DEPENDENCIES.md を期待値とする） ==="
DEP="$REPO/DEPENDENCIES.md"
if [ ! -f "$DEP" ]; then echo "ERROR: DEPENDENCIES.md が無い"; exit 1; fi

# 2 節: DEPENDENCIES.md の表の各行を読み、ファイルが無い／LFS のポインタのまま／hash が違う／OK を数える。
PASS=0; FAIL=0; PTR=0
# 表形式: | `path` | size | purpose | `SHA256` |
#   バッククォートを外し、1列目がパスらしく 4列目が64桁hexの行だけを対象にする
while IFS='|' read -r _ c1 c2 c3 c4 _rest; do
  path="$(echo "$c1" | tr -d ' `')"
  want="$(echo "$c4" | tr -d ' `' | tr 'A-F' 'a-f')"
  case "$path" in app/*) ;; *) continue ;; esac
  echo "$want" | grep -qE '^[0-9a-f]{64}$' || continue

  f="$REPO/$path"
  if [ ! -f "$f" ]; then
    printf "  MISSING   %-52s\n" "$path"; FAIL=$((FAIL+1)); continue
  fi
  sz=$(wc -c < "$f")
  # LFS ポインタの判定: 極小サイズ かつ 先頭が LFS スペック行
  if [ "$sz" -lt 1024 ] && head -c 60 "$f" | grep -q "git-lfs.github.com/spec"; then
    printf "  LFS-POINTER %-50s (%s bytes) ★実体が取得されていない\n" "$path" "$sz"
    PTR=$((PTR+1)); FAIL=$((FAIL+1)); continue
  fi
  got="$(sha256sum "$f" | cut -d' ' -f1)"
  if [ "$got" = "$want" ]; then
    printf "  OK        %-52s %s MiB
" "$path" "$(awk -v b=$sz "BEGIN{printf \"%8.2f\", b/1048576}")"
    PASS=$((PASS+1))
  else
    printf "  HASH-NG   %-52s\n            want=%s\n            got =%s\n" "$path" "$want" "$got"
    FAIL=$((FAIL+1))
  fi
done < "$DEP"

echo ""
echo "  照合結果: OK=$PASS  NG=$FAIL  (うち LFS ポインタのまま=$PTR)"
if [ "$PTR" -gt 0 ]; then
  echo "  → 対処: cd $REPO && git lfs install && git lfs pull"
fi

echo ""
echo "=== 3. ビルド前提の確認 ==="
# 期待値はリポジトリ側の設定から読む
AGP="$(grep -oE "com.android.application' version '[0-9.]+" "$REPO/build.gradle" 2>/dev/null | grep -oE '[0-9.]+$')"
GRADLE="$(grep -oE 'gradle-[0-9.]+-bin' "$REPO/gradle/wrapper/gradle-wrapper.properties" 2>/dev/null)"
NDKV="$(grep -oE 'ndk.version=[0-9.]+' "$REPO/gradle.properties" 2>/dev/null | cut -d= -f2)"
CSDK="$(grep -oE 'compileSdk [0-9]+' "$REPO/app/build.gradle" 2>/dev/null | grep -oE '[0-9]+')"
CMAKEV="$(grep -oE "version '[0-9.]+'" "$REPO/app/build.gradle" 2>/dev/null | grep -oE '[0-9.]+' | head -1)"
printf "  リポジトリの要求: AGP=%s / %s / NDK=%s / compileSdk=%s / CMake=%s\n" "$AGP" "$GRADLE" "$NDKV" "$CSDK" "$CMAKEV"

# chk: 名前とパスを受け取り、そのパスがあるかを OK / MISSING で出す。
chk() { if [ -e "$2" ]; then printf "  OK      %-26s %s\n" "$1" "$2"; else printf "  MISSING %-26s %s\n" "$1" "$2"; fi; }
chk "NDK $NDKV"            "$SDK/ndk/$NDKV"
chk "platforms android-$CSDK" "$SDK/platforms/android-$CSDK"
chk "cmake $CMAKEV"        "$SDK/cmake/$CMAKEV"
chk "build-tools"          "$SDK/build-tools"
chk "JDK 17"               "$JDK/bin/java.exe"

echo ""
echo "=== 4. local.properties ==="
LP="$REPO/local.properties"
if [ -f "$LP" ]; then echo "  既存:"; sed 's/^/    /' "$LP"; else echo "  未作成"; fi

echo ""
echo "=== 5. 次のアクション（JVM を使うためユーザー実行が必要） ==="
cat <<'ACT'
  PowerShell で:
    $env:JAVA_HOME="D:\Android\jdk17"
    cd D:\AI\source\n_chap5\EdgeSLM_Android\third_party\Edge_ZipDepth-SM8350
    .\gradlew.bat assembleDebug

  初回は Gradle 8.13 本体(約130MB)と AGP/Kotlin/OpenCV の依存取得が走るため
  数分〜十数分かかる。以降はキャッシュされる。
ACT
} 2>&1 | tee "$OUT"

echo ""
echo "-----------------------------------------------"
echo "保存先: $OUT"
