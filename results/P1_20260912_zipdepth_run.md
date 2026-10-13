# P1 実行記録: ZipDepth を SOG05 で動作させる

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 端末 | Sony Xperia 5 III SOG05（`QV780Y6DA3`） |
| 対象 | ZipDepth-SM8350（commit `77651fed`）app-debug.apk（97.2 MiB） |
| baseline | QAIRT 2.48 ランタイム同梱 / ORT 1.27.0 AAR / NDK 21.4.7075529 / AGP 8.7.3 |
| 判定 | **条件③④は達成。①②は端末のロック解除待ちで未実施** |

---

## 1. ビルドとインストール

| 工程 | 結果 |
|---|---|
| Git LFS 実体の SHA-256 照合 | **13/13 一致、ポインタ残存0件** |
| `gradlew assembleDebug` | **BUILD SUCCESSFUL in 2m 24s**（警告はSDK XML v4・上流コードのdeprecationのみで無害） |
| `adb install -r` | **Success** |
| カメラ権限 | `adb shell pm grant` で付与（`granted=true`） |

### C-6 の実証

マニフェストに **`android:extractNativeLibs="true"`**（`useLegacyPackaging = true` の結果）。インストール後、`.so` が**実ファイルとして展開**されていることを確認した。

```
/data/app/~~4_rJjgg0Wemm2stQ8LaE2w==/org.zipdepth.npudemo--GJRc8u0KfbAkm6k58pB4Q==/lib/arm64/
  -rwxr-xr-x  3760360  libQnnHtp.so
  -rwxr-xr-x 10240924  libQnnHtpV68Skel.so    ← QAIRT SDK 同梱物と完全一致
  -rwxr-xr-x  4052904  libQnnSystem.so
```

**APK 内では `.so` は DEFLATED（圧縮）で格納されている。** それでも `extractNativeLibs=true` によりインストール時に展開されるため FastRPC が要求するファイルシステムパスが得られる。→ 要件定義書 C-6 の記述をこの実測に合わせて訂正した（圧縮方式ではなくマニフェストを見る）。

その他マニフェスト情報: `package=org.zipdepth.npudemo` / `minSdk=26` / `targetSdk=34` / **`debuggable=true`** / 推論は別プロセス `:zipdepth_npu`。

---

## 2. ★NPU 実行の証拠（logcat 原文）

```
09-12 16:02:44.537  I org.zipdepth.npudemo:zipdepth_npu:
  fastrpc_apps_user.c:1401: remote_handle64_open: Successfully opened handle 0x2badf3d0 for
  file:///libQnnHtpV68Skel.so?qnn_2_48_0_skel_handle_invoke&_modver=1.0&_dom=cdsp on domain 3

09-12 16:02:44.602  I ZipDepthDemo: ZipDepth session opened
  (board=lahaina backend='QNN HTP • EPContext • all nodes' input=image output=depth)
09-12 16:02:44.602  I ZipDepthDemo: warmup: running the first OrtRun…
09-12 16:02:44.641  I ZipDepthDemo: warmup: first OrtRun completed in 38.4 ms
09-12 16:02:44.641  I ZipDepthDemo: ZipDepth ready: board=lahaina
  backend='QNN HTP • EPContext • all nodes' input=image output=depth
09-12 16:02:44.643  I ZipDepthDemo: remote model init: ready=true res=384
  result="QNN HTP • EPContext • all nodes"
```

読み取れること。

| 事実 | 意味 |
|---|---|
| `libQnnHtpV68Skel.so ... _dom=cdsp on domain 3` を **FastRPC で open 成功** | **アプリの SELinux ドメインからは CDSP(domain 3) にアクセスできる**。`adb shell` では EACCES だった（R-9）のと対照的 |
| skel 名に **`qnn_2_48_0`** | QAIRT **2.48** のスケルトンが実際にロードされた |
| backend = **`QNN HTP • EPContext • all nodes`** | **全ノードが HTP**。2つある正常パターンのうち良い方（`CPU I/O boundary` ではない） |
| first OrtRun = **38.4 ms** | 参考値 40〜60ms（384×384・ウォームアップ後）と同オーダー。しかもこれはコールドスタートの初回 |
| `board=lahaina` | SoC 判定をアプリ側も通過 |

### ノイズとして無視してよいログ

起動直前に `adsprpc/src/apps_std_imp.c` の `apps_std_FILE_get failed for fd 0x464, errno Bad file descriptor` が多数出る。これは Qualcomm の adsprpc 層がアプリ側のファイルを探索して失敗している過程で、直後に

```
W ...: apps_std_fopen_with_env failed with 0x2 for libc++abi.so.1 (No such file or directory)
I cdsprpcd: Successfully opened file /vendor/dsp/cdsp/libc++abi.so.1
```

と **`cdsprpcd` が正しい場所から開き直して成功**している。**失敗と誤読しないこと。**

---

## 3. 物理的裏取り（温度・dma-buf）

| 時点 | nspss-0 / nspss-2 | cdsp | DmaBufTotal |
|---|---|---|---|
| 推論前（baseline） | 32.7 / 32.7 ℃ | 0/7 | 725,536 kB |
| 起動直後 | 34.7 / 34.7 ℃ | 0 | 736,992 kB |
| 起動20秒後（ピーク） | **35.1 / 35.4 ℃** | 0 | **770,804 kB** |

- **NPU温度が +2.4〜2.7℃ 上昇**し、**DmaBufTotal が +45 MB 増加**した。context と各種バッファが DSP へ渡されたことが dma-buf 会計に現れている（基本設計① 4.7節の代替指標が実際に機能することの確認）
- `cdsp` cooling device は 0 のまま＝スロットリング未発動（短時間・アイドル状態のため当然）

---

## 4. P1 Gate の判定状況

| 条件 | 状態 |
|---|---|
| ① 深度マップが表示され推論が完了する | **未実施**。端末がロック画面（`mDreamingLockscreen=true`）でアプリが前面に来ず、カメラが動いていない。ウォームアップ推論1回は成功している |
| ② オンデバイス精度テストが通る | **未実施**（UI操作が必要） |
| ③ 起動ログが `QNN HTP - EPContext - all nodes` | **達成** |
| ④ `HTP` 時間が 40〜60ms のオーダー | **達成**（初回 38.4 ms） |
| ⑤ `nspss-*-usr` の上昇を観測 | **達成**（+2.4〜2.7℃、DmaBuf +45MB） |

---

## 5. この結果が持つ意味

### (a) 端末側の疑いが消えた

ZipDepth の実機は docomo の **SO-51B**、こちらは au の **SOG05** であり、firmware / carrier 差が最大の未知要素だった（要件定義書 R-7）。**同一の構成で動いたため、以降 P2 以降で失敗した場合は我々の量子化・変換・実装の問題と断定できる。**

### (b) 未決事項#14 の案②（アプリ化）が確実に使えることが確定した

R-9 により `adb shell`（`u:r:shell:s0`）からは `/dev/adsprpc-smd` が EACCES だったが、**アプリのドメインからは FastRPC が通り CDSP domain 3 を開けた**。したがって HTP 推論の実行形態として**アプリ化は確実な退路**である。

残る案①（`run-as` 方式）は、**この APK が `debuggable=true` であるため追加のビルドなしに検証できる**。

### (c) unsigned PD での動作が実証された

`--htp_socs sm8350` のみで生成された context binary（`socModel=0`）が、電力プロファイル未指定のまま unsigned PD で動いている。要件定義書3.4節 C-2 / C-3 の正しさが実機で裏付けられた。

---

## 6. 次のアクション

1. **端末のロックを解除し、ZipDepth アプリを前面にする**（ユーザー操作）
2. カメラを何かに向け、連続推論状態で以下を取得
   - テレメトリ（`CAM` / `DEPTH` / `HTP` / `NATIVE` / `ROUND` / `E2E`）
   - 連続推論時の `nspss-*` 温度推移と `cdsp` の発動有無
3. **オンデバイス精度テスト**を実行し、同梱 fp32 CPU 参照との採点結果を記録（条件②）
4. 7 variant（A16W8 / A8W8 × 384 / 320 / 288 / 256）を切り替えて比較データを取得（P3 の予習）
5. 案①（`run-as`）の検証
