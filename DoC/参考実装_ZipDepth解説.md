# 参考実装 ZipDepth 解説

| 項目 | 内容 |
|---|---|
| 文書名 | 参考実装 ZipDepth 解説 |
| プロジェクト名 | EdgeSLM_Android |
| 版数 | 1.0 |
| 作成日 | 2026-09-12 |
| 対象 | `third_party/Edge_ZipDepth-SM8350`（commit `77651fed`、2026-07-14） |
| 上流 | https://github.com/meh301/ZipDepth-Quantized-Qualcomm-QNN-SM8350 |

## 改訂履歴

| 版数 | 日付 | 変更内容 |
|---|---|---|
| 1.0 | 2026-09-12 | 初版。P1 実施時点のソース読解結果をまとめた |

---

## 0. この文書の目的

時間が経つと「ZipDepth って何だったか」「Gradle / APK とは何か」が分からなくなるため、**後から読んで思い出せる**ことを目的に書いた。1章で位置づけ、2章で用語、3章以降でソースの中身を扱う。

---

## 1. ZipDepth とは何で、なぜ使うのか

### 1.1 2階層あることに注意

「ZipDepth」という言葉は2つのものを指す。

| | 何か |
|---|---|
| **ZipDepth（上流のモデル）** | 学術研究の**単眼深度推定モデル**。1枚のカラー画像から各ピクセルの相対的な奥行きを推定する CNN。https://github.com/fabiotosi92/ZipDepth |
| **ZipDepth-SM8350（我々が使うリポジトリ）** | 上記モデルを **Snapdragon 888 / HTP v68 向けに量子化・コンパイルして、Android アプリとして動かした実装**。第三者（meh301 氏）による公開物 |

我々が参照しているのは後者である。

### 1.2 なぜこれを使うのか

**SOG05 と同じ SM8350 / HTP v68 で、実際に NPU 推論が動いている唯一の公開アプリ**だからである。本プロジェクトにとって2つの価値がある。

| 価値 | 内容 |
|---|---|
| **① P1 の検証対象** | 「我々の端末（au版 SOG05）で NPU が使えるか」を、**自分のコードを一切持ち込まずに**確認できる。提供元が違えば原因の切り分けが一意になる（下表） |
| **② 変換パイプラインの参照実装** | `tools/variants/` に、**v68 向け context binary を作る手順が動く形で残っている**。しかも「これを破ると端末で error 1002 になる」という禁止事項が文書化されている。要件定義書3.4節の C-1〜C-6 はここから取った |

P1 で「誰が何を提供するか」。

| 要素 | 提供元 |
|---|---|
| アプリのソース | **ZipDepth（第三者）** |
| 量子化済みモデル / context binary | **ZipDepth（第三者）** |
| QNN ランタイムライブラリ | **ZipDepth に同梱**（QAIRT 2.48 のセット） |
| **端末** | **我々**（SOG05） |

→ 失敗すれば端末側（firmware / carrier 差）、成功すれば端末の疑いが消え、**以降の失敗は我々の実装の問題と断定できる**。

---

## 2. 用語集

### 2.1 ビルドとパッケージ

| 用語 | 意味 |
|---|---|
| **Gradle** | **ビルドを自動化する道具**。「どのライブラリを使うか」「どうコンパイルするか」を `build.gradle` に書いておくと、依存ライブラリの取得からコンパイル、パッケージングまで一括でやってくれる。Android 開発の標準。`gradlew.bat assembleDebug` がその実行コマンドで、`gradlew` は「Gradle 本体を自動で落としてから実行するラッパー」 |
| **APK** | **Android Package** の略。**Android アプリの配布・インストール単位となる1個のファイル**。実体は ZIP で、中に ①コンパイル済みの Java/Kotlin コード ②ネイティブライブラリ（`.so`） ③画像やモデルなどの資産（assets） ④設定（AndroidManifest.xml） が入っている。`adb install app-debug.apk` で端末に入れる |
| **AAR** | **Android ARchive**。Android 用のライブラリ配布形式（Java の JAR に Android 固有の資産を足したもの）。ZipDepth は `onnxruntime-android-qnn-1.27.0.aar` を同梱している＝ONNX Runtime 本体をこの形で取り込んでいる |
| **AGP** | Android Gradle Plugin。Gradle に Android 用のビルド手順を教えるプラグイン。ZipDepth は **8.7.3** を要求 |
| **NDK** | Native Development Kit。**Android 向けに C / C++ をコンパイルする道具**。ZipDepth は **21.4.7075529（r21e）** を要求（新しい版に上げてはいけない。3.3節） |
| **CMake** | C/C++ のビルド手順を記述する道具。NDK と組み合わせて使う |
| **Kotlin** | Android の主要言語（Java の後継的な位置づけ）。UI やカメラ制御を書いている |
| **JNI** | Java Native Interface。**Kotlin/Java から C の関数を呼ぶための橋渡し**の仕組み |

### 2.2 Qualcomm NPU 周り

| 用語 | 意味 |
|---|---|
| **NPU / HTP** | NPU は Neural Processing Unit。Qualcomm では **HTP（Hexagon Tensor Processor）**と呼ぶ。SD888 の世代は **v68** |
| **CDSP** | Compute DSP。HTP が載っているサブシステム。OS から見た「NPU のある相手」 |
| **FastRPC** | **CPU 側から CDSP 側の関数を呼ぶ仕組み**（リモート手続き呼び出し）。`/dev/adsprpc-smd` を開いて通信する。これが開けないと NPU は使えない（要件定義書 R-9） |
| **skel（スケルトン）** | CDSP 側で動く実体。`libQnnHtpV68Skel.so` が v68 用。**CPU側の `.so` ではなく DSP で実行されるコード**である点に注意 |
| **QNN** | Qualcomm Neural Network。Qualcomm の推論ランタイム API |
| **QAIRT** | Qualcomm AI Runtime。QNN / SNPE / Genie をまとめた SDK の現行名称。ZipDepth は **2.48** を使用 |
| **QDQ** | Quantize-DeQuantize。ONNX 上で量子化を表現する形式。演算の前後に「量子化」「逆量子化」ノードを挟む |
| **context binary** | **QNN が事前にコンパイルしたグラフの実行形式**。これを作っておけば端末側でのコンパイルが不要になる |
| **EPContext** | 上記 context binary を **ONNX ファイルの中に埋め込んだ形式**。ORT から見ると「EPContext という1個のノードを持つ ONNX」に見える。**このノードは QNN EP しか実行できないため、CPU に黙ってフォールバックすることが原理的に起きない** |
| **unsigned PD** | unsigned Protection Domain。署名なしで DSP を使う実行形態。一般アプリはこれを使うが、**電力設定などの特権操作ができない**（3.4節） |

---

## 3. ディレクトリ構成

```
Edge_ZipDepth-SM8350/
├── README.md                   アプリの説明・性能参考値・テレメトリの読み方
├── DEPENDENCIES.md             ★同梱バイナリ13件の一覧とSHA-256（LFS実体の照合に使う）
├── build.gradle                AGP 8.7.3 / Kotlin 2.1.10 を宣言
├── gradle.properties           ★ndk.version=21.4.7075529 を固定
├── gradle/wrapper/...          Gradle 8.13 を使う指定
├── local.properties            （我々が作成）sdk.dir=D:/Android/sdk
├── settings.gradle             依存の取得元（google / mavenCentral）
│
├── app/
│   ├── build.gradle            ★compileSdk 34 / minSdk 26 / arm64-v8a のみ /
│   │                             CMake 3.22.1 / useLegacyPackaging = true
│   ├── libs/
│   │   └── onnxruntime-android-qnn-1.27.0.aar   ONNX Runtime 本体（QNN EP入り）
│   └── src/main/
│       ├── AndroidManifest.xml  権限・プロセス構成
│       ├── java/org/zipdepth/npudemo/
│       │   ├── MainActivity.kt              1557行  カメラ・UI・テレメトリ表示
│       │   ├── CameraCalibrationController.kt 651行  チェッカーボード校正（本検証では不要）
│       │   ├── ZipDepthService.kt             189行  ★NPU推論を行う別プロセスのサービス
│       │   └── ZipDepthNative.kt              114行  JNI の宣言（Kotlin→C の窓口）
│       ├── cpp/
│       │   ├── CMakeLists.txt                  9行  ネイティブのビルド定義
│       │   ├── zipdepth_jni.c                855行  ★中核。ORT/QNN の実装本体
│       │   └── ort/*.h                              ORT C API ヘッダ（AAR 1.27.0 と対応）
│       ├── jniLibs/arm64-v8a/                       ★QNN ランタイム（QAIRT 2.48 のセット）
│       │   ├── libQnnHtp.so            3.59 MiB     HTP バックエンド（CPU側）
│       │   ├── libQnnHtpPrepare.so    83.84 MiB     グラフ準備
│       │   ├── libQnnHtpV68Skel.so     9.77 MiB     ★v68 の DSP 側実体
│       │   ├── libQnnHtpV68Stub.so     0.71 MiB     CPU側スタブ
│       │   └── libQnnSystem.so         3.87 MiB     context メタデータ/ランタイム
│       └── assets/
│           ├── zipdepth.onnx            6.62 MiB    ★既定モデル（A16W8 384、EPContext）
│           ├── models/zd_*.onnx         6件          ★切替可能な variant
│           └── testset/img_*.rgb + 参照深度          精度テスト用（384×384 RGB888 6枚）
│
├── tools/
│   ├── make_testset.py         精度テスト用データの生成
│   └── variants/               ★★我々が最も参照する部分
│       ├── README.md           **v68で error 1002 を起こす禁止事項の一次情報**
│       ├── zd_qnnquant.py      fp32 ONNX → QDQ ONNX（量子化レシピ）
│       ├── compile_variants.sh QDQ → DLC → context binary → EPContext
│       ├── wrap_epctx.py       EPContext ラッパー（既知正解スキーマ）
│       └── check_meta.py       socModel=0 等の検証
└── calibration_target/         カメラ校正用チェッカーボード（本検証では不要）
```

---

## 4. 動作の流れ

### 4.1 2プロセス構成

```
┌─ プロセス1: org.zipdepth.npudemo ───────────────┐
│  MainActivity.kt                                │
│   Camera2 で YUV_420_888 フレームを受け取る       │
│   → ネイティブで RGB 変換・向き補正・正方形中央   │
│     クロップ・384×384 リサイズ                   │
│   → 共有メモリへ置く                             │
│   → 画面に「入力クロップ」と「深度マップ」を表示  │
└──────────────┬──────────────────────────────────┘
               │ Binder（小さな制御メッセージのみ）
               │ 画像データは共有メモリ経由
┌──────────────▼─ プロセス2: :zipdepth_npu ───────┐
│  ZipDepthService.kt → ZipDepthNative.kt (JNI)   │
│   → zipdepth_jni.c                              │
│       ORT セッション（QNN EP / EPContext）       │
│       → FastRPC → CDSP(domain 3) → HTP v68      │
│       → 深度の正規化・カラーマップ               │
└─────────────────────────────────────────────────┘
```

**カメラと NPU を別プロセスに分けている。** README によれば、Camera2 のバッファ所有権を QNN プロセスから切り離す意図。またカメラは独立した生産者として動き続け、**NPU が詰まってもカメラは待たない**（latest-frame backpressure: 古い入力は捨てて最新に置き換える）。

### 4.2 JNI の境界

`ZipDepthNative.kt` が宣言し、`zipdepth_jni.c` が実装する関数は8つ。

```
nativeInit            セッション初期化（ORT + QNN EP のセットアップ）
nativeModelInputSize  モデルの入力サイズ取得
nativePrepareRgb      RGB 前処理
nativePreview         プレビュー用の変換
nativeProcess         推論本体（YUV入力）
nativeProcessRgb      推論本体（RGB入力）
nativeProcessRgbRaw   推論本体（生RGB入力）
nativeShutdown        後片付け
```

`CMakeLists.txt` は9行だけで、`zipdepth_jni.c` 1本を `libzipdepth_demo.so` にビルドし、`log` / `android` / `dl` をリンクしている。**ORT は AAR 側の `libonnxruntime.so` を `dl` 経由で使う構成。**

---

## 5. ★NPU を使うための設定（最重要・我々が流用する部分）

`zipdepth_jni.c` の `open_qnn_session()` が核心である。**「NPU を有効にする設定」はここに集約されている。**

```c
SetSessionLogSeverityLevel(options, 2);
SetIntraOpNumThreads(options, 1);          // グラフ全体がHTPなのでCPUスレッドは1で足る
SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);

const char *keys[]   = { "backend_path", "offload_graph_io_quantization", "qnn_context_priority" };
const char *values[] = { backend_path,   "0",                             "high" };
SessionOptionsAppendExecutionProvider(options, "QNN", keys, values, 3);

AddSessionConfigEntry(options, "session.disable_cpu_ep_fallback", "1");   // strict時
```

読み取れる設計上の要点。

| 設定 | 意味と我々への示唆 |
|---|---|
| `backend_path` | `<アプリのネイティブライブラリディレクトリ>/libQnnHtp.so` を**文字列パスで**渡している（`snprintf` で組み立て）。**これが C-6（`extractNativeLibs=true`）が必要な直接の理由**。`.so` が APK 内に埋まったままではこのパスが存在しない |
| **`offload_graph_io_quantization = "0"`** | **グラフ入出力の量子化/逆量子化を CPU に逃がさない**。これが `QNN HTP • EPContext • all nodes` と `CPU I/O boundary` を分ける設定。我々も `all nodes` を狙うなら `0` にする |
| `qnn_context_priority = "high"` | コンテキスト優先度。per-run の対応設定が無いためプロバイダオプションで渡す |
| **`session.disable_cpu_ep_fallback = "1"`** | **CPU フォールバックを禁止**。要件定義書 FR-5 が要求するのと同じ設定が、参考実装でも使われている |
| `SetIntraOpNumThreads(1)` | 全ノードが HTP なので CPU スレッドは不要 |

### 5.1 電力プロファイルについて（コード内コメントが我々の前提を裏付けている）

```c
// No HTP burst/power-vote options: this app runs in an UNSIGNED protection
// domain, where the HTP setPowerConfig (HAP_Power DCVS) call is not permitted
// and fails the Run. We leave the DSP at its default DCVS.
```

**unsigned PD では `setPowerConfig`（HAP_Power DCVS）が許可されず、Run が失敗する。** そのため burst / 電力votes は一切設定せず、DSP を既定の DCVS に任せている。

これは要件定義書3.4節 C-3（生成時に perf_profile を埋め込むとロードが失敗する）とは**別レイヤの同じ制約**である。整理すると、

| レイヤ | 症状 |
|---|---|
| **生成時**に backend_ext config で `perf_profile` を埋め込む | **ロード時**に `createFromBinary` が error 1002（C-3） |
| **実行時**に `htp_performance_mode` / 電力votes を設定する | **Run 時**に失敗する（本節） |

→ **どちらの経路でも電力プロファイルは制御できない。** 基本設計① 2.2節の「測定値は常にデフォルト電力プロファイル」という前提は、参考実装のコードによって裏付けられた。

---

## 6. P1 での実測値（2026-09-12、SOG05）

| 項目 | 実測 |
|---|---|
| 起動ログ | **`QNN HTP • EPContext • all nodes`**（全ノードHTP） |
| FastRPC | `libQnnHtpV68Skel.so?qnn_2_48_0_skel_handle_invoke&_dom=cdsp on domain 3` を **open 成功** |
| 初回 OrtRun（コールド） | **38.4 ms** |
| 連続推論時の OrtRun | **36.0 / 36.8 / 38.8 / 39.0 / 42.8 ms**（中央値 ≒ **39 ms**。参考値 40〜60ms と同等かやや良い） |
| 前処理 | 0.2〜0.5 ms（**推論の 1% 未満**。ボトルネックは完全に NPU 側） |
| total（前処理＋推論＋後処理） | 38.0〜44.7 ms |
| 深度値の範囲 | `[0.0096 .. 0.1000]` とフレームごとに変化 → 実際のカメラ入力を処理している |
| 推論頻度 | ログはフレームID 20 ごとに出力され、間隔 約0.88秒 → **約 22〜23 回/秒** |
| NPU 温度（`nspss-0-usr`） | アイドル 32.7℃ → **連続推論時 55〜59℃**（**+25℃**） |
| CPU（`cpuss-1-usr`）/ GPU | 64〜68℃ / 54〜57℃ |
| **スロットリング** | **`thermal-cpufreq-7` が開始8秒後に 0→1**（大コアの周波数制限が発動）。一方 **`cdsp` は 0 のまま**（NPU自体は未スロットリング） |
| `DmaBufTotal` | 568,588〜770,804 kB の範囲で大きく変動（カメラバッファの影響。基本設計① 4.7節で指摘したノイズが実際に現れた） |

**推論頻度 22〜23回/秒 × 39ms ≒ 0.9 秒/秒** なので NPU はほぼ連続稼働しており、+25℃ の温度上昇と整合する。

**早期の知見**: NPU は 59℃ でもスロットリングしていないのに、**CPU（大コア）が 8 秒で制限に入った**。エッジ推論の熱問題が必ずしも NPU 側から来るわけではないことを示す実例で、P3 の設計に反映する価値がある。

---

## 7. 我々が流用するもの / しないもの

| 対象 | 扱い |
|---|---|
| `tools/variants/` の4スクリプト | **基本設計②の仕様の出発点**（要件定義書3.4節 C-1〜C-6 の出典） |
| `open_qnn_session()` のセッション設定 | **基本設計③（端末側実行形態）の参考実装**。特に `disable_cpu_ep_fallback` と `offload_graph_io_quantization` |
| 2プロセス構成（カメラ/NPU分離） | 参考。我々はカメラを使わないため、そのままは不要 |
| `wrap_epctx.py` の EPContext スキーマ | **既知正解**として基本設計② S8 に記録済み |
| カメラ校正（`CameraCalibrationController.kt` / `calibration_target/`） | **不要**。深度の絶対値を使う用途のための機能 |
| ZipDepth モデル自体 | P1 限り。P2 以降は我々のモデル（BERT系 → Qwen2.5-0.5B） |

---

## 8. 注意点

| # | 注意 |
|---|---|
| 1 | **Git LFS が必須。** 同梱バイナリは LFS 管理で、LFS 無しでクローンすると130バイト程度のポインタになる。**ビルドは成功し、端末実行時にネイティブクラッシュする**ため原因が分かりにくい。`scripts/p1_verify_zipdepth.sh` で SHA-256 照合すること |
| 2 | **バージョンはセットで揃える。** `DEPENDENCIES.md` に「replacing one QNN library independently can cause initialization errors, incorrect graph placement, or a native crash」と警告がある。AAR 同梱の QNN 2.42 をあえて排除し、jniLibs の QAIRT 2.48 セットで統一している |
| 3 | **NDK は 21.4.7075529 から上げない。** `-DANDROID_STL=none` で r21 の C++ ランタイムを注入しない設計で、新しいランタイムは OpenCV の AAR が供給している |
| 4 | **起動時の `adsprpc ... Bad file descriptor` は無害。** Qualcomm の adsprpc 層がファイルを探索して失敗する過程で、直後に `cdsprpcd` が `/vendor/dsp/cdsp/` から開き直して成功している。失敗と誤読しないこと |
| 5 | A8 variant は**速度測定専用**で出力品質が落ちると README が警告している |
