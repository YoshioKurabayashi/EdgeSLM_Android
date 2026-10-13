# EdgeSLM Runner v0.1 受け入れ記録

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 端末 | Sony Xperia 5 III SOG05（SM8350 / HTP v68 / Android 13） |
| 対象 | `android_runner` v0.1（`probe` / `bench`） |
| モデル | ZipDepth `zipdepth.onnx`（A16W8 / 384×384 / EPContext）＝**既知良好な第三者製** |
| ランタイム | 同梱 QAIRT 2.48 セット + ORT 1.27.0 AAR（**ZipDepth と MD5 一致を確認済み**） |
| 判定 | **合格（4条件すべて達成）** |

---

## 1. 合格条件の判定

| # | 条件 | 結果 |
|---|---|---|
| ① | `probe` が `QNN HTP - EPContext - all nodes` 相当を返す | **達成**。`"backend":"QNN HTP - EPContext - all nodes"` |
| ② | `disable_cpu_ep_fallback=1` を有効にしてもセッション生成が成功する | **達成**。`"strict_no_cpu_fallback": true` |
| ③ | `bench` の `OrtRun` 中央値が P1 の ≒39ms と同オーダー | **達成**。**median 35.77 ms** |
| ④ | `nativeVersions()` が ORT の版を返す | **達成**。`onnxruntime 1.27.0 / ort_api_version 27` |

## 2. 計測値

```
warmup  : 38.0, 36.6, 37.6 ms   （破棄。ただし記録は残す）
samples : 37.1, 35.8, 38.2, 35.6, 35.7, 35.7, 35.8, 34.0, 36.2, 35.7 ms
stats   : median=35.77  p90=37.19  min=33.98  max=38.18  stddev=1.08  n=10
power_profile : default(unsigned_pd)
input   : 1,769,472 bytes（1×3×384×384 float32、seed固定のランダム）
```

**ウォームアップ3回のうち初回が最も遅い**（38.0ms）という既知の傾向が、我々の実装でも再現した。基本設計① 2.1節「単発実行から結論を出さない」の妥当性を裏づける。

P1 での ZipDepth 実測（中央値 ≒39ms）よりやや速いが、ZipDepth はカメラ駆動で同一プロセス内に他の処理があるのに対し、こちらは隔離したループである。**同オーダーであることが確認できれば十分**であり、両者の差を性能差として解釈しない。

## 3. 入出力の確認

```
inputs : image  [1,3,384,384] float32  static=true
outputs: depth  [1,1,384,384] float32  static=true
```

動的次元がないことをコード側でも確認している（QNN EP は動的形状を扱えないため、
`static=false` なら `bench` は明示的に失敗する設計）。

---

## 4. ★到達までに踏んだ障害（4件）

v0.1 は一度で通らず、4つの問題を順に潰した。**いずれもドキュメントからは辿り着けず、
参考実装（ZipDepth）との差分比較で特定した。**

### 障害1: `android.permission.WAKE_LOCK` の宣言漏れ

```
java.lang.SecurityException: Neither user 10362 nor current process has android.permission.WAKE_LOCK
    at RunnerService.onStartCommand(RunnerService.kt:56)
```

Service が起動直後にクラッシュ。マニフェストに追加し、あわせて **WakeLock 取得失敗でも
ジョブを続行する**ようにした（WakeLock は「あると望ましい」もので、これが原因で検証が
止まるのは割に合わない）。

### 障害2: サブディレクトリの所有者が `shell` になっていた

`adb shell mkdir` で `models` / `jobs` を作ったため所有者が `shell`（`drwxrws--- shell ext_data_rw`）
になり、アプリから入れず `job file not found` になった。

→ **サブディレクトリはアプリ自身に作らせる。** `files` 直下はアプリ所有かつ group が
`ext_data_rw` なので、アプリが作ったディレクトリには adb からも push できる。

### 障害3（誤診）: FastRPC のパスを「設定した」こと

当初「ZipDepth が `ADSP_LIBRARY_PATH` を設定しているから必要だ」と判断して実装したが、
ORT のログに次が出た。

```
W qnn_backend_manager.cc:308] Using existing ADSP_LIBRARY_PATH setting of <path>,
  which may cause the HTP backend to fail.
```

**ORT の QNN EP は自分で設定する仕様**で、既に設定されていると自分の設定をスキップする。

ただし**これは真因ではなかった**（設定してもしなくても失敗した。ZipDepth は同じ警告を
出したうえで成功している）。既定では ORT に任せ、比較用に `job.json` の
`runtime.set_fastrpc_paths` で切り替えられる形に落ち着かせた。

### 障害4（真因）: `<uses-native-library>` の宣言漏れ

**Android 12+（API 31以降）では、vendor のネイティブライブラリを使うアプリは
マニフェストで明示宣言しなければならない。**

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
<uses-native-library android:name="libadsprpc.so" android:required="false" />
```

宣言が無いとアプリのリンカ名前空間に含まれず、**FastRPC クライアント `libcdsprpc.so` に
到達できない**。

**症状が真因を示さない点が最大の罠である。**

```
E qnn_execution_provider.cc:1046 GetCapability] QNN SetupBackend failed
  Failed to create device. Error: QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values
→ E EPContext node 'ZipDepth_QNN_ctx' is not compatible with any execution provider
  added to the session. Available: [QNNExecutionProvider, CPUExecutionProvider]
```

「**config が不正**」「**EP が無い**」と言われるが、実態は「**DSP に繋がれない**」。

---

## 5. 切り分けの過程（排除した仮説）

真因に至るまでに、次を**実測で**排除した。順序としてはもっと早くマニフェストを
比較すべきだった。

| 仮説 | 検証方法 | 結果 |
|---|---|---|
| provider option が不正 | `backend_path` のみ / `offload_graph_io_quantization` のみ / `qnn_context_priority` のみ で実行 | ✗ すべて同じ失敗 |
| SoC 指定が必要 | `soc_model=30` / `htp_arch=68` / 両方 | ✗ すべて同じ失敗 |
| `ADSP_LIBRARY_PATH` | 自前設定あり / なし の両方 | ✗ どちらも失敗。**ZipDepth は同じ警告を出して成功** |
| ライブラリの差異 | 両アプリの installed `.so` を MD5 比較 | ✗ **5ファイルすべて一致** |
| `libc++_shared.so` の欠如 | QNN 各 `.so` の ELF NEEDED を確認 | ✗ `libc/libm/libdl/liblog`(+`libcdsprpc.so`) のみ。依存なし |
| Activity / グラフィクス初期化 | Service を直接起動（Activity 非経由） | ✗ 同じ失敗 |
| 端末の状態異常 | ZipDepth を再実行 | ✗ **正常動作（41.9ms）**。端末は健全 |
| **マニフェストの差異** | 両 APK の manifest を全項目 diff | **○ これが真因** |

### 誤った前提

P0-c で**シェルから**の `dlopen("libcdsprpc.so")` が成功していたため「ライブラリは使える」
と判断していた。しかし**シェルバイナリはアプリの名前空間制限の対象外**であり、
条件の異なる観測を根拠にしていた。

**同じ操作でも実行主体が違えば結論は変わる**（R-9 で SELinux ドメインについて学んだのと
同じ構図）。

---

## 6. 意味

**「我々の器は正しい」ことが確定した。** 以降、基本設計②のパイプラインで作った自作モデルを
同じ器に入れて失敗した場合、**原因はパイプライン側と断定できる**。

P1 で端末の疑いを消し、v0.1 で runner の疑いを消した。残る未知はモデル生成だけである。
