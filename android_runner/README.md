# EdgeSLM Runner（android_runner）

SOG05（SM8350 / HTP v68）上で推論を実行する**器**。UI は持たない。

基本設計③（`../DoC/基本設計③_実行用アプリ設計.md`）の実装。

## なぜアプリなのか

**HTP(NPU) は `untrusted_app` ドメインのプロセスからしか使えない**ため（要件定義書 R-9）。
`adb shell`（`u:r:shell:s0`）からも `run-as`（`u:r:runas_app:s0`）からも `/dev/adsprpc-smd` は開けない。

UI は作らない。adb から駆動する殻を1枚被せているだけで、**推論の中身は C++**（最終目標の「C++ 推論サービス」そのもの）。

## 構成

```
app/src/main/
├── java/org/edgeslm/runner/
│   ├── RunnerActivity.kt   UIなし入口。stopped 状態からの起動と、
│   │                       フォアグラウンド文脈からの FGS 起動のためだけに存在
│   ├── RunnerService.kt    Foreground Service。ジョブを読み JNI を呼び結果を書くだけ
│   ├── NativeBridge.kt     JNI 宣言（引数・戻り値はすべて JSON 文字列）
│   └── JsonInject.kt       端末固有パス（base_dir / native_lib_dir）の注入
├── cpp/
│   ├── edgeslm_jni.cpp     JNI シム。ロジックを置かない
│   ├── runner/session.*    ORT/QNN セッション。dlopen(libonnxruntime.so) 方式
│   ├── runner/job.*        task 分岐と計測プロトコル
│   ├── ort/*.h             ORT C API ヘッダ（AAR 1.27.0 と対応）
│   └── third_party/json.hpp nlohmann/json 3.11.3
├── jniLibs/arm64-v8a/      QNN ランタイム（QAIRT 2.48 セット・SHA-256 照合済み）
└── libs/                   onnxruntime-android-qnn-1.27.0.aar
```

## v0.2 のスコープ

| task | 状態 |
|---|---|
| `probe` | 実装済み。セッションを開いて配置情報を返す |
| `bench` | 実装済み。ウォームアップ + 反復 + 統計 |
| **`op_check`** | **v0.2 で実装**（P2b）。`models[]` を順に開き4値で判定する |
| `soak` / `generate` | **未実装**。呼ぶと「未実装」と明示して返す（黙って別動作をしない） |

### v0.2 で足したもの

| 変更 | 理由 |
|---|---|
| **`op_check` task** | 単体 op を次々に試す。**対象は `job.json` 側の `models[]` にあるので、op を足しても APK を作り直さなくてよい** |
| **多入力の `Session::run`** | 固定長KVキャッシュは `x` / `past_k` / `past_v` の3入力になる。v0.1 は単入力しか扱えず明示的に弾いていた |
| **`Session::reopen`** | 同じ Session でモデルを張り替える。**OrtEnv は作り直さない**——QNN バックエンドはプロセス全体の状態を持つため、モデルごとに Env を作って壊すと不安定になる |
| **整数入力の生成規則** | int64 を乱数バイトで埋めると天文学的な値になり、Gather の索引なら範囲外・`position_ids` なら無意味な位置になる。型ごとに作り分け、`input.int_max` で上限を渡す |
| 無視していた `OrtStatus*` の処理 | `-Wunused-result` を消すためだけでなく、「設定できていたのか」を後から確認できるようにするため。非致命的な失敗は `(non-fatal)` としてログに残す |

`op_check` の判定は「通る / 通らない」の2値にせず、**次の手が決まる粒度**まで割る。

| 判定 | 意味 | 次の手 |
|---|---|---|
| `pass_strict` | 全ノードが HTP | そのまま使える |
| `pass_cpu_boundary` | 開けたが CPU フォールバックあり | delegate しきれていない箇所を探す |
| `open_failed` | ロードで落ちた | C-1〜C-8 / op 非対応を疑う |
| `run_failed` | 開けたが実行で落ちた | 入力の形・型を疑う |

```bash
bash ../scripts/run_runner.sh push-ops                # P2b の単体 op モデルを一括で送る
bash ../scripts/run_runner.sh run jobs/p2b_op_check.json
```

## 設計上の約束

| 約束 | 理由 |
|---|---|
| **モデル・ジョブ仕様は APK の外**（`/sdcard/Android/data/org.edgeslm.runner/files/`） | 差し替えに再ビルドが要らない。**ビルドはユーザー実行**になるため回数を減らすことが最優先 |
| **JNI の引数・戻り値はすべて JSON 文字列** | パラメータ追加のたびにシグネチャが変わると再ビルドになる |
| **Kotlin にロジックを置かない** | 同上＋最終目標が C++ 推論サービスであるため |
| **端末の状態監視を持たない** | 温度・クロック・dma-buf はホスト側ハーネスが adb 経由で取る。計測項目を増やしてもアプリを作り直さずに済む |
| **電力プロファイルを設定しない** | unsigned PD では `setPowerConfig` が許可されず Run が失敗する（C-3）。全結果に `power_profile="default(unsigned_pd)"` を刻む |
| **グラフ配置は推測しない** | `disable_cpu_ep_fallback=1` で開けたかどうかでラベルを決める（strict成功→`all nodes`） |

## ビルド（ユーザー実行）

```powershell
$env:JAVA_HOME="D:\Android\jdk17"; cd D:\AI\source\n_chap5\EdgeSLM_Android\android_runner; .\gradlew.bat assembleDebug
```

## 実行（ホストから）

```bash
bash ../scripts/run_runner.sh install
bash ../scripts/run_runner.sh push-model ../third_party/Edge_ZipDepth-SM8350/app/src/main/assets/zipdepth.onnx
bash ../scripts/run_runner.sh run jobs/probe_zipdepth.json
bash ../scripts/run_runner.sh run jobs/bench_zipdepth.json
```

## v0.1 の合格条件（基本設計③ 8.3節）

1. `probe` が **`QNN HTP - EPContext - all nodes`** を返す
2. `disable_cpu_ep_fallback=1` を有効にしてもセッション生成が成功する
3. `bench` の `OrtRun` 中央値が **P1 の ≒39ms と同オーダー**
4. `nativeVersions()` が ORT の版を返す

初回テストは **ZipDepth と同じ材料**（同じモデル・同じ QNN ライブラリ・同じ ORT AAR）で行う。
**持ち込むのは自分のコードだけ**にして、失敗時の原因を runner 側に限定するため。
