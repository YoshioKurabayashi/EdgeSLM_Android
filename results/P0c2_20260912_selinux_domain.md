# 未決事項#14 の決着: HTP 推論の実行形態

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 端末 | Sony Xperia 5 III SOG05（Android 13 / SELinux Enforcing） |
| 目的 | R-9（`adb shell` から FastRPC が使えない）の回避策として、**案①「`run-as` 方式」が成立するか**を判定する |
| 結論 | **案①は不成立。案②（アプリプロセス内で推論を動かす）が必須** |

---

## 1. 測定結果

同一の `/dev/adsprpc-smd` に対する open の可否を、3つの実行主体で比較した。

| 実行主体 | SELinux ドメイン | uid | `/dev/adsprpc-smd` の open |
|---|---|---|---|
| `adb shell` | `u:r:shell:s0` | 2000 (shell) | **Permission denied** |
| **`run-as org.zipdepth.npudemo`** | **`u:r:runas_app:s0:c105,c257,c512,c768`** | 10361 (u0_a361) | **Permission denied** |
| **アプリ本体プロセス** | **`u:r:untrusted_app:s0:c105,c257,c512,c768`** | 10361 (u0_a361) | **成功（fd 90, 91 が open 済み）** |

アプリの NPU プロセスの fd を確認した実際の出力。

```
lr-x------ 1 u0_a361 u0_a361 64 2026-09-12 16:15 90 -> /dev/adsprpc-smd
lr-x------ 1 u0_a361 u0_a361 64 2026-09-12 16:15 91 -> /dev/adsprpc-smd
```

---

## 2. 原因

**`run-as` は uid はアプリのものになるが、SELinux ドメインは `untrusted_app` ではなく `runas_app` になる。**

`adsprpc_device`（DSP デバイスノードの SELinux タイプ）へのアクセス権は `untrusted_app` にのみ与えられており、`shell` にも `runas_app` にも無い。したがって uid を揃えても FastRPC は開けない。

**DAC（ファイル権限）ではないことの根拠**: `/dev/adsprpc-smd` の権限は `crw-rw-r-- system:system` で **other に read がある**。それでも `O_RDONLY` が EACCES になるため、拒否しているのは SELinux である。

---

## 3. 測定手法についての注意（自分が一度誤った点）

最初に `dd if=/dev/adsprpc-smd bs=1 count=0` の出力を `tail` で絞って判定したところ、**`shell` でも成功したように見えた**。これは誤りで、`dd` は

```
dd: /dev/adsprpc-smd: Permission denied      ← この行を tail で隠していた
0+0 records in
0+0 records out
0 bytes (0 B) copied, 0.000454 s, 0 B/s      ← 失敗時もこのサマリを出す
```

のように**失敗してもサマリ行を出す**（終了コードは 1）。`dd` のサマリだけを見て成否を判断してはならない。

確実な open テストは次の2つ。

```sh
# ① シェルのリダイレクト（open のみ。read しない）
if exec 3</dev/adsprpc-smd; then echo OPEN_OK; exec 3<&-; else echo OPEN_FAIL; fi

# ② C プログラムで open(2) を直接呼び errno を見る（scripts/p0c_fastrpc_probe.sh）
```

---

## 4. 3案の評価（確定）

| 案 | 内容 | 判定 |
|---|---|---|
| **①`run-as` 方式** | デバッグ可能APKの文脈でバイナリを実行し app のドメインを得る | **× 不成立**。`runas_app` ドメインには `adsprpc_device` 権限がない |
| **②アプリプロセス内で推論を動かす** | 推論コアをアプリプロセス（`untrusted_app`）の中で実行する | **○ 実証済み**。ZipDepth がまさにこれで、P1 で動作確認済み |
| ③CPU限定 | HTP 検証は ZipDepth に任せ、自前実装は CPU のみ | △ 学習目的（NPUの割り当て理解）を満たさないため採らない |

---

## 5. 案②の最小形（推奨）

「アプリ化」と言っても、UI を作り込む必要はない。**最小の推論サービス APK を用意し、adb から駆動する**形で、CLI に近い運用を保てる。

```
┌─ ホストPC ───────────────────────────────────┐
│  adb shell am start-foreground-service ...   │  ← コマンドで駆動
│  adb shell am broadcast -a ... --es prompt   │
│  adb logcat / adb pull results.jsonl         │  ← 結果はログとファイルで回収
└────────────┬─────────────────────────────────┘
             │
┌────────────▼─ 端末: 最小APK（untrusted_app）─┐
│  Service（Kotlin、UIなし）                    │
│    ↓ JNI                                     │
│  ★C++ 推論コア（我々の実装）                   │
│    ORT + QNN EP → FastRPC → HTP v68          │
└──────────────────────────────────────────────┘
```

**この形は当初の目標と矛盾しない。** 本プロジェクトの最終目標は「**C++ で作成した推論サービスを搭載**して軽量 SLM をローカル完結で動かす」ことであり、**JNI 経由で C++ 推論コアをアプリプロセス内で動かす構成は、まさにその目標形そのもの**である。ZipDepth も同じ構造（`:zipdepth_npu` という別プロセスのサービス + `zipdepth_jni.c`）を採っている。

変わるのは「**adb shell から直接起動するネイティブ CLI**」という**外形だけ**で、C++ 推論サービスという中身は変わらない。

### 必要になる追加作業

| 項目 | 規模 |
|---|---|
| 最小 APK（Service のみ、UI なし） | 小。ZipDepth の `ZipDepthService.kt`(189行) / `ZipDepthNative.kt`(114行) が雛形になる |
| JNI 境界の定義 | 小。ZipDepth の8関数構成を参考にする |
| `useLegacyPackaging = true`（C-6） | 設定1行 |
| **ビルドがユーザー実行になる** | **JVM 制約により Gradle は代行できない**（環境構築記録4.2節）。反復のたびにビルドを依頼することになる |

**最後の項目が運用上の主な影響である。** C++ の実装・量子化・変換・計測は私が回せるが、**APK のビルドだけは毎回お願いすることになる**。

---

## 6. 反映先

| 文書 | 変更 |
|---|---|
| 要件定義書 4.1 / 4.2節 | 成果物の形。**「Android アプリ化は対象外」を見直す必要がある**（HTP 推論に必須のため） |
| 要件定義書 未決事項#14 | 本文書で決着 |
| 要件定義書 R-9 | 対処方針を「案②」に確定 |
| 基本設計③（端末側実行形態） | **本文書を入力として新規作成する** |
