# P2: 量子化・変換パイプラインの疎通（自作モデルで NPU 実行）

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 端末 | Sony Xperia 5 III SOG05（SM8350 / HTP v68） |
| 対象 | 自作の小型 CNN（`testnet`、Conv×2 + BN×2 + Relu + Pool + MatMul） |
| 量子化 | **A16W8**（QUInt16 活性化 / QInt8 per-channel 重み / MinMax） |
| baseline | QAIRT 2.48.0.260626 + ORT 1.30.0（ホスト量子化） / ORT 1.27.0（端末実行） |
| 判定 | **疎通成功。C-1〜C-5 すべて PASS** |

---

## 1. 何を確認したか

**ホスト側パイプライン（基本設計②）で作ったモデルが、我々の runner（基本設計③）で
NPU 実行できること。** つまり S1〜S8 の全工程が実際に機能することを端から端まで確認した。

```
S1 自作 fp32 ONNX（11ノード、BatchNormalization×2）
 → S2 BN畳み込み（13ノード）→ quant_pre_process（8ノード）   [C-5 OK]
 → S3 キャリブレーション（ランダム32 + 合成8 = 40サンプル）
 → S4 get_qnn_qdq_config → quantize（26ノード: DQ×13 / Q×7）  [C-1 OK]
 → S5 qairt-converter --preserve_io_datatype → DLC          [C-4 OK]
 → S6 qnn-context-binary-generator --htp_socs sm8350        [C-2/C-3 OK]
 → S7 qnn-context-binary-utility → meta.json
 → S8 EPContext ラッパー（53.4 KiB）
 → adb push → runner で実行
```

## 2. 端末での実行結果

```
backend : QNN HTP - EPContext - all nodes
strict  : True                      （disable_cpu_ep_fallback=1 で成功）
io      : input [1,3,32,32] float32 -> output [1,4] float32
warmup  : 1.083, 0.464, 0.283 ms
samples : 0.279 0.277 0.272 0.275 0.298 0.277 0.274 0.278 0.276 0.278 ms
stats   : median=0.277  p90=0.281  min=0.272  max=0.298  stddev=0.007  n=10
power   : default(unsigned_pd)
```

**ウォームアップの初回が3回目の 3.9 倍遅い**（1.083ms → 0.283ms）。
ZipDepth 級のモデルでは 38.0 → 37.6ms と差が小さかったが、小さいモデルほど
固定費（グラフ finalize 等）の比率が大きくなるため差が顕著に出る。
基本設計① 2.1節「単発実行から結論を出さない」の妥当性を改めて裏づける。

## 3. C-1〜C-5 の機械的検証

```
OK  C-1 float セグメント残存ゼロ : 計算ノード間を直接流れる float テンソルはない
OK  C-5 BatchNormalization 残存ゼロ
OK  C-2 socModel==0              : socModel=0
OK  C-3 backend_extensions 未使用 : config 指定なし
OK  C-4 --preserve_io_datatype    : C-1 pass 済みモデルに付与
総合: PASS
```

生成された context binary のメタデータ（`qnn-context-binary-utility` 出力）:

```
info.socModel                      = 0     ← C-2。soc_id を刻んでいない
info.contextMetadata.info.dspArch  = 68    ← v68 を対象にコンパイルされている
info.platformInfo...deviceId       = 0
```

**ZipDepth の README が「出荷済み動作版は socModel=0 と dsp_arch v68 のみ」と
書いていた形と一致した。**

### 検証器そのものの妥当性確認

C-1 検出器が「実際に違反を見つけられるか」を、意図的に不正なモデルで確認した。

| 入力 | 結果 |
|---|---|
| 完全QDQモデル（正常系） | PASS |
| 前処理後の fp32 モデル | **NG: 7箇所**を検出。`b1: Conv(conv1) → Relu(relu1)` のように**テンソル名・producer・consumer を列挙** |
| 元の fp32 モデル | **NG: C-1 で10箇所、C-5 で BatchNorm 2個**を検出 |

「件数だけでは直せない」ため、違反箇所を具体的に出す設計にしている。

---

## 4. 途中で判明した誤り（文書を訂正）

### 誤り: 「QAIRT 環境に onnx / onnxruntime は不要」

環境構築時に `check-python-dependency` の `required_packages` を読み、
「QAIRT の依存リストに onnx は含まれない → 量子化は本体環境、変換は QAIRT 環境」
と結論していた。**これは不正確だった。**

`qairt-converter` は ONNX フロントエンドで `import onnx` するため、**QAIRT 専用環境にも
`onnx` が必要**である。

```
ERROR - Encountered Error: Error loading onnx, Message: No module named 'onnx'
  File ".../qti/aisw/converters/onnx/onnx_to_ir.py", line 26, in <module>
    import onnx
```

→ `onnx==1.17.0` を QAIRT 環境へ追加（**`numpy==1.26.4` を固定したまま**。
QAIRT の要求を壊さないため）。`protobuf` も同時に入る。

**「依存リストに書かれていない＝不要」ではない。** required_packages は
check スクリプトが検査する対象の一覧であって、全依存の網羅ではなかった。

### 他に踏んだもの

| 事象 | 対処 |
|---|---|
| BN 畳み込みで initializer 名が衝突（`bn1_scale` が既存） | 一意名を生成するヘルパーを追加 |
| `quant_pre_process` が `sympy` を要求 | 本体環境に `sympy` を追加（形状が埋まらないと per-channel 量子化が成立しない） |
| 16bit 量子化で **opset が 17→21 に自動更新**される | 警告として記録。EPContext ラッパーは別に構築するため影響なし |

---

## 5. 観察

### quant_pre_process は op 種別まで変える

```
元         11ノード  Conv×2 BatchNormalization×2 Relu×2 MaxPool GAP Reshape MatMul Add
BN畳み込み後 13ノード  Conv×2 Mul×2 Add×3 Relu×2 MaxPool GAP Reshape MatMul
前処理後     8ノード  Conv×2 Relu×2 MaxPool GAP Reshape Gemm
```

Mul/Add が Conv の重みに吸収され、MatMul+Add が **Gemm に融合**された。
**`init_overrides` のテンソル名を前処理「後」のモデルから取らなければならない**
（基本設計② S2 補足3）ことが、実物で確認できた。元モデルから拾えば
名前が一致せず、**上書きが黙って無効になる**。

---

## 6. 意味と次

未知が順に消えている。

```
✅ P1    端末の疑いが消えた      （ZipDepth が SOG05 で動作）
✅ v0.1  runner の疑いが消えた   （自作コードで同じ結果を再現）
✅ P2    パイプラインの疑いが消えた（自作モデルが NPU で動作）
```

**3つの層すべてで「自分の側は正しい」ことを確認できた。** 以降 Transformer で失敗した
場合、原因は「Transformer 固有の事情」（対応 op・精度指定・静的形状・KVキャッシュ）に
限定して調べられる。

次は **P2b（Transformer 構成要素の重点検証）**。基本設計② 6.1節の順序で
MatMul → RMSNorm → LayerNorm → RoPE → Softmax → Attention → KVキャッシュ を
個別に通す。runner 側に `op_check` task（v0.2）を追加する必要がある。
