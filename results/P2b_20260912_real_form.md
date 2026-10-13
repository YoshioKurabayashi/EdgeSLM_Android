# P2b 続: 本物の export 形で通す（C-9 / C-10 の発見）

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 端末 | Sony Xperia 5 III SOG05（SM8350 / HTP v68） |
| きっかけ | 本物の Qwen2.5-0.5B を export して、**手書き13モデルの推測が4箇所外れていた**ことが判明 |
| 結果 | **22/22 `pass_strict`**。制約を2つ（C-9 / C-10）追加し、検証器の穴を1つ塞いだ |

---

## 1. 発端: 推測が外れていた

P2b の13モデルは「HuggingFace の export はこう出すだろう」という**私の推測**で書いていた。
本物（`attn_implementation="eager"` / opset 17 / 静的形状）を export して op を数えたら、
**4箇所が違っていた。**

| 部位 | 手書き（推測） | **本物の export** |
|---|---|---|
| GQA のヘッド複製 | `Concat` ×rep | **`Unsqueeze`→`Expand`→`Reshape`** |
| RoPE の rotate_half | `Split` | **`Slice` ×2** |
| RMSNorm の除算 | `Sqrt`→`Div` | **`Sqrt`→`Reciprocal`→`Mul`** |
| 因果マスク | `Add`（定数 -60.0） | `Add`（定数 **-3.4028235e+38**） |

本物にだけ出る op: **`Expand` / `Reciprocal` / `Slice` / `Unsqueeze`**（いずれも P2b 未検証）
手書きにだけある op: `Div` / `Split` / `LayerNormalization` / `RotaryEmbedding` / `SimplifiedLayerNormalization`

融合形（`LayerNormalization` / `RotaryEmbedding` / `SimplifiedLayerNormalization`）が
本物に**出てこない**ことも確認できた。P2b でこれらを「使えない」と判定したのは、
実害がなかったということでもある。

---

## 2. C-9: 因果マスクの定数は量子化できない

### 何が起きるか

因果マスクは attention のスコアに足される行列で、上三角が大きな負の値になっている。
HuggingFace はここに **`-3.4028235e+38`（float32 の最小値）** を使う。

量子化は「**最小値〜最大値を N 段階に割る**」処理なので:

| | 範囲 | 16bit での1段階 |
|---|---|---|
| 本来のスコアだけ | 約 -20 〜 +20 | 0.0006 |
| **-3.4e38 が混ざると** | -3.4e38 〜 +20 | **5e33** |

**全分解能がマスクの1点に食われ、実際のスコアが丸ごと潰れる。**

### 対処

`preprocess.clamp_extreme_constants()` で **-60.0** に置き換える。
softmax を通せば `exp(-60) ≒ 8.8e-27` で遮断として十分であり、
かつ量子化の分解能を食わない。小さすぎると遮断が甘くなり、
大きすぎる（-1e4 等）と分解能を失うため、**両方を見て決める値**である。

### 検証

本物の Qwen（2層）で置換前後の logits を比較した:

```
logits shape (1, 16, 151936)
  最大絶対差  0.000000e+00
  平均絶対差  0.000000e+00
  argmax 一致 True
```

**fp32 では完全にビット一致。** グラフの意味は変わっていない。

---

## 3. C-10: v68 は `Expand` を含む context binary をロードできない

### 症状

`block_real`（実 export 形の Transformer 1ブロック）は**ホストでは context 生成まで成功**
したが、端末でロードに失敗した。

```
LoadCachedQnnContextFromBuffer Failed to create context from binary. Error code: 1002
```

**ホストの `qnn-context-binary-generator` は素通しする。** 端末に持って行くまで分からない。

### 切り分け

「変数を1つに絞る」で単体モデルを作って実機で1つずつ試した。

| モデル | 中身 | 実機 |
|---|---|---|
| `unsqueeze_only` | Unsqueeze のみ | **OK** |
| **`expand_only`** | **Expand のみ** | **NG err 1002** |
| `gqa_concat` | Concat で複製 | **OK** |
| `gqa_repeat` | Unsqueeze + Expand | NG err 1002 |

→ **`Expand` 単独が原因。`Unsqueeze` は無罪。`Concat` は代替として使える。**

### 対処

`preprocess.rewrite_expand_to_concat()` で書き換える。
`Expand` が「サイズ1の軸を n 倍に広げる」形なら、**同じテンソルを n 個 `Concat` する**のと
結果は完全に同じである。GQA のヘッド複製がまさにこの形。

```
gqa_expand  [1,2,1,16,32] -> [1,2,2,16,32]   axis2 を 2回 Concat
```

ブロードキャストがそれ以外の形（次元数そのものが増える等）なら**書き換えず残す**。
黙って別の意味のグラフにするより、端末で落ちて気付くほうがよい。

### 検証

fp32 で書き換え前後を比較: **最大絶対差 0.000000e+00**（完全一致）。
書き換え後、`block_real` を含む全モデルが実機で `pass_strict` になった。

---

## 4. C-1 検証器の穴を塞いだ

`gqa_repeat` を作ったとき、C-1 検証器は **PASS を返したのに context 生成が落ちた**。

```
"QNN_DATATYPE_FLOAT_32_converted_k_QNN_DATATYPE_FLOAT_16" generated: could not create op
```

QDQ を見ると、**先頭の `Unsqueeze` がグラフ入力 `k`（float32）を量子化せずに受けて**いた。
量子化器は形状 op を「そのまま通す」扱いにするためである。

**検証器はグラフ I/O を丸ごと除外していた。** `--preserve_io_datatype` で I/O が float32
に保たれる前提で書いたのだが、それで正当なのは**境界そのもの**だけで、
「float の入力が `QuantizeLinear` を経ずに計算ノードへ入る」のは立派な違反だった。
P2 のモデルは先頭が Conv（必ず量子化される）だったため穴が見えていなかった。

修正後: **24件中、実際に落ちた1件だけを検出**（誤検知なし）。
`gqa_repeat` は S5/S6 を無駄にせず S2〜S4 の段で止まるようになった。

なお `gqa_repeat` 自体も直した。単体化のために入力を直接 `Unsqueeze` へ入れていたが、
**実際の Transformer では複製は projection の後ろに来る**。
単体化のために現実と違う形にすると、判定そのものが無意味になる。

---

## 5. 最終結果: 22/22

```
summary: {"pass_strict": 22, "pass_cpu_boundary": 0, "failed": 0, "total": 22}
```

| モデル | 中央値 (ms) | | モデル | 中央値 (ms) |
|---|---:|---|---|---:|
| **block_real**（実形） | **0.419** | | matmul_weight | 0.323 |
| block（推測形） | 0.424 | | matmul_act (mixed) | 0.351 |
| attention (mixed) | 0.432 | | rmsnorm | 0.369 |
| kvcache (mixed) | 0.414 | | rmsnorm_real | 0.352 |
| embed | 0.260 | | rope | 0.467 |
| silu | 0.346 | | rope_slice | 0.382 |
| softmax | 0.365 | | layernorm | 0.345 |
| unsqueeze_only | 0.338 | | expand_only | 0.343 |
| gqa_concat | 0.351 | | gqa_repeat | 0.349 |

（**この数字は性能ではない**。全モデルが 0.26〜0.47ms に収まっており、
ノード数に比例していない＝固定費を測っている。`P2b_20260912_op_matrix_device.md` 2章参照）

---

## 6. 教訓

### 「動く形」を自分で書くと、自分の思い込みを検証してしまう

P2b の13モデルは丁寧に作ったが、**4箇所が本物と違った**。しかもそのうち1つ（`Expand`）は
**v68 で動かない op** だった。手書きだけで進んでいたら、実モデルに移った段階で
「1ブロックは動いたのに実モデルが動かない」という形で遭遇し、
数千ノードから原因を探すことになっていた。

→ **推測で作った検証は、本物と突き合わせるまで検証ではない。**

### ホストで通ることは端末で動くことを意味しない（再確認）

`Expand` はホストの context 生成を素通りして、端末のロードで落ちた。
C-1〜C-8 と同じ構図である。**端末で試すまで終わっていない。**

### 症状が真因を示さない（3回目）

| 表示 | 真因 |
|---|---|
| `EPContext node is not compatible...` | `<uses-native-library>` の宣言漏れ（C-7） |
| `ORT Tensor data size does not match` | ラッパーの I/O 型が float32 決め打ち |
| **`Failed to create context from binary. Error code: 1002`** | **`Expand` が1個入っていた** |

err 1002 は「context binary が作れない」としか言わない。**単体モデルを1つずつ
試して切り分けるしかない。** そのための `op_check` が効いた（再ビルドなしで
4モデルを1回の実行で判定できた）。

---

## 7. 次

本物の Qwen2.5-0.5B を、このパイプライン（C-9 / C-10 の書き換え込み）で通す。

```
⬜ 1層だけ  … 本物の重み・本物の形状（hidden 896 / FFN 4864）で通るか
⬜ 精度      … 実テキストでキャリブレーションし fp32 と比較
⬜ 24層      … 容量の壁（PD / VTCM / SMMU）が出るか
```

**精度はまだ一度も測っていない。** ここまでは全部「動くか」の話である。
