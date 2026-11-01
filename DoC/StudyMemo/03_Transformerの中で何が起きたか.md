# 3. Transformer の中で何が起きたか — residual stream と RMSNorm

[← 目次へ](../StudyMemo.md) | [← 前: 2. 量子化の基礎](02_量子化の基礎.md)

> **ソースと動かし方**（2026-10-14 追記）: この章の各節を、どのファイルのどの関数がやっているか・どのコマンドで確かめられるか・どの結果ファイルに数字があるかは [13 章](13_ソースと動かし方.md) の **13.3.3**。実際のコードの抜き出しと 1 行ずつの解説は [14 章](14_コードで読む検証.md) の **14.2**。スマホも GPU も要らない演習は 13.2。 各節の終わりには、その節の処理をしている実際のソースの枠がある（2026-10-17〜。`sync_src` で自動で更新）。

---

## 3.1 Transformer の1層と residual stream

1層をかなり単純化すると:

```
入力 x
 │
 ├──────────────┐
 ↓              │
RMSNorm         │
 ↓              │
Attention       │
 ↓              │
   + ←──────────┘     ← 残差加算
 ↓
新しい x
 │
 ├──────────────┐
 ↓              │
RMSNorm         │
 ↓              │
FFN             │
 ↓              │
   + ←──────────┘
 ↓
次の層へ
```

横を流れている `x` を **residual stream** と呼ぶ。

各層はモデル全体の状態をゼロから作り直すのではなく、

```
元の x + Attention が計算した情報 + FFN が計算した情報
```

という形で**少しずつ情報を書き足す**。

```
Layer 0
  x                     = [0.2, -0.5, 1.1, ...]
  Attention の寄与       = [0.01, 0.3, -0.2, ...]
  新しい x               = [0.21, -0.2, 0.9, ...]
```

Qwen2.5-0.5B ではこれを **24層**繰り返す。

<!-- src: app/pipeline/export_qwen_decode.py | for i, layer in enumerate\(self.layers\): | return \(self.lm_head\(h\), \*presents\) | omit=x = layer.input_layernorm\(h\)..h = res \+ attn.o_proj\(ctx\) | note=QwenDecodeStep.forward。1 層の中の残差加算 -->
**ソース** `app/pipeline/export_qwen_decode.py` L145〜L213（QwenDecodeStep.forward。1 層の中の残差加算） — md5 9f3431d3・`sync_src` が自動で更新

```python
  145|         for i, layer in enumerate(self.layers):
  146|             # ② 層 i: 入力の past を取り出し、正規化 → q/k/v の射影（[1,1,ヘッド,hd] にしてから [1,ヘッド,1,hd] へ）
  147|             past_k, past_v = caches[2 * i], caches[2 * i + 1]
  148|             attn = layer.self_attn
  149|             res = h
  150|             x = layer.input_layernorm(h)
     |             …（中略: 53 行。L151〜L203）…
  204|             h = res + attn.o_proj(ctx)
  205| 
  206|             # ⑤ MLP（正規化 → gate/up/down）→ 残差を足す。HF の Qwen2DecoderLayer と同じ順
  207|             res = h
  208|             x = layer.post_attention_layernorm(h)
  209|             h = res + layer.mlp(x)
  210| 
  211|         # ⑥ 最後の RMSNorm → lm_head で語彙ぶんの logits [1,1,V]
  212|         h = self.norm(h)
  213|         return (self.lm_head(h), *presents)
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `res = h` | 層の入口の x を取っておく（横を流れる residual stream） |
| `x = layer.input_layernorm(h)` | Attention の前の RMSNorm。FFN の前は `post_attention_layernorm` |
| `h = res + attn.o_proj(ctx)` | Attention の寄与を足し戻す（1 回目の残差加算） |
| `h = res + layer.mlp(x)` | FFN の寄与を足し戻す（2 回目）。各層は書き足すだけで、作り直さない |

---

## 3.2 residual stream の実測値 — massive activation

この residual stream の `max|x|` を層ごとに測った（Qwen2.5-0.5B、実テキスト入力）。

| 層 | max \|x\| |
|---|---:|
| 0 | 6.2 |
| 1 | 7.4 |
| **2** | **749** |
| 3 | 1604 |
| 5〜20 | 約 1630 |
| 23 | 110 |

（2026-10-17 補足: この表は 2 つの記録を合わせたもの。0・1・5〜20・23 層は `results/P3_20260913_full24.md` 3 節（そこでは 5〜20 層は「約 1677」）、2・3 層は `results/P3_20260913_rmsnorm_fix.md` 1 節の分布の表。下の分布の表とは測ったデータが違うため、同じ層でも最大が違う（0 層 6.2 と 3.6、23 層 110 と 40.1）と推定。下の表の p99 の列は結果ファイルに見つからず照らせなかった）

**層2 で急に2桁跳ね上がる。**

この「**少数の次元だけ桁違いに大きい値**」を **massive activation** と呼ぶ。
LLM で広く観測される現象で、異常ではない（モデルが何かを符号化するのに使っている）。

### 分布で見ると構造が分かる

| 層 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| 0 | 0.067 | 0.224 | 0.60 | 3.6 |
| 1 | 0.106 | 0.337 | 0.81 | 4.8 |
| **2** | 0.151 | 0.567 | 4.52 | **749** |
| 3 | 0.187 | 0.627 | 4.27 | 1604 |
| 20 | 0.759 | 2.173 | 5.50 | 1634 |
| 23 | 1.024 | 2.791 | 6.00 | 40.1 |

**典型的な値（p50）は 0.07〜1.0 程度なのに、max は 1600。** 約1600倍の開きがある。
`p99` が 4〜6 なので、**桁違いなのは本当に上位 0.1% だけ**である。

<!-- src: app/pipeline/split_layers_n.py | if tag == "eval": | 最大 \{np.abs\(x\).max\(\) | note=今のソースで層ごとの入力の大きさを出す所（main の中） -->
**ソース** `app/pipeline/split_layers_n.py` L147〜L150（今のソースで層ごとの入力の大きさを出す所（main の中）） — md5 30989e1e・`sync_src` が自動で更新

```python
  147|             if tag == "eval":
  148|                 x = store[ins[L]]
  149|                 print(f"  層{L:2d}  入力 {x.shape}  |値| p50 {np.percentile(np.abs(x),50):8.4f}"
  150|                       f"  p99.9 {np.percentile(np.abs(x),99.9):10.2f}  最大 {np.abs(x).max():10.2f}")
```
<!-- /src -->

上の 2 つの表を作ったスクリプトは残っていない（記録だけ）。今のソースで同じ種類の数字を出すのは `split_layers_n.py`（24 層 decode の層ごとの本物の入力について p50・p99.9・最大を出す。2026-09-14 に足した道具）。

| 目印 | 何をしているか・なぜ |
|---|---|
| `np.percentile(np.abs(x),50)` | 典型的な値（p50）。上の表の p50 の列にあたる |
| `np.abs(x).max()` | 最大値。massive activation はこちらにだけ現れる（p50 と比べて初めて「上位の一部だけが桁違い」と分かる） |

---

## 3.3 RMSNorm は何をしているのか

**数字の全体的な大きさを揃える**処理である。

$$
y = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}} \times w
$$

`w` を一旦無視して、$x=[3,4]$ で手を動かす:

```
x²        = [9, 16]
mean      = (9+16)/2 = 12.5
sqrt      = √12.5 ≈ 3.54
x / 3.54  ≈ [0.85, 1.13]
```

つまり「大きい x → 適度な大きさへ」正規化してから Attention / FFN に渡す。
これがないと、24層積むうちに値が発散したり消えたりする。

### なぜ `x²` を計算するのか

RMS = **R**oot **M**ean **S**quare（二乗平均平方根）だから。

```
x → square → x² → mean → 平均 → sqrt → 平方根
```

**この `x²` が今回の主役である。**

<!-- src: app/study/ex02_rmsnorm_rewrite.py | ^def rms_orig | return x / np.sqrt | note=演習 2 の RMSNorm（本物は transformers の中の Qwen2 の RMSNorm） -->
**ソース** `app/study/ex02_rmsnorm_rewrite.py` L32〜L36（演習 2 の RMSNorm（本物は transformers の中の Qwen2 の RMSNorm）） — md5 5861412e・`sync_src` が自動で更新

```python
   32| def rms_orig(x, eps=1e-6, quant=None):
   33|     x2 = x ** 2
   34|     if quant:
   35|         x2 = q_dq(x2, *quant)
   36|     return x / np.sqrt(x2.mean(-1, keepdims=True) + eps)
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `x2 = x ** 2` | 二乗。この `x²` が量子化されるのが問題の場所（3.4） |
| `x / np.sqrt(x2.mean(-1, keepdims=True) + eps)` | 二乗平均（最後の軸＝チャンネル方向）の平方根で割る。`w`（gamma）は省いてある |

---

## 3.4 massive activation があると `x²` で何が起きるか

residual が例えばこう（実測に近い値）:

```
x = [1677, 20, 1, -0.5, 0.2, ...]
```

RMSNorm は二乗する:

| x | x² |
|---:|---:|
| **1677** | **2,812,329** |
| 20 | 400 |
| 1 | 1 |
| 0.5 | 0.25 |
| 0.2 | 0.04 |

**元の差が 1677 : 0.2 = 8,385倍だったのが、二乗で 7,000万倍になる。**

**二乗は差を広げる演算**なので、massive activation があると破壊的に効く。

<!-- src: app/pipeline/preprocess.py | 理由は `x²` のレンジである。 | 正規化の分母そのものが狂う。 | note=normalize_rmsnorm_input の説明 -->
**ソース** `app/pipeline/preprocess.py` L542〜L548（normalize_rmsnorm_input の説明） — md5 11bc914b・`sync_src` が自動で更新

```python
  542|     理由は `x²` のレンジである。残差には少数の巨大チャンネル（±1677）があるため
  543| 
  544|         x²  のレンジ = [0, 約280万]  →  uint16 で 1段階 **42.7**
  545| 
  546|     となり、**通常チャンネルの x²（0.04 程度）が丸ごと 0 に潰れる**。
  547|     トークンによって巨大値の有無が違うので、巨大値を持たないトークンでは
  548|     `mean(x²)` が壊れ、正規化の分母そのものが狂う。
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `巨大チャンネル（±1677）` | residual の巨大値（3.2） |
| `x²  のレンジ = [0, 約280万]` | 1677² ≈ 281 万。二乗で差が広がる |

---

## 3.5 per-tensor 量子化と合わさって壊れる

`x²` のレンジが $[0,\ 2{,}800{,}000]$ なので、uint16 の1段階は:

$$
s = \frac{2{,}800{,}000}{65{,}535} \approx 42.7
$$

表現できる値は `0, 42.7, 85.4, ...` だけ。

```
x² = 1    を表したい  → 0 か 42.7 しか選べない → 0 に潰れる
x² = 10   を表したい  → 同じく 0
x² = 400  を表したい  → 0 か 42.7 か ... 約9段階
```

**通常チャンネルの `x²` が丸ごと 0 に潰れる。**

### なぜそれで壊れるのか

`mean(x²)` は巨大チャンネルに支配されるので、**巨大値を持つトークンでは
平均はそれなりに出る**。問題は**巨大値を持たないトークン**で、
そちらは `x²` が全部 0 に潰れ、**`mean(x²) ≈ 0` になって正規化の分母が壊れる**。

massive activation は**特定のトークン位置**（attention sink など）に集中するので、
「一部のトークンだけ正規化が壊れる」という形で効く。

<!-- src: app/study/ex02_rmsnorm_rewrite.py | == 2\. x² を uint16 の per-tensor で量子化したとき | 出力が eps で割った巨大な値になる | note=演習 2 の 2 -->
**ソース** `app/study/ex02_rmsnorm_rewrite.py` L64〜L75（演習 2 の 2） — md5 5861412e・`sync_src` が自動で更新

```python
   64|     print("\n== 2. x² を uint16 の per-tensor で量子化したとき（較正の範囲は 4 トークン全体の最小〜最大）")
   65|     ref = rms_orig(x, eps=1e-6)
   66|     x2_range = (0.0, float((x ** 2).max()))
   67|     xs2_range = (0.0, 1.0)                                  # max で割ると必ず [0, 1]
   68|     o = rms_orig(x, eps=1e-6, quant=x2_range)
   69|     n = rms_maxnorm(x, eps=1e-12, quant=xs2_range)
   70|     print(f"  x² の範囲 {x2_range[1]:,.0f} → 1 段 {x2_range[1] / 65535:.1f}      xs² の範囲 1 → 1 段 {1 / 65535:.1e}")
   71|     print("  トークン |  元の式（量子化）の誤差 | max で割った式（量子化）の誤差")
   72|     for t in range(4):
   73|         # 書き換え後の出力は xs/rms(xs) = x/rms(x)（スケール不変）なので、そのまま元の参照と比べられる
   74|         print(f"     {t}     |  {rel(ref[t], o[t]):>18.1%}    | {rel(ref[t], n[t]):>18.4%}")
   75|     print("  → 巨大チャンネルの無いトークン（1〜3）は、元の式では x² が全部 0 に丸まり、出力が eps で割った巨大な値になる")
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `x2_range = (0.0, float((x ** 2).max()))` | 較正の範囲は 4 トークン全体の最大＝巨大チャンネルの二乗で決まる（per-tensor） |
| `o = rms_orig(x, eps=1e-6, quant=x2_range)` | その範囲で x² を量子化して RMSNorm を計算する |
| `巨大チャンネルの無いトークン（1〜3）` | x² が全部 0 に丸まり mean(x²) ≈ 0 → 分母が eps だけになって壊れる（本文の「一部のトークンだけ正規化が壊れる」） |

---

## 3.6 「p50 が何段階に載るか」で見る

ここで一度、**私（Claude）が間違えた説明**を置いておく。同じ罠を踏まないため。

### 間違った説明

> 「各層が residual に加える寄与は ±10〜20 なので、1段階 0.051 に埋もれる」

**成立しない。** $10 \div 0.051 \approx 196$ 段階ある。埋もれない。

### 正しい見方

見るべきは最大値ではなく**分布**だった。residual stream について:

| 層 | p50 | max | 1段階 | **p50 が何段階に載るか** |
|---|---:|---:|---:|---:|
| 0 | 0.067 | 3.6 | 0.00009 | **755** |
| 1 | 0.106 | 4.8 | 0.00014 | **752** |
| **2** | 0.151 | **749** | 0.0121 | **12.5** |
| 3 | 0.187 | 1604 | 0.0259 | **7.2** |
| 20 | 0.759 | 1634 | 0.0264 | 28.7 |
| 23 | 1.024 | 40.1 | 0.0011 | 906 |

**層2以降、典型的なチャンネル（p50）は 7〜29段階しか使えていない。**
16bit テンソルなのに、**実質3〜5ビット**である。

> **max だけ見ずに p50 / p90 / p99 を見る。**

<!-- src: なし | 考え方の説明（p50 ÷ 1段階 の割り算）。数字は results/P3_20260913_rmsnorm_fix.md 1 節。分布を測る今のソースは 3.2 節の枠 -->
> **ソース**: なし — 考え方の説明（p50 ÷ 1段階 の割り算）。数字は results/P3_20260913_rmsnorm_fix.md 1 節。分布を測る今のソースは 3.2 節の枠
<!-- /src -->

---

## 3.7 なぜ24層で初めて壊れたのか

| 層数 | residual の max | `x²` の max | 1段階 | 結果（top-1） |
|---|---:|---:|---:|---|
| 2 | 7.4 | 約 55 | 0.0008 | **正常** 80.7% |
| 6 | 1631 | 約 266万 | 40.6 | **壊れる** 6.2% |
| 24 | 約 1677 | 約 280万 | 42.7 | **完全崩壊** 4.5% |

2層が正常だったのは、**層0〜1 の residual が ±7.4 しかない**から。
層2 で 749 に跳ねるので、**3層以上なら壊れる**（6層で確認）。

（★2026-10-17 訂正: 24 層の行は旧「1634 / 約 267万 / 42.7」。1634² ≈ 267 万なら 1 段階は 40.7 で、42.7 と合わない。42.7 は量子化のログの `pow_13 range=2,801,222`（≈ 1674²）から来る値で、`results/P3_20260913_full24.md` 3 節でも 24 層の残差の最大は約 1677。1634 は分布の表（3.2）の 20 層の値。2・6 層の行の数字は結果ファイルで照らせなかった）

### 「24層だから誤差が累積した」のではない

最初はこう考えた:

```
1層で誤差 → 2層 → 3層 → ... → 24層 → 誤差が累積
```

しかし RMSNorm を直したら:

```
2層  top-1 80.7%
24層 top-1 80.7%        ← 層数を増やしても落ちていない
```

**層数そのものは主要因ではなかった。** 正しい因果は:

```
深い層で massive activation が出る
 ↓
square で巨大化（2.8M）
 ↓
per-tensor 量子化と相性が悪い
 ↓
RMSNorm 内部だけ破綻する
```

> **「深いから壊れる」ではなく「深い層で特定の条件が揃うと壊れる」。**
> 因果の違いは対策の当たり所を決めるので重要。
> 「累積」だと思うと「層ごとに誤差を減らす」方向に行くが、実際は
> 「特定の op の Dynamic Range を直す」だけで済んだ。

<!-- src: app/pipeline/oracle_exclude.py | ^PATTERNS = \{ | ^\} | note=Oracle 実験で量子化から外す op の組 -->
**ソース** `app/pipeline/oracle_exclude.py` L46〜L52（Oracle 実験で量子化から外す op の組） — md5 32423156・`sync_src` が自動で更新

```python
   46| PATTERNS = {
   47|     "none": [],
   48|     "residual": ["Add"],
   49|     "rmsnorm": ["Pow", "ReduceMean", "Sqrt", "Reciprocal"],
   50|     "residual+rmsnorm": ["Add", "Pow", "ReduceMean", "Sqrt", "Reciprocal"],
   51|     "matmul_act": [],       # 特別扱い（下記）
   52| }
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `"residual": ["Add"],` | 残差加算（Add）だけ量子化から外す → 6 層で 6.2% のまま |
| `"rmsnorm": ["Pow", "ReduceMean", "Sqrt", "Reciprocal"],` | RMSNorm の中だけ外す → 74.4%。「深いから壊れる（累積）」ではなく「RMSNorm の中が壊れる」と分かった実験（4 章） |

---

## 3.8 Attention 側にも同じ構造がある

RMSNorm ほど派手ではないが、Attention のスコアにも Dynamic Range の問題がある。

**QK^T のスコア**（1/√d でスケール後）を層ごとに測ると:

| 層 | scaled スコアの範囲 | 幅 |
|---|---|---:|
| **0** | [-22.9, **+1564**] | **1587** |
| 1 | [-26.3, +232] | 258 |
| 2 | [-29.8, +74] | 104 |
| 23 | [-6.4, +24] | 30 |

**layer 0 だけ桁違いに広い。** さらにヘッドごとに見ると:

```
head 1: [268.7, 1600.4]  幅 1332   ← これだけ異常
head 6: [443.7,  521.8]  幅   78
head 9: [ 17.7,   71.5]  幅   54
（他の12ヘッドは幅 54〜272）
```

行（トークン）ごとの幅は**中央値 7.37**。つまり
**本当に必要な分解能は 7 程度なのに、1587 のレンジに合わせた目盛りを使わされている。**

（2026-10-17 補足: 行ごとの幅 7.37 を出した元の記録 `results/P3_20260912_accuracy_layer1.md` 3.3 節は、較正データでのスコアの範囲 [-26.7, +1600.4]・幅 1627 と比べている。上の表の layer 0（[-22.9, +1564]・幅 1587）は別の文での実測（`results/P3_20260913_mixed_matmul.md` 4 節で、自作 ONNX と HF の値が一致）。どちらも layer 0 で、言っていることは同じ。layer 1・2・23 の行は結果ファイルで照らせなかった）

これが [4章](04_原因の特定と指標の読み方.md) の softmax 増幅に繋がる。

<!-- src: app/pipeline/export_qwen_decode.py | scores = torch.matmul\(q, kk.transpose\(2, 3\)\) \* self.scale | +3 | note=QwenDecodeStep.forward の中の QK^T -->
**ソース** `app/pipeline/export_qwen_decode.py` L199〜L201（QwenDecodeStep.forward の中の QK^T） — md5 9f3431d3・`sync_src` が自動で更新

```python
  199|             scores = torch.matmul(q, kk.transpose(2, 3)) * self.scale   # [1,nh,1,L]
  200|             scores = scores + attn_mask
  201|             probs = torch.softmax(scores, dim=-1)
```
<!-- /src -->

<!-- src: app/pipeline/preprocess.py | C-8 の回避で QK\^T の出力は 8bit に | mul_8  （スケール後）  実測 | note=fold_scale_into_matmul の説明（layer 0 の実測） -->
**ソース** `app/pipeline/preprocess.py` L229〜L233（fold_scale_into_matmul の説明（layer 0 の実測）） — md5 11bc914b・`sync_src` が自動で更新

```python
  229|     を掛ける形で export される。ところが C-8 の回避で QK^T の出力は 8bit に
  230|     落とさざるを得ず、**スケール前の広いレンジのまま量子化される**:
  231| 
  232|         matmul_1（QK^T 生値）  実測 [-183, +12515]  → 8bit で 1段階 50.9
  233|         mul_8  （スケール後）  実測 [ -23,  +1564]
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `* self.scale` | 1/√d のスケール（head_dim 64 なら 1/8）。表の「scaled スコア」はこれを掛けた後 |
| `scores = scores + attn_mask` | マスク（0 か −60）を足してから softmax に入る。スコアの分解能がそのまま softmax の入力の分解能になる（4.6） |
| `matmul_1（QK^T 生値）  実測 [-183, +12515]` | スケール前の layer 0 の実測 |
| `mul_8  （スケール後）  実測 [ -23,  +1564]` | スケール後。上の表の layer 0 [-22.9, +1564] と同じ値 |

---

[← 目次へ](../StudyMemo.md) | [次: 4. 原因の特定と指標の読み方 →](04_原因の特定と指標の読み方.md)
