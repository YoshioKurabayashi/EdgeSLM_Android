# 5. Graph Rewrite — 答えは同じ、途中計算だけ NPU 向け

[← 目次へ](../StudyMemo.md) | [← 前: 4. 原因の特定と指標の読み方](04_原因の特定と指標の読み方.md)

> **ソースと動かし方**（2026-10-14 追記）: この章の各節を、どのファイルのどの関数がやっているか・どのコマンドで確かめられるか・どの結果ファイルに数字があるかは [13 章](13_ソースと動かし方.md) の **13.3.5**。実際のコードの抜き出しと 1 行ずつの解説は [14 章](14_コードで読む検証.md) の **14.2**。スマホも GPU も要らない演習は 13.2。 各節の終わりには、その節の処理をしている実際のソースの枠がある（2026-10-17〜。`sync_src` で自動で更新）。

---

## 5.0 考え方

モデルの数式をそのまま NPU に押し込む必要はない。

| | 変わるか |
|---|---|
| 数学的な答え | **変わらない** |
| NPU が実行する演算 | **変わる** |
| 量子化のしやすさ / 対応可否 | **改善する** |

> **数式としては同じ答えを出す別の計算方法へ変形し、
> ハードウェアにとって都合のよい形にする。**

書き換えるのは **ONNX（設計図）**である
（[1. レイヤ構造](01_レイヤ構造とツールチェーン.md)）。
ランタイムやハードを変えているのではない。

本プロジェクトで入れた書き換えは5つ。実装は `app/pipeline/preprocess.py` と
`force_mixed_matmul.py`。

| # | 書き換え | 何のため |
|---|---|---|
| **C-9** | マスク定数 `-3.4e38` → `-60` | 量子化できない値を消す |
| **C-10** | `Expand` → `Concat` の繰り返し | v68 が `Expand` をロードできない |
| **C-11** | RMSNorm を max で正規化 | `x²` の Dynamic Range を潰す |
| **C-8 対応** | `QK^T = (K Q^T)^T` で左右入替 | 8bit にする側を選ぶ |
| （診断用） | スケール移動 / softmax 安定化 | **効かなかった**。記録として残す |

（★2026-10-17 訂正: 旧「実装は `app/pipeline/preprocess.py` と `force_mixed_matmul.py`」。右辺を uint8 にする後処理は、今のソースの本番の流れでは `downgrade_actact_rhs.py`（`force_mixed_matmul.py` は当時の検証用。`app/pipeline/README.md`）。また表の最後の行のうち「スケール移動」は、診断用に外してあるのではなく、`preprocess.run` の既定 `fold_scale=True` で本番でも毎回かかっている（下の枠）。効かなかったのは確か（5.6）だが、厳密に等価な変形なので害も無く、外していない。softmax 安定化は既定で off）

<!-- src: app/pipeline/preprocess.py | ^def run\(src: Path, folded_path | rms = normalize_rmsnorm_input\(m0\) if norm_rmsnorm else \[\] | note=S2 前処理の本体。書き換えを呼ぶ順 -->
**ソース** `app/pipeline/preprocess.py` L743〜L783（S2 前処理の本体。書き換えを呼ぶ順） — md5 11bc914b・`sync_src` が自動で更新

```python
  743| def run(src: Path, folded_path: Path, prepped_path: Path,
  744|         mask_replace: float = MASK_REPLACE, fold_scale: bool = True,
  745|         stabilize: bool = False, swap_matmuls: set[str] | None = None,
  746|         norm_rmsnorm: bool = False) -> dict:
  747|     """S2 全体。結果の要約を返す。
  748| 
  749|     引数: src（export した ONNX）、folded_path（BN 畳み込み後の書き出し先）、prepped_path（前処理後の
  750|           書き出し先＝S4 の入力）、mask_replace（C-9 の置き換え先）、fold_scale（③）、stabilize（④）、
  751|           swap_matmuls（⑤ の対象の名前）、norm_rmsnorm（⑥）。
  752|     戻り値: 書き換えごとの記録・BatchNorm の数の推移・形状推論の方式・出力パスの辞書。
  753|     書き換えが 1 つでもあれば、書き換え後を <folded の名前>_clamped.onnx に保存し、以降はそれを元にする。
  754|     最後に BatchNorm が残っていれば C-5 違反として止める。
  755|     """
  756|     # ① 極端な定数の置き換えを**最初に**やる。BN 畳み込みや定数畳み込みが
  757|     #    これらの値を別の initializer へ伝播させる前に潰しておく必要がある。
  758|     m0 = onnx.load(str(src))
  759|     clamped = clamp_extreme_constants(m0, replace_with=mask_replace)
  760|     # ② Expand → Concat の書き換え（C-10）。量子化より前にやる:
  761|     #    量子化後だと Q/DQ を挟んだ形を壊さずに書き換えるのが難しくなる。
  762|     expanded = rewrite_expand_to_concat(m0)
  763|     # ③ スケールを MatMul の前へ移す。
  764|     #    ★2026-10-18 訂正: 旧「C-8 の回避で 8bit にせざるを得ない attention スコアの分解能を、
  765|     #    等価変形だけで 8倍にできる。」。実測では効果なし（30.3819% → 30.3904%、StudyMemo 5.6 節）。
  766|     #    量子化されるのはスケール後の値で、移してもそのレンジは変わらないため。等価で害も無いので既定で残している。
  767|     scaled = fold_scale_into_matmul(m0) if fold_scale else []
  768|     # ④ softmax の前で行ごとの最大値を引く。**既定では無効**。
  769|     #    数学的には等価で、numpy 上の単独試算では劇的に効く（softmax 誤差
  770|     #    62.8% → 3.2%）のだが、**実際のパイプラインでは逆に悪化した**
  771|     #    （層全体の相対L2 30.4% → 41.6%、2026-09-12 実測）。
  772|     #    理由: スコアは **MatMul の出力時点で既に 8bit 化されている**ため、
  773|     #    その後に行最大を引いても手遅れで、ReduceMax/Sub の量子化誤差だけが乗る。
  774|     #    **対策を打つ位置が間違っていた。** 記録のため実装は残すが既定は off。
  775|     softmaxes = stabilize_softmax(m0) if stabilize else []
  776|     # ⑤ QK^T の左右を入れ替える。C-8 で 8bit にせざるを得ないのは常に右辺なので、
  777|     #    レンジの狭い方（Q）を右辺へ持ってくる。既定は off（効果を測ってから判断）。
  778|     #    （2026-10-18 追記: 効果は測定済み（top-1 66.5% → 80.7%）。本番は p2b_ops が方式名に swap が
  779|     #    あるとき spec の swap_matmuls を渡して入れる）
  780|     swapped = swap_actact_matmul_operands(m0, only=swap_matmuls) if swap_matmuls else []
  781|     # ⑥ RMSNorm をトークンごとの max で正規化してから計算する（C-11）。
  782|     #    x² のレンジが 280万 → 1 になり、通常チャンネルが潰れなくなる。
  783|     rms = normalize_rmsnorm_input(m0) if norm_rmsnorm else []
```
<!-- /src -->

<!-- src: app/pipeline/p2b_ops.py | r = preprocess.run\(src | +3 | note=本番（p2b_ops prepare）での呼び方 -->
**ソース** `app/pipeline/p2b_ops.py` L173〜L175（本番（p2b_ops prepare）での呼び方） — md5 f6a45c7e・`sync_src` が自動で更新

```python
  173|         r = preprocess.run(src, d / f"{name}_folded.onnx", prepped,
  174|                            swap_matmuls=set(swap) if swap else None,
  175|                            norm_rmsnorm="rms" in scheme)
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `clamp_extreme_constants(m0, replace_with=mask_replace)` | ① C-9。最初にやる（定数の畳み込みで極端な値がほかの初期化子へ広がる前に） |
| `expanded = rewrite_expand_to_concat(m0)` | ② C-10。量子化より前にやる（Q/DQ を挟んだ後では形を壊さずに書き換えにくい） |
| `scaled = fold_scale_into_matmul(m0) if fold_scale else []` | ③ スケール移動。既定 `fold_scale=True`。`p2b_ops` は `fold_scale` を渡さないので本番でもかかる |
| `softmaxes = stabilize_softmax(m0) if stabilize else []` | ④ softmax 安定化。既定 off（悪化したため。5.6） |
| `swapped = swap_actact_matmul_operands(m0, only=swap_matmuls) if swap_matmuls else []` | ⑤ C-8 の左右入替。spec の `swap_matmuls` に名前があるものだけ |
| `norm_rmsnorm="rms" in scheme)` | ⑥ C-11 は方式の名前に `rms` が入っているときだけ（例 `a16w8_rms_swap`） |

---

## 5.1 C-11: RMSNorm を max で正規化（今回の本命）

### 問題

`x²` のレンジが約 280万 → uint16 で 1段階 **42.7** →
通常チャンネルの `x²`（0.04 程度）が **0 に潰れる**
（[3章](03_Transformerの中で何が起きたか.md)）。

### 書き換え

```
x = [1677, 20, 1]
     ↓
m = max(|x|) = 1677
     ↓
xn = x / m = [1, 0.0119, 0.000596]     ← 全部 -1 〜 +1 に収まる
     ↓
xn² = [1, 0.000142, 0.00000036]        ← 最大値が 1
```

`xn²` の最大値が 1 なので、uint16 の1段階は $1/65535 \approx 0.000015$。

| | 1段階 |
|---|---:|
| 変更前（`x²`） | **42.7** |
| 変更後（`xn²`） | **0.000015** |

**約280万倍の改善。**

### ONNX グラフでどう書いたか

```
変更前:
  Pow(x,2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul(x, ·) → Mul(w)

変更後:
  m   = Add( ReduceMax( Abs(x) ), 1e-6 )     ← 下限を足す（後述）
  rmx = Reciprocal(m)
  xs  = Mul(x, rmx)
  Pow(xs,2) → ReduceMean → Add(eps') → Sqrt → Reciprocal → Mul(xs, ·) → Mul(w)
```

`ReduceMax` の軸は最後の1軸（チャンネル方向）なので、**トークンごとの max** になる。

### なぜ答えが変わらないのか

RMSNorm は**スケール不変**である。$x' = x/m$ とすると、

$$
\frac{x/m}{\sqrt{\mathrm{mean}((x/m)^2)}}
= \frac{x/m}{\sqrt{\mathrm{mean}(x^2)/m^2}}
= \frac{x/m}{\sqrt{\mathrm{mean}(x^2)}/m}
= \frac{x}{\sqrt{\mathrm{mean}(x^2)}}
$$

$m$ が**分子と分母で打ち消し合う**。元の式と完全に一致する。

### もう一つの効用: 分母がゼロに潰れない

$m = \max|x|$ で割ると、**最大要素は必ず 1 になる**。したがって

$$
\mathrm{mean}(xn^2) \ge \frac{1}{C} \qquad (C = \text{チャンネル数})
$$

Qwen なら $C = 896$ なので $\mathrm{mean}(xn^2) \ge 0.00112$。
**0 近くに潰れることがない**ので、正規化の分母が壊れる心配も消える。

### 効果（実測）

| | top-1 | KL | 相対L2 |
|---|---:|---:|---:|
| 6層 従来 | 6.2% | 9.61 | 139.7% |
| **6層 RMSNorm 正規化** | **74.4%** | **0.085** | **12.5%** |
| 24層 従来 | 4.5% | 10.59 | 129.0% |
| **24層 RMSNorm 正規化** | **80.7%** | 0.548 | 34.9% |

**6層の 74.4% は、Oracle で「RMSNorm を量子化しなければこうなる」と
分かっていた値に完全一致した。**

> **Oracle で得られた理想精度を、Graph Rewrite だけで再現できた。
> 原因仮説と対策が一致した** ＝ 仮説が正しかったことの強い証拠。

<!-- src: app/pipeline/preprocess.py | # ★max に小さい下限を足す。 | pw.input\[0\] = xs | note=normalize_rmsnorm_input の中。ノードを足す所 -->
**ソース** `app/pipeline/preprocess.py` L611〜L627（normalize_rmsnorm_input の中。ノードを足す所） — md5 11bc914b・`sync_src` が自動で更新

```python
  611|         # ★max に小さい下限を足す。
  612|         #   足さないと **全ゼロのテンソルで 1/0 = inf** になり、
  613|         #   qairt-converter が
  614|         #     `"quantization max"=inf which exceeds the size of type: float`
  615|         #   で落ちる（2026-09-13。較正の合成パターンに全ゼロが含まれていた）。
  616|         #   実データの max は 1000 オーダーなので 1e-6 の影響は無視できる。
  617|         flo = _unique(f"{base}_floor", {}, g)
  618|         g.initializer.append(numpy_helper.from_array(np.float32(1e-6), flo))
  619|         mxr = f"{base}_maxr"
  620|         extra.append(helper.make_node("Abs", [x], [ab], f"{base}_abs_n"))
  621|         extra.append(helper.make_node("ReduceMax", [ab, ax], [mxr], f"{base}_max_n",
  622|                                       keepdims=1, noop_with_empty_axes=0))
  623|         extra.append(helper.make_node("Add", [mxr, flo], [mx], f"{base}_maxfloor_n"))
  624|         extra.append(helper.make_node("Reciprocal", [mx], [rmx], f"{base}_rmax_n"))
  625|         extra.append(helper.make_node("Mul", [x, rmx], [xs], f"{base}_xs_n"))
  626| 
  627|         pw.input[0] = xs
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `helper.make_node("Abs", [x], [ab]` | \|x\| |
| `helper.make_node("ReduceMax", [ab, ax], [mxr]` | 最後の軸（`ax` = [-1]）で max を取る → トークンごとの m |
| `np.float32(1e-6)` | m の下限。全ゼロのトークンで 1/0 = inf になり、qairt-converter が落ちたため |
| `helper.make_node("Mul", [x, rmx], [xs]` | xs = x × (1/m)。xs は [-1, 1] に収まる |
| `pw.input[0] = xs` | 元の `Pow(x, 2)` の入力を xs に差し替える。後ろの `Mul(x, ·)` も xs に差し替える（この枠の後ろ） |

---

## 5.2 ε の扱い — ここは厳密解を取り逃した

実際の RMSNorm には $\epsilon$ がある:

$$
y = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}
$$

$x' = x/m$ にすると $\mathrm{mean}(x'^2) = \mathrm{mean}(x^2)/m^2$ なので、
**厳密に等価にするには $\epsilon' = \epsilon/m^2$ にする必要がある**:

$$
\frac{x/m}{\sqrt{\mathrm{mean}((x/m)^2)+\epsilon/m^2}}
= \frac{x/m}{\frac{1}{m}\sqrt{\mathrm{mean}(x^2)+\epsilon}}
= \frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}
$$

### 実装では近似した

`normalize_rmsnorm_input()` は **$\epsilon$ を固定の小さい値（1e-12）に
置き換える**近似をしている。「$m$ がデータ依存なので定数にできない」と判断したため。

**しかし $m$ はグラフ内のテンソルなので、`eps * rmx * rmx` として
実装できたはずである**（$rmx = 1/m$）。ここは改善余地として残っている。

### 近似の影響（実測）

fp32 で書き換え前後を比べると:

```
最大絶対差 5.075e-02   相対L2 2.197e-03（= 0.22%）   top-1 一致 100%
```

**0.22% のずれ**がある。量子化誤差（9〜35%）に比べれば小さいので採用したが、
**「厳密な等価ではない」と知って使っている**ことが大事。

<!-- src: app/pipeline/preprocess.py | # eps を小さくする（m² 倍されるため元の値では効きすぎる） | add.input\[i\] = e2 | note=normalize_rmsnorm_input の中 -->
**ソース** `app/pipeline/preprocess.py` L628〜L637（normalize_rmsnorm_input の中） — md5 11bc914b・`sync_src` が自動で更新

```python
  628|         # eps を小さくする（m² 倍されるため元の値では効きすぎる）
  629|         for add in consumers.get(rm[0].output[0], []):
  630|             if add.op_type != "Add":
  631|                 continue
  632|             for i, inp in enumerate(add.input):
  633|                 if inp in inits and np.asarray(inits[inp]).size == 1:
  634|                     e2 = _unique(f"{base}_eps", {}, g)
  635|                     g.initializer.append(numpy_helper.from_array(
  636|                         np.array(eps_new, dtype=np.asarray(inits[inp]).dtype), e2))
  637|                     add.input[i] = e2
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `if add.op_type != "Add":` | ReduceMean の次の Add（`+ eps`）を探す |
| `np.array(eps_new, dtype=np.asarray(inits[inp]).dtype)` | eps を固定の小さい値（関数の引数 `eps_new` の既定 1e-12）に置き換える。ε/m² にすれば厳密だった（`eps * rmx * rmx`）が、ここは定数で近似した |

---

## 5.3 C-8 対応: `QK^T = (K Q^T)^T` で左右を入れ替える

### 制約

v68 の MatMul は **`u16 × u8` しか通らない**
（[2章](02_量子化の基礎.md)）。つまり **8bit にできるのは常に右辺**。

### 問題

Attention の `MatMul(Q, K^T)` では右辺が K で、

| | レンジ | uint8 の1段階 |
|---|---|---:|
| **K**（右辺） | ±130 | **0.99** |
| Q（左辺） | [-10, 6] | **0.064** |

**粗い方（K）を 8bit にせざるを得ない**のが損。**15倍の差**がある。

### 書き換え

$$
A \times B = (B^T \times A^T)^T
$$

は厳密に等価なので、

```
変更前: MatMul(Q, K^T)            → 右辺 = K   （1段階 0.99）
変更後: Transpose(MatMul(K, Q^T)) → 右辺 = Q^T （1段階 0.064）
```

`Transpose` が3つ増えるが、**右辺に置くテンソルを選べる**ようになる。

### 効果（end-to-end、2層）

| | top-1 | fp32 の1位が何位か | v68 |
|---|---:|---:|---|
| 両側 A8（当初） | 50.0% | 20.64位 | 動く |
| `u16×u8`（入替なし） | 66.5% | 1.95位 | 動く |
| **`u16×u8` + 左右入替** | **80.7%** | **1.34位** | **動く** |
| A16W8 素 | 85.2% | 1.20位 | **動かない** |

**到達不可能な上限（85.2%）まで 4.5ポイント**。当初の 50.0% から見れば別物。

### 実機では速くもなった

| | 実機 median |
|---|---:|
| 層1枚: 従来（両側A8） | 1.178 ms |
| 層1枚: `u16×u8`（入替なし） | 0.923 ms |
| 埋め込み + 2層 + lm_head: 当初（両側A8） | 17.42 ms |
| **埋め込み + 2層 + lm_head: 左右入替 + `u16×u8`** | **14.29 ms** |

（★2026-10-17 訂正: 旧表は「従来（両側A8） 1.178 ms / 左右入替 + `u16×u8` 0.923 ms」。`results/P3_20260913_mixed_matmul.md` では、0.923 ms は層1枚の**入替なし**の `u16×u8`（2 節）で、左右入替を入れた実機の数字は 2 層 end-to-end の 17.42 → 14.29 ms（6・7 節）だった）

**Transpose を3つ足したのに速くなった。** 型変換ノードが減ったためと推測されるが未確認。

<!-- src: app/pipeline/preprocess.py | perm = list\(range\(r\)\) | f"\{base\}_tr_out", perm=perm\)\) | note=swap_actact_matmul_operands の中 -->
**ソース** `app/pipeline/preprocess.py` L509〜L519（swap_actact_matmul_operands の中） — md5 11bc914b・`sync_src` が自動で更新

```python
  509|         perm = list(range(r))
  510|         perm[-1], perm[-2] = perm[-2], perm[-1]
  511| 
  512|         base = node.name or f"mm{len(changed)}"
  513|         at, bt, st = f"{base}_At", f"{base}_Bt", f"{base}_scoresT"
  514|         new_nodes.append(helper.make_node("Transpose", [b], [bt], f"{base}_tr_B", perm=perm))
  515|         new_nodes.append(helper.make_node("Transpose", [a], [at], f"{base}_tr_A", perm=perm))
  516|         # B^T × A^T = (A × B)^T
  517|         new_nodes.append(helper.make_node("MatMul", [bt, at], [st], f"{base}_swapped"))
  518|         new_nodes.append(helper.make_node("Transpose", [st], [node.output[0]],
  519|                                           f"{base}_tr_out", perm=perm))
```
<!-- /src -->

<!-- src: app/pipeline/find_swap_matmuls.py | qk, pv, other, weight = \[\], \[\], \[\], \[\] | other.append\(n.name\) | note=classify の中。入れ替える対象（QK^T）を形で見分ける -->
**ソース** `app/pipeline/find_swap_matmuls.py` L115〜L126（classify の中。入れ替える対象（QK^T）を形で見分ける） — md5 97ae616d・`sync_src` が自動で更新

```python
  115|     qk, pv, other, weight = [], [], [], []
  116|     for n in g.node:
  117|         if n.op_type != "MatMul":
  118|             continue
  119|         if is_weight(n.input[0]) or is_weight(n.input[1]):
  120|             weight.append(n.name)
  121|         elif reaches_softmax(n.output[0]):
  122|             qk.append(n.name)
  123|         elif from_softmax(n.input[0]) or from_softmax(n.input[1]):
  124|             pv.append(n.name)
  125|         else:
  126|             other.append(n.name)
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `perm[-1], perm[-2] = perm[-2], perm[-1]` | 最後の 2 軸を入れ替える転置 |
| `helper.make_node("MatMul", [bt, at], [st]` | Bᵀ × Aᵀ。右辺が元の左辺（Q）の転置になる |
| `f"{base}_tr_out", perm=perm` | 結果をもう一度転置して元の A×B に戻す。Transpose が 3 つ増える |
| `elif reaches_softmax(n.output[0]):` | 入れ替える対象（QK^T）は、出力が Mul/Add を経て Softmax に届く MatMul。P×V は対象外（右辺 V のレンジが元々狭い） |

---

## 5.4 C-10: `Expand` → `Concat`

### 問題

**v68 は `Expand` を含む context binary をロードできない。**
ホストの context 生成は素通しするので、**端末で初めて分かる**:

```
Failed to create context from binary. Error code: 1002
```

HuggingFace の GQA ヘッド複製が `Unsqueeze`→`Expand`→`Reshape` で出るので、
**実モデルでは必ず踏む**。

### 切り分け方（単体モデルで1つずつ）

| モデル | 実機 |
|---|---|
| `unsqueeze_only` | **OK** |
| **`expand_only`** | **err 1002** |
| `gqa_concat` | **OK** |

→ **`Expand` 単独が原因。`Unsqueeze` は無罪。`Concat` は代替として使える。**

### 書き換え

`Expand` が「サイズ1の軸を n 倍に広げる」形なら、
**同じテンソルを n 個 `Concat` する**のと結果は同一。

```
gqa_expand  [1,2,1,16,64] -> [1,2,7,16,64]   →  axis2 方向に 7回 Concat
```

fp32 で**最大絶対差 0**（完全一致）を確認済み。

### ここで踏んだ罠: target はブロードキャスト形式でありうる

dynamo の export は次の形を出した:

```
入力    [1, 2, 1, 16, 64]
target  [1, 1, 7, 1, 1]     ← 完全な出力形状ではない
結果    [1, 2, 7, 16, 64]
```

私の判定は target を出力形状と見なしていたため
「軸1が 2 vs 1 で不一致」と誤判定し、**48件すべて取りこぼした**。

**ブロードキャスト結果（`max(a, t)`）を自分で計算する**よう修正して直った。

> **気付かなければ 54分の context 生成を無駄にし、
> 端末で err 1002 を見てから戻ることになっていた。**
> 生成を始める前に中身を確認したのが効いた。

<!-- src: app/pipeline/preprocess.py | # ★target は\*\*ブロードキャスト形式\*\*でありうる。 | cur = out$ | note=rewrite_expand_to_concat の中 -->
**ソース** `app/pipeline/preprocess.py` L188〜L207（rewrite_expand_to_concat の中） — md5 11bc914b・`sync_src` が自動で更新

```python
  188|         # ★target は**ブロードキャスト形式**でありうる。
  189|         #   実際 dynamo の export は `[1,2,1,16,64]` を `[1,1,7,1,1]` で広げる形を出した
  190|         #   （結果は `[1,2,7,16,64]`）。target をそのまま出力形状と見なすと
  191|         #   「軸1が 2 vs 1 で不一致」と誤判定して**書き換えを取りこぼす**
  192|         #   （2026-09-13 に踏んだ。54分の context 生成を無駄にしかけた）。
  193|         #   ブロードキャスト結果を自分で計算する。
  194|         out_shape = [max(a, t) for a, t in zip(in_shape, target)]
  195|         axes = [(i, o) for i, (a, o) in enumerate(zip(in_shape, out_shape))
  196|                 if a == 1 and o > 1]
  197|         if not axes or any(a != o and a != 1 for a, o in zip(in_shape, out_shape)):
  198|             new_nodes.append(node)
  199|             continue
  200| 
  201|         cur = src
  202|         for k, (axis, rep) in enumerate(axes):
  203|             out = node.output[0] if k == len(axes) - 1 else f"{node.output[0]}_cat{k}"
  204|             new_nodes.append(helper.make_node(
  205|                 "Concat", [cur] * rep, [out],
  206|                 f"{node.name or 'expand'}_to_concat{k}", axis=axis))
  207|             cur = out
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `out_shape = [max(a, t) for a, t in zip(in_shape, target)]` | target をそのまま出力の形と見なさず、ブロードキャストの結果を自分で計算する（48 件の取りこぼしの直し） |
| `if a == 1 and o > 1]` | サイズ 1 の軸を広げる所だけを対象にする。それ以外の形は書き換えずに残す |
| `"Concat", [cur] * rep, [out],` | 同じテンソルを rep 個つなぐ＝ Expand と同じ結果 |

---

## 5.5 C-9: マスク定数 `-3.4e38` → `-60`

### 問題

HuggingFace の因果マスクは **`-3.4028235e+38`（float32 の最小値）**を使う。

量子化は「最小値〜最大値を N 段階に割る」処理なので、この値が1つ混ざると:

| | 範囲 | uint16 の1段階 |
|---|---|---:|
| 本来のスコアだけ | 約 -20 〜 +20 | 0.0006 |
| **-3.4e38 が混ざると** | -3.4e38 〜 +20 | **5e33** |

**全分解能がマスクの1点に食われ、実際のスコアが丸ごと潰れる。**

### なぜこれが一番危険か

**エラーにならない。** そのまま量子化も変換もロードも通り、
**精度だけが静かに壊れる。**

### 書き換え

有限値（**-60.0**）へ置き換える。softmax を通せば
$e^{-60} \approx 8.8 \times 10^{-27}$ で遮断として十分。

- 小さすぎる（-10 など）→ 遮断が甘くなる
- 大きすぎる（-1e4 など）→ 分解能を食う

**両方を見て決める値**である。

### 検証

本物の Qwen（2層）で置換前後の logits を比較:

```
最大絶対差  0.000000e+00
平均絶対差  0.000000e+00
argmax 一致 True
```

**fp32 では完全にビット一致。** グラフの意味は変わっていない。

<!-- src: app/pipeline/preprocess.py | ^# 量子化できない極端な値の既定の置き換え先。 | ^EXTREME = 1e30 | note=置き換え先の値と、その理由 -->
**ソース** `app/pipeline/preprocess.py` L68〜L77（置き換え先の値と、その理由） — md5 11bc914b・`sync_src` が自動で更新

```python
   68| # 量子化できない極端な値の既定の置き換え先。
   69| #
   70| # 因果マスクは softmax の前にスコアへ足される。遮断できればよいので、
   71| # exp(x) が実質 0 になる程度の有限値で足りる（exp(-60) ≒ 8.8e-27）。
   72| # 小さくしすぎると遮断が甘くなり、大きくしすぎると量子化の分解能を食う:
   73| #   本来のスコア範囲が ±20 程度のとき
   74| #     置き換え -60  → 範囲 [-60, 20]  16bit で 1段階 0.0012
   75| #     置き換え -1e4 → 範囲 [-1e4, 20] 16bit で 1段階 0.15    ← 実データが潰れる
   76| MASK_REPLACE = -60.0
   77| EXTREME = 1e30
```
<!-- /src -->

<!-- src: app/pipeline/preprocess.py | neg = np.isneginf\(arr\) | new\[pos\] = -replace_with | note=clamp_extreme_constants の中 -->
**ソース** `app/pipeline/preprocess.py` L105〜L111（clamp_extreme_constants の中） — md5 11bc914b・`sync_src` が自動で更新

```python
  105|         neg = np.isneginf(arr) | (arr <= -threshold)
  106|         pos = np.isposinf(arr) | (arr >= threshold)
  107|         if not (neg.any() or pos.any()):
  108|             continue
  109|         new = arr.astype(np.float64).copy()
  110|         new[neg] = replace_with
  111|         new[pos] = -replace_with
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `MASK_REPLACE = -60.0` | 置き換え先。exp(-60) ≈ 8.8e-27 で遮断には十分 |
| `置き換え -1e4 → 範囲 [-1e4, 20] 16bit で 1段階 0.15` | 大きすぎると分解能を食う例 |
| `(arr <= -threshold)` | `EXTREME = 1e30` を超える負の値（-3.4e38 や -inf）を見つける |
| `new[neg] = replace_with` | -60 に置き換える。黙って変えないよう、変えたものは `changed` に記録する |

---

## 5.6 効かなかった書き換え（再試行を防ぐために残す）

### スケールを MatMul の前へ移す → 効果なし

`Mul(MatMul(q,k), 1/8)` → `MatMul(Mul(q,1/8), k)`。
数学的に厳密に等価（fp32 で差 0 を確認）。QK^T の生値レンジ 12,515 が 1,600 になる。

**結果: 30.3819% → 30.3904%（変わらず）。**

**読み違えていた。** 量子化されるのは**スケール後**の値であって、
移動してもスケール後のレンジは変わらない。

### softmax の前で行最大を引く → 悪化

`softmax(x) = softmax(x - rowmax(x))` は厳密に等価（fp32 で差 0 を確認）。
numpy 上の単独試算では劇的に効く:

| | 1段階 | softmax後 L2誤差 | argmax 一致 |
|---|---:|---:|---:|
| そのまま 8bit | 6.38 | 62.8% | 29.2% |
| **行最大を引いて 8bit** | **0.26** | **3.2%** | **90.5%** |

**ところがパイプラインに入れると 30.4% → 41.6% に悪化した。**

理由: **スコアは MatMul の出力時点ですでに 8bit 化されている。**
その後に行最大を引いても手遅れで、追加した ReduceMax/Sub の量子化誤差だけが乗る。

> **対策を打つ位置が間違っていた。**
> 「数学的に正しい変形」でも、**パイプライン上のどこに入るか**で効果が変わる。

<!-- src: app/pipeline/preprocess.py | スケールを MatMul の\*\*前\*\*へ移せば | 近似は一切していない。 | note=fold_scale_into_matmul の説明（当時の見込み） -->
**ソース** `app/pipeline/preprocess.py` L235〜L237（fold_scale_into_matmul の説明（当時の見込み）） — md5 11bc914b・`sync_src` が自動で更新

```python
  235|     スケールを MatMul の**前**へ移せば、量子化されるのはスケール後の値になり、
  236|     同じ 8bit でも**分解能が 8倍**になる。`(a·s)×b = (a×b)·s` なので**厳密に等価**で、
  237|     近似は一切していない。
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `分解能が 8倍` | 当時の見込み。実際は効かなかった。量子化されるのはスケール後の値（`mul_8`）で、移してもスケール後の範囲は変わらなかった |
| `(a·s)×b = (a×b)·s` | 厳密に等価なので fp32 で差 0。害も無いので、今も `preprocess.run` の既定で入る（5.0 の枠の ③） |

---

## 5.7 等価性は必ず数値で確かめる

「数学的に等価」と思っても、実装でミスをする。毎回 **fp32 で書き換え前後を比較**
してから先へ進めた。

| 書き換え | fp32 での差 |
|---|---|
| C-9（マスク定数） | **0**（ビット一致） |
| C-10（Expand→Concat） | **0** |
| 左右入替（`QK^T`） | **0** |
| スケール移動 | **0** |
| softmax 前の行最大減算 | **0** |
| **C-11（RMSNorm 正規化）** | **0.22%**（ε を近似したため） |

**差ゼロを確認できたものと、そうでないものを区別して扱う。**

### 自前で書いた配線も本物と突き合わせる

decode グラフでは attention を自前で書き直した（KV キャッシュを明示的な
テンソルにするため）。**HF と1トークンずつ突き合わせて確認した:**

```
6ステップ  最大絶対差 3.266e-05  top-1 全一致 True
```

マスク入力版の prefill も:

```
最大絶対差 0.000e+00  top-1 一致 True
```

> **「重みは本物、配線だけ自前」なら、配線が合っているかは必ず測れる。**
> ここを飛ばすと、以降の精度議論がすべて無意味になる。

### 検証側のバグにも注意

最初の decode 検証で「最大絶対差 20.5」が出て「配線バグ」と判定した。
**実際は自前が2層・HF が24層で比べていた**（検証側のバグ）。
HF 側も2層にしたら **4.3e-05** だった。

> **検証側が壊れていると、正しい実装を「バグ」と誤判定する。**

<!-- src: app/pipeline/export_qwen_decode.py | ^def verify\(step: QwenDecodeStep | t1.append\(bool | note=自前の配線を HF と突き合わせる -->
**ソース** `app/pipeline/export_qwen_decode.py` L376〜L412（自前の配線を HF と突き合わせる） — md5 9f3431d3・`sync_src` が自動で更新

```python
  376| def verify(step: QwenDecodeStep, model, cache_len: int, n_check: int = 6) -> dict:
  377|     """**自前の配線が HF と一致するか確かめる。**
  378| 
  379|     HF に同じトークン列を1つずつ食わせ、各ステップの logits を比べる。
  380|     ここが合わなければ、以降の精度議論はすべて無意味になる。
  381| 
  382|     （2026-10-18 追記）HF 側は毎回「先頭から n 個」をキャッシュなしで流し、最後の位置の logits を取る。
  383|     自前側は 1 トークンずつ流し、出力の present を次の past に回す（端末での使い方と同じ）。
  384|     戻り値: max_abs_diff（全ステップの最大絶対差）・per_step・top1_match（全ステップで 1 位が一致したか）・steps
  385|     """
  386|     tok_ids = [9707, 11, 1879, 0, 3555, 374]          # 適当な実トークン列
  387|     tok_ids = tok_ids[:n_check]
  388|     nl, nkv, hd = len(step.layers), step.nkv, step.hd
  389|     # ★HF 側も同じ層数にする。
  390|     #   これを忘れて「自前2層 vs HF24層」を比べ、配線バグと誤判定した（2026-09-13）。
  391|     saved = model.model.layers
  392|     model.model.layers = step.layers
  393|     caches = [torch.zeros(1, nkv, cache_len, hd) for _ in range(2 * nl)]
  394|     diffs, t1 = [], []
  395| 
  396|     with torch.no_grad():
  397|         # HF 側は素の forward に全履歴を渡して最後の位置の logits を取る
  398|         for n in range(1, len(tok_ids) + 1):
  399|             ids = torch.tensor([tok_ids[:n]])
  400|             ref = model(input_ids=ids, use_cache=False).logits[:, -1:, :]
  401| 
  402|             # 自前側は1トークンずつ
  403|             c, sn = rope_cos_sin(model, [n - 1])
  404|             mask = torch.full((1, 1, 1, cache_len), MASK_NEG)
  405|             mask[0, 0, 0, cache_len - n:] = 0.0        # 有効なのは後ろ n 個
  406|             out = step(torch.tensor([[tok_ids[n - 1]]]),
  407|                        torch.from_numpy(c), torch.from_numpy(sn), mask, *caches)
  408|             mine, caches = out[0], list(out[1:])
  409| 
  410|             d = float((ref - mine).abs().max())
  411|             diffs.append(d)
  412|             t1.append(bool(ref.argmax(-1).item() == mine.argmax(-1).item()))
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `model.model.layers = step.layers` | HF 側も同じ層数にする。これを忘れて「自前 2 層 vs HF 24 層」を比べ、配線バグと誤判定した（最大絶対差 20.5） |
| `ref = model(input_ids=ids, use_cache=False).logits[:, -1:, :]` | HF は全履歴を毎回渡して、最後の位置の logits を取る |
| `mask[0, 0, 0, cache_len - n:] = 0.0` | 自前は 1 トークンずつ。有効なのは後ろ n 個（左シフト＋マスク） |
| `d = float((ref - mine).abs().max())` | 最大絶対差。24 層・KV 128 で 3.266e-05、2 層・KV 32 で 4.29e-05（`work/qwen_decode/decode_L*_C*.meta.json` の `verify`） |

---

[← 目次へ](../StudyMemo.md) | [次: 6. ログの読み方と原則 →](06_ログの読み方と原則.md)
