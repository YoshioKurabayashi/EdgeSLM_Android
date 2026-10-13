# P2b: Transformer 構成要素の op 対応表（ホスト側 S1〜S8）

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 対象 | Transformer 構成要素 13種の単体 ONNX |
| baseline | QAIRT 2.48.0.260626 + ORT 1.30.0（ホスト量子化） |
| ターゲット | SM8350 / HTP **v68** |
| 段階 | **ホスト側 S1〜S8 のみ**。端末実行は runner v0.2 のビルド後 |
| 判定 | **v68 の壁を1点に特定。回避策も確認済み** |

---

## 1. 結論（先に）

> **HTP v68 は「活性化 × 活性化」の MatMul を 16bit で受け付けない。**
> これは attention の QK^T と PV に直撃する。
> **A8 に落とすか、その2箇所だけ A8 にする（混合精度）と通る。**

```
<E> [4294967295] has incorrect Value 68, expected >= 73.
<E> QnnBackend_validateOpConfig failed 3110
<E> Failed to validate op matmul with error 0xc26
```

失敗した4モデル（`matmul_act` / `attention` / `kvcache` / `block`）は
**すべてこの1つのエラー**で、他に落ちた原因は無い。つまり
**v68 で Transformer が載らない理由は、いま判明している範囲では1点だけ**である。

これは要件定義書 3.3節の「v68 は 16a16w の matmul が使えない（V73以上）」を
**自分の手で再現し、影響範囲を特定した**ことになる。文献上の制約が、
「attention の2つの MatMul に効く」という具体に変わった。

---

## 2. 対応表

段を分けて記録している。同じ「通らない」でも対処がまったく違うため。

| 段 | 落ちたときの意味 | 対処の方向 |
|---|---|---|
| S2 前処理 | ONNX の形が悪い | モデルの書き方を変える |
| S4 量子化 | 量子化器が op を知らない | 分解する / 精度指定 |
| S5 変換 | `qairt-converter` が op を知らない | **op を分解する** |
| S6 context生成 | **HTP コンパイラが受け付けない** | **精度を変える / 分解する** ← v68 の壁はここ |
| 端末 | ロード or 実行で落ちる | C-1〜C-7 の世界 |

### 2.1 A16W8（本線として想定していた方式）

| # | 要素 | S2 | S4 | S5 | S6 | 判定 |
|---|---|:--:|:--:|:--:|:--:|---|
| 1a | **matmul_weight**（重み固定） | OK | OK | OK | OK | **通る** |
| 1b | **matmul_act**（両辺が活性化） | OK | OK | OK | **NG** | **v68 の壁** |
| 2 | **rmsnorm**（素の op へ分解） | OK | OK | OK | OK | **通る** |
| 2b | rmsnorm_fused（`SimplifiedLayerNormalization`） | **NG** | - | - | - | 作れない（後述） |
| 3 | **layernorm**（opset 17 標準） | OK | OK | OK | OK | **通る** |
| 4 | **rope**（素の op へ分解） | OK | OK | OK | OK | **通る** |
| 4b | rope_fused（`com.microsoft.RotaryEmbedding`） | OK | OK | **NG** | - | 分解が必要 |
| 5 | **softmax** | OK | OK | OK | OK | **通る** |
| 5b | **silu**（`x*sigmoid(x)`） | OK | OK | OK | OK | **通る** |
| 6 | **attention**（手分解） | OK | OK | OK | **NG** | **v68 の壁** |
| 7 | **kvcache**（decode 1ステップ / 多入出力） | OK | OK | OK | **NG** | **v68 の壁** |
| 8 | **embed**（Gather + 形状操作 / int64索引） | OK | OK | OK | OK | **通る** |
| 9 | **block**（Transformer 1ブロック） | OK | OK | OK | **NG** | **v68 の壁** |

### 2.2 回避策の検証

失敗した4つを別方式で作り直した。

| 方式 | matmul_act | attention | kvcache | block |
|---|:--:|:--:|:--:|:--:|
| A16W8 | NG | NG | NG | NG |
| **A8W8**（全部 8bit） | **OK** | **OK** | **OK** | **OK** |
| **A16W8 混合**（該当 MatMul だけ A8） | **OK** | **OK** | **OK** | **OK** |

**混合精度が成立した**のが重要である。全体を A8 に落とさずに済む。

混合が本当に混合になっているかを、QDQ モデルの zero-point の型で確認した
（**「上書きが黙って無効になる」のが量子化で最も起きやすい失敗**のため）:

| モデル | 方式 | 活性化 Q ノードの型 |
|---|---|---|
| block | A16W8 | UINT16 × 40 |
| block | A8W8 | UINT8 × 40 |
| **block** | **A16W8 混合** | **UINT16 × 34 / UINT8 × 12** |
| attention | A16W8 混合 | UINT16 × 14 / UINT8 × 12 |

8bit に落としたのは attention の2つの MatMul に関わる**6テンソルだけ**:

```
QK^T : att_qt（Q） att_kk（K複製） att_sc（スコア）
PV   : att_pr（確率） att_vv（V複製） att_ct（文脈）
```

重み固定の MatMul（Q/K/V/O projection、FFN の gate/up/down）は **16bit のまま**。
`app/pipeline/p2b_ops.py` の `act_act_matmul_tensors()` が
「入力のどちらも initializer でない MatMul」を機械的に拾ってこれを作る。
**両者を区別せず全部 8bit にすると、精度を無駄に捨てる。**

---

## 3. 個別に分かったこと

### 3.1 `rmsnorm_fused` は「作れない」のであって「非対応」ではない

```
ValidationError: No Op registered for SimplifiedLayerNormalization with domain_version of 17
```

`SimplifiedLayerNormalization` は **ORT が内部で融合して作る op** であって、
標準 opset の op ではない。したがって**ファイルとして書き出して持ち込む道が無い**。
「v68 で非対応」ではなく「この経路では存在し得ない」が正しい。

→ RMSNorm は **素の op へ分解した形（#2）で通る**ので、実害はない。
ONNX opset 23 の `RMSNormalization` を使う道は、opset を上げる判断とセットになる。

### 3.2 `rope_fused` は converter の段で落ちる

`com.microsoft.RotaryEmbedding` は量子化（S4）までは通るが、`qairt-converter` が
QNN 検証で弾く。一方 **素の op へ分解した RoPE（#4）は S6 まで通る**。

→ **RoPE は分解して使う。** 「①そのまま通る ②分解すれば通る ③通らない」の
**②に該当する唯一の要素**だった。

### 3.3 int64 の索引は通る

`embed`（Gather + Reshape + MatMul、`ids` が int64）が S6 まで通った。
埋め込み表引きが整数量子化グラフの中で成立することが確認できた。

### 3.4 多入出力も S6 まで通る

`kvcache` は入力3・出力3（`x` / `past_k` / `past_v` → `y` / `present_k` / `present_v`）。
A8 で通ったので、**固定長KVキャッシュの形そのものは v68 で作れる**。

---

## 4. 踏んだ落とし穴

### 4.1 成功ログにも出る ERROR を真因として報告しかけた

```
87.7ms [ ERROR ] WindowsFileIO couldn't open libcdsprpc.dll
```

最初この行を失敗原因として集計表に出してしまった。**この行は成功した実行の
ログにも同じように出る**（ホストの x86 には端末側 FastRPC ライブラリが無い、
というだけ）。真因は QNN 自身が出す `<E>` 行のほうにある。

→ 集計スクリプトを、`<E>` 行を優先し既知の無害行を除外するよう修正した
（`scripts/p2b_build_ops.sh` の `qnn_cause()`）。

**中間まとめ 5章「型4: 症状と真因の混同」と同じ型**を、また踏んだ。
ログの最初のエラー行は真因とは限らない。

### 4.2 失敗しても「生成した」と書かれる

失敗した実行でも、最終行はこう出る:

```
Cache binary for SM8350 copied to work\ops\matmul_act\ctx\ctx_matmul_act/matmul_act_ctx.SM8350
```

**実際にはファイルは作られていない**（`ls` で確認済み）。
`dd` がエラーでもサマリ行を出すのと同じ型。
`build_context.sh` は最初から `ls *.bin` の実在確認で成否を見ているため
この件で誤判定はしていないが、**ログの最終行を成否の根拠にしてはいけない**。

---

## 5. これが意味すること

### 5.1 Qwen2.5-0.5B を v68 で動かす道は残っている

Transformer 1ブロックが **A8W8 でも A16W8混合でも S8 まで通った**。
P4/P5 で問題になるのは op の対応可否ではなく、**容量と精度**に絞られた。

### 5.2 本線の量子化方式を改める必要がある

要件定義書は本線を **A16W8** としていたが、**素の A16W8 では attention が載らない**。
本線は **「A16W8 を基本とし、活性化×活性化の MatMul のみ A8」** に改める。

これは机上の想定ではなく、**同一モデルを3方式で作って比較した実測**である。

### 5.3 精度の評価が次の論点になる

A8 に落とした2箇所が精度にどう効くかは**まだ測っていない**。
attention のスコアと確率は分布が偏る（softmax 後は 0..1 に集中する）ため、
8bit でも足りる可能性はあるが、**根拠は無い**。P3 で fp32 参照との比較を行う。

---

## 6. 次

**端末での実行（runner v0.2）**。ホストで `ready_for_device` になった **15本**を
`op_check` task で順に開き、次の4値で判定する:

| 判定 | 意味 |
|---|---|
| `pass_strict` | 全ノードが HTP。そのまま使える |
| `pass_cpu_boundary` | 開けたが CPU フォールバックあり。delegate しきれていない |
| `open_failed` | ロードで落ちた（C-1〜C-7 / op 非対応） |
| `run_failed` | 開けたが実行で落ちた |

**ホストで通ることと端末で動くことは別**である。P1 で
「変換は通ったのに端末で `error 1002`」という形の失敗が起こりうることは
C-1〜C-5 として分かっている。ここが最後の未確認区間になる。

runner v0.2 のビルドが必要（NDK clang で構文・警告は確認済み。
`-Wall -Wextra` で警告ゼロ）。
