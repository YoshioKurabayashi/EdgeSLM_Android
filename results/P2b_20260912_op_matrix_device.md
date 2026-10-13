# P2b: Transformer 構成要素の端末実行（runner v0.2 / op_check）

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-09-12 |
| 端末 | Sony Xperia 5 III SOG05（SM8350 / HTP v68） |
| runner | **v0.2**（`op_check` task・多入力対応・セッション張り替え） |
| ORT | 1.27.0（端末） |
| 対象 | ホストで `ready_for_device` になった **15本** |
| 電力 | `default(unsigned_pd)`（C-3。burst は使えない） |
| 判定 | **15/15 `pass_strict`** |

---

## 1. 結果

```
summary: {"pass_strict": 15, "pass_cpu_boundary": 0, "failed": 0, "total": 15}
```

**全モデルが `QNN HTP - EPContext - all nodes` / `disable_cpu_ep_fallback=1` で成功。**
CPU へ落ちたノードはひとつもない。

| モデル | 方式 | 判定 | OrtRun 中央値 (ms) |
|---|---|---|---:|
| matmul_weight | A16W8 | pass_strict | **0.284** |
| matmul_act | A16W8混合 | pass_strict | 0.334 |
| matmul_act | A8W8 | pass_strict | 0.292 |
| rmsnorm | A16W8 | pass_strict | 0.306 |
| layernorm | A16W8 | pass_strict | 0.324 |
| rope | A16W8 | pass_strict | **0.264** |
| softmax | A16W8 | pass_strict | 0.307 |
| silu | A16W8 | pass_strict | 0.335 |
| embed | A16W8 | pass_strict | 0.307 |
| attention | A16W8混合 | pass_strict | 0.347 |
| attention | A8W8 | pass_strict | 0.300 |
| kvcache | A16W8混合 | pass_strict | 0.332 |
| kvcache | A8W8 | pass_strict | 0.378 |
| **block** | **A16W8混合** | **pass_strict** | **0.372** |
| **block** | **A8W8** | **pass_strict** | 0.388 |

### 意味

**Transformer 1ブロックが、KVキャッシュ込みで、SOG05 の NPU 上で全ノード動いた。**

ホストで通ったものが端末でもすべて動いた、というのが要点である。C-1〜C-8 は
「変換は通ったのに端末で落ちる」類の制約なので、**ここを通るまで確定とは言えなかった**。

---

## 2. この数字を性能として読んではいけない

| モデル | ノード数 | 中央値 (ms) |
|---|---:|---:|
| matmul_weight | 1 | 0.284 |
| rope | 6 | 0.264 |
| attention | 20 | 0.347 |
| **block** | **39** | **0.372** |

**ノード数が39倍でも所要時間は1.3倍**にしかならない。P2 の `testnet`（0.277ms）とも
ほぼ同じである。つまりこの規模では**計算ではなく固定費（呼び出し・転送・同期）を
測っている**。

→ **この表から「RoPE は Softmax より速い」等を読み取ってはいけない。**
すべて 0.26〜0.39ms のノイズの中にあり、**A16W8混合 と A8W8 の差も判別できない**。
op ごとの実コストが見えるのは、本物の規模（hidden 896 / 24層）に載せてからになる。

分かったのは「**1回の推論呼び出しに約 0.3ms の床がある**」という事実のほうで、
これは decode ループの設計（1トークンごとに呼ぶのか、まとめるのか）に効いてくる。

ウォームアップの初回は全モデルで 1.0〜1.4ms、2回目で 0.35〜0.48ms に落ちる。
P2 の `testnet` で見た 3.9倍の初回ペナルティと同じ傾向である。

---

## 3. 1本だけ落ちた（そして直した）

初回の実行では `embed` が `run_failed` になった。**14/15 だった。**

```
ExecuteGraph ORT Tensor data size does not match QNN tensor data size
```

### 原因は v68 ではなく自分のラッパー

`wrap_epctx.py` が EPContext ONNX の I/O を **float32 決め打ち**で宣言していた。
`--preserve_io_datatype` でグラフ I/O が float32 に保たれる前提で書いたためだが、
**埋め込み表引きの索引 `ids` は量子化されず int64 のまま残る**。

| | 宣言 | 実体 |
|---|---|---|
| ONNX 側 | float32 → 4byte × 16 = **64 byte** | |
| QNN 側 | | int64 → 8byte × 16 = **128 byte** |

バイト数が合わず Run で落ちた。型を QDQ モデルから引き継ぐよう修正し、
**S8 だけ作り直して（S5/S6 の成果物は再利用）15/15 になった。**

### 症状の型として新しい

C-1〜C-8 はすべて**ロード時**に落ちる。これは**ロードは成功して Run で落ちる**。

| 落ちる場所 | 意味 |
|---|---|
| ホストの context 生成 | C-8（op が v68 に無い） |
| 端末のロード | C-1〜C-7（構成・パッケージング・PD） |
| **端末の Run** | **I/O の型・形状の宣言が実体と違う** ← 今回 |

`describe` の出力に `"name":"ids", "type":"float32"` と出ていたのが手がかりだった。
**runner が I/O を JSON で全部返す設計にしていたので、1回の実行で原因が特定できた。**

---

## 4. C-8 の扱いについて

端末側では A16W8混合 と A8W8 の**どちらも動いた**。ホスト側で確定した
「C-8 を避ければ通る」が、実機でも成立することを確認したことになる。

ただし**精度はまだ測っていない**。乱数重み・乱数入力なので、この15本から
精度について言えることは何もない。A8 に落とした2箇所（QK^T / PV）が
実際にどれだけ効くかは、**本物の重みと本物の入力**で測るしかない。

---

## 5. P2b の到達点

```
✅ ホスト S1〜S8   13種の単体 op。v68 の壁を C-8 の1点に特定し、回避策を実証
✅ 端末 実行       15/15 pass_strict。Transformer 1ブロックが NPU で全ノード動作
❌ 精度           未着手
❌ 実モデル       未着手（自作の乱数重み・縮小形状のみ）
```

**「op は載る」ことは確定した。** 残るのは**精度**と**規模**で、どちらも
本物の Qwen2.5-0.5B を通さないと答えが出ない。

次は **本物の Qwen2.5-0.5B の ONNX export**。
まず確かめるべきは「**私が手で書いた13個のグラフが、実際の export と合っていたか**」
である。GQA のヘッド複製を `Concat` で書いたが実際は `Expand`/`Tile` かもしれず、
マスクも `Add` ではなく `Where`/`Trilu` かもしれない。
**そこが違えば、この15本の結果はそのままは転用できない。**
（C-8 だけは QK^T / PV がどんな実装でも活性化×活性化なので、export の形に依存しない。）
