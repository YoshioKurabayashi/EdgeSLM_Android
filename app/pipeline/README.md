# app/pipeline — 道具の索引

このディレクトリは **「モデルを NPU で動かす」ための道具**と、
**「なぜ合わないのかを調べる」ための道具**が混在している。
どちらなのかを先に示す。各ファイルの冒頭にも同じ説明がある。

---

## 1. 本番の流れ（この順に通る）

```
export → S2 前処理 → S3/S4 量子化 → （後処理） → S5 DLC変換 → S6 context生成 → 配備 → 実機
```

| 段 | ファイル | すること |
|---|---|---|
| export | `export_qwen_decode.py` | HF の重みを使って decode / prefill グラフを書き出す。**配線が HF と一致するか自分で検算する**（`--verify-only`） |
| データ | `make_decode_calib.py` / `make_prefillmask_data.py` / `make_calib_data.py` | 較正・評価データ。**実データで作る**（乱数だとレンジが桁違いになる） |
| S2〜S4 | `p2b_ops.py` | 前処理と量子化の本体。spec（`work/ops/<name>/<name>.spec.json`）を読む |
| ↳ S2 | `preprocess.py` | C-9 / C-10 / C-11 / 左右入替などの Graph Rewrite |
| ↳ S4 | `quantize_qdq.py` | ORT の QDQ 量子化 |
| ↳ S4 検査 | `weight_percg.py` | **重みが本当に per-channel になったか数える** |
| ↳ S4 修正 | `fix_norm_gamma.py` | **C-13。RMSNorm の gamma を符号付き16bitに収める**（自動） |
| 後処理 | `downgrade_actact_rhs.py` | **C-8。act×act の MatMul の右辺を uint8 に落とす** |
| S5〜S7 | `scripts/build_context.sh` | DLC 変換と context 生成（**約1時間**） |
| 検査 | `dlc_audit.py` | **DLC の型とエンコーディングを突き合わせる**（push 前ゲート） |
| S8・配備 | `deploy_pm64.py` | 包む → md5 3点照合 → push → probe。**上の検査を内包する** |
| 測定 | `device_accuracy.py` | 実機で推論して fp32 と比べる。**生出力を残す** |
| 測定 | `ab_host_device.py` | **fp32 / ホストQDQ / 実機 の3者比較**（主指標はホスト↔実機） |

### 最低限の手順

```bash
python -m app.pipeline.p2b_ops prepare --name <name> --scheme a16w8_rms_swap
python -m app.pipeline.downgrade_actact_rhs \
    --qdq work/ops/<name>/<name>_a16w8_rms_swap_qdq.onnx \
    --prepped work/ops/<name>/<name>_prepped.onnx \
    --out work/ops/<name>/<name>_final_qdq.onnx
# ★ここでホストに読ませて動くか確かめる（過去に壊れた QDQ を実機へ送ったことがある）
bash scripts/build_context.sh work/ops/<name>/<name>_final_qdq.onnx work/ops/<name>/ctx <tag>
python -m app.pipeline.deploy_pm64 --tag <name> --ctx-sub ctx --short <tag> \
    --qdq work/ops/<name>/<name>_final_qdq.onnx --model-name <name>_epctx.onnx
python -m app.pipeline.device_accuracy --model models/ops/<name>_epctx.onnx ...
```

---

## 2. 端末で使うもの

| ファイル | すること |
|---|---|
| `make_rope_table.py` | RoPE の cos/sin テーブル。**モデルごとに head_dim が違う**（Qwen2.5 は 64 / Qwen3 は 128） |
| `make_tokenizer_assets.py`（別プロジェクト） | 端末側トークナイザの資材 |

---

## 3. 調べるための道具（本番の流れには入らない）

| ファイル | 何を分けるか |
|---|---|
| `oracle_qdq.py` | 量子化を**部分的に外して**、どこが効いているかを見る |
| `split_layer.py` | 本物の層を **Attention 側 / FFN 側**に割る。**本物の activation で較正する** |
| `split_layers_n.py` | 層ごとに切り出す。**伝播とその層が作った分を分ける** |
| `decode_error_profile.py` | decode の誤差を (ステップ × 層) で見る |
| `sliding_decode_host.py` | **端末と同じ生成ループをホストで再現**する |
| `sink_compare_host.py` | attention sink が効くかを **fp32 だけで**判定する |
| `context_policy_host.py` | 文脈の保持方針（sink / 固定 prefix）を比べる |
| `prefix_recall_test.py` | **先頭に置いた事実を後で思い出せるか**を1トークンで測る |

### 使えないと分かった道具（同じ道を辿らないための記録）

| ファイル | なぜ使えないか |
|---|---|
| `qnn_cpu_compare.py` | QNN の **CPU backend は `Gather` も `RmsNorm` も持っていない**。このモデルでは組めない |
| `break_rmsnorm_fusion.py` | **融合は崩せなかった**。真因は融合ではなく gamma のエンコーディング不整合（C-13）だった |
| `fix_cache_qparams.py` | past/present の qparam 統一。**効果が無いと分かった**（見えた改善は1プロンプトの偶然） |

---

## 4. この順番を守る理由

**高価な工程の前に検査する。** context 生成は約1時間かかる。

```
S4 量子化の直後   … per-channel の本数 / C-13 のはみ出し   ← ここで止める
S5 のあと         … DLC の型とエンコーディング             ← ここで止める
push の前         … md5 3点照合                            ← ここで止める
```

**ONNX の検査と DLC の検査は両方要る。** 融合されると ONNX のノードは消えるので、
ONNX だけ見ていても気付けない（C-13 はこれで発見が丸一日遅れた）。

---

## 5. 落とし穴（実際に踏んだもの）

| | |
|---|---|
| `ByteSize()` は 2GB 超で**例外を投げる** | 上限チェックのつもりの行が落ちる。`preprocess._fits_protobuf()` で包む |
| サイズ判定は **`.onnx` だけでなく隣の `.data` も数える** | 外部データ形式のモデルを「小さい」と誤判定する |
| 名前の**列挙は、モデルが変わると壊れる** | C-13 が Qwen3 の `q_norm` / `k_norm` を取りこぼした。「その性質を持つもの」で拾う |
| scale / zero_point の初期化子は**他ノードと共有されている** | その場で書き換えると無関係なテンソルを巻き添えにする。新しい初期化子を作る |
| `convert.log` は**当てにならない** | 起動時の警告が混ざる。中身は `qairt-dlc-info` で見る |
| 速度は**発熱で大きく動く** | 同じモデルが1日で 16〜58ms。比較は必ず同一セッション内で |

詳しくは `DoC/StudyMemo/`（全10章）と `results/P*.md`。
