# EdgeSLM_Android

Sony Xperia 5 III SOG05（Snapdragon 888 / SM8350 / HTP v68）上で、Qualcomm NPU へのSLMポーティング要素技術を習得するための検証プロジェクト。

## 位置づけ

- 目的は**製品構築ではなく要素技術の習得**。NPUの割り当て・量子化・静的形状化・KVキャッシュのNPU向け設計を実測で理解する
- [EdgeSLM](../EdgeSLM)（Python/PC検証）・[EdgeSLM-cpp](../EdgeSLM-cpp)（C++移植）・[EdgeSLM_Offload](../EdgeSLM_Offload)（部分オフロード検証）とは**独立した別プロジェクト**。既存3プロジェクトのファイルは変更しない
- Git管理の対象外
- **主成果物はコードではなく実測記録**（`DoC/フェーズ記録_Pn.md`）
- **2026-09-19 以降は参照専用。** 次の検証（Qwen2.5-1.5B の搭載）は [EdgeSLM_AndroidScale](../EdgeSLM_AndroidScale) で行う

## 最新の調査（2026-09-17・**Qwen3 調査はクローズ**）

**[P14: Qwen3-0.6B 移植調査のクローズ](results/P14_20260917_qwen3_closing.md) — 到達点・確定した制約・未解決の整理。**
Qwen3-0.6B は **28層が実機で全ノード NPU 動作**（872MB / 95.8ms per token / fp32↔実機 top-1 75.0%）し、
fp32 では **System Prompt を守り会話履歴も参照できる**（Qwen2.5 は不可）。
しかし実機の自由対話は崩れる。原因は HTP の計算間違いではなく、**位置0 の巨大活性（8142）に合わせた
per-tensor の粗い格子**で、後続トークンが実質3〜4段でしか表現できず、Host と HTP の **±1〜2段**の
丸め差が信号の数十%になること。**v68 の C-8 と per-tensor A16 の組み合わせでは解けなかった**ため、
Qwen2.5-0.5B を Reference Model として残しクローズした。

以下は調査の全経過（新しい順）。

[P16: 「実機がよければよい」の検証](results/P16_20260917_single_model_4x.md) — クローズ後の追試。Host で唯一きれいだった 4x は**位置0 だけ baseline モデルで守る2モデル構成**に依存しており、実機（context binary 1本）では作れない形だった。1モデルにすると**層2 の scale ただ1つの変更で完全崩壊**（1位保持 85.7%→**0.0%**、KL 0.195→18.24、文字化け）。位置0 の 8142 が飽和し、KV の先頭スロットとして最後まで汚染が残るため。**実機ビルド140分の手前で止められた。**

[P15: 引き継ぎ・クローズ・デモ整備](results/P15_20260917_takeover_closing_demo.md) — 別セッションから引き継いだ時点の状態（3件とも失敗/中断）と切り分け（条件A の失敗は介入内容ではなく float タップによる部分モデルの作り方／条件B は C-8 で構成自体が不可）。クローズに伴う記録の整備と、デモアプリの不具合修正（**モデル切り替えで会話がリセットされず、system prompt 抜きの差分を新モデルへ渡していた**）。実機で Qwen2.5 26.6 tok/s・Qwen3 10.4 tok/s を確認。

### 第3段階: 層帯の絞り込みと最小介入（P13a〜P13d）

[P13d: L11-17 介入の実機検証](results/P13d_20260917_v_l11_17_device_probe.md) — **条件B は v68 で成立せず**（C-8 でグラフ検証失敗、`node_matmul_23`）。**条件A は Host↔HTP 差を改善せず**（L17 write 1.74%→2.09%、L18 入口 8.23%→8.92%）→ 28層への適用は No-Go。そもそも L11-17 の V は Host↔HTP では既に一致率 99.6%以上・clip 0% で、**P13a/P13b が見つけた「V の有害状態」は Host が FP32 からずれる問題であって HTP のずれではない**（2軸が別物だった）。残る HTP 差は層境界の残差（L18 入口 8.23%、scale 0.113）に集中し、P12j の add_26 と同じ構造。

[P13c: L11-17 の V だけへの最小介入](results/P13c_20260917_v_l11_17_intervention.md) — baseline のまま L11-17 の V 量子化だけを変更。条件A（range拡張）・条件B（8bit→16bit）とも **Host 自由生成を壊さず**、3ターンとも「音楽をかけて」に正答。1位保持 85.7%→87.5%/89.3%、点A は条件Bで −0.91→+0.35 と正転、バナナは条件Aで +4.45→+6.06。このセッションで他の変更が軒並み turn1 を壊してきたのとは対照的。

[P13b: V は L11-17 に集中、K は L0-6×L14-20 の相互作用](results/P13b_20260917_kv_combo_swap.md) — V は L11-17（7層）だけで全28層一括（+2.202）を上回る回復（**+2.349**）。K は L0-6 単独では悪化（−1.544）だが **L14-20 と組むと符号反転して改善**（+1.064）。L7-13・L21-27 との組では反転しない。

[P13a: layer block × K/V 分解](results/P13a_20260917_layer_block_kv_swap.md) — V は L7-13（+1.27）・L14-20（+1.88）が主で分散型かつ概ね加法的。K は**ブロック合計 −0.744 に対し全K一括 +1.860 と符号すら合わない**強い相互作用。

### 第2段階: 真因は KV の累積差（P12u〜P12z）

[P12z: 大域 K/V swap ゲート試験](results/P12z_20260916_global_kv_swap_gate.md) — KV を丸ごと入れ替えると **K だけでも V だけでも判定が負→正に反転**。局所 swap が3回連続で失敗していたのと対照的で、「累積した KV 状態そのものが判定を動かす」が強く支持された。

[P12y: position45・layer14 の V 局所 swap](results/P12y_20260916_local_swap_pos45_L14V.md) — P12x で見つけた局所異常（oracle だけ L2 誤差 71.7%）を1点だけ差し替えても、3条件とも判定への効果は無視できる大きさ（oracle はむしろ +0.0188 と良化方向）。

[P12x: KV 累積差を position×layer×K/V で追跡](results/P12x_20260916_kv_trajectory_probe.md) — 誤差は79ステップかけて育つのではなく**位置1-2 で既に大きい**（baseline 位置1・L27 の V で 93.9%）。V は K より一貫して大きい。3条件の誤差の大きさはほぼ同水準なのに生成品質は大きく違う＝**大きさより向き**（P12h と整合）。

[P12w: add_26 の因果プローブ](results/P12w_20260916_add26_causal_probe.md) — linear_20 の量子化を完全に除いても add_26 は FP32 にほとんど近づかず（31.06%→29.83%）、**この1位置を差し替えるだけでは判定が動かない**。真因は上流か KV の累積差にあると判断。

[P12v: linear_20 oracle（量子化を外す）](results/P12v_20260916_linear20_oracle.md) — QDQ を1組外して完全精度を通すと **baseline より明確に悪化**（1位保持 85.7%→76.8%、点B +3.35→−1.02、turn1 は「エアコンを鳴らすか？」）。「細かくすれば直る」という仮説の反証。

[P12u: D0'（FP32 位置0 ＋ KV range 修正）](results/P12u_20260916_d0prime_fp32pos0_kvwide.md) — 旧 D0 の悪化が KV の clip との交絡ではないかという仮説は**否定**。KV-wide16 を足しても turn1 は旧 D0 とほぼ一字一句同じ失敗文のまま。

### 第1段階: 発生源の特定と range 調整 → 打ち切り（P12m〜P12t）

[P12t: セッション総括](results/P12t_20260916_session_summary_layer2_htp_tradeoff.md) — P12l〜P12s の総括。**range 幅だけを動かすアプローチは打ち切り**。教訓: 集計指標は自由生成の健全性を予測しない／teacher forcing と自由生成は別物／range と品質の関係は非単調／失敗の形は条件ごとに違う。

[P12s: 4x range の実機評価](results/P12s_20260916_layer2_linear20_4x_device.md) — Host を壊さない唯一の 4x を実機へ。add_26 の Host↔HTP 誤差は 21.95%→**18.67%** にとどまり、事前基準（15%以下）に届かず。

[P12r: linear_20 の range 掃引（1x〜8x）](results/P12r_20260916_linear20_range_sweep.md) — 7点中、teacher forcing と自由生成の両方で安定したのは **4x だけ**。3.75x は集計指標が全変種中最良なのに自由生成は失敗。**単調な関係ではなく孤立した安定点**。

**P12q: 層2 だけの変種（L2-M / L2-L / L2-ML）** — 記録 md は未作成（データは `results/P12q_20260916_layer2_only_variants/`、内容は P12t に集約）。L2-ML は 1位保持 87.5% と baseline 超えなのに**自由生成は崩れる**という、指標と生成の乖離を最初に示した実験。

[P12p: mul と linear の分離](results/P12p_20260916_mul_linear_split.md) — 全体劣化の主犯は **linear 側**（1位保持 76.8%）。mul 単独は集計指標では baseline 並みか一部上回る。ただし「音楽」「BANANA」の失敗は**どちらか一方の問題ではない**。

[P12o: 位置0 だけ既存範囲・位置1以降を narrow](results/P12o_20260916_pos0_wide_pos1_narrow.md) — P12n の全面崩壊は**位置0 を保護すれば消える**（1位保持 80.4%）。ただし baseline 相当には戻らず、点B が +3.35→−0.43、自由生成でも turn1 が英語で自問自答を始める。

[P12n: 28層へ機械的に拡張](results/P12n_20260916_mlp2_all_layers_later.md) — 同型56テンソルへ広げ全位置に適用すると **Host だけで完全崩壊**（1位保持 0.0%、単語にならないトークン列）。136分の本番ビルド前に Host で止められた。

[P12m: 層2 の2本だけ narrow 化 → 実機で大きく改善](results/P12m_20260916_layer2_kv_narrow.md) — `mul_41`/`linear_20` だけを後続用範囲にすると、add_26 の Host↔HTP 誤差が **21.95%→8.61%**。「発生源に絞った調整」は効くことを実機で確認。

### 前段: 量子化レンジの調査（P12c〜P12l）

[P12l: KV範囲を広げる実験](results/P12l_20260916_kv_range_recal.md) — KV系252グループを全部広げるとbaselineが悪化（1位保持85.7→78.6%、おうむ返し）。主因はC-8で8bitにしたGQA複製の28グループ。16bitのみ広げると1位保持87.5%・margin保持51.8%まで戻るが、点Bと日本語System Promptは悪化。飽和を消す利得と8bitの分解能がトレードオフ。

[P12k: 位置0の巨大値とFP32事前計算prefix、KV入力範囲の飽和](results/P12k_20260916_fp32_prefix.md) — 巨大値はトークンによらず位置0だけ（チャットでは常に8142.7）。FP32位置0事前計算＋後続用範囲は自由生成3ターンが通ったが最悪marginは悪化。system全体の事前計算は集計最良でも自由生成失敗。合否は各回答の最初のトークンのmarginと対応。新発見: KVキャッシュの入力範囲が狭く、対話のほぼ全位置で新しいKVが飽和（「22度」の「2」でpast_v_14の8倍）。

[P12j: Host↔HTPの差の発生源は層2 MLP](results/P12j_20260916_layer2_htp_probe.md) — 層2を切り出して実機と整数コードを比較。層2 MLPのmul_41（SiLU×up）とlinear_20（down_proj）は先頭の巨大値に合わせたscaleのため、後続トークンではほぼ3値（±1段以内が93.5%/86.4%）。整数差は最大2〜3段なのに実数誤差16.8%/55.1%。RMSNormは実機で融合され中間テンソルが消える。

[P12i: 範囲変更を1ブロックずつ戻すA/B](results/P12i_20260915_block_revert_ab.md) — 7ブロックのどれを戻しても全評価点でbaseline以上にならず、効果は強く非加法的（点B: 加法予測+0.93、全部戻す実測+3.35）。点Bを最も戻したのは層2 MLPの2本（巨大値の発生源）。層14-27の範囲変更は単独では点Bにほぼ無関係。評価点56では変種間の差の大半が1〜3点分。

[P12h: KV誤差の向き](results/P12h_20260915_kv_error_direction.md) — 層ごとの大きさ・attentionへの効き方ではfocusedの方が良い。判定「音−エ」への一次近似（FP32勾配×attention出力のずれ）では全層 baseline −2.23 / focused −3.71（実測 −1.46 / −3.43）。差は層15-21で、focusedは誤差は小さいが判定を悪くする向きの成分が増えた。

[P12g: 逆転はKV由来・決め手は層14-20](results/P12g_20260915_kv_step_swap.md) — KV×最終ステップの3×3で、focusedが積んだKVだけで「音」−「エ」の差が5.07→1.64。層14-20のKVだけFP32に戻すと3.63まで回復。focusedのKVは全層でbaselineよりFP32に近いのに答えを壊す（誤差の向きが効く）。

[P12f: 自由生成が分岐する瞬間の解剖](results/P12f_20260915_divergence_probe.md) — focused_splitの「音楽」崩れはclipではない（168本すべて0、陽性対照あり）。FP32は同じ履歴でも「音」を選ぶ。focusedは活性誤差が小さいのに逆転し、Host QDQ自体が残差をFP32から約33%ずらしていることが判明。

[P12e: 先頭・後続の量子化範囲分離を実モデルで検証](results/P12e_20260915_temporal_qdq.md) — 168組の限定分離でHostの残差誤差と予測一致率は改善したが、自由生成では音楽の指示にエアコンの説明を反復。先頭を既存モデルに戻しても解消せず、実機への適用は見送り。

[P12d: 対話・生成・KV256入替後の残差分布](results/P12d_20260915_dialogue_activation_ranges.md) — 評価2,119tokenで巨大値は先頭だけ。後続用の1.5倍範囲候補は観測した層3・層6で範囲外0。チャット開始値は約8,140で既存上限を超えるため、先頭側も較正の見直しが必要。実機精度の改善は未検証。

[P12c: 残差activation最大値7257の追跡](results/P12c_20260915_activation_outliers.md) — 公式FP32でも先頭tokenのchannel 35に巨大値が発生することを確認。Hostと保存済み実機出力の整数差を再計算し、add_53の31.53%誤差を再現。量子化方式の改善は次の検証課題。

## ドキュメント

**これから同じことをやる人へ**: [DoC/検証環境と手法.md](DoC/検証環境と手法.md) に、必要なソフト・SDK・変換手順・精度の測り方・落とし穴・工数の目安を1枚でまとめてある。


| 文書 | 内容 |
|---|---|
| **[results/P2b_20260912_op_matrix_host.md](results/P2b_20260912_op_matrix_host.md)** | **Transformer 構成要素の op 対応表（ホスト）**。v68 で載らないのは「活性化×活性化の MatMul を 16bit で」の1点だけ、という特定と、A8 / 混合精度という回避策の実証 |
| **[../EdgeSLM_AndroidDemo/results/D2_20260913_tokenizer_verify.md](../EdgeSLM_AndroidDemo/results/D2_20260913_tokenizer_verify.md)** | **★トークナイザの端末移植と HF との一致検証**。`NFC` 正規化の欠落（27/31）を見つけて修正し、端末で **encode 31/31 / decode 往復 31/31**。これで「ローカル完結」が数値で裏付けられた |
| **[results/P7_20260913_decode_cache_io.md](results/P7_20260913_decode_cache_io.md)** | **★キャッシュ I/O 仮説は「単発では外れ・ループでは当たり」**。機構は再量子化ではなく**範囲の食い違いによる clip**。量子化パラメータを揃えて**24ステップ生成の一致が 9/24 → 19/24**。P6 の「精度不足」は**評価セットの偏り**による誤りで、実機 top-1 は **60.4%** |
| **[results/P6_20260913_decode.md](results/P6_20260913_decode.md)** | **★decode（seq=1 + KVキャッシュ）が実機で動いた**。「`rms_norm` 融合が作れない」は**誤診**で、真因は **KVキャッシュの float 入力が `Slice` に入って量子化されないこと**（C-1）。順序を変えるだけで通った。`could not create op` の因果の読み方も |
| **[results/P5_20260913_window64.md](results/P5_20260913_window64.md)** | **★窓を 16→64 に拡張**。`lm_head` を最終位置だけに掛けて logits を 38.9MB→0.6MB に。**窓の外にある情報を復元できるようになった**（`8317` の例）。一方で**実機側の誤差が窓と共に増える**ことを定量化（29.0%→46.5%） |
| **[results/P4_20260913_prefillmask.md](results/P4_20260913_prefillmask.md)** | **マスク入力版 prefill の実機検証**。マスクをグラフ入力にするのは v68 で可。ただし**短い入力の答えはほぼ変わらなかった**（6/8 → 6/8）。同じ推論結果が測る位置で **24.2% / 63.3% / 75.0%** に見える。**err 1002 の真因は生成途中のバイナリ**だった |
| **[results/P4_20260913_generation.md](results/P4_20260913_generation.md)** | **★端末の NPU で文章を生成した記録**。`The capital city of France is` → **` Paris.`**、fp32 と 6/6 一致。パディングを品質要因と見たこと（**[P4_…_prefillmask.md](results/P4_20260913_prefillmask.md) で撤回**）、decode（seq=1+KV）が v68 で作れなかったことも |
| **[results/P3_20260913_full24_device.md](results/P3_20260913_full24_device.md)** | **★フル24層が実機で動いた記録**。772MB が全ノード HTP、prefill 16トークン 37ms、実機 top-1 73.3%。C-12（`embed_mode=0`）の発見も |
| **[results/P3_20260913_rmsnorm_fix.md](results/P3_20260913_rmsnorm_fix.md)** | **24層破綻の真因は RMSNorm だった（C-11）**。oracle 実験で「残差は 0 効果 / RMSNorm で 6.2%→74.4%」と確定し、スケール不変性を使った等価変形で 4.5%→80.7% |
| **[results/P3_20260913_full24.md](results/P3_20260913_full24.md)** | **24層への到達と、そこで現れた壁**。2GB 問題は opset 21 export で解決したが、**24層は残差の massive activations で数値的に破綻**（ホストの時点で）。v73 でも解決しない可能性 |
| **[results/P3_20260913_error_decompose.md](results/P3_20260913_error_decompose.md)** | **誤差の内訳**。fp32→ホストQDQ→実機 に分解。**HTP 側の誤差が量子化より大きい**が、HTP が本質的に不正確なわけではない（重み固定 MatMul なら 0.11%） |
| **[results/P3_20260913_device_accuracy.md](results/P3_20260913_device_accuracy.md)** | **実機での精度**（runner v0.3）。**ホストのシミュレーションは楽観的だった**（top-1 80.7% → 実機 **63.1%**）。ただし方式の順位は保たれ、左右入替は実機でも +16.5ポイント |
| **[results/P3_20260913_mixed_matmul.md](results/P3_20260913_mixed_matmul.md)** | **C-8 は思っていたより狭い制約だった**。`u16×u8` は v68 で通る（非対称）。top-1 一致 50.0%→**66.5%**、fp32 top1 の順位 20.6位→**1.95位**。壊れているのは **layer 0 だけ**という発見も |
| **[results/P3_20260912_endtoend.md](results/P3_20260912_endtoend.md)** | **end-to-end の精度と容量**。**440MB の本物の LLM（埋め込み+2層+lm_head）が NPU 全ノードで動作**。ただし次トークン予測の top-1 一致は **85.2%（A16W8素）→ 50.0%（v68 で動く形）** |
| **[results/P3_20260912_accuracy_layer1.md](results/P3_20260912_accuracy_layer1.md)** | **本物の Qwen2.5-0.5B 層1枚の精度**。実機で動くが誤差 30.4%。原因は **C-8 の回避で attention スコアを 8bit にせざるを得ないこと**と特定。試した対策3つと、効かなかった理由も記録 |
| **[results/P2b_20260912_real_form.md](results/P2b_20260912_real_form.md)** | **本物の export 形で通した記録**。手書きモデルが4箇所外れていたこと、**C-9**（マスク定数 -3.4e38）と**C-10**（v68 は `Expand` をロードできない）の発見、C-1 検証器の穴の修正。実機 22/22 |
| **[results/P2b_20260912_op_matrix_device.md](results/P2b_20260912_op_matrix_device.md)** | **同・端末実行**。15/15 `pass_strict`。**Transformer 1ブロックが KVキャッシュ込みで NPU 全ノード動作**。この規模の所要時間は固定費であって性能ではない、という読み方の注意も |
| **[DoC/StudyMemo.md](DoC/StudyMemo.md)** | **学習メモの目次**。6編に分割: [01 レイヤ構造とツールチェーン](DoC/StudyMemo/01_レイヤ構造とツールチェーン.md)（ONNX / ORT / QNN EP / HTP の切り分け・EPContext）/ [02 量子化の基礎](DoC/StudyMemo/02_量子化の基礎.md)（QDQ の数式・`step=42.7`・per-tensor / per-channel・A16W8・`u16×u8`）/ [03 Transformer の中で何が起きたか](DoC/StudyMemo/03_Transformerの中で何が起きたか.md)（residual stream / RMSNorm / massive activation）/ [04 原因の特定と指標の読み方](DoC/StudyMemo/04_原因の特定と指標の読み方.md)（Oracle 実験・**指標の罠3つ**）/ [05 Graph Rewrite](DoC/StudyMemo/05_GraphRewrite.md)（等価変形4例・ε の扱い）/ [06 ログの読み方と原則](DoC/StudyMemo/06_ログの読み方と原則.md)（**実ログ6例**・6つの原則） |
| **[DoC/試行錯誤の記録.md](DoC/試行錯誤の記録.md)** | **何を試してなぜ失敗しどう直したか**。誤りを10の型に整理（「成果物ができた＝成功」6件／症状が真因を示さない4件／見る場所を間違えた6件 等）。**効かなかった対策**も再試行防止のため記録 |
| **[DoC/検証結果まとめ.md](DoC/検証結果まとめ.md)** | **最初に読む文書**。到達点・実測値（性能/容量/熱/精度）・**v68 制約 C-1〜C-12**・**踏んだ誤りの型**・未決事項・成果物の地図を横断的に集約 |
| [DoC/要件定義書.md](DoC/要件定義書.md) | 目的・スコープ・機能要件・フェーズ計画。3.3節に事前調査で確認した技術前提を集約（**3.3.1 固定事実＝再調査不要 / 3.3.2 バージョン依存事実＝使用版変更時に再確認 / 3.3.3 P1のビルド要件**）。**3.4節は v68 で失敗する禁止事項 C-1〜C-7 + エラーコード表**で、実装前の必読箇所 |
| [DoC/基本設計①_計測ハーネス設計.md](DoC/基本設計①_計測ハーネス設計.md) | FR-1 / FR-11 の実装設計。計測スキーマ・測定プロトコル（単発測定の禁止）・全レコードに付与する測定条件。**4章のデータ源は P0-a の実測待ちで TBD** |
| [DoC/基本設計②_量子化・変換パイプライン設計.md](DoC/基本設計②_量子化・変換パイプライン設計.md) | FR-4 / FR-6 / FR-7 / FR-13 の実装設計。fp32 ONNX → QDQ → DLC → context binary → EPContext の工程定義、**C-1〜C-5 の機械的検証**（5章）、Transformer 構成要素の重点検証計画（6章）。**ホスト側のみで端末に依存しない** |
| [DoC/基本設計③_実行用アプリ設計.md](DoC/基本設計③_実行用アプリ設計.md) | 端末側の実行形態。**UIなし最小 Service + JNI + C++ 推論コア**の構成、ジョブ仕様、ファイル配置、段階的な作り方。**設計の主軸は「再ビルドを要する変更を最小化する」**（Gradle ビルドがユーザー実行になるため） |
| [DoC/参考実装_ZipDepth解説.md](DoC/参考実装_ZipDepth解説.md) | **ZipDepth とは何か・ソース構成・NPUを使う設定の実体**。**2章に用語集**（Gradle / APK / AAR / NDK / JNI / FastRPC / skel / EPContext 等）。6章に P1 の実測値。後から「これ何だったか」を思い出すための文書 |
| [DoC/環境構築記録.md](DoC/環境構築記録.md) | **環境構築の全記録**。配置・バージョン・再現手順に加え、**4章「つまずいた点と解決」**（QAIRT が Windows では Python 3.10 必須である件など8件）と付録の再構築チェックリスト |

## 技術構成（第一ルート）

```
Qwen2.5-0.5B
  → static shape ONNX（prefill/decode 分離・固定長KVキャッシュ）
  → QDQ量子化 A16W8 + 混合（attention の活性化×活性化 MatMul のみ A8）※P2b の実測で確定
  → qairt-converter
  → qnn-context-binary-generator --htp_socs sm8350
  → gen_qnn_ctx_onnx_model.py（EPContext ONNX 生成）
  → ONNX Runtime + QNN EP（htp_arch=68, soc_model=30）
  → HTP v68 / Hexagon 780
```

ExecuTorch + QNN は**実現性判定のオラクル兼フォールバック**として併用する（要件定義書2.2節）。

## バージョン baseline（3本持つ）

| 用途 | baseline |
|---|---|
| P1: ZipDepth 再現 | QAIRT **2.48** + ORT 1.27.0（実証構成の固定） |
| P2以降: 標準検証 | QAIRT **2.49.40** + onnxruntime-qnn v2.5.0 |
| MatMulNBits / W4 評価 | QAIRT **2.50.40** + onnxruntime-qnn **v2.6.0**（HTPが MatMulNBits の uint16/int16 QDQ 活性化に対応） |

## フェーズ

| | フェーズ | 性格 |
|---|---|---|
| P0 | 足場（端末プロファイル・NDK・計測・QAIRT入手） | — |
| P1 | ZipDepth 再現による配管確認 | **Gate** |
| P1.5 | ExecuTorch v68 recipe で Qwen2.5-0.5B を移植 | 実現性スパイク |
| P2 | ORT 本線パイプライン確立 | **Gate** |
| P2b | Transformer 構成要素の重点op検証 | **最重要Gate** |
| P3 | CPU / HTP / 熱の比較 | 判断点 |
| P4 | Decode グラフ（AR-1） | **Gate** |
| P5 | Prefill グラフ（AR-N）+ Hybrid | **Gate** |
| P6 | 最適化（A8W8 / 混合 / W4） | 任意 |

各 Gate の合格条件は要件定義書 7.1節に数値基準として定義してある。

## 押さえておくべき制約（詳細は要件定義書3.3節）

- **HTP v68 は整数量子化が基本**（fp16 は v69以降）
- **v68 の MatMul は `u16×u16` と `u8×u16` が不可、`u16×u8` は可**（C-8。非対称）。8bit にできるのは常に右辺なので、`A×B=(B^T×A^T)^T` でレンジの狭い方を右辺へ置く
- **v68 は `Expand` を含む context binary をロードできない**（C-10）。HuggingFace の GQA 複製が `Expand` を出すため**実モデルでは必ず踏む**。`Concat` へ書き換える
- **因果マスクの `-3.4e38` は量子化できない**（C-9）。有限値へ置き換える
- **動的形状は非対応** → 静的形状化と固定長KVキャッシュが必須
- **LPBQ / 16a4w_block は V69以上** → v68では使えない。通常の 16a4w は別方式で使用可能
- **weight sharing の公式対応は v73以上 + SA8295P** → SM8350 では使えない前提（worst case = 重み2コピー）でメモリ設計し、実効可否は P5 で確認
- **メモリ制約は階層で扱う**（HTP PD / FastRPC-SMMU / VTCM / システムRAM）。v68 で数百MB級の context・shared weight がロード失敗した事例はあるが**固定的な閾値ではない**（1Bクラスが動いた報告もある）。SOG05 の実効上限は未確定で FR-8 で実測する
- **SOG05 の VTCM は 4MB**（SA8295 は同じ v68 でも 8MB）。これは tiling・spill/fill・性能の制約であり、**PD容量とは別レイヤ**
- **GroupQueryAttention は HTP 非対応**（GPUのみ） → まず ONNX export 時に fused GQA を出さない方針を取る
- **ORT 自身の x86 EPContext 生成は v68 で失敗する**（未修正）→ QAIRT の `qnn-context-binary-generator` を使う

## 開発環境（2026-09-12 構築）

| 項目 | 場所 | 状態 |
|---|---|---|
| adb / fastboot 37.0.1 | winget 管理（ユーザーPATHに登録済み） | ✅ |
| Android cmdline-tools 15859902 | `D:\Android\sdk\cmdline-tools\latest` | ✅ SHA-256 一致確認 |
| Temurin JDK 17.0.20.1 | `D:\Android\jdk17` | ✅ 管理者権限不要のZIP版。**システムの JAVA_HOME は変更していない**（既定の `java` は Oracle 22 のまま） |
| Android SDK パッケージ（platform-tools / android-34 / build-tools 34.0.0 / cmake 3.22.1 / **NDK 21.4.7075529**） | `D:\Android\sdk` | ✅ 5パッケージすべて `package.xml` 付きで導入確認。**arm64 クロスコンパイルの動作も検証済み**（ELF `e_machine=0xB7`） |
| QAIRT SDK 2.48.0.260626 | `D:\QAIRT\v2.48.0.260626\qairt\2.48.0.260626` | ✅ 2,157 MiB。`qairt-converter` / `qnn-context-binary-generator` 等の起動確認済み |
| QAIRT 専用 Python 環境 | `D:\QAIRT\qairtenv` | ✅ **Python 3.10.18**（3.12 は不可。要件定義書3.2節） |
| プロジェクト本体 venv | `EdgeSLM_Android\.venv` | ✅ Python 3.12.11 / onnxruntime 1.30.0 |

### Android SDK パッケージの導入（ユーザー実行が必要）

Claude Code の実行環境では **JVM プロセスだけが外部TCP接続できない**ため（curl/Python は可）、`sdkmanager` と Gradle の依存解決は代行できない。以下をユーザー自身のターミナルで実行する。

```bash
JAVA_HOME="D:/Android/jdk17" "D:/Android/sdk/cmdline-tools/latest/bin/sdkmanager.bat" --sdk_root=D:/Android/sdk "platform-tools" "platforms;android-34" "build-tools;34.0.0" "cmake;3.22.1" "ndk;21.4.7075529"
```

この制約により、**P1 の APK ビルドはユーザー実行**となる。一方 P2 以降の本線（ORT + QAIRT パイプライン）は Python とネイティブ exe で構成されるため代行可能。

## スクリプト

| スクリプト | 用途 |
|---|---|
| [scripts/p0a_collect.sh](scripts/p0a_collect.sh) | **P0-a**: 端末素性の一括取得。thermal zone 構成・QNN/FastRPC ランタイムの在否・`/data/local/tmp` の実行可否など、基本設計①の入力を1回で収集し `results/P0a_<timestamp>.txt` に保存する |
| [scripts/p0b_ndk_hello.sh](scripts/p0b_ndk_hello.sh) | **P0-b**: NDK で arm64 バイナリをクロスビルドし、ELFヘッダを検査して端末に転送・実行する。PC側のビルド検証は端末なしでも動く（端末未接続なら転送をスキップ） |
| [scripts/p0c_fastrpc_probe.sh](scripts/p0c_fastrpc_probe.sh) | **P0-c**: `adb shell` から FastRPC(CDSP) を開けるかの実測。**R-9（CLI から NPU が使えない）の根拠**を取る |
| [scripts/p1_verify_zipdepth.sh](scripts/p1_verify_zipdepth.sh) | **P1準備**: ZipDepth の Git LFS 実体を `DEPENDENCIES.md` の SHA-256 と照合し、ビルド前提（NDK/SDK/CMake/JDK）を確認する。期待値はハードコードせずリポジトリから読む |

端末を接続したら次を実行する。

```bash
bash scripts/p0a_collect.sh
```

## 実装

| 場所 | 内容 |
|---|---|
| [android_runner/](android_runner/README.md) | **EdgeSLM Runner v0.1**（基本設計③の実装）。UIなし Service + JNI + C++ 推論コア。`probe` / `bench` を実装。**モデルとジョブ仕様は APK の外**に置き、差し替えに再ビルドを要さない |
| [scripts/run_runner.sh](scripts/run_runner.sh) | Runner をホストから駆動する（`install` / `push-model` / `run`） |

## 外部参照物

| 場所 | 内容 |
|---|---|
`third_party/Edge_ZipDepth-SM8350/` | [ZipDepth-Quantized-Qualcomm-QNN-SM8350](https://github.com/meh301/ZipDepth-Quantized-Qualcomm-QNN-SM8350) のクローン（322MB、Git LFS 実体込み）。**SOG05 と同じ SM8350 / HTP v68 で NPU 推論が動いている唯一の公開アプリ**で、P1 の検証対象かつ変換パイプライン（`tools/variants/`）の参照実装。`local.properties` は作成済み、LFS 実体は SHA-256 で全13件照合済み |

## 現在の状況

2026-09-12 時点。**P0 / P1 / runner v0.1 / P2 まで完了**。

```
✅ P0    足場（端末プロファイル・NDK・計測ハーネス設計・QAIRT導入）
✅ P1    配管Gate    — ZipDepth（第三者製）が SOG05 で NPU 動作    OrtRun 中央値 ≒39ms
✅ v0.1  自作 runner — 同じモデルを自作コードで実行                median 35.77ms
✅ P2    パイプライン — 自作モデルを自作変換で作り NPU 実行         median 0.277ms
✅ P2b   Transformer 構成要素の重点検証   **実 export 形**を含む端末 22/22 合格
✅ P3    本物の Qwen                **全24層 772MB が実機動作**（37ms / 実機 top-1 73.3%）
🔶 P4    文章生成                   **端末だけで生成成功**（fp32 と 6/6 一致）。KVキャッシュ decode は未達
```

**P2b でいちばん重要な発見**: v68 で Transformer が載らない理由は、いま判明している
範囲では **1点だけ**だった。**活性化 × 活性化の MatMul を 16bit で受け付けない**
（`has incorrect Value 68, expected >= 73`）。これは attention の QK^T と PV に直撃する。
**その2箇所だけ A8 に落とせば Transformer 1ブロックが通る**ことまで実証した。
本線の量子化方式は「素の A16W8」から「**A16W8 + 該当 MatMul のみ A8**」へ改める。
詳細は [results/P2b_20260912_op_matrix_host.md](results/P2b_20260912_op_matrix_host.md)。

**「変数を1つに絞る」原則**で、端末・runner・パイプラインの3層の疑いを別々に消した。
以降 Transformer で失敗した場合、原因は **Transformer 固有の事情**（対応op・精度指定・
静的形状・KVキャッシュ）に限定して調べられる。

この過程で v68 の禁止事項は **C-1〜C-7** まで増えた（当初 C-5 まで）。追加分は
**C-6: `useLegacyPackaging = true`**（FastRPC は実ファイルパスを要求する）と
**C-7: `<uses-native-library>` の宣言**（Android 12+ で vendor ライブラリを名前空間に入れる）。
どちらも公式ドキュメントには無く、参考実装との差分から突き止めた。

**現在地と知見の集約は [DoC/検証結果まとめ.md](DoC/検証結果まとめ.md) を読む。**
実測値・確定した制約・**判断を誤った5つの型**をまとめてある。

**P2b は完了し、P3 で本物の Qwen2.5-0.5B のデコーダ層1枚が実機で動いた**（median 0.958ms）。

**容量は足りた**。埋め込み(151,936語彙) + 2層 + lm_head を量子化した **440MB の
EPContext が、全ノード HTP でロードできた**（ロード 4.15秒 / prefill 16トークン 17.4ms）。

**足りないのは精度**で、**原因は C-8（v68 は 16bit×16bit の MatMul 不可）の回避策そのもの**:

| | **ホスト** top-1 | **実機** top-1 | v68 |
|---|---:|---:|---|
| A16W8 素 | 85.2% | — | 動かない |
| **`u16×u8` + 左右入替** | 80.7% | **63.1%** | **動く** |
| 両側 A8（当初の回避策） | 50.0% | 46.6% | 動く |

**実機で測ったら、ホストの数字は楽観的だった**（80.7% → 63.1%）。
ただし**方式の順位は保たれ**、左右入替は実機でも +16.5ポイント効いている。
なお A16W8 素は v68 で動かないため実機値が無く、**精度面での世代差は
この測定からは出せない**（ホスト値と実機値を直接比べてはいけない）。

**C-8 は当初の理解より狭い制約だった。** MatMul の対応は非対称で、
**左辺 uint16・右辺 uint8 なら通る**（`u8×u16` は通らない）。さらに
`A×B = (B^T×A^T)^T` の等価変形で**右辺に置くテンソルを選べる**ので、
レンジの狭い Q を 8bit 側に持ってくると**分解能が 15倍**になる。

結果、**到達不可能な上限 85.2% に対し 80.7%（差 4.5ポイント）**まで回復した。
当初の 50.0% から見れば別物である。実機も 17.42ms → **14.29ms** と速くなった。
なお**壊れているのは layer 0（と 1）だけ**で、層2〜23 は 8bit でも argmax 一致 95.7%。

因果の鎖:

```
C-8 → attention スコアを 8bit にせざるを得ない
    → スコアのレンジ 1627 に対し必要な分解能は 7
      → softmax が壊れる（attention の argmax 一致 29%）
        → 層あたり誤差 5.5% → 30.4%
          → 次トークン予測の top-1 一致 85.2% → 50.0%
```

**これは実装の巧拙ではなく v68 のアーキテクチャ制約**である。詳細と、試して
効かなかった対策3つは [results/P3_20260912_accuracy_layer1.md](results/P3_20260912_accuracy_layer1.md)。

ただし**確定したのは「op が載る」ことだけ**である。使ったのは自作の乱数重み・縮小形状
（hidden 128 / 1ブロック / vocab 512）であり、**精度は一切測っていない**し、
本物の規模（hidden 896 / 24層 / vocab 151,936）も試していない。

次の着手は **実機での精度測定**。ここまでの精度はすべて**ホスト CPU 上の
QDQ シミュレーション**であり、HTP の演算順序・丸めの違いは含まれていない。
P1〜P2b で「ホストで通ることと端末で動くことは別」を3回学んだ以上、精度も実機で
確かめる。runner に出力を返す機能（v0.3）が要るので**ビルドを1回お願いすることになる**。

**未決事項#6（8 Gen 2以降の端末を追加調達するか）の材料は揃った** —
v68 固有の制約2つ（C-8 / weight sharing）はいずれも v73 で解ける一方、
SD888 でも「動く」ところまでは到達しており、要素技術の習得という目的は果たせている。
