# StudyMemo — 目次

| | |
|---|---|
| 目的 | **ログを見て「なぜそう判断できるのか」が自分で説明できる**状態になること |
| 題材 | 2026-09-13 の「24層が壊れて、RMSNorm を書き換えて直った」一連のデバッグ（1〜10章）と、
2026-09-15〜17 の「Qwen3 が実機で崩れ、原因は特定できたが v68 では直せなかった」記録（11章） |
| 関連 | [試行錯誤の記録.md](試行錯誤の記録.md)（何を試したか） / [検証結果まとめ.md](検証結果まとめ.md)（結論） |
| 全体の構成図 | [コンポーネント構成図.html](コンポーネント構成図.html)（2026-10-23。PC の変換パイプライン・受け渡し・端末の実行アプリ・調べる道具を 1 枚に。各部品に対応する章の番号つき。ブラウザで開く） |

1本が長くなりすぎたので分割した。**上から順に読むと積み上がる**が、
知りたい所だけ読んでも通じるようにしてある。

---

## 読む順番

| # | 文書 | 何が分かるか |
|---|---|---|
| **0** | [用語集](StudyMemo/00_用語集.md) | **知らない語をその場で引く。** 分野別の一覧（土台／変換の流れ／量子化／モデルの中身／指標／速さと文脈）・つまずきやすい語の詳しい解説・**C-1〜C-13 の早見**・紛らわしい対・略語表 |
| **1** | [レイヤ構造とツールチェーン](StudyMemo/01_レイヤ構造とツールチェーン.md) | **ONNX / ORT / QNN EP / HTP がそれぞれ何者か。** どこで何が起きているか。EPContext と「コンパイル済みバイナリ」 |
| **2** | [量子化の基礎](StudyMemo/02_量子化の基礎.md) | **QDQ の数式**（scale / zero point）。`step=42.7` の正体。per-tensor と per-channel。A16W8 と `u16×u8` |
| **3** | [Transformer の中で何が起きたか](StudyMemo/03_Transformerの中で何が起きたか.md) | **residual stream / RMSNorm / massive activation / Dynamic Range。** 実測値で見る「p50 が何段階か」 |
| **4** | [原因の特定と指標の読み方](StudyMemo/04_原因の特定と指標の読み方.md) | **Oracle 実験。** top-1 / 相対L2 / KL の意味と、**実際に踏んだ指標の罠3つ**。softmax による増幅 |
| **5** | [Graph Rewrite](StudyMemo/05_GraphRewrite.md) | **数学的に等価な変形で NPU に合わせる。** 4つの実例と、ε の扱い、等価性の確かめ方 |
| **6** | [ログの読み方と原則](StudyMemo/06_ログの読み方と原則.md) | **実際のログ6例**を自分で読む。覚えておくべき8つの原則（★2026-10-17 訂正: 旧「7つ」。6.2 節は ①〜⑧ の 8 つ） |
| **7** | [重みの量子化](StudyMemo/07_重みの量子化.md) | **誤差の主因はこちらだった。** per-tensor と per-channel、`TensorQuantOverrides` の書き方、`tie_word_embeddings` の罠 |
| **8** | [比較を成立させる](StudyMemo/08_比較を成立させる.md) | **測り方の作法。** サンプル数・teacher forcing・分布・分母・道具の検算・**設定ではなく生成物**・**新機構は無効化して既知の方式と突き合わせる** |
| **9** | [実機に合わせる](StudyMemo/09_実機に合わせる.md) | **ホストで正しくても実機で正しいとは限らない。** 融合 op・DLC まで見る・誤差を形で見る・**C-13** |
| **11** | [解けない問題を見極める](StudyMemo/11_解けない問題を見極める.md) | **直せなかった記録。** Qwen3 が実機で崩れた理由（位置0 の巨大活性で後続が実質3値）・**2つの軸の混同**・集計指標は生成品質を予測しない・**やめる判断** |
| **10** | [文脈をどう持つか](StudyMemo/10_文脈をどう持つか.md) | **固定サイズのキャッシュで長く動かす。** 文脈長のスケーリング・RoPE の位置・attention sink・固定 prefix・**会話をまたぐキャッシュ持ち回し**・**品質はモデルか移植か** |
| **12** | [新しいモデルを載せるかの判断](StudyMemo/12_新しいモデルを載せるかの判断.md) | **次のモデル・端末を、実装する前に安い順に判断する。** 端末で確認すること（HTP の世代・メモリの3つの壁・帯域）・世代で変わること／変わらないこと・速さの見積もり方・AI Hub で分かること／分からないこと・判断の関門 |
| **13** | [ソースと動かし方](StudyMemo/13_ソースと動かし方.md) | **1〜12 章の内容を実際のソースとコマンドに紐づける**（2026-10-14）: 全体の地図（PC の道具・端末のアプリ・環境の使い分け）、**スマホも GPU も要らない演習 5 本**（`app/study/`: QDQ と step・RMSNorm の書き換え・指標と Oracle・C-13・固定長 KV キャッシュ）、章ごとの対応表（節 → 関数と行・コマンド・結果ファイル）、端末のアプリを読む順番、一番大事なこと 1〜12 とソース |
| **14** | [コードで読む検証](StudyMemo/14_コードで読む検証.md) | **検証ごとに「どの道具を、どのコマンドで動かし、中でどのコードが何をしているか」を、実際のソースの抜き出しと 1 行ずつの解説で**（2026-10-14）: 端末で動かして出力を持ち帰る道、Graph Rewrite（C-11・C-9・C-10・C-8）、量子化の本体（`p2b_ops.prepare`）、重みの per-channel と生成物の検査、Oracle・指標・誤差の分解、C-13 と DLC の検査、decode の固定長キャッシュ。**抜き出しはソースから自動で作る**（`app/study/make_code_tour.py`。ソースがずれると止まる） |

---

## 一番大事なことを9行で

1. **bit 数ではなく Dynamic Range を見る。** 16bit でも、280万の幅を表すなら1段階 42.7 になる。
2. **推定で対策を決めない。** Oracle 実験（一箇所だけ理想条件に戻す）は実装より圧倒的に安い。
3. **数式としては同じ答えを出す別の計算方法へ変形し、ハードに都合のよい形にする。** これがポーティングの核心。
4. **設定値ではなく生成物を真実として扱う。** `per_channel=True` と指定していたのに
   **生成物は全部 per-tensor**で、それが誤差の主因だった（[7章](StudyMemo/07_重みの量子化.md)）。
5. **比較が成立しているかを先に確かめる。** サンプル数・軌道・分布・分母・道具の検算
   （[8章](StudyMemo/08_比較を成立させる.md)）。
6. **ONNX で止めず、HTP が実際に実行する DLC まで確認する。** 融合されると ONNX のノードは
   消えるので、そこで止めると永久に気付けない（[9章](StudyMemo/09_実機に合わせる.md)）。
7. **理論上よさそうな最適化を全部入れず、問題が実測されたものだけ追加する。**
   attention sink を実機へ持っていかずに済んだのは fp32 で先に測ったから
   （[10章](StudyMemo/10_文脈をどう持つか.md)）。
8. **新しい機構を作ったら、無効化したときに既知の方式とビット単位で一致するかを先に確かめる。**
   KVキャッシュの持ち回しで**バグを6件**踏んだが、**目視では1つも見つからなかった**
   （[8.8節](StudyMemo/08_比較を成立させる.md)）。
9. **「直らないもの」を追わないために、上限（fp32）を測る。**
   会話が噛み合わないのは移植ではなくモデルの能力だった（[10.10節](StudyMemo/10_文脈をどう持つか.md)）。
10. **「Host が FP32 からずれる」と「実機が Host からずれる」は別の軸。**
    混ぜると、別の問題を解こうとしてしまう（[11.6節](StudyMemo/11_解けない問題を見極める.md)）。
11. **合否は自由生成そのもので決める。** 1位保持率や margin 保持率が baseline を上回るのに
    生成が崩れる例が3つ出た（[11.8節](StudyMemo/11_解けない問題を見極める.md)）。
12. **解が閉じたら、棄却した仮説と事前に決めた基準を残して止める。**
    それがクローズの中身（[11.9節](StudyMemo/11_解けない問題を見極める.md)）。

---

## この題材で何が起きたのか（1枚で）

```
Qwen2.5-0.5B を 24層まるごと NPU に載せたら、出力が壊れた（top-1 4.5%）
  │
  ├─ 推定: residual stream の massive activation（±1677）が原因では？
  │     ↓
  │   Oracle 実験: residual だけ量子化を外す → 6.2% のまま。**0 効果**
  │                RMSNorm だけ外す         → 6.2% → **74.4%**
  │     ↓
  │   真因は RMSNorm 内部の `x²`（レンジ 280万 → 1段階 42.7）だった
  │
  └─ 対策: RMSNorm はスケール不変なので、トークンごとの max|x| で割ってから二乗する
        ↓
      `x²` のレンジが 280万 → 1 になり、1段階が 42.7 → 0.000015（280万倍）
        ↓
      24層 top-1 4.5% → **80.7%**（Oracle の上限と一致）
        ↓
      実機で文章生成に成功。`The capital city of France is` → **` Paris.`**（fp32 と 6/6 一致）
```

## ソースとの対応（節ごと・自動で作る）

各学習の章の節の終わりに置いたソースの枠の一覧（2026-10-17〜）。`<!-- srcindex -->` の中は `& $PY2 -B -m app.study.sync_src` が書き直す。

<!-- srcindex -->
| 章 | 節 | 載せているソース |
|---|---|---|
| [1. レイヤ構造とツールチェーン — どこで何が起きているか](StudyMemo/01_レイヤ構造とツールチェーン.md) | 1.1 混同しやすい4つを分ける | `android_runner/app/src/main/cpp/runner/session.cpp` L250〜L254（strict のときだけ CPU に回すのを禁じる）<br>`android_runner/app/src/main/cpp/runner/session.cpp` L397〜L407（Session::reopen の中。どちらで開けたかがラベルになる）<br>`android_runner/app/src/main/cpp/runner/job.cpp` L347〜L350（op_check（単体 op の判定）の 4 区分） |
|  | 1.2 QDQ は「量子化の設計図」 | `app/pipeline/error_decompose.py` L53〜L60（ホストの CPU EP で QDQ をそのまま実行する）<br>`app/pipeline/error_decompose.py` L121〜L124（main の中。3 つの相対 L2） |
|  | 1.3 ONNX に RMSNorm という op は無い | `app/pipeline/preprocess.py` L591〜L602（normalize_rmsnorm_input の中。export された RMSNorm を形で探す）<br>`app/pipeline/break_rmsnorm_fusion.py` L5〜L33（融合を崩そうとした道具の説明（崩せなかった）） |
|  | 1.4 本プロジェクトの実際の流れ | `app/pipeline/export_qwen_decode.py` L515〜L518（① decode の書き出し（main の中））<br>`scripts/build_context.sh` L92〜L102（⑦ context の生成（ホストで先にコンパイルする））<br>`app/pipeline/wrap_epctx.py` L81〜L117（⑧ EPContext の ONNX を作る）<br>`app/pipeline/deploy_pm64.py` L196〜L199（本番の配備で embed=False を渡す所） |
|  | 1.5 どの層の制約なのかを見分ける | `app/pipeline/force_mixed_matmul.py` L14〜L23（ファイルの説明）<br>`app/pipeline/quantize_qdq.py` L103〜L106（run の中） |
|  | 1.6 ★その原則に自分が引っかかった例（decode） | `app/pipeline/export_qwen_decode.py` L171〜L191（QwenDecodeStep.forward の中）<br>`app/pipeline/p2b_ops.py` L196〜L216（prepare の中。float のグラフ入力を拾う） |
| [2. 量子化の基礎 — `step=42.7` の正体](StudyMemo/02_量子化の基礎.md) | 2.1 QuantizeLinear / DequantizeLinear の数式 | `app/study/ex01_qdq_step.py` L22〜L29（演習 1 の式（本物の量子化は ORT の中で行われる）） |
|  | 2.2 scale はどう決まるか | `app/pipeline/downgrade_actact_rhs.py` L94〜L96（本番の道具で scale と zero point を計算し直す所（run の中））<br>`app/pipeline/make_calib_data.py` L97〜L138（実データ較正（main の中））<br>`app/pipeline/calib.py` L243〜L245（build_multi_reader の中。P2b まで使っていた乱数較正） |
|  | 2.3 bit 数より Dynamic Range | `app/study/ex01_qdq_step.py` L57〜L67（演習 1 の 2。同じ 16bit でも範囲で 1 段が決まる） |
|  | 2.4 per-tensor と per-channel ← 今回の根っこ | `app/pipeline/weight_percg.py` L151〜L166（per-channel の指定を作る）<br>`app/pipeline/weight_percg.py` L169〜L194（できあがった QDQ の重みを数える） |
|  | 2.5 symmetric と asymmetric | `app/pipeline/quantize_qdq.py` L96〜L98（run の中。get_qnn_qdq_config に渡す型） |
|  | 2.6 A16W8 とは実際に何をしているか | `app/pipeline/quantize_qdq.py` L38〜L43（方式の名前と活性化の型） |
|  | 2.7 `u16 × u8` とは何を意味するか | `app/pipeline/p2b_ops.py` L106〜L118（act_act_matmul_tensors の説明）<br>`app/pipeline/downgrade_actact_rhs.py` L98〜L110（run の中。右辺の Q/DQ だけを uint8 に差し替える） |
|  | 2.8 QDQ 形式と C-1 の関係 | `app/pipeline/verify_artifacts.py` L83〜L90（check_c1_no_float_segments の説明）<br>`app/pipeline/verify_artifacts.py` L111〜L125（同じ関数の中） |
| [3. Transformer の中で何が起きたか — residual stream と RMSNorm](StudyMemo/03_Transformerの中で何が起きたか.md) | 3.1 Transformer の1層と residual stream | `app/pipeline/export_qwen_decode.py` L145〜L213（QwenDecodeStep.forward。1 層の中の残差加算） |
|  | 3.2 residual stream の実測値 — massive activation | `app/pipeline/split_layers_n.py` L147〜L150（今のソースで層ごとの入力の大きさを出す所（main の中）） |
|  | 3.3 RMSNorm は何をしているのか | `app/study/ex02_rmsnorm_rewrite.py` L32〜L36（演習 2 の RMSNorm（本物は transformers の中の Qwen2 の RMSNorm）） |
|  | 3.4 massive activation があると `x²` で何が起きるか | `app/pipeline/preprocess.py` L542〜L548（normalize_rmsnorm_input の説明） |
|  | 3.5 per-tensor 量子化と合わさって壊れる | `app/study/ex02_rmsnorm_rewrite.py` L64〜L75（演習 2 の 2） |
|  | 3.6 「p50 が何段階に載るか」で見る | なし（考え方の説明（p50 ÷ 1段階 の割り算）。数字は results/P3_20260913_rmsnorm_fix.md 1 節。分布を測る今のソースは 3.2 節の枠） |
|  | 3.7 なぜ24層で初めて壊れたのか | `app/pipeline/oracle_exclude.py` L46〜L52（Oracle 実験で量子化から外す op の組） |
|  | 3.8 Attention 側にも同じ構造がある | `app/pipeline/export_qwen_decode.py` L199〜L201（QwenDecodeStep.forward の中の QK^T）<br>`app/pipeline/preprocess.py` L229〜L233（fold_scale_into_matmul の説明（layer 0 の実測）） |
| [4. 原因の特定と指標の読み方 — Oracle 実験](StudyMemo/04_原因の特定と指標の読み方.md) | 4.1 Oracle 実験とは | `app/pipeline/oracle_exclude.py` L3〜L19（ファイルの説明） |
|  | 4.2 結果 | `app/pipeline/oracle_exclude.py` L98〜L114（main の中。パターンごとに量子化し直して測る）<br>`app/pipeline/oracle_qdq.py` L104〜L133（できあがった QDQ から Q/DQ の組を外す版） |
|  | 4.3 なぜ SmoothQuant を使わなかったのか | なし（考え方の説明。SmoothQuant はこのプロジェクトでは実装していない（実装したのは 5.1 節の RMSNorm の書き換えだけ）） |
|  | 4.4 指標の読み方 | `app/pipeline/accuracy_logits.py` L65〜L102（指標を計算する関数） |
|  | 4.5 ★指標の罠（実際に3回踏んだ） | `app/pipeline/error_decompose.py` L68〜L78（罠 1 の直し）<br>`app/pipeline/device_accuracy.py` L213〜L225（罠 2 の直し（main の中）） |
|  | 4.6 「その先で何に使われるか」まで見る — softmax の増幅 | `app/pipeline/preprocess.py` L323〜L336（stabilize_softmax の説明。この節の測定の記録） |
|  | 4.7 ヘッドごとに見ると原因が絞れる（追加の診断） | なし（記録だけ（numpy での試算。results/P3_20260912_accuracy_layer1.md 4 節）でスクリプトが残っていない） |
| [5. Graph Rewrite — 答えは同じ、途中計算だけ NPU 向け](StudyMemo/05_GraphRewrite.md) | 5.0 考え方 | `app/pipeline/preprocess.py` L743〜L783（S2 前処理の本体。書き換えを呼ぶ順）<br>`app/pipeline/p2b_ops.py` L173〜L175（本番（p2b_ops prepare）での呼び方） |
|  | 5.1 C-11: RMSNorm を max で正規化（今回の本命） | `app/pipeline/preprocess.py` L611〜L627（normalize_rmsnorm_input の中。ノードを足す所） |
|  | 5.2 ε の扱い — ここは厳密解を取り逃した | `app/pipeline/preprocess.py` L628〜L637（normalize_rmsnorm_input の中） |
|  | 5.3 C-8 対応: `QK^T = (K Q^T)^T` で左右を入れ替える | `app/pipeline/preprocess.py` L509〜L519（swap_actact_matmul_operands の中）<br>`app/pipeline/find_swap_matmuls.py` L115〜L126（classify の中。入れ替える対象（QK^T）を形で見分ける） |
|  | 5.4 C-10: `Expand` → `Concat` | `app/pipeline/preprocess.py` L188〜L207（rewrite_expand_to_concat の中） |
|  | 5.5 C-9: マスク定数 `-3.4e38` → `-60` | `app/pipeline/preprocess.py` L68〜L77（置き換え先の値と、その理由）<br>`app/pipeline/preprocess.py` L105〜L111（clamp_extreme_constants の中） |
|  | 5.6 効かなかった書き換え（再試行を防ぐために残す） | `app/pipeline/preprocess.py` L235〜L237（fold_scale_into_matmul の説明（当時の見込み）） |
|  | 5.7 等価性は必ず数値で確かめる | `app/pipeline/export_qwen_decode.py` L376〜L412（自前の配線を HF と突き合わせる） |
| [6. ログの読み方と原則](StudyMemo/06_ログの読み方と原則.md) | 6.1 実際のログを6例読む | `app/pipeline/error_decompose.py` L139〜L156（main の中。分解と直交の検算を出す所）<br>`app/pipeline/deploy_pm64.py` L127〜L148（生成器の出力が書き終わるまで待つ）<br>`app/pipeline/deploy_pm64.py` L208〜L221（main の中。md5 を 3 点で照合する）<br>`scripts/p2b_build_ops.sh` L54〜L63（ログから原因の 1 行を選ぶ関数） |
|  | 6.2 覚えておくべき8つの原則 | `app/pipeline/export_qwen_decode.py` L490〜L494（main の中）<br>`app/pipeline/weight_percg.py` L213〜L238（生成物の per-channel を期待と比べる）<br>`app/pipeline/dlc_audit.py` L124〜L135（audit の中。型とエンコーディングを突き合わせる） |
|  | 6.3 この検証で使った道具一覧（再現用） | なし（道具の一覧。各道具の中身は 1〜6 章の各節の枠と、13 章の表・14 章にある） |
| [7. 重みの量子化 — 主因はこちらだった](StudyMemo/07_重みの量子化.md) | 7.0 この章の位置づけ | なし（章の導入（2章 2.4節の訂正）。根拠のソースは 7.1〜7.5 節の枠） |
|  | 7.1 誤差を「重み」と「活性化」に分ける | `app/pipeline/oracle_qdq.py` L104〜L155（strip の前半（活性化の Q/DQ を外す所）） |
|  | 7.2 なぜ重みが悪いのか — per-channel が1本も効いていなかった | `app/pipeline/quantize_qdq.py` L88〜L98（パイプラインが渡していた指定）<br>`app/pipeline/weight_percg.py` L169〜L209（生成物の重みを数える） |
|  | 7.3 per-tensor だと何が起きるか（実データ） | なし（試算（168 本の per-tensor／per-channel の再構成誤差）のスクリプトは残っていない。同じ計算は演習 app/study/ex01_qdq_step.py の 3 で確かめられる） |
|  | 7.4 直し方 — `TensorQuantOverrides` に axis を書く | `app/pipeline/weight_percg.py` L124〜L166（matmul_weight_axes の後半と build_overrides）<br>`app/pipeline/p2b_ops.py` L271〜L281（prepare の S4（量子化の前）） |
|  | 7.5 結果 | `app/pipeline/error_decompose.py` L121〜L139（fp32 → ホストQDQ → 実機 の分解）<br>`android_runner/app/src/main/cpp/runner/job.cpp` L423〜L432（op_check の判定（単体 MatMul を v68 が受け付けるか）） |
|  | 7.6 これで整理し直せること | なし（整理（7.1〜7.5 節の結果の言い直し）。根拠のソースは 7.1〜7.5 節の枠） |
|  | 7.7 ★古い版は消さずに残す | `../EdgeSLM_AndroidDemo/demo_app/app/src/main/java/org/edgeslm/demo/DemoActivity.kt` L374〜L397（デモ画面のモード（デモのプロジェクト。読むだけ）） |
|  | 7.8 v68 では重みを16bitにはできない | `app/pipeline/downgrade_actact_rhs.py` L88〜L96（C-8 の対応（act×act の MatMul の右辺を uint8 に）） |
| [8. 比較を成立させる — 測り方の作法](StudyMemo/08_比較を成立させる.md) | 8.0 この章の位置づけ | なし（章の導入（壊し方の一覧）。根拠のソースは 8.1〜8.8 節の枠） |
|  | 8.1 ①② サンプル数 — 分解能を先に計算する | `app/pipeline/make_decode_calib.py` L54〜L59（評価用の文）<br>`app/pipeline/make_decode_calib.py` L99〜L114（各文の先頭から --steps ステップを記録する） |
|  | 8.2 ③ 軌道を揃える（teacher forcing） | `app/pipeline/decode_error_profile.py` L190〜L211（teacher forcing の本体） |
|  | 8.3 ④ 影響の試算は実データの分布で | `app/pipeline/fix_cache_qparams.py` L9〜L19（docstring に残る当時の見積もり） |
|  | 8.4 ⑤ 相対指標は分母を確かめる | `app/pipeline/error_decompose.py` L63〜L78（絶対 L2 と、分母を指定できる相対 L2） |
|  | 8.5 ⑥ 道具には検算を用意する | `app/pipeline/oracle_qdq.py` L172〜L213（strip の @weights（畳むだけで外れない実装）） |
|  | 8.6 ★設定値ではなく生成物を真実として扱う | `app/pipeline/weight_percg.py` L213〜L238（verify）<br>`app/pipeline/p2b_ops.py` L327〜L337（prepare の S4（量子化の直後）） |
|  | 8.7 指標は併記する | `app/pipeline/accuracy_logits.py` L65〜L102（metrics（cos は無い））<br>`app/pipeline/ab_host_device.py` L55〜L79（metrics（5つをそろえて返す）） |
|  | 8.8 ★新しい機構を作ったら、**無効化したときに既知の方式と一致するか**を先に確かめる | `../EdgeSLM_AndroidDemo/demo_app/app/src/main/cpp/runner/job.cpp` L688〜L712（turns（同じセッションで複数ターン。デモのプロジェクト））<br>`../EdgeSLM_AndroidDemo/demo_app/app/src/main/cpp/runner/job.cpp` L902〜L913（UI が通る経路の差分（バグ5））<br>`../EdgeSLM_AndroidDemo/demo_app/app/src/main/cpp/runner/job.cpp` L1115〜L1122（バグ2（二重投入）を直した所） |
|  | 8.9 まとめ: 比較の前に確かめる8項目 | なし（章のまとめ（根拠は 8.1〜8.8 節）） |
| [9. 実機に合わせる — ホストで正しくても実機で正しいとは限らない](StudyMemo/09_実機に合わせる.md) | 9.0 この章の問い | なし（章の導入（問いの提示）。数字の出どころは 7.5 節） |
|  | 9.1 まず指標を変えた | `app/pipeline/ab_host_device.py` L138〜L163（fp32 / ホストQDQ / 実機 の3者比較） |
|  | 9.2 二分探索で場所を絞る | `app/pipeline/split_layer.py` L72〜L92（本物の層を割り、本物の活性で較正データを作る） |
|  | 9.3 ★誤差の「形」を見る | `app/pipeline/fix_norm_gamma.py` L3〜L13（docstring に残る実測） |
|  | 9.4 真因 — C-13 | `app/pipeline/fix_norm_gamma.py` L81〜L112（audit（生成物を数える））<br>`app/pipeline/fix_norm_gamma.py` L115〜L151（run（符号付き対称で入れ直す）） |
|  | 9.5 24層での効果 | なし（結果の表（数字は results/P9_20260914_pm64_c13_ab.json・P9_20260914_dec24_c13_ab.json・P9_20260914_dec2b_c13_ab.json）。比べた道具は 9.1 節の枠の ab_host_device.py） |
|  | 9.6 覚えておくこと | `app/pipeline/dlc_audit.py` L98〜L140（audit（DLC の型とエンコーディングを突き合わせる）） |
|  | 9.7 自動化した | `app/pipeline/p2b_ops.py` L339〜L361（prepare の S4 の直後（C-13 を自動で直す））<br>`app/pipeline/deploy_pm64.py` L58〜L78（配備の 1.5 段（push の前に DLC を検査）） |
|  | 9.8 別のモデルへ持っていくとき | `app/pipeline/fix_norm_gamma.py` L54〜L78（対象の拾い方） |
| [10. 文脈をどう持つか — 固定サイズのキャッシュで長く動かす](StudyMemo/10_文脈をどう持つか.md) | 10.0 この章の位置づけ | なし（章の導入（この章は移植ではなく推論エンジンの設計の話であること）） |
|  | 10.1 2つの方式 | `app/pipeline/export_qwen_decode.py` L190〜L204（QwenDecodeStep.forward（1 トークンの decode）） |
|  | 10.2 文脈長を伸ばす — どこまで伸ばせるか | `app/pipeline/export_qwen_decode.py` L90〜L105（QwenDecodeStep の形（容量 L・KV ヘッド・head_dim）） |
|  | 10.3 RoPE の位置 — 容量とは別物 | `app/pipeline/make_rope_table.py` L48〜L60（RoPE の表を作る）<br>`../EdgeSLM_AndroidDemo/demo_app/app/src/main/cpp/runner/job.cpp` L1152〜L1172（generate_kv の位置の付け方（デモのプロジェクト）） |
|  | 10.4 ★窓が埋まると崩れる。原因は移植ではない | `app/pipeline/context_policy_host.py` L59〜L80（窓なし fp32 で基準の系列を作る）<br>`app/pipeline/sliding_decode_host.py` L87〜L103（端末と同じ窓ずらしの生成ループ） |
|  | 10.5 attention sink は効かなかった | `app/pipeline/context_policy_host.py` L110〜L132（先頭 N 固定の有無だけを変えて比べる） |
|  | 10.6 ★固定 prefix は効く。ただしサイズを中身に合わせる | `app/pipeline/prefix_recall_test.py` L52〜L59（先頭の事実・無関係な文・質問）<br>`app/pipeline/prefix_recall_test.py` L107〜L129（score（答えのトークンの順位と NLL）） |
|  | 10.7 実装で2回間違えた — 検算の作り方 | `app/pipeline/sink_compare_host.py` L76〜L81（R(-1)：位置を 1 つ戻す回転）<br>`app/pipeline/sink_compare_host.py` L108〜L142（run_mode（2 回の誤りを直した所）） |
|  | 10.8 対話はできるのか（実測） | なし（端末のデモアプリでの手作業の実測（記録は results/P12_20260914_prefill_cache.md）。直す前の「毎回全履歴を入れ直す」経路は、10.9 節の枠の再利用が効かない場合（skip が 0）） |
|  | 10.9 会話をまたいでキャッシュを持ち回す | `../EdgeSLM_AndroidDemo/demo_app/app/src/main/cpp/runner/session.h` L43〜L75（KvCarry（会話をまたぐキャッシュの状態））<br>`../EdgeSLM_AndroidDemo/demo_app/app/src/main/cpp/runner/job.cpp` L1332〜L1366（ターンの終わりに保存する所） |
|  | 10.10 ★会話の品質は、移植ではなくモデルの問題だった | なし（fp32 と実機の答えの比較の記録（results/P12_20260914_prefill_cache.md）。比べたスクリプトはリポジトリに残っていない） |
|  | 10.11 この章のまとめ | なし（章のまとめ（根拠は 10.2〜10.10 節）） |
| [11. 解けない問題を見極める — Qwen3 が実機で崩れた理由と、そこで止めた判断](StudyMemo/11_解けない問題を見極める.md) | 11.0 この章の問い | なし（章の導入（問いの提示）） |
|  | 11.1 症状と、最初に切るべき切り口 | なし（4 段の比較は記録だけでスクリプトが残っていない（results/P12b_20260915_qwen3_device_chat.md の道具の表では scratchpad/q3_chat.py。リポジトリの外）） |
|  | 11.2 発生源を1点に絞る | `app/pipeline/layer2_probe.py` L237〜L255（compare（Host と実機の整数コードを比べる）） |
|  | 11.3 ★整数の一致率は、誤差の大きさを表さない | なし（考え方の説明。整数差と実数誤差を並べる計算は 11.2 節の枠（real_rel）） |
|  | 11.4 なぜ scale がこんなに粗いのか — 位置0 の巨大活性 | `app/pipeline/massive_origin_check.py` L66〜L84（位置0 の巨大活性を公式 HF の fp32 で調べる） |
|  | 11.5 直そうとして起きたこと | `app/pipeline/linear20_range_sweep.py` L74〜L105（linear_20 の範囲だけを margin 倍にした QDQ を作る） |
|  | 11.6 ★2つの軸を混同しない | `app/pipeline/v_l11_17_device_probe.py` L315〜L345（compare（L11-17 の V を Host と実機で比べる）） |
|  | 11.7 ハードの制約が解を閉じることがある | `app/pipeline/v_l11_17_device_probe.py` L303〜L307（(a) 条件B は実機の比較から外す）<br>`app/pipeline/kv_range_recal.py` L150〜L181（(b) KV 系の範囲を広げる） |
|  | 11.7b ★「実機に載せられる形か」を先に確かめる | `app/pipeline/single_model_4x.py` L57〜L59（変種の定義）<br>`app/pipeline/single_model_4x.py` L81〜L94（build（1 モデルで 4x にする）） |
|  | 11.8 測り方で踏んだ落とし穴（4つ） | `app/pipeline/v_l11_17_device_probe.py` L264〜L279（観測は量子化後の整数コードで取る） |
|  | 11.9 やめる判断 | なし（記録（棄却した仮説と判定基準の一覧は results/P14_20260917_qwen3_closing.md・results/P12t_20260916_session_summary_layer2_htp_tradeoff.md）） |
|  | 11.10 持ち帰り | なし（章のまとめ（根拠は 11.1〜11.9 節）） |
|  | 11.11 この題材の記録 | なし（記録の一覧） |
| [12. 新しいモデルを載せるかの判断 — 実装する前に、安い順に確かめる](StudyMemo/12_新しいモデルを載せるかの判断.md) | 12.0 この章の問い | なし（考え方の説明（章の問いと、見るものの一覧）） |
|  | 12.1 端末で確認すること（一覧） | `scripts/p0a_collect.sh` L51〜L70（端末の素性を一度に集める（P0-a）） |
|  | 12.2 HTP の世代で決まること | `scripts/build_context.sh` L92〜L102（S6 の context 生成） |
|  | 12.3 メモリ — 載るか | `app/pipeline/wrap_epctx.py` L81〜L112（EPContext の ONNX で包む） |
|  | 12.4 帯域 — 速さを見積もる | なし（見積もりの考え方（式）。実測は別プロジェクト EdgeSLM_AndroidGemma4 の results/P6_20260921_latency.md、KV 長の式は 10.2 節（results/P10_20260914_context_length.md）） |
|  | 12.5 モデルで確認すること — 形と op | `app/pipeline/verify_artifacts.py` L141〜L156（check_c1_no_float_segments（C-1 の自動検査）） |
|  | 12.6 モデルで確認すること — 量子化に耐えるか | `app/pipeline/make_calib_data.py` L133〜L138（較正データの分布を出す所） |
|  | 12.7 AI Hub で分かること・分からないこと | なし（外部サービス（Qualcomm AI Hub Workbench）の公式ドキュメントの話で、この検証では使っていない） |
|  | 12.8 判断の手順 — 安い順に通し、落ちたら止める | `android_runner/app/src/main/cpp/runner/session.cpp` L397〜L407（strict（CPU に回すのを禁じて）開く） |
|  | 12.9 同じ端末に載せた4つのモデル | なし（章のまとめ（根拠は 1〜11 章と、別プロジェクト EdgeSLM_AndroidScale・EdgeSLM_AndroidGemma4 の記録）） |
|  | 12.10 持ち帰り | なし（章のまとめ（持ち帰り）） |
|  | 12.11 この題材の記録 | なし（記録の一覧） |
<!-- /srcindex -->
