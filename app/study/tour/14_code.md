<<OUT DoC/StudyMemo/14_コードで読む検証.md>>
# 14. コードで読む検証 ―― 道具・コマンド・実際のソース

> **このファイルは生成物**です（`app/study/make_code_tour.py` が `app/study/tour/14_code.md` と実際のソースから作る。2026-10-14）。
> ソースの抜き出し・行番号・結果の値は作った時点のものそのものです（最後の 14.9 にソースの md5）。
> 作り直し: `& $PY2 -B -m app.study.make_code_tour`（`--check` で、抜き出しの目印が全部ソースに見つかるかだけを確かめる）。

13 章は「どのファイルの何行目を見ればよいか」の地図でした。この章はその先の**中身**です。検証ごとに

1. **何を確かめたか**　2. **道具とコマンド**　3. **流れ**　4. **実際のソース**（行番号つきの抜き出しと、「どの行が何をしているか・なぜ」の表）　5. **結果**

の順で読みます。表の「目印」は抜き出しの中のその行の書き出しです（行番号はソースを直すと変わるため）。

| 節 | やったこと | 主なソース |
|---|---|---|
| 14.1 | （全部の検証の共通部分）量子化したモデルを端末で動かし、出力を持ち帰る道 | `device_accuracy.py`・`job.cpp`・`session.cpp` |
| 14.2 | **Graph Rewrite**: RMSNorm を max で割る（C-11）・マスク定数（C-9）・Expand → Concat（C-10）・左右入替（C-8） | `preprocess.py` |
| 14.3 | **量子化の本体**: 前処理 → 量子化 → 検査を段ごとに記録する | `p2b_ops.py`・`quantize_qdq.py` |
| 14.4 | **重みの per-channel と「生成物を数える」** | `weight_percg.py` |
| 14.5 | **原因の特定**: Oracle 実験・指標・誤差の分解 | `oracle_exclude.py`・`accuracy_logits.py`・`error_decompose.py` |
| 14.6 | **実機だけ壊れる（C-13）と DLC の検査**、C-8 の右辺 8bit | `fix_norm_gamma.py`・`dlc_audit.py`・`downgrade_actact_rhs.py` |
| 14.7 | **decode と固定長の KV キャッシュ** | `export_qwen_decode.py` |
| 14.8 | 自分で書き換えを 1 つ足すときの手順 | — |

---

## 14.1 端末で動かして出力を持ち帰る道（共通部分）

### 流れ

```text
PC: device_accuracy.py
  1. 評価データ（npz）を 1 サンプルずつ .bin に展開 → adb push files/inputs/
  2. job（JSON）を作る: task "infer"、model、batches（サンプルごとの入力ファイル）
  3. adb push files/jobs/<job_id>.json → am start RunnerActivity … --es job jobs/<job_id>.json
                                           └ RunnerService → C++ の Session::open（strict で開く）→ run_job → infer
  4. out/last_result.json を待ち、job_id を確かめ、出力の .bin を pull
  5. fp32 の ONNX をホストで動かして比べる（error_decompose.py なら ホスト QDQ も挟んで 3 者）
```

```powershell
& $PY2 -m app.pipeline.device_accuracy --model models/ops/<名前>_epctx.onnx --eval-npz <評価.npz> --fp32 <fp32.onnx>
& $PY2 -m app.pipeline.error_decompose --fp32 <fp32.onnx> --qdq <実機と同じ QDQ> --eval-npz <評価.npz> --device-dir work/devacc --out results/<記録>.json
```

### PC 側: job を作って送る（`device_accuracy.py`）

<<CODE app/pipeline/device_accuracy.py | ^def export_inputs\(npz | ^    return batches>>

<<CODE app/pipeline/device_accuracy.py | ^    job = \{$ | raise SystemExit\(f"別のジョブの結果が返った | omit=^    import time..^    local_res = >>

| 目印 | 何をしているか・なぜ |
|---|---|
| `a = np.ascontiguousarray(z[name][i:i + 1])`・`p.write_bytes(a.tobytes())` | 入力を**生のバイト列**にする。端末の C++ は形と型を `describe` から知っているので、バイト列だけ渡せばよい |
| `"task": "infer"`・`"batches": batches` | 端末の `job.cpp` の `infer` を呼ぶ。`batches` の 1 つが 1 サンプル（入力名 → ファイル） |
| `"options": {"offload_graph_io_quantization": "0", …}` | 入出力の量子化・逆量子化を CPU に回さない（全部 NPU で。CPU に回すと strict で開けない） |
| `rm -f … last_result.json …` | 前の結果を消してから動かす（前の job の結果を取り違えない。6.1 節 例 6・検証環境と手法 6 章） |
| `am start -n org.edgeslm.runner/.RunnerActivity … --es job …` | アプリを起動して job を渡す |
| `if res.get("job_id") != args.job_id: raise` | **完了は成果物で判定する**（標準出力の文言で判定して、成功を失敗と誤判定したことがある） |

### 端末側: CPU に回すのを禁じて開く（`session.cpp`）・job を実行する（`job.cpp`）

<<CODE android_runner/app/src/main/cpp/runner/session.cpp | ① まず strict（CPU フォールバック禁止）で開く | backend_label_ = strict_succeeded_ >>

<<CODE android_runner/app/src/main/cpp/runner/session.cpp | strict の場合のみ CPU フォールバックを禁止する | disable_cpu_ep_fallback"\);>>

まず `session.disable_cpu_ep_fallback = 1`（NPU で動かせないノードがあれば開くのを失敗させる）で開き、開けたら「全ノード NPU」と記録します
（結果の `describe.strict_no_cpu_fallback: true`）。失敗したら CPU に回すのを許して開き直し、「どこかが CPU で動いている」と分かるようにします。

<<CODE android_runner/app/src/main/cpp/runner/job.cpp | if \(task == "infer"\) \{ | out\["completed"\] = n_ok; | omit=if \(!loaded\) \{..std::vector<std::vector<uint8_t>> outs; | omit=json written = json::array\(\);..rec\["outputs"\] = written;>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `for (const auto& in : desc["inputs"])` | 入力は**モデルの入力の順**に読む（job の中の順ではない） |
| `if (buf.size() != count * esz)` | 期待のバイト数と違えば止める（形・型の取り違えを、黙って変な値で動かさない） |
| `session.run(inputs, &ms, &outs)` | NPU で 1 回動かす。`ort_run_ms` に時間 |
| （中略した所） | 出力を `out/<job_id>_<id>_<出力名>.bin` に書き、形・型と一緒に記録する（PC の `error_decompose.py` がこの名前で探す） |

---

## 14.2 Graph Rewrite（`preprocess.py`）

### 何を確かめたか

24 層が壊れた原因（RMSNorm の中の `x²` の範囲。4 章）を、**数式として同じ別の計算**に書き換えて直した（5 章）。
ほかにも、v68 で通らない形（C-9 マスク定数・C-10 Expand）と、8bit にせざるを得ない側を選ぶ左右入替（C-8）を同じ所で書き換えます。

### 呼ぶ順（`run`）

<<CODE app/pipeline/preprocess.py | ^def run\(src: Path, folded_path | rms = normalize_rmsnorm_input\(m0\) if norm_rmsnorm else \[\]>>

| 目印 | 何をしているか・なぜ |
|---|---|
| ① `clamp_extreme_constants` を最初に | 後の定数畳み込みが ±3.4e38 を別の場所に広げる前に潰す |
| ② `rewrite_expand_to_concat` を量子化より前に | Q/DQ が挟まった後だと、形を壊さずに書き換えるのが難しい |
| ③ `fold_scale_into_matmul` | 注意のスコアのスケールを MatMul の前に移す（効果なし。5.6 節） |
| ④ `stabilize_softmax`（既定 off） | numpy の試算では効いたのに、実際は悪化した。スコアは MatMul の出力の時点で 8bit 化済みで手遅れだった（5.6 節） |
| ⑤ `swap_actact_matmul_operands`（名前で指定したものだけ） | QK^T の左右を入れ替える |
| ⑥ `normalize_rmsnorm_input`（方式名に `rms` があるとき） | RMSNorm を max で割ってから計算する |

### C-11: RMSNorm を max で割ってから計算する（`normalize_rmsnorm_input`）

<<CODE app/pipeline/preprocess.py | ^def normalize_rmsnorm_input | ^    return changed$ | from=^def swap_actact_matmul_operands | omit=RMSNorm を「トークンごとの最大値で割ってから」..    g = model.graph | omit=# 追加ノードを入れてトポロジ順に直す..        del g.node\[:\]>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `if pw.op_type != "Pow": continue`・指数が 2 か・`ReduceMean` に入るか | ONNX に RMSNorm という op は無い（1.3 節）。**「x の 2 乗を平均する」という形**で RMSNorm を見つける（名前で探すとモデルが変わると取りこぼす） |
| `g.initializer.append(… 1e-6 …)`（`_floor`） | max に小さい下限を足す。全ゼロのテンソルで 1/0 = inf になり、変換器が `quantization max=inf` で落ちたため |
| `Abs` → `ReduceMax`（`keepdims=1`、軸 −1）→ `Add` → `Reciprocal` → `Mul` | `m = max|x|`（トークンごと）、`xs = x * (1/m)`。xs は [−1, 1] に収まる |
| `pw.input[0] = xs` | 二乗するのを x から xs に差し替える。xs² は [0, 1] なので uint16 の 1 段が 1.5e−5（元は 42.7） |
| `add.input[i] = e2`（eps を 1e−12 に） | 割った後の平均は 1/m² 倍なので、元の eps のままだと効きすぎる（近似。厳密には eps/m²。演習 ex02 の 3） |
| `x * rsqrt(...) の x を xs に差し替える` | 最後の掛け算も xs に。`xs / rms(xs) = x / rms(x)`（スケール不変）なので結果は同じ |
| （最初の中略） | 関数の説明文（なぜ必要か・oracle の数字・スケール不変の式）。中身は 5.1 節と同じなので省いた。ソースを開くと読める |
| （2 つ目の中略） | 足したノードを入れて、依存の順（トポロジ順）に並べ直す |

結果（6 層、176 位置。`results/P3_rms6.json`）: 書き換える前 <<VAL results/P3_rms6.json | "%.1f%%" % (d[0]["top1_agreement"] * 100)>> → 書き換えた後 <<VAL results/P3_rms6.json | "%.1f%%" % (d[1]["top1_agreement"] * 100)>>（top-1 一致）。
Oracle の上限（RMSNorm の量子化を外した場合）と同じ所まで戻りました（14.5 節）。

### C-9: マスク定数（`clamp_extreme_constants`）

<<CODE app/pipeline/preprocess.py | ^MASK_REPLACE = -60.0 | +2>>

<<CODE app/pipeline/preprocess.py | neg = np.isneginf\(arr\) \| \(arr <= -threshold\) | init.CopyFrom\(numpy_helper.from_array\(new, init.name\)\) | from=^def clamp_extreme_constants>>

HF の因果マスクは float32 の最小値（−3.4e38）。量子化は「最小〜最大を N 段に割る」ので、これが 1 つでも入ると全部の段がそこに食われ、本物のスコアが同じ値に潰れます。
−60 なら `exp(−60) ≒ 8.8e−27` で softmax の後は 0 と同じ（演習 ex02 の 4）。**変えたことは必ず記録する**（`changed` に入れて返す）。

### C-10: Expand → Concat（`rewrite_expand_to_concat`）

<<CODE app/pipeline/preprocess.py | in_shape = shapes\[src\] | "concat_axes": \[\{"axis": a, "repeat": r\} for a, r in axes\], | from=^def rewrite_expand_to_concat>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `out_shape = [max(a, t) for a, t in zip(in_shape, target)]` | Expand の目標の形は**ブロードキャストの形**でありうる（`[1,1,7,1,1]` で広げる）。目標をそのまま出力の形と見ると取りこぼす（54 分の context 生成を無駄にしかけた） |
| `axes = [(i, o) … if a == 1 and o > 1]` | 「サイズ 1 の軸を n 倍にする」軸だけ |
| `if not axes or any(a != o and a != 1 …)` | それ以外の広げ方は書き換えない（黙って別の意味のグラフにするより、端末で落ちて気づく方がよい） |
| `helper.make_node("Concat", [cur] * rep, …)` | 同じテンソルを rep 個つなぐ＝ Expand と同じ結果。v68 は Expand を含む context binary を読めない（err 1002）が Concat は読める |

### C-8: QK^T の左右入替（`swap_actact_matmul_operands`）

<<CODE app/pipeline/preprocess.py | def is_weight\(t: str\) -> bool: | "new_rhs": f"\{a\} \(転置\)", "rank": r\}\) | from=^def swap_actact_matmul_operands>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `is_weight`: 形だけ変える op を辿って大元が初期化子か | lm_head の重みは Transpose を経て MatMul に入る。素朴に「入力名が初期化子か」だけ見ると重みの MatMul まで入れ替えて落ちた |
| `perm[-1], perm[-2] = perm[-2], perm[-1]` | 最後の 2 軸を入れ替える転置 |
| `MatMul([bt, at])` → `Transpose` | `A×B = (Bᵀ×Aᵀ)ᵀ`。v68 の act×act の MatMul は **右辺だけ** 8bit にできる（`u16×u8`）。K（±130）ではなく Q（[−10, 6]、15 倍細かい）を右辺に持ってくる |

---

## 14.3 量子化の本体（`p2b_ops.py` の `prepare`・`quantize_qdq.py` の `run`）

### 道具とコマンド

```powershell
# 前処理と量子化（spec は work/ops/<名前>/<名前>.spec.json。方式 a16w8_rms_swap ＝ A16W8 ＋ RMSNorm の書き換え ＋ 左右入替）
& $PY2 -m app.pipeline.p2b_ops prepare --name <名前> --scheme a16w8_rms_swap
# C-8: act×act の MatMul の右辺を uint8 に（14.6 節）
& $PY2 -m app.pipeline.downgrade_actact_rhs --qdq work/ops/<名前>/<名前>_a16w8_rms_swap_qdq.onnx --prepped work/ops/<名前>/<名前>_prepped.onnx --out work/ops/<名前>/<名前>_final_qdq.onnx
# ★ここでホストに読ませて動くか確かめる（壊れた QDQ を実機へ送ったことがある）
& $PY2 -c "import onnxruntime as ort; ort.InferenceSession('work/ops/<名前>/<名前>_final_qdq.onnx')"
```

`prepare` は段（S2 前処理・S4 量子化・C-1 の検査）ごとに成否を記録し、**落ちても例外を外へ投げずに「どの段で落ちたか」を残します**（1.5 節: S5 で落ちたなら ONNX の書き換え、S6 なら v68 の対応、と次の手が違う）。

### S2 → S3/S4 の分かれ道

<<CODE app/pipeline/p2b_ops.py | # --- S2 前処理 --- | norm_rmsnorm="rms" in scheme\) | from=^def prepare\(work>>

<<CODE app/pipeline/p2b_ops.py | if scheme.endswith\("_swap"\) or "rms" in scheme: | "kept_u16": lhs \+ outs\} | from=^def prepare\(work>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `swap = spec.get("swap_matmuls") if "swap" in scheme` | 左右入替は**名前で指定した MatMul だけ**（前処理の時点では範囲が分からないので、どれを入れ替えるかは spec に書く） |
| `norm_rmsnorm="rms" in scheme` | 方式名に `rms` があれば C-11 の書き換え |
| `if "rhs8" in scheme:` … `QuantType.QUInt8` | act×act の MatMul の右辺を 8bit に指定する。★もとは分岐が排他で、`a16w8_rms_swap` だけでは u16×u16 が残り context 生成が落ちた |
| `converts = False` | ORT に「型をそろえる変換ノード」を挿させない（勝手に 8bit にそろえられないように） |

### 重みの per-channel・グラフ入力の量子化・較正

<<CODE app/pipeline/p2b_ops.py | ★重みを per-channel で量子化させる（spec に per_channel_weights | st\["forced_input_quant"\] = gin>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `weight_percg.build_overrides(prepped, overrides)` | 重みを per-channel にする指定（14.4 節）。`get_qnn_qdq_config(per_channel=True)` だけでは MatMul の重みに効かなかった |
| `gin = _float_graph_inputs(prepped)` → `QUInt16` を指定 | グラフの入力が形だけ変える op（Slice など）に直接入ると、量子化器は float のまま残す（C-1 違反）。decode の KV キャッシュの入力 48 本がこれだった |

<<CODE app/pipeline/p2b_ops.py | r = quantize_qdq.run\(prepped, qdq, scheme=base_scheme | "unsafe_after": 0\} | from=^def prepare\(work>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `quantize_qdq.run(…)` | ORT の量子化（下の抜き出し） |
| `weight_percg.verify(qdq, expect_per_channel=…, strict=True)` | **設定ではなく生成物を見る**。頼んだ本数が本当に per-channel になったかを、約 1 時間の context 生成の**前**に確かめる |
| `fix_norm_gamma.audit(qdq)` → `run` → もう一度 `audit` | C-13（14.6 節）。直した後にもう一度数え、はみ出しが残っていれば止める |

<<CODE app/pipeline/p2b_ops.py | # --- C-1 / C-5（モデル単体で判定できる分）--- | st\["verdict"\] = "ready_for_convert" | from=^def prepare\(work>>

C-1（float の区間が残っていないか）を割ったまま進めても、端末で error 1002 になるだけなので、ここで止めます。

### ORT の量子化（`quantize_qdq.run`）

<<CODE app/pipeline/quantize_qdq.py | cfg = get_qnn_qdq_config\( | raise RuntimeError\($ | omit=# get_qnn_qdq_config は extra_options を受け取らないので、..    out_qdq.parent.mkdir>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `get_qnn_qdq_config(…)` | QNN EP 向けの制約を織り込んだ設定を ORT に作らせる。op の一覧を手で渡すと float の区間が残る（C-1） |
| `activation_type=ACTIVATION[scheme]`・`weight_type=QuantType.QInt8` | A16W8 ＝ 活性化 uint16・重み int8（2.6 節） |
| `init_overrides=init_overrides` | ★テンソル名は**前処理の後のモデル**から取る（前処理が名前を変えるので、元のモデルの名前だと上書きが黙って効かない） |
| `nodes_to_exclude=` | Oracle 実験だけで使う（外すと float が残るので本番では使わない。14.5 節） |
| `if len(_m.graph.node) == 0 …: raise` | **「ファイルができた＝成功」と読まない**。2GB を超えるモデルで、保存が例外を出さずに空の成果物を残したことがある |

---

## 14.4 重みの per-channel と「生成物を数える」（`weight_percg.py`）

### 何が起きていたか

`per_channel=True` を渡していたのに、生成物の重みは **390 本すべて per-tensor** でした（7.2 節）。ORT は MatMul の重みについては
`TensorQuantOverrides` に `axis` が書かれているときだけ per-channel にします。

### どの重みに axis を付けるか（`matmul_weight_axes`）

<<CODE app/pipeline/weight_percg.py | ^def matmul_weight_axes | ^    return out$ | omit=def trace_init\(name: str, depth: int = 0\)..    skipped: dict\[str, set\[str\]\] = \{\}>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `if n.op_type != "MatMul"`・`trace_init(n.input[1])` | MatMul の**右辺**に入っている初期化子（重み）を、Transpose などを辿って見つける |
| `if len(t.dims) != 2 or … < min_elems` | 2 次元で、ある程度大きいものだけ |
| `uses = terminal_ops(nm)`・`if uses != {"MatMul"}` | ★MatMul の右辺**だけ**に使われる重みに限る。`tie_word_embeddings` で lm_head の重みは埋め込みの Gather とも共有されていて、axis を付けると実行時に落ちた（7.4 節） |
| `out[nm] = 1` | `MatMul(x[…, K], W[K, N])` の出力チャンネルは N なので axis = 1 |

<<CODE app/pipeline/weight_percg.py | ^def build_overrides | ^    return ov, axes>>

書式は `{"重みの名前": [{"axis": 1}]}`（リストの中は辞書。二重リストにすると `ValueError`）。

### 生成物を数える（`inspect_weights`・`verify`）

<<CODE app/pipeline/weight_percg.py | ^def inspect_weights | ^        "rows": rows,>>

<<CODE app/pipeline/weight_percg.py | ^def verify\(qdq_path | ^    return info$>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `if nd.op_type != "DequantizeLinear" or nd.input[0] not in inits` | 重みは「初期化子（量子化済みの整数）→ DequantizeLinear」の形で入っている |
| `n_sc = int(np.prod(sc.dims))`・`"per_channel": n_sc > 1` | **scale が何個あるか**が真実。1 個なら per-tensor、N 個なら per-channel |
| `verify`: `if info["big_per_channel"] != expect_per_channel` | 期待の本数と違えば例外で止める。「per_channel=True を渡した」は根拠にならない |

```powershell
& $PY2 -m app.pipeline.weight_percg --prepped work/ops/<名前>/<名前>_prepped.onnx        # 量子化の対象になる重みを数える
& $PY2 -m app.pipeline.weight_percg --qdq work/ops/<名前>/<名前>_final_qdq.onnx --expect 168   # 生成物を検査（違えば異常終了）
```

---

## 14.5 原因の特定: Oracle 実験・指標・誤差の分解

### 何を確かめたか

24 層が壊れたとき、原因は「残差の巨大な値」と**推定**していました。対策を作る前に、**量子化を 1 か所だけ外して（理想の条件にして）精度が戻るか**を測りました（Oracle 実験、4 章）。

```powershell
& $PY2 -m app.pipeline.oracle_exclude --prepped <prepped.onnx> --fp32 <fp32.onnx> --calib-npz <較正.npz> --eval-npz <評価.npz> --workdir work/oracle6 --out results/P3_oracle6.json
```

### 外す所の決め方と、量子化し直して測るループ（`oracle_exclude.py`）

<<CODE app/pipeline/oracle_exclude.py | ^PATTERNS = \{ | ^    return \[n.name for n in model.graph.node if n.op_type in ops and n.name\]>>

<<CODE app/pipeline/oracle_exclude.py | for pat in args.patterns: | rows.append\(m\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `PATTERNS`: `"residual": ["Add"]`・`"rmsnorm": ["Pow", "ReduceMean", "Sqrt", "Reciprocal"]` | 外す所は **op の種類**で選ぶ（ノードの名前はモデルごとに違う） |
| `quantize_qdq.run(…, nodes_to_exclude=excl or None)` | 指定したノードだけ量子化しないで量子化し直す。外した所は float のまま（**v68 では C-1 違反で動かない**。原因を確定するためのホストの診断） |
| `metrics(np.concatenate(ref, 0), …)` | fp32 と比べて top-1・KL など（下の `metrics`） |

結果（`results/P3_oracle6.json`）:

<<JSON results/P3_oracle6.json | [{"外した所": r["pattern"], "外したノード": r["excluded_nodes"], "top-1 一致": round(r["top1_agreement"], 3), "KL": round(r["kl_fp32_to_quant"], 3)} for r in d]>>

残差（Add）を外しても top-1 は変わらず、RMSNorm を外すと大きく戻る ＝ 推定は外れで、主因は RMSNorm（4.2 節）。これが 14.2 節の書き換えにつながりました。
QDQ の Q/DQ の組を直接外す版（量子化し直さない・多入力のモデルも扱える）は `oracle_qdq.py` の `strip`（7 章の「重みだけ・活性化だけ float」に使った）。

### 指標（`accuracy_logits.metrics`）

<<CODE app/pipeline/accuracy_logits.py | ^def metrics\(ref | ^    \}$>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `top1 = (r.argmax(-1) == t.argmax(-1)).mean()` | 次の語の 1 位が fp32 と同じ割合。生成に一番直結する |
| `np.argpartition(-r, k, …)[:, :k]` | 上位 k 語の集合の重なり（貪欲でない生成の余裕） |
| `kl = … pr * (log pr − log pt)` | 確率分布のずれ（fp32 ‖ 量子化）。0 に近いほどよい |
| `rank = (t > t[…, best]).sum(-1) + 1` | fp32 の 1 位が量子化側で何位か。1 位を外しても 2 位なら実害は小さい |
| `logits_rel_l2` | 参考（層の測定とつなげるため）。**L2 が大きくても 1 位が保たれれば実害は小さい**（演習 ex03 の 1） |

### 誤差の分解（`error_decompose.py`）

<<CODE app/pipeline/error_decompose.py | ^def _rel\(a: np.ndarray | ^    return float\(np.linalg.norm\(x - y\) / d\) if d else 0.0>>

<<CODE app/pipeline/error_decompose.py | # \*\*分母はすべて ‖fp32‖ に揃える\*\* | t_err = \[_rel\(r, d, r\) for r, d in zip\(ref, dev\)\]>>

fp32 → ホスト QDQ → 実機 の間に「ホストの CPU で同じ QDQ モデルを動かした結果」を挟むと、量子化の誤差と、実機で足された誤差（QNN への変換・融合・丸め）に分けられます。
★分母は常に ‖fp32‖（最初は分母がそろっておらず、比べられない数字を並べていた。4.5 節 罠 1）。L2 なので 3 つは足し算の関係ではなく、直交に近ければ二乗和（`quadrature_check`）。

<<JSON results/P3_error_decompose.json | [{"モデル": r["label"], "量子化": round(r["quant_rel_l2"], 4), "実機の追加分": round(r["qnn_htp_rel_l2"], 4), "合計": round(r["total_rel_l2"], 4)} for r in d]>>

重みが固定の MatMul なら実機の追加分は 0.1% 前後で、HTP が本質的に不正確なわけではない（1 層では実機の追加分が量子化より大きい。6.1 節 例 5）。

---

## 14.6 実機だけ壊れる（C-13）・DLC の検査・C-8 の右辺 8bit

### C-13: RMSNorm の gamma（`fix_norm_gamma.py`）

**ONNX では正しいのに実機だけ壊れた**原因は、変換器が RMSNorm を融合 op `RmsNorm` にして gamma を**符号付き 16bit（`sFxp_16`）**として渡すのに、
ORT は **uint16** で量子化していたことでした（量子化値が 32767 を超える要素が黙って頭打ち。9 章・演習 ex04）。

<<CODE app/pipeline/fix_norm_gamma.py | ^DEFAULT_PATTERN = r"norm\\.weight" | ^SIGNED_MAX = 32767>>

<<CODE app/pipeline/fix_norm_gamma.py | ^def run\(src: Path, dst | "requant_rel_l2": err\}\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `DEFAULT_PATTERN = r"norm\.weight"` | 対象は `norm.weight` を**含む**もの全部。最初は名前を列挙していて、Qwen3 の `q_norm`・`k_norm` を取りこぼした |
| `g = (q - zp) * sc` | いまの量子化値から gamma の実数に戻す |
| `new_sc = |g|max / SIGNED_MAX`・`np.clip(np.rint(g / new_sc), -32768, SIGNED_MAX).astype(np.int16)` | **符号付き対称**で入れ直す。符号付きとして読まれても値が変わらない（入れ直しの誤差 0.002% 前後） |
| `inits[w].CopyFrom(…)` | 重み・scale・zero_point の初期化子を書き換える |

### DLC の検査（`dlc_audit.py`）

融合されると ONNX のノードは消えるので、ONNX だけ見ていても気づけません。**HTP が実際に実行する DLC** を `qairt-dlc-info` で開いて確かめます（9.6 節 ①）。

<<CODE app/pipeline/dlc_audit.py | ^LIMITS = \{ | "sFxp_32": \(-2\*\*31, 2\*\*31 - 1\), "uFxp_32": \(0, 2\*\*32 - 1\)\}>>

<<CODE app/pipeline/dlc_audit.py | if t in LIMITS and e\["scale"\] > 0: | problems.append\(row\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `LIMITS` | 型ごとに表せる量子化値の範囲（`sFxp_16` は −32768〜32767） |
| `q_max = e["max"] / e["scale"] - e["offset"]` | エンコーディング（min・max・scale・offset）が要求する量子化値。★QNN の約束は `実数 = (量子化値 + offset) × scale`。符号を逆にすると全テンソルが「不整合」に見える（一度間違えた。9.6 節 ③） |
| `if q_max > hi + 1.5 or q_min < lo - 1.5` | 型の範囲からはみ出していれば問題（丸めの 1.5 段は許す） |

```powershell
& $PY2 -m app.pipeline.dlc_audit --dlc work/ops/<名前>/ctx/<tag>.dlc        # push の前のゲート（deploy_pm64.py も中で呼ぶ）
```

### C-8: act×act の MatMul の右辺を uint8 に（`downgrade_actact_rhs.py`）

<<CODE app/pipeline/downgrade_actact_rhs.py | for name in rhs: | "range": \[lo, hi\], "new_zp": new_z\}\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `lo, hi = (0 - z) * s, (65535 - z) * s` | いまの uint16 の量子化が表している実数の範囲 |
| `new_s = (hi - lo) / 255.0` | 同じ範囲を 256 段（uint8）で表す scale |
| `sn, zn = f"{name}_scale_u8", …`・`n.input[1] = sn` | ★scale・zero_point の初期化子は**ほかのノードと共有されている**。その場で書き換えると無関係なテンソルまで変わる（lm_head の DQ が uint16/uint8 混在になり読めなくなった）。新しい初期化子を作って、このノードだけ差し替える |

量子化の設定（`TensorQuantOverrides` で uint8 を指定）だけでは uint16 のまま出てくることがあったので、**できあがった QDQ を後から書き換えて**います。

---

## 14.7 decode と固定長の KV キャッシュ（`export_qwen_decode.py`）

### 何を確かめたか

NPU（v68）は動的形状を扱えないので、HF の伸びるキャッシュは使えません。**重みは本物のまま、配線だけ自前で**、長さ L の固定のキャッシュを入出力に持つ 1 語ぶんのグラフを書き出しました（1.6 節・10 章）。

```powershell
& $PY2 app/pipeline/export_qwen_decode.py --layers 24 --cache-len 64            # decode（seq=1＋固定長キャッシュ）
& $PY2 app/pipeline/export_qwen_decode.py --layers 24 --cache-len 64 --verify-only   # HF と logits を突き合わせるだけ
```

<<CODE app/pipeline/export_qwen_decode.py | def forward\(self, input_ids, cos, sin, attn_mask, \*caches\): | return \(self.lm_head\(h\), \*presents\) | omit="""cos / sin は\*\*ホストが計算して渡す\*\*。..        h = self.embed\(input_ids\) | omit=# 左シフトして末尾に書く（長さ L のまま）。..            pk = torch.cat>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `cos, sin` が入力 | RoPE の cos/sin は位置だけで決まる表引きなので、**ホストが作って渡す**（端末に Cos/Sin の op を持ち込まない） |
| `if hasattr(attn, "q_norm")` | Qwen3 はヘッドごとの RMSNorm がある（Qwen2.5 には無い） |
| `pk = torch.cat((past_k, k), dim=2)[:, :, 1:, :]` | 左シフトして末尾に書く（長さ L のまま）。★「連結してから捨てる」順にしたのは、past が **Slice に直接入ると量子化器が float のまま残す**ため（C-1 違反で context binary が作れなかった）。「ずらしてから足す」と結果は同じ（演習 ex05 の 1）。**ハードではなくツールの制約**（1.6 節） |
| `pk.repeat_interleave(self.rep, dim=1)` | GQA: KV のヘッドを Q のヘッドの数に増やす（export では Expand になり、14.2 節の C-10 で Concat に書き換える） |
| `scores = scores + attn_mask` | マスクは 0 か −60（C-9）。ホストが毎回作る（有効なのは後ろ n 個） |

<<CODE app/pipeline/export_qwen_decode.py | tok_ids = \[9707, 11, 1879, 0, 3555, 374\] | t1.append\(bool\(ref.argmax\(-1\).item\(\) == mine.argmax\(-1\).item\(\)\)\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `model.model.layers = step.layers` | ★HF 側も同じ層数にする（自前 2 層と HF 24 層を比べて、配線の誤りと誤判定したことがある） |
| `ref = model(input_ids=ids, use_cache=False).logits[:, -1:, :]` | HF は毎回全部の履歴を入れて最後の位置の答え |
| `mask[0, 0, 0, cache_len - n:] = 0.0`・`out = step(…)` | 自前は 1 語ずつ、キャッシュを持ち回して |
| `d = float((ref - mine).abs().max())` | ここが合わなければ、以降の精度の議論は全部意味が無い（8.8 節「無効化して既知の方式と一致するか」と同じ考え方） |

---

## 14.8 自分で書き換えを 1 つ足すとき

14.2 節の型どおりに、例えば新しい Graph Rewrite を足すなら:

1. **関数を足す**（`preprocess.py`）: 対象を**形（op の並び）で**見つけ、新しいノード・初期化子を足し、変えたことを `changed` に入れて返す。名前の衝突は `_unique` で避ける
2. **`run` の呼ぶ順に入れる**: 量子化より前に。ほかの書き換えとの順番の理由をコメントに書く（14.2 節の ①〜⑥）
3. **数値で等価か確かめる**: 書き換え前後の fp32 モデルを同じ入力で動かし、最大差・相対 L2・top-1 を見る（5 章「等価変形は必ず数値で確かめる」）
4. **Oracle で上限を知る**: 書き換えの対象の量子化を外したときの精度（14.5 節）。書き換えがそこまで戻せば十分
5. **生成物を見る**: `weight_percg --qdq`・`fix_norm_gamma --audit`・`dlc_audit`（14.4・14.6 節）。融合されると ONNX のノードは消える
6. **記録**: `results/` に結果、`DoC/StudyMemo/` に学んだこと、このテンプレート（`app/study/tour/14_code.md`）に抜き出しを足して作り直す

---

## 14.9 この章で抜き出したソースの md5

<<MD5_TABLE>>
