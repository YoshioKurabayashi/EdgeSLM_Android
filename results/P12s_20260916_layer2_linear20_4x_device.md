# P12s — linear_20 の4x rangeは実機で18.67%までしか改善せず、「広げすぎ」ゾーンに入った

2026-09-16 / Qwen3-0.6B 28層 KV256 / 実機 Xperia 5 III（SM8350 / HTP v68）/ 入力は P12j/P12m と同じ step41

## 結論

**P12rでHostの自由生成が完全に正常化した4x range（margin=6.0）を層2部分モデルで実機に流すと、
add_26 の Host↔HTP 実数誤差は 21.95%（baseline）→ 18.67%（4x）にしか下がらなかった。**
narrow（1x, margin=1.5）の 8.61% と比べると改善幅はごく小さい。

事前に決めていた判定基準（P12rの直後にユーザーと合意）:

| 4xの結果 | 判定 |
|---|---|
| 15%以下 | 前向き。28層本番候補へ |
| 18〜20% | rangeを広げすぎ。2.5x/3x/3.5xを追加探索 |
| 10〜14% | 前向き。3x付近の細かい探索は省略して次へ |

**実測 18.67% は「18〜20%」ゾーンに入り、「広げすぎ」の判定になった。**

## 1. 方法

- P12mと同じ層2部分モデル（`work/q3_l2/l2ctl_qdq.onnx`、add_17→add_26を切り出した3出力版）を複製
- `linear_20_scale`/`linear_20_zero_point` だけを P12r の4x（margin=6.0）の値に差し替え。
  `mul_41` は baseline のまま変更していない
  - 旧: scale=0.11333262920379639 / zero=1489（表現範囲 −168.75〜7258.5）
  - 新: scale=0.0007177746156230569 / zero=21833（表現範囲 −15.67〜31.37）
- Host（ORT CPU）で baseline / 4x それぞれの add_26 を計算し、同じ入力を実機（QNN HTP, strict=True）で走らせて比較
- ビルド手順・比較方法は P12m（`layer2_kv_narrow.py`）と同一。新スクリプトは
  `app/pipeline/layer2_linear20_4x_device.py`

## 2. 結果

| 変種 | add_26 Host↔HTP 実数誤差 | 整数コード差max |
|---|---:|---:|
| baseline（既存範囲） | 21.95% | 2 |
| **linear_20 4x** | **18.67%** | 2 |
| narrow（1x, P12m参考値） | 8.61% | 2 |

Host側の add_26 自体は baseline と4xでほぼ同一（最大コード差1）。実機の差だけが縮んでいる。

## 3. 読み方

- **P12rでHost品質が完全に回復した4xでも、HTP誤差の改善は限定的（21.95%→18.67%、相対15%減）。**
  narrow（1x）が8.61%まで下げていたのに対し、4xはその半分にも届かない
- P12rで見えたトレードオフ曲線（狭いほどHTPに有利・広いほどHost品質に有利）が、実機データでも
  そのまま裏付けられた。**4xはHost品質を守る側に寄りすぎて、HTP改善効果を大きく失っている**
- 事前に決めていた判定ルールにより、**次は2.5x/3x/3.5xの追加探索**に進む。2x（Host自由生成△、
  軽い文字化けあり）と4x（Host自由生成○、HTP改善小さい）の間に、両方を満たす点があるかを見る

## 4. 次

`app/pipeline/linear20_range_sweep.py` を拡張し、margin=2.25(1.5x)/2.5x/3x/3.5x相当の
中間点を追加でHost評価する。Host自由生成が○になる最も狭い点を選び、それを層2部分モデルで
再度実機評価する。

## 5. 成果物

- `app/pipeline/layer2_linear20_4x_device.py`（build / wrap / compare）
- `results/P12s_20260916_layer2_linear20_4x_device/`: `build_audit.json` / `epctx_meta.json` / `compare.json`
- モデル: `work/q3_l2/l2ctl_4x_qdq.onnx` / `l2ctl_4x_epctx.onnx` / `ctx_l2ctl_4x/`
