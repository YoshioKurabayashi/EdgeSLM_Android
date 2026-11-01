r"""Oracle 実験を QDQ モデルの上で直接やる（再量子化なし・多入力対応）。

## なぜ作ったか

`oracle_exclude.py` は **量子化をやり直して**除外対象を変えるため、
①毎回数分かかる ②**単一入力のモデルしか扱えない**（decode は入力52本）。

QDQ 形式は「float のグラフに Q/DQ を挟んだもの」なので、
**挟まっている Q/DQ を外せばその箇所の量子化を無効化できる**。
ホストの CPU はどうせ float で計算するので、外した分の丸めが消えるだけである。
これは「現実には使えない理想条件を与えて原因を特定する」Oracle そのもの
（[DoC/StudyMemo/04_原因の特定と指標の読み方.md]）。v68 では C-1 違反になるので
**診断専用**である。

## 何を外せるか

    --ops Pow ReduceMean Sqrt Reciprocal   … その op の**出力**の Q/DQ を外す
    --graph-inputs past_                   … 接頭辞が一致するグラフ入力の Q/DQ を外す
    --graph-outputs present_               … 同・グラフ出力側

## 使い方

    python -m app.pipeline.oracle_qdq \
        --qdq work/ops/qwen_dec24/qwen_dec24_final_qdq.onnx \
        --fp32 work/qwen_dec24/qwen05b_decode_L24_C64.onnx \
        --eval-npz work/qwen_dec24/eval_dec24.npz \
        --workdir work/oracle_dec24 \
        --case none --case "rmsnorm:Pow,ReduceMean,Sqrt,Reciprocal" \
        --case "residual:Add" --case "softmax:Softmax"
"""
# ---------------------------------------------------------------------------
# 【補足（学習用、2026-10-18 追記）】上の docstring は --help にそのまま出るので、補足・訂正はこのコメントに書く。
# ★2026-10-18 訂正: docstring の「何を外せるか」の旧「--ops … / --graph-inputs past_ / --graph-outputs present_」という
#   引数は無い。実際の指定はすべて --case "名前:中身" で、中身は op 名のカンマ区切り・@in=接頭辞・@out=接頭辞・
#   @layer=12|22（層は | で区切る）・@t=テンソル名・@weights（parse_case() と --case の help の通り）。
#   「使い方」の例は今の引数の形で正しい。
# どの段か: StudyMemo 4.2 節（QDQ の上で直接やる Oracle）・7.1 節（重みと活性化に分ける）・8.5 節（@weights の落とし穴）。
# 結果: results/P8_20260913_oracle_*.json（例: oracle_weights.json では none 68.8%・重みだけ float 68.8%・
#   活性化だけ float 64.6%。ただし @weights は丸め済みの値を畳むだけで重みの量子化は外れていない ― 下の ①-d の注意）。
# 入力:  --qdq（量子化済みのモデル）・--fp32（基準）・--eval-npz（評価の入力。キーはモデルの入力名）。
# 出力:  --workdir/strip_<名前>.onnx（＋ .data）と、--out に case ごとの指標の JSON。
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def assign_layers(g) -> dict[str, int]:
    """各ノードを層に割り当てる（**データフローで伝播させる**）。

    重み由来の初期化子には `layers.N.` が付いている。そこを起点に、
    **トポロジカル順に「入力の層番号の最大値」を前へ流す。**

    ★ノード名の順序で決めてはいけない。重みの `DequantizeLinear` は
      グラフの先頭にまとめて並んでおり、計算ノードと交互にはならない。
      それで「直前に見た層タグを引き継ぐ」とすると、1つの層に
      6000ノードが集まる（2026-09-13 に実際にそうなった）。

    層の境目のノード（residual の Add など）は**後ろの層に寄る**。
    割り当て結果は件数を出して確かめること。
    """
    import re  # noqa
    lay_of_init: dict[str, int] = {}
    for t in g.initializer:
        m = re.search(r"layers\.(\d+)\.", t.name)
        if m:
            lay_of_init[t.name] = int(m.group(1))

    tensor_layer: dict[str, int] = dict(lay_of_init)
    out: dict[str, int] = {}
    for node in g.node:                      # ONNX の node はトポロジカル順
        best = -1
        for i in node.input:
            v = tensor_layer.get(i, -1)
            if v > best:
                best = v
        m = re.search(r"layers\.(\d+)\.", node.name)
        if m:
            best = max(best, int(m.group(1)))
        if best >= 0:
            out[node.name] = best
            for o in node.output:
                tensor_layer[o] = best
    return out


def _maps(g):
    """テンソル名 → それを作るノード（prod）と、テンソル名 → それを使うノードの一覧（cons）を作る。"""
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return prod, cons


def strip(model, ops: list[str], in_prefix: str | None, out_prefix: str | None,
          layers: list[int] | None = None, tensors: list[str] | None = None,
          weights: bool = False):
    """指定した箇所の `QuantizeLinear → DequantizeLinear` ペアを外す。

    戻り値は (model, 外したペア数)。

    【説明（2026-10-18 追記）】引数: ops = 出力の Q/DQ を外す op の種類、in_prefix/out_prefix = グラフ入力・出力の接頭辞、
    layers = 層番号（層単位の Oracle）、tensors = テンソル名、weights = 重みの DQ を定数に畳むか。
    外し方: Q と DQ のノードを消し、DQ の出力を使っていたノードの入力を Q の前の float のテンソルに付け替える。
    ホストの CPU は float で計算するので、外した所だけ丸めが無くなる（v68 では C-1 違反になる、診断専用）。
    """
    g = model.graph
    prod, cons = _maps(g)
    drop: set[str] = set()
    rewire: dict[str, str] = {}      # 旧テンソル名 -> 新テンソル名
    out_rewire: dict[str, str] = {}  # グラフ出力の付け替え
    pairs = 0

    def unwrap_after(tensor: str) -> bool:
        """tensor に付いている Q→DQ を外して、tensor を直接使わせる。"""
        q = next((c for c in cons.get(tensor, []) if c.op_type == "QuantizeLinear"), None)
        if q is None:
            return False
        dq = next((c for c in cons.get(q.output[0], []) if c.op_type == "DequantizeLinear"), None)
        if dq is None:
            return False
        drop.add(q.name)
        drop.add(dq.name)
        rewire[dq.output[0]] = tensor
        return True

    # ① 指定 op の出力
    for n in list(g.node):
        if n.op_type in ops:
            for o in n.output:
                if unwrap_after(o):
                    pairs += 1

    # ①-b 指定した層に属するノードの出力（層単位の Oracle）
    if layers:
        owner = assign_layers(g)
        want = set(layers)
        hit = 0
        for n in list(g.node):
            if owner.get(n.name, -1) in want and n.op_type not in ("QuantizeLinear",
                                                                   "DequantizeLinear"):
                hit += 1
                for o in n.output:
                    if unwrap_after(o):
                        pairs += 1
        print(f"      層 {sorted(want)} に属する計算ノード {hit} 個")

    # ①-c 名前で指定したテンソル（`present_v_12` など）
    if tensors:
        for t in tensors:
            if unwrap_after(t):
                pairs += 1
            else:
                # 出力側にある場合はそちらを試す
                dq = prod.get(t)
                if dq is not None and dq.op_type == "DequantizeLinear":
                    q = prod.get(dq.input[0])
                    if q is not None and q.op_type == "QuantizeLinear":
                        drop.add(q.name); drop.add(dq.name)
                        out_rewire[t] = q.input[0]
                        pairs += 1

    # ①-d 重みの量子化を外す（初期化子 → DequantizeLinear の形）
    #
    # ★活性化の Q/DQ を外しても**重みは量子化されたまま**である。
    #   活性化は「float → Q → DQ → 計算」だが、重みは
    #   「量子化済み初期化子 → DQ → 計算」で前段に Q が無いため、
    #   `unwrap_after` では外れない。A16W8 の W8 を評価するにはここを見る。
    #
    # ★注意（2026-10-18 追記）: ここで作る float の定数は「量子化済みの値 q を DQ した値」
    #   (q - zp) * scale なので、**重みの丸め誤差はそのまま残る**（元の fp32 の重みには戻らない）。
    #   外れるのは計算グラフの中の重みの DequantizeLinear だけで、精度への効果は測れない。
    #   重みの量子化の影響を測るには、元の fp32 の重みを入れ直す必要がある（StudyMemo 7.1 節）。
    if weights:
        from onnx import numpy_helper as _nh
        init_map = {t.name: t for t in g.initializer}
        added = []
        for nd in list(g.node):
            if nd.op_type != "DequantizeLinear":
                continue
            if nd.input[0] not in init_map:
                continue
            q = _nh.to_array(init_map[nd.input[0]]).astype(np.float64)
            sc = _nh.to_array(init_map[nd.input[1]]).astype(np.float64)                 if nd.input[1] in init_map else None
            if sc is None:
                continue
            zp = (_nh.to_array(init_map[nd.input[2]]).astype(np.float64)
                  if len(nd.input) > 2 and nd.input[2] in init_map else 0.0)
            axis = next((a.i for a in nd.attribute if a.name == "axis"), None)
            if sc.ndim == 0:
                deq = (q - zp) * sc
            else:
                # per-channel。axis に合わせて形を整えてから掛ける
                ax = axis if axis is not None else 0
                shape = [1] * q.ndim
                shape[ax] = sc.size
                deq = (q - np.reshape(zp, shape)) * np.reshape(sc, shape)
            nm = nd.output[0] + "_dequantized_const"
            added.append(_nh.from_array(deq.astype(np.float32), nm))
            drop.add(nd.name)
            rewire[nd.output[0]] = nm
            pairs += 1
        g.initializer.extend(added)
        print(f"      重みの DequantizeLinear {len(added)} 個を定数へ畳んだ")

    # ② グラフ入力
    # （past_* のような入力は、入力の直後に Q→DQ が付いている。それを外して float のまま使わせる）
    if in_prefix:
        for vi in g.input:
            if vi.name.startswith(in_prefix) and unwrap_after(vi.name):
                pairs += 1

    # ③ グラフ出力（Q→DQ の前にある float を出力名にする）
    # （present_* のような出力は、出力の手前に Q→DQ がある。出力の名前を Q の前の float に付け替える）
    if out_prefix:
        for vo in g.output:
            if not vo.name.startswith(out_prefix):
                continue
            dq = prod.get(vo.name)
            if dq is None or dq.op_type != "DequantizeLinear":
                continue
            q = prod.get(dq.input[0])
            if q is None or q.op_type != "QuantizeLinear":
                continue
            drop.add(q.name)
            drop.add(dq.name)
            out_rewire[vo.name] = q.input[0]
            pairs += 1

    nodes = [n for n in g.node if n.name not in drop]
    # ★out_rewire の分もノード入力に適用する。
    #   外した DQ の出力がグラフ出力でありつつ**別のノードの入力でもある**ことがある
    #   （decode の `present_*` が次段に使われている形）。片方だけ直すと
    #   "Node input 'present_v_0' is not ... output of a previous node" で落ちる。
    all_rewire = dict(rewire)
    all_rewire.update(out_rewire)
    for n in nodes:
        for i, nm in enumerate(n.input):
            if nm in all_rewire:
                n.input[i] = all_rewire[nm]
    # ★グラフ出力名も付け替える。
    #   外した DQ の出力がそのままグラフ出力になっている場合があり
    #   （decode の `present_*` がそれ）、ここを忘れると
    #   "Graph output (present_k_0) does not exist in the graph" で落ちる。
    for vo in g.output:
        if vo.name in out_rewire:
            vo.name = out_rewire[vo.name]
        elif vo.name in rewire:
            vo.name = rewire[vo.name]
    del g.node[:]
    g.node.extend(nodes)
    return model, pairs


def run_model(path: Path, npz: Path, out_name: str = "logits") -> np.ndarray:
    """モデルを CPU で 1 サンプルずつ動かし、out_name の出力をまとめて返す。

    npz はモデルの入力名をキーに持ち、最初の軸がサンプル数（1 サンプルずつ [i:i+1] で切って与える）。
    戻り値は axis=1 でつないだ配列（logits [1, T, 語彙] を T の向きに並べる形。推定: decode では T=1 なので [1, サンプル数, 語彙]）。
    """
    z = np.load(npz)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    outs = [o.name for o in s.get_outputs()]
    idx = outs.index(out_name) if out_name in outs else 0
    n = z[names[0]].shape[0]
    got = [s.run(None, {k: z[k][i:i + 1] for k in names})[idx] for i in range(n)]
    return np.concatenate(got, axis=1)


def parse_case(spec: str):
    """指定を解く。

        "名前:op1,op2"        … その op の出力
        "名前:@in=past_"      … 接頭辞が一致するグラフ入力
        "名前:@out=present_"  … 同・グラフ出力
        "名前:@layer=12,22"   … その層に属する計算ノードの出力（層単位 Oracle）
        "名前:@t=present_v_12" … 名前を指定したテンソル

    ★2026-10-18 訂正: 旧「"名前:@layer=12,22"」は誤り。中身はカンマで区切って読むので、層は「|」で区切る
    （"名前:@layer=12|22"。--case の help と同じ）。カンマで書くと「22」が op 名として扱われる。
    書かれていなかった指定: "名前:@weights" … 重みの DequantizeLinear を定数に畳む（丸め誤差は残る）。
    "none"（コロン無し）… 何も外さない基準。戻り値: (名前, ops, in 接頭辞, out 接頭辞, layers, tensors, weights)。
    """
    if ":" not in spec:
        return spec, [], None, None, None, None, False
    name, rest = spec.split(":", 1)
    ops, ip, op_, layers, tensors, weights = [], None, None, None, None, False
    for part in rest.split(","):
        part = part.strip()
        if part == "@weights":
            weights = True
        elif part.startswith("@in="):
            ip = part[4:]
        elif part.startswith("@out="):
            op_ = part[5:]
        elif part.startswith("@layer="):
            layers = (layers or []) + [int(x) for x in part[7:].split("|") if x]
        elif part.startswith("@t="):
            tensors = (tensors or []) + [part[3:]]
        elif part:
            ops.append(part)
    return name, ops, ip, op_, layers, tensors, weights


def main() -> None:
    """fp32 の基準を 1 回計算し、--case ごとに Q/DQ を外したモデルを作って比べ、表を出す。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--eval-npz", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out-name", default="logits")
    ap.add_argument("--case", action="append", default=[],
                    help='"名前:op1,op2" / "@in=接頭辞" / "@out=接頭辞" / '
                         '"@layer=12|22" / "@t=テンソル名" / "@weights" / "none"')
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    import sys
    # app/ を import の探し先に足して pipeline.accuracy_logits の指標を使う（python -m でもスクリプト直でも動くように）。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.accuracy_logits import metrics

    print("=== fp32 参照 ===")
    ref = run_model(args.fp32, args.eval_npz, args.out_name)

    rows = []
    for spec in (args.case or ["none"]):
        name, ops, ip, op_, layers, tensors, weights = parse_case(spec)
        if name == "none" and not any((ops, ip, op_, layers, tensors, weights)):
            path, pairs = args.qdq, 0
        else:
            m = onnx.load(str(args.qdq))
            m, pairs = strip(m, ops, ip, op_, layers, tensors, weights)
            path = args.workdir / f"strip_{name}.onnx"
            # 2GB を超えうるので重みは外部データ（.data）で保存する。
            onnx.save(m, str(path), save_as_external_data=True,
                      location=path.name + ".data", all_tensors_to_one_file=True)
        test = run_model(path, args.eval_npz, args.out_name)
        mt = metrics(ref, test)
        rows.append({"case": name, "spec": spec, "pairs_removed": pairs, "metrics": mt})
        print(f"  {name:<22} 外した Q/DQ {pairs:4d} 組  "
              f"top-1 {mt['top1_agreement']:6.1%}  相対L2 {mt['logits_rel_l2']:7.2%}")

    print()
    print(f"{'除外した箇所':<24} {'組数':>5} {'top-1':>8} {'相対L2':>9} {'KL':>8} {'1位の順位':>10}")
    print("-" * 72)
    for r in rows:
        mt = r["metrics"]
        print(f"{r['case']:<24} {r['pairs_removed']:5d} {mt['top1_agreement']:8.1%} "
              f"{mt['logits_rel_l2']:9.2%} {mt['kl_fp32_to_quant']:8.3f} "
              f"{mt['rank_of_fp32_top1_mean']:10.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
