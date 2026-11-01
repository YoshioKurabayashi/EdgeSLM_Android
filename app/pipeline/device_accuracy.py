"""実機での精度を測る（P3 / runner v0.3）

## なぜホストのシミュレーションでは足りないのか

ここまでの精度はすべて **ホスト CPU 上で QDQ モデルを走らせた値**である。
Q/DQ ノードが実際に量子化・逆量子化するので**量子化による情報の損失**は出るが、
**HTP が実際にどう計算するか**は含まれていない:

- 積和の順序と中間精度（HTP は固定小数点のアキュムレータを持つ）
- 丸めの方式
- タイリングやレイアウト変換に伴う差

このプロジェクトでは「ホストで通ることと端末で動くことは別」を
C-7 / C-10 / I/O 型の3回学んでいる。**精度も実機で確かめる。**

## 手順

    1. 評価データ(npz) → 生バイト列(.bin) に展開して端末へ push
    2. runner の `infer` task で推論し、出力を .bin で書かせる
    3. 端末から pull して numpy で読み戻し、fp32 参照と比較

出力の形と型は `describe` が返すものを使う（端末側が書いた JSON に含まれる）。

## 入力と出力・動かし方（2026-10-18 追記）

- 入力: 端末に置いた EPContext モデル（--model。端末の models/ からの相対）・評価セットの npz・
  基準の fp32 ONNX（PC で走らせる）。端末は adb でつながり、runner アプリ（org.edgeslm.runner）が入っていること
- 出力: --work の下に、入力の .bin（inputs/）・job の JSON・端末の結果 last_result.json・
  端末の出力 .bin（<job_id>_<番号>_<出力名>.bin。ab_host_device.py がこれを読む）。
  --out を付けると要約の JSON（例: results/P3_device_accuracy.json。記録は results/P3_20260913_device_accuracy.md）
- 動かし方（StudyMemo 14.1）:
    python -m app.pipeline.device_accuracy --model models/ops/<名前>_epctx.onnx --eval-npz <評価.npz> --fp32 <fp32.onnx>
- 使った節: StudyMemo 6.3・4 章（罠 2: パディングで指標が薄まる）・14.1（PC から端末へ job を渡す道）
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

DTYPE = {"float32": np.float32, "float16": np.float16, "int64": np.int64,
         "int32": np.int32, "uint8": np.uint8, "int8": np.int8,
         "uint16": np.uint16, "int16": np.int16}


def _adb(args: list[str], **kw) -> subprocess.CompletedProcess:
    """adb を呼んで結果（標準出力・標準エラー・終了コード）を返す。失敗しても例外にしない（呼ぶ側が見る）。

    ★MSYS2_ARG_CONV_EXCL: Git Bash（MSYS2）の環境から呼ぶと、"/data/local/tmp" のような引数が
    Windows のパスに書き換えられて端末に届かない。書き換えない接頭辞をここで指定している。
    """
    env = {"MSYS2_ARG_CONV_EXCL": "/data/local/tmp;/sys;/vendor;/dev;/system;/proc;/sdcard;/storage"}
    import os
    e = dict(os.environ)
    e.update(env)
    return subprocess.run(["adb", *args], capture_output=True, text=True, env=e, **kw)


#: npz に入っていても**モデルの入力ではない**補助キー。
#  有効トークン数のような評価用メタデータを同じ npz に同梱できるようにするため。
#  除外し忘れると端末へ未知の入力名を渡してジョブが失敗する。
META_KEYS = ("n_valid",)


def export_inputs(npz: Path, out_dir: Path, limit: int | None = None) -> list[dict]:
    """npz を1サンプルずつ生バイト列に展開する。

    引数: npz = 評価セット（入力名ごとに [N, …]）、out_dir = .bin を書くフォルダ、limit = 先頭何件まで。
    戻り値: job の "batches" に入れる dict の list。
      [{"id": "000", "inputs": {"input_ids": "inputs/input_ids_000.bin", ...}}, ...]
    1 件ずつ [1, …] の形で、型はそのまま（tobytes なのでヘッダの無い生の並び）。端末側は
    モデルの入力の形と型（describe）を知っているので、生の並びだけで読める。
    """
    z = np.load(npz)
    names = [n for n in z.files if n not in META_KEYS]
    n = z[names[0]].shape[0]
    if limit:
        n = min(n, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = []
    for i in range(n):
        files = {}
        for name in names:
            a = np.ascontiguousarray(z[name][i:i + 1])
            p = out_dir / f"{name}_{i:03d}.bin"
            p.write_bytes(a.tobytes())
            files[name] = f"inputs/{p.name}"
        batches.append({"id": f"{i:03d}", "inputs": files})
    return batches


def main() -> None:
    """実機で推論し、出力を持ち帰って fp32 と比べる本体（冒頭の手順 1〜3 をそのまま 5 段で行う）。

    段: ① 入力を .bin へ → ② 端末へ push → ③ job を送り、アプリを起動して結果を待つ
        → ④ 端末の出力 .bin を pull → ⑤ 同じ入力で fp32 を PC で走らせ、相対 L2・cos・top-1 を出す
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="端末上の EPContext モデル（models/... 相対）")
    ap.add_argument("--eval-npz", type=Path, required=True)
    ap.add_argument("--fp32", required=True, help="参照となる fp32 ONNX")
    ap.add_argument("--job-id", default="devacc")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--work", type=Path, default=Path("work/devacc"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--positions", choices=["all", "valid", "last"], default="all",
                    help="どの位置で指標を取るか。"
                         "all=全位置 / valid=有効トークンのみ / last=最終位置のみ。"
                         "valid と last は npz に n_valid が必要。"
                         "★パディングのあるモデルでは all は薄まる（指標の罠）")
    args = ap.parse_args()

    # runner アプリの外部ファイル置き場。jobs/・inputs/・out/ はここからの相対で job に書く
    dev_base = "/sdcard/Android/data/org.edgeslm.runner/files"
    args.work.mkdir(parents=True, exist_ok=True)

    # ① 入力を 1 件ずつの .bin に展開する
    print("=== 1. 入力を生バイト列へ展開 ===")
    batches = export_inputs(args.eval_npz, args.work / "inputs", args.limit)
    print(f"  {len(batches)} サンプル")

    # ② 入力の .bin を 1 本ずつ端末へ送る（失敗したら止める）
    print("=== 2. 端末へ push ===")
    for p in sorted((args.work / "inputs").glob("*.bin")):
        r = _adb(["push", str(p.resolve()), f"{dev_base}/inputs/{p.name}"])
        if r.returncode != 0:
            raise SystemExit(f"push 失敗: {p.name}\n{r.stderr}")
    print(f"  {len(batches) * len([n for n in np.load(args.eval_npz).files if n not in META_KEYS])} ファイル")

    # job の JSON（端末の RunnerService → job.cpp が読む）。task=infer で batches を 1 件ずつ推論し、出力を out/ に書く。
    #   ep: QNN の HTP で動かす。offload_graph_io_quantization=0 は入出力の量子化を CPU に出さない指定。
    #   intra_op_num_threads=1: CPU 側の並列は使わない（HTP で動かすので不要）
    job = {
        "job_id": args.job_id,
        "task": "infer",
        "model": args.model,
        "ep": {"provider": "QNN", "backend": "libQnnHtp.so",
               "options": {"offload_graph_io_quantization": "0",
                           "qnn_context_priority": "high"},
               "session": {"intra_op_num_threads": 1, "log_severity_level": 2}},
        "output": {"dir": "out"},
        "batches": batches,
    }
    job_path = args.work / f"{args.job_id}.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    # ③ job を送り、前の結果を消してからアプリを起動し、last_result.json ができるまで 2 秒ごとに待つ（最大 90 回＝180 秒）
    print("=== 3. 端末で推論 ===")
    # ★run_runner.sh を subprocess で呼ぶのはやめた。
    #   Python(Windows exe) から `bash` が解決できず**黙って何も起きない**まま
    #   「結果が見つからない」と出て、原因が分かりにくかった（2026-09-13）。
    #   adb を直接叩く方が依存が少なく、失敗も見える。
    _adb(["push", str(job_path.resolve()), f"{dev_base}/jobs/{job_path.name}"])
    _adb(["shell", f"rm -f {dev_base}/out/last_result.json {dev_base}/out/last_error.json"])
    _adb(["shell", "am", "start", "-n", "org.edgeslm.runner/.RunnerActivity",
          "-a", "org.edgeslm.runner.RUN", "--es", "job", f"jobs/{job_path.name}"])
    import time
    for _ in range(90):
        time.sleep(2)
        if _adb(["shell", f"[ -f {dev_base}/out/last_result.json ] && echo yes"]).stdout.strip():
            break
    else:
        raise SystemExit("端末側がタイムアウトした")
    local_res = args.work / "last_result.json"
    _adb(["pull", f"{dev_base}/out/last_result.json", str(local_res.resolve())])

    # ★完了判定は**標準出力の文言ではなく成果物**で行う。
    #   run_runner.sh の出力は環境によって取りこぼすことがあり、
    #   実際に端末側は成功しているのに「失敗」と誤判定した（2026-09-13）。
    #   job_id が一致する結果ファイルがあるかどうかを見る。
    res = json.loads(local_res.read_text(encoding="utf-8"))
    if res.get("job_id") != args.job_id:
        raise SystemExit(f"別のジョブの結果が返った: {res.get('job_id')}")
    if not res.get("records"):
        print(json.dumps(res, ensure_ascii=False)[:800])
        raise SystemExit("records が無い")
    bad = [r_ for r_ in res["records"] if "error" in r_]
    if bad:
        print(f"  {len(bad)} 件が失敗: {bad[0].get('error','')[:160]}")

    # ④ 結果の records の各出力（path・name・type・shape）を pull する。got は（番号, 出力の情報, 手元のパス）
    print("=== 4. 結果を回収 ===")
    got = []
    for rec in res["records"]:
        if "outputs" not in rec:
            continue
        for o in rec["outputs"]:
            local = args.work / Path(o["path"]).name
            rr = _adb(["pull", f"{dev_base}/{o['path']}", str(local.resolve())])
            if rr.returncode != 0:
                raise SystemExit(f"pull 失敗: {o['path']}\n{rr.stderr}")
            got.append((rec["id"], o, local))
    print(f"  {len(got)} ファイル")

    # ⑤ 同じ入力で fp32 を PC の CPU で走らせ、端末の出力と比べる
    print("=== 5. fp32 参照と比較 ===")
    import onnxruntime as ort
    z = np.load(args.eval_npz)
    names = list(z.files)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.fp32, so, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    n_valid = z["n_valid"] if "n_valid" in z.files else None
    if args.positions != "all" and n_valid is None:
        raise SystemExit(f"--positions {args.positions} には npz の n_valid が必要")

    def slice_positions(arr: np.ndarray, i: int) -> np.ndarray:
        """[1,S,V] を評価対象の位置だけに絞る。

        ★右詰めパディング前提（build_mask と同じ約束）。
        パディング位置の出力は意味を持たないので、混ぜると指標が薄まる。
        生成で実際に使うのは**最終位置だけ**である。
        """
        if arr.ndim != 3 or args.positions == "all":
            return arr
        if args.positions == "last":
            return arr[:, -1:, :]
        off = arr.shape[1] - int(n_valid[i])
        return arr[:, off:, :]

    # ★出力は**名前で対応付ける**。decode モデルは logits + present_k/v で5本あり、
    #   全部を出力[0] と比べると形が合わず落ちる（2026-09-13 に踏んだ）。
    out_names = [o.name for o in sess.get_outputs()]
    primary = out_names[0]
    per_output: dict[str, list[float]] = {}

    # ref_cache: 同じサンプルの fp32 は出力が何本あっても 1 回だけ走らせる
    rel, cos, top1 = [], [], []
    ref_cache: dict[int, list] = {}
    for sid, o, local in got:
        i = int(sid)
        if i not in ref_cache:
            ref_cache[i] = sess.run(None, {n: z[n][i:i + 1] for n in in_names})
        oname = o.get("name", primary)
        if oname not in out_names:
            raise SystemExit(f"端末の出力名 '{oname}' が fp32 モデルに無い（{out_names}）")
        ref = ref_cache[i][out_names.index(oname)]
        # 端末の出力は、端末が返した型と形で読み戻す（DTYPE は型名 → numpy の型の表）
        dev = np.frombuffer(local.read_bytes(), dtype=DTYPE[o["type"]]).reshape(o["shape"])
        # 主出力以外は相対L2だけ記録して、要約の top-1 などには混ぜない
        if oname != primary:
            a_ = ref.astype(np.float64).ravel()
            b_ = dev.astype(np.float64).ravel()
            per_output.setdefault(oname, []).append(
                float(np.linalg.norm(a_ - b_) / (np.linalg.norm(a_) + 1e-30)))
            continue
        ref = slice_positions(ref, i)
        dev = slice_positions(dev, i)
        a = ref.astype(np.float64).ravel()
        b = dev.astype(np.float64).ravel()
        rel.append(float(np.linalg.norm(a - b) / np.linalg.norm(a)))
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        # ★logits かどうかは**語彙サイズ級かで判定する**。
        #   当初 shape[-1] > 100 にしていたため、層出力 [1,16,896] の
        #   hidden 次元に対する argmax を「top-1 一致」として出してしまった（無意味）。
        if ref.ndim == 3 and ref.shape[-1] >= 10000:
            top1.append(float((ref.argmax(-1) == dev.argmax(-1)).mean()))

    if per_output:
        # KVキャッシュ出力は**次のステップの入力になる**ので、ここの誤差は
        # ステップを重ねると累積する。主出力と分けて見る。
        summary_extra = {k: {"rel_l2_mean": float(np.mean(v)),
                             "rel_l2_max": float(np.max(v)), "n": len(v)}
                         for k, v in sorted(per_output.items())}
    else:
        summary_extra = {}

    # 要約: 相対 L2・cos の平均と最悪、端末の OrtRun の時間（ms）の平均。logits のときだけ top-1 一致。
    #   ★rel_l2 はここでは基準 a（fp32）で割る（ab_host_device.metrics と同じ向き）
    summary = {"samples": len(rel), "positions": args.positions,
               "primary_output": primary, "other_outputs": summary_extra,
               "rel_l2_mean": float(np.mean(rel)), "rel_l2_max": float(np.max(rel)),
               "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)),
               "ort_run_ms_mean": float(np.mean([r_["ort_run_ms"] for r_ in res["records"]
                                                 if "ort_run_ms" in r_]))}
    if top1:
        summary["top1_agreement"] = float(np.mean(top1))

    print(f"  サンプル {summary['samples']} 件")
    print(f"  相対L2誤差   平均 {summary['rel_l2_mean']:.4%}  最大 {summary['rel_l2_max']:.4%}")
    print(f"  cos類似度    平均 {summary['cosine_mean']:.6f}  最小 {summary['cosine_min']:.6f}")
    if top1:
        print(f"  top-1 一致   {summary['top1_agreement']:.1%}")
    for k, v in summary.get("other_outputs", {}).items():
        print(f"  {k:14s} 相対L2 平均 {v['rel_l2_mean']:.4%}  最大 {v['rel_l2_max']:.4%}")
    print(f"  OrtRun       平均 {summary['ort_run_ms_mean']:.2f} ms")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
