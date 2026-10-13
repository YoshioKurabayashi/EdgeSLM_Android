r"""窓64のマスク入力版を端末へ配備して検証する（S8 → push → 精度 → 生成）。

## なぜ1本にまとめるか

context 生成に1時間以上かかるので、**出来た直後の手順でミスをしたくない**。
2026-09-13 に「生成途中のバイナリを包んで push し、err 1002 で1時間分の切り分けを
やり直した」ことがある（results/P4_20260913_prefillmask.md 5節）。そこで

  1. 生成器の出力が**安定していること**を確かめる（サイズと mtime を2回見る）
  1.5 **DLC を検査する**（C-13。型とエンコーディングの不整合。`dlc_audit.py`）
  2. 包む（`embed_mode=0`。C-12）
  3. push して **md5 を3点（生成器の出力 / 配った側 / 端末）で照合**
  4. probe で全ノード HTP を確認
  5. 最終位置の精度を測る
  6. 生成を回してホスト参照と突き合わせる

をこの順で固定する。どこかで落ちたら止まる。

## 使い方

    python -m app.pipeline.deploy_pm64                      # 窓64 per-tensor（従来）
    python -m app.pipeline.deploy_pm64 --tag qwen_pm64pc    # 窓64 per-channel
    python -m app.pipeline.deploy_pm64 --skip-push          # 既に push 済みのとき

`--tag` は `work/ops/<tag>/` 以下の名前をまとめて切り替えるだけで、手順は同じである。
**per-tensor 版のファイルは1つも触らない**ので、端末には両方が並んで残る。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV_RUNNER = "/sdcard/Android/data/org.edgeslm.runner/files"


def audit_dlc(p: "Paths") -> None:
    """**HTP が実際に実行する DLC** を検査する（C-13 の再発防止）。

    ONNX 側で正しく量子化していても、融合 op の重みが DLC で別の型として
    渡されることがある（2026-09-14: RMSNorm の gamma が uint16 -> sFxp_16 で
    32767 超が黙って頭打ち。RMSNorm 単体で相対L2 13.54%）。
    **ONNX を見ているだけでは気付けない**ので、push の前にここで止める。
    """
    dlc = p.ctx_dir / f"{p.short}.dlc"
    if not dlc.is_file():
        print(f"  （DLC が無いので省略: {dlc}）")
        return
    from pipeline import dlc_audit
    a = dlc_audit.audit(dlc)
    print(f"  DLC {dlc.name}: エンコーディング付きテンソル {a['tensors']} 本")
    if a["problems"]:
        for r in a["problems"][:5]:
            print(f"    ★ {r['name']}: {r['problem']}")
        raise SystemExit(f"DLC の型とエンコーディングが不整合（{len(a['problems'])} 本）。"
                         "C-13 を確認すること")
    print("  OK  型とエンコーディングの不整合なし")


class Paths:
    """`--tag` から実際のファイル名を組み立てる。

    生成器は `qwen_pm64` → `qwenpm64` のようにアンダースコアを落とした名前を使う。
    """

    def __init__(self, tag: str, ctx_sub: str = "ctx", short: str | None = None) -> None:
        self.tag = tag
        short = short or tag.replace("_", "")   # qwen_pm64pc -> qwenpm64pc
        self.short = short
        base = ROOT / "work/ops" / tag
        self.ctx_dir = base / ctx_sub
        self.gen_bin = self.ctx_dir / f"ctx_{short}" / f"{short}_ctx.SM8350.bin"
        self.side_bin = self.ctx_dir / f"{short}_ctx.SM8350.bin"
        self.epctx = self.ctx_dir / f"{tag}_epctx.onnx"
        self.qdq = base / f"{tag}_final_qdq.onnx"
        self.model_rel = f"models/ops/{tag}_epctx.onnx"

ENV = dict(os.environ)
ENV["MSYS2_ARG_CONV_EXCL"] = "/data/local/tmp;/sys;/vendor;/dev;/system;/proc;/sdcard;/storage"


def adb(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *a], capture_output=True, text=True, env=ENV)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def wait_stable(p: Path, checks: int = 2, gap: float = 10.0) -> None:
    """サイズと mtime が動かなくなるまで待つ。

    ★これを省くと「書き換え途中のファイル」を掴む（2026-09-13 に踏んだ）。
    """
    last = None
    for i in range(checks * 6):
        if not p.is_file():
            print(f"  まだ無い: {p}")
            time.sleep(gap)
            continue
        st = (p.stat().st_size, int(p.stat().st_mtime))
        if st == last:
            print(f"  安定: {st[0]:,} bytes")
            return
        last = st
        print(f"  監視中: {st[0]:,} bytes  mtime {st[1]}")
        time.sleep(gap)
    raise SystemExit(f"安定しない: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--tag", default="qwen_pm64",
                    help="work/ops/<tag>/ を対象にする（例: qwen_pm64pc）")
    ap.add_argument("--ctx-sub", default="ctx",
                    help="work/ops/<tag>/ の下の context ディレクトリ（例: ctx_c13）")
    ap.add_argument("--qdq", type=Path, help="包むときの QDQ（省略時は <tag>_final_qdq.onnx）")
    ap.add_argument("--short", help="生成器が使った短い名前（省略時は tag から _ を除いたもの）")
    ap.add_argument("--model-name", help="端末に置く epctx のファイル名（省略時は <tag>_epctx.onnx）")
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "app"))

    P = Paths(args.tag, args.ctx_sub, args.short)
    if args.qdq:
        P.qdq = args.qdq
    if args.model_name:
        P.epctx = P.ctx_dir / args.model_name
        P.model_rel = f"models/ops/{args.model_name}"
    GEN_BIN, SIDE_BIN, EPCTX, QDQ = P.gen_bin, P.side_bin, P.epctx, P.qdq
    MODEL_REL = P.model_rel
    print(f"対象: {args.tag}  ->  {MODEL_REL}")

    if not args.skip_push:
        step("1. 生成器の出力が安定しているか")
        wait_stable(GEN_BIN)

        # 生成器は `ctx/ctx_<short>/` の下に置く。配る側は `ctx/` 直下に揃えたいので
        # **安定を確認したあとで**コピーする（先にコピーすると途中のものを掴む）。
        if not SIDE_BIN.is_file() or SIDE_BIN.stat().st_size != GEN_BIN.stat().st_size:
            shutil.copyfile(GEN_BIN, SIDE_BIN)
            print(f"  コピー: {GEN_BIN.name} -> {SIDE_BIN}")

        step("1.5 DLC を検査する（C-13）")
        audit_dlc(P)

        step("2. EPContext で包む（embed_mode=0 / C-12）")
        from pipeline import wrap_epctx
        r = wrap_epctx.run(GEN_BIN, QDQ, EPCTX, args.tag, embed=False)
        print(f"  {EPCTX.name}  {r['bytes']} bytes  embed_mode={r['embed_mode']}")
        for i in r["inputs"]:
            print(f"    in  {i['name']:10s} {i['shape']} {i['dtype']}")
        for o in r["outputs"]:
            print(f"    out {o['name']:10s} {o['shape']} {o['dtype']}")

        step("3. push して md5 を3点照合")
        h_gen, h_side = md5(GEN_BIN), md5(SIDE_BIN)
        print(f"  生成器の出力 {h_gen}")
        print(f"  配る側       {h_side}")
        if h_gen != h_side:
            raise SystemExit("包む前後で中身が違う。生成が終わっていない可能性がある")
        for p in (SIDE_BIN, EPCTX):
            res = adb("push", str(p.resolve()), f"{DEV_RUNNER}/models/ops/{p.name}")
            if res.returncode != 0:
                raise SystemExit(f"push 失敗: {p.name}\n{res.stderr}")
            print(f"  pushed {p.name}")
        out = adb("shell", f"md5sum {DEV_RUNNER}/models/ops/{SIDE_BIN.name}").stdout.split()
        print(f"  端末         {out[0] if out else '?'}")
        if not out or out[0] != h_gen:
            raise SystemExit("端末のバイナリが一致しない")

    step("4. probe（全ノード HTP か）")
    job = {"job_id": f"{args.tag}_probe", "task": "probe", "model": MODEL_REL,
           "ep": {"provider": "QNN", "backend": "libQnnHtp.so",
                  "options": {"offload_graph_io_quantization": "0",
                              "qnn_context_priority": "high"},
                  "session": {"intra_op_num_threads": 1, "log_severity_level": 2}}}
    jp = ROOT / f"work/qwen_pm64/{args.tag}_probe.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    adb("push", str(jp.resolve()), f"{DEV_RUNNER}/jobs/{jp.name}")
    adb("shell", f"rm -f {DEV_RUNNER}/out/last_result.json")
    adb("shell", "am", "start", "-n", "org.edgeslm.runner/.RunnerActivity",
        "-a", "org.edgeslm.runner.RUN", "--es", "job", f"jobs/{jp.name}")
    for _ in range(60):
        time.sleep(2)
        if adb("shell", f"[ -f {DEV_RUNNER}/out/last_result.json ] && echo y").stdout.strip():
            break
    else:
        raise SystemExit("probe がタイムアウト")
    res = json.loads(adb("shell", f"cat {DEV_RUNNER}/out/last_result.json").stdout)
    d = res.get("describe", {})
    print(f"  backend = {d.get('backend')}")
    print(f"  strict  = {d.get('strict_no_cpu_fallback')}")
    for i in d.get("inputs", []):
        print(f"    in  {i['name']:10s} {i['shape']} {i['type']}")
    for o in d.get("outputs", []):
        print(f"    out {o['name']:10s} {o['shape']} {o['type']}")
    if not d.get("ok"):
        raise SystemExit("セッションを開けなかった。logcat を見る")

    print("\n次にやること:")
    print("  精度: python -m app.pipeline.device_accuracy --model " + MODEL_REL +
          " --eval-npz work/qwen_pm64/eval_pm64.npz"
          " --fp32 work/qwen_pm64/qwen05b_prefillmask_L24_S64_last.onnx"
          " --job-id pm64acc --limit 19 --positions last"
          " --work work/devacc_pm64 --out results/P5_20260913_pm64_device.json")


if __name__ == "__main__":
    main()
