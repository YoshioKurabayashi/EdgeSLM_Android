"""端末上の NPU で文章を生成する（P4 の最短経路）

## なぜ KV キャッシュを使わないのか

本来の decode は `seq=1 + KVキャッシュ` だが、**v68 では RMSNorm が seq=1 で
融合 op を作れず context 生成に失敗する**ことが分かった（`rms_norm_node___... could not create op`）。
これは別課題として残し、**既に実機で動いている 24層 prefill モデル（seq=16）**で
生成を成立させる。

    プロンプト（16トークン窓）→ NPU → 最後の位置の logits → argmax → 次トークン
    窓を1つずらして繰り返す

毎トークンでモデル全体を再計算するので遅い（1トークン 37ms + 呼び出し overhead）。
**「動く」ことの確認が目的**で、速度は KV キャッシュが通ってから改善する。

文脈は 16トークンに限られる。これも seq を増やしたモデルを作れば伸ばせる。

## トークナイザはまだホスト側

当初目標の「スマホのローカル完結」を厳密に満たすには**端末に移す必要がある**
（CPU で十分）。現時点では PC でトークン化し、端末へはトークンIDを渡している。
**未達であることを明示しておく。**
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

DEV_BASE = "/sdcard/Android/data/org.edgeslm.runner/files"


def _adb(args: list[str]) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e["MSYS2_ARG_CONV_EXCL"] = ("/data/local/tmp;/sys;/vendor;/dev;/system;/proc;"
                               "/sdcard;/storage")
    return subprocess.run(["adb", *args], capture_output=True, text=True, env=e)


def step_on_device(ids: np.ndarray, model: str, work: Path, job_id: str) -> np.ndarray:
    """16トークンを1回 NPU に通し、logits を返す。"""
    work.mkdir(parents=True, exist_ok=True)
    bin_path = work / "ids.bin"
    bin_path.write_bytes(np.ascontiguousarray(ids.astype(np.int64)).tobytes())
    _adb(["push", str(bin_path.resolve()), f"{DEV_BASE}/inputs/gen_ids.bin"])

    job = {"job_id": job_id, "task": "infer", "model": model,
           "ep": {"provider": "QNN", "backend": "libQnnHtp.so",
                  "options": {"offload_graph_io_quantization": "0",
                              "qnn_context_priority": "high"},
                  "session": {"intra_op_num_threads": 1, "log_severity_level": 3}},
           "output": {"dir": "out"},
           "batches": [{"id": "g", "inputs": {"input_ids": "inputs/gen_ids.bin"}}]}
    jp = work / f"{job_id}.json"
    jp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    _adb(["push", str(jp.resolve()), f"{DEV_BASE}/jobs/{jp.name}"])
    _adb(["shell", f"rm -f {DEV_BASE}/out/last_result.json {DEV_BASE}/out/{job_id}_g_logits.bin"])
    _adb(["shell", "am", "start", "-n", "org.edgeslm.runner/.RunnerActivity",
          "-a", "org.edgeslm.runner.RUN", "--es", "job", f"jobs/{jp.name}"])

    for _ in range(60):
        time.sleep(1)
        if _adb(["shell", f"[ -f {DEV_BASE}/out/last_result.json ] && echo y"]).stdout.strip():
            break
    else:
        raise RuntimeError("端末がタイムアウトした")

    out = work / "logits.bin"
    _adb(["pull", f"{DEV_BASE}/out/{job_id}_g_logits.bin", str(out.resolve())])
    if not out.is_file():
        rj = work / "res.json"
        _adb(["pull", f"{DEV_BASE}/out/last_result.json", str(rj.resolve())])
        raise RuntimeError(f"logits を取得できなかった: {rj.read_text(encoding='utf-8')[:300]}")
    return np.frombuffer(out.read_bytes(), dtype=np.float32).reshape(1, len(ids[0]), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/ops/qwen24ext_epctx.onnx")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="日本の首都は")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--work", type=Path, default=Path("work/gen"))
    ap.add_argument("--also-fp32", type=str, default="",
                    help="比較する fp32 ONNX（同じ窓で生成して並べる）")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)

    pid = tok(args.prompt, return_tensors="np").input_ids[0].tolist()
    print(f"プロンプト: {args.prompt!r}  → {len(pid)} トークン {pid}")
    if len(pid) >= args.seq:
        raise SystemExit(f"プロンプトが窓（{args.seq}）以上。短くすること")

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def window(seq_ids: list[int]) -> tuple[np.ndarray, int]:
        """右詰めで窓に収める。最後の有効位置の index も返す。"""
        w = seq_ids[-args.seq:]
        last = len(w) - 1
        if len(w) < args.seq:
            w = [pad] * (args.seq - len(w)) + w
            last = args.seq - 1
        return np.array([w], dtype=np.int64), last

    # ---- 端末で生成 ----
    seq_dev = list(pid)
    t0 = time.time()
    for i in range(args.max_new):
        ids, last = window(seq_dev)
        lg = step_on_device(ids, args.model, args.work, f"gen{i}")
        nxt = int(lg[0, last].argmax())
        seq_dev.append(nxt)
        print(f"  [{i + 1:2d}] {nxt:6d}  {tok.decode(seq_dev[len(pid):])!r}")
        if nxt == tok.eos_token_id:
            break
    dev_text = tok.decode(seq_dev[len(pid):])
    elapsed = time.time() - t0

    result = {"prompt": args.prompt, "device_text": dev_text,
              "device_tokens": seq_dev[len(pid):],
              "seconds": round(elapsed, 1), "window": args.seq}

    # ---- 同じ窓方式で fp32 と比較 ----
    if args.also_fp32:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(args.also_fp32, so, providers=["CPUExecutionProvider"])
        seq_ref = list(pid)
        for _ in range(args.max_new):
            ids, last = window(seq_ref)
            lg = sess.run(None, {"input_ids": ids})[0]
            nxt = int(lg[0, last].argmax())
            seq_ref.append(nxt)
            if nxt == tok.eos_token_id:
                break
        result["fp32_text"] = tok.decode(seq_ref[len(pid):])
        result["fp32_tokens"] = seq_ref[len(pid):]
        n = min(len(result["device_tokens"]), len(result["fp32_tokens"]))
        result["token_match"] = sum(
            1 for a, b in zip(result["device_tokens"][:n], result["fp32_tokens"][:n]) if a == b)
        result["compared"] = n

    print()
    print("=" * 60)
    print(f"プロンプト : {args.prompt}")
    print(f"端末(NPU)  : {dev_text!r}")
    if "fp32_text" in result:
        print(f"fp32(PC)   : {result['fp32_text']!r}")
        print(f"トークン一致: {result['token_match']}/{result['compared']}")
    print(f"所要       : {elapsed:.1f} 秒（{args.max_new} トークン。窓 {args.seq}）")
    print("=" * 60)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
