# -*- coding: utf-8 -*-
"""P12x: KVキャッシュの累積差をtime(position)×layer×K/Vで追跡する。

P12wで、点Bの判定位置のadd_26を単体で差し替えても判定は壊れないことが分かった。
「linear_20の1step出力精度は最終失敗を説明する主変数ではない」ということであり、
autoregressiveモデル特有の「1stepの誤差はわずかでも、それが約79stepにわたって
hidden→K/V生成→attention→次hiddenと伝播・蓄積する」ことが真の原因という仮説に移る。

この①では、baseline / 4x / oracle を FP32 と**同じ入力トークン列**（点Bの79トークン）で、
**各モデル自身のKVキャッシュ**を使ってteacher forcingし（P12wとは違い、他モデルの文脈に
固定しない）、各position・各layer・K/VについてFP32比のL2相対誤差を記録する。
どのposition・どのlayerから差が急成長するかを特定するのが目的。

## 使い方

  python kv_trajectory_probe.py --phase capture   # 4モデル×79token×28層×K/Vを採取
  python kv_trajectory_probe.py --phase compare    # position×layerの誤差成長表を出す
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from temporal_qdq_experiment import Runner, tables, tokenizer, ROOT  # noqa: E402
from block_revert_ab import eval_points  # noqa: E402
from add26_causal_probe import sess, FOURX_FULL  # noqa: E402
from linear20_oracle import ORACLE_FULL  # noqa: E402
from temporal_qdq_experiment import SOURCE, OPS  # noqa: E402

import numpy as np  # noqa: E402

OUT = ROOT / "results/P12x_20260916_kv_trajectory_probe"
WORK = ROOT / "work/p12x_kv_trajectory"
N_LAYERS = 28

VARIANTS = {
    "fp32": OPS / "qwen3_dec28c256_prepped.onnx",
    "baseline": SOURCE,
    "4x": FOURX_FULL,
    "oracle": ORACLE_FULL,
}


def emit(o):
    print(json.dumps(o, ensure_ascii=False), flush=True)


def capture():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    pts = eval_points(tok)
    p = next(p for p in pts if p["name"] == "point_B_music")
    ids = p["ids"]
    n = len(ids)
    emit({"phase": "capture", "tokens": n})

    cos, sin = tables()
    # 配列: [condition][layer][K/V] -> (n_positions, 1024)
    store = {name: {"k": np.zeros((n, N_LAYERS, 1024), np.float32),
                     "v": np.zeros((n, N_LAYERS, 1024), np.float32)} for name in VARIANTS}
    baseline_sess = sess(SOURCE)  # "4x"の位置0保護に使う(P12r/P12sと同じ、位置0はbaselineのまま)
    for name, path in VARIANTS.items():
        s = sess(path) if name != "4x" else sess(FOURX_FULL)
        r = Runner(s, cos, sin)
        for pos, tid in enumerate(ids):
            step_s = baseline_sess if (name == "4x" and pos == 0) else s
            got, _ = r.feed(tid, step_s)
            for L in range(N_LAYERS):
                store[name]["k"][pos, L] = got[f"present_k_{L}"][:, :, -1, :].ravel()
                store[name]["v"][pos, L] = got[f"present_v_{L}"][:, :, -1, :].ravel()
        del s, r
        emit({"phase": "capture", "variant": name, "done": True})
    del baseline_sess

    npz = {}
    for name in VARIANTS:
        npz[f"{name}_k"] = store[name]["k"]
        npz[f"{name}_v"] = store[name]["v"]
    np.savez(WORK / "kv_trajectory.npz", **npz)
    emit({"phase": "capture", "wrote": str(WORK / "kv_trajectory.npz")})


def compare():
    OUT.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(WORK / "kv_trajectory.npz"))
    n = d["fp32_k"].shape[0]
    layers_shown = [0, 2, 6, 8, 12, 14, 16, 20, 24, 27]
    positions_shown = sorted(set([0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, n - 1]))
    positions_shown = [p for p in positions_shown if p < n]

    result = {"positions": positions_shown, "layers": layers_shown, "tables": {}}
    for name in ("baseline", "4x", "oracle"):
        for kv in ("k", "v"):
            fp32_arr = d[f"fp32_{kv}"].astype(np.float64)  # (n, L, 1024)
            var_arr = d[f"{name}_{kv}"].astype(np.float64)
            table = {}
            for pos in positions_shown:
                row = {}
                for L in layers_shown:
                    a, b = var_arr[pos, L], fp32_arr[pos, L]
                    l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))
                    row[str(L)] = round(l2 * 100, 2)
                table[str(pos)] = row
            result["tables"][f"{name}_{kv}"] = table

    (OUT / "kv_error_growth.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    emit({"phase": "compare", "wrote": str(OUT / "kv_error_growth.json")})

    for name in ("baseline", "4x", "oracle"):
        for kv in ("k", "v"):
            print(f"\n=== {name} / {kv.upper()}: FP32比 L2相対誤差(%) ===")
            header = "  pos  " + " ".join(f"L{L:>3d}" for L in layers_shown)
            print(header)
            for pos in positions_shown:
                row = result["tables"][f"{name}_{kv}"][str(pos)]
                vals = " ".join(f"{row[str(L)]:5.1f}" for L in layers_shown)
                print(f"  {pos:4d}  {vals}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["capture", "compare"], required=True)
    a = ap.parse_args()
    {"capture": capture, "compare": compare}[a.phase]()


if __name__ == "__main__":
    main()
