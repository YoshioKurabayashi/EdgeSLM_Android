"""work/ の中間成果物を整理する

## 方針

**再生成できるものだけ削り、作り直しが高い成果物は残す。**

残すもの:

| 対象 | 理由 |
|---|---|
| `work/ops/qwen_full/qwen24_ctx.SM8350.bin` | **生成に54分**。最重要 |
| `work/ops/qwen_full/qwen24ext_epctx.onnx` | 端末が読むモデル本体（442バイト） |
| `work/ops/qwen_full/qwen_full_v2_qdq.onnx` | context を作り直すときの入力（量子化 2分45秒） |
| `work/qwen21/_base24.onnx(.data)` | 精度測定の fp32 参照 |
| `*.npz` | 較正・評価データ（作り直すとモデル読み込みが必要） |
| `*.spec.json` / `status_*.json` / `*meta*.json` | 何をどう作ったかの記録 |
| `work/ops/*/*_epctx.onnx` | P2b の単体 op モデル（小さく `op_check` で使う） |
| `work/ops/qwen_pm/ctx/qwenpm_ctx.SM8350.bin` + `qwen_pm_epctx.onnx` | マスク入力版 prefill。**生成に約80分** |
| `work/ops/qwen_pm/qwen_pm_final_qdq.onnx` | 同・context を作り直すときの入力 |
| `work/qwen_pm/qwen05b_prefillmask_L24_S16.onnx(.data)` | 同・精度測定の fp32 参照 |
| `work/ops/qwen_pm64/ctx/qwenpm64_ctx.SM8350.bin` + `qwen_pm64_epctx.onnx` | **窓64**。生成に54分 |
| `work/ops/qwen_pm64/qwen_pm64_final_qdq.onnx` | 同・context を作り直すときの入力 |
| `work/qwen_pm64/qwen05b_prefillmask_L24_S64_last.onnx(.data)` | 同・精度測定の fp32 参照 |

消すもの（すべて再生成可能）:

| 対象 | 再生成の手段 |
|---|---|
| `*_prepped.onnx(.data)` / `*_folded*.onnx` | S2 をやり直す（数分） |
| `*.dlc` | S5 をやり直す（context binary があれば不要） |
| `work/oracle6/` | 診断用。結果は `results/` に記録済み |
| `work/dev*/` の `.bin` | 端末出力。数値は `results/` に記録済み |
| opset18 の古い export | opset21 版で置き換え済み |
| 役目を終えた QDQ 変種 | 比較結果は `results/` に記録済み |
| `ctx_*/` （生成器の出力ディレクトリ） | **EPContext ONNX の隣に同じバイナリを置いてある**ので2重に持たない。消す前に md5 の一致を確認する |

Windows では `du` の挙動が揃わないので、サイズ計算は Python で行う。
（2026-10-18 追記）学習用の補足:

- 段: 本番の流れの外。2026-09-13 ごろ（P3〜P5 のあと）に work/ が大きくなったので作った片付けの道具
- 入力: プロジェクトの work/ の中身（下の PATTERNS に合うもの）。出力: 画面の一覧と、--execute のときの実際の削除
- 動かし方: python -m app.pipeline.cleanup_work（一覧だけ）→ 中身を見てから python -m app.pipeline.cleanup_work --execute
- 注意（コードから読めること）:
  - 残すものの守りは「KEEP と同じパスか」だけ。表の「残すもの」の *.npz・*.spec.json などは、PATTERNS に合わないから残る。
    PATTERNS がフォルダに合うと、その中のファイルは KEEP にあっても丸ごと消える
  - 「消す前に md5 の一致を確認する」はこのコードでは行わない（PATTERNS の理由の欄に「確認済み」と書いてあるだけ）
  - 消したものは戻せない（ごみ箱を通らない）
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# プロジェクトの直下（このファイルは app/pipeline/ にあるので 2 つ上）
ROOT = Path(__file__).resolve().parents[2]

# glob パターンと、なぜ消してよいかの理由
PATTERNS: list[tuple[str, str]] = [
    ("work/ops/*/*_prepped.onnx", "S2 の中間（数分で再生成）"),
    ("work/ops/*/*_prepped.onnx.data", "S2 の中間（数分で再生成）"),
    ("work/ops/*/*_folded.onnx", "S2 の中間"),
    ("work/ops/*/*_folded.onnx.data", "S2 の中間"),
    ("work/ops/*/*_folded_clamped.onnx", "S2 の中間"),
    ("work/ops/*/*_folded_clamped.onnx.data", "S2 の中間"),
    ("work/ops/*/ctx*/*.dlc", "S5 の中間（context binary があれば不要）"),
    ("work/oracle6", "oracle 実験の成果物（結果は results/P3_*_rmsnorm_fix.md）"),
    ("work/devacc", "端末出力（結果は results/P3_*_device_accuracy.md）"),
    ("work/devacc_first", "決定性確認用"),
    ("work/devacc2", "端末出力"),
    ("work/devacc3", "端末出力"),
    ("work/devacc4", "端末出力"),
    ("work/devacc5", "端末出力"),
    ("work/devrep", "決定性確認用"),
    ("work/devmm", "端末出力（誤差分解）"),
    ("work/dev24", "端末出力（24層）"),
    ("work/dev_*", "端末出力（op ごとの誤差分解）"),
    ("work/qwen/qwen05b_prefill_L24_S16.onnx", "opset18。opset21 版で置換済み"),
    ("work/qwen/qwen05b_prefill_L24_S16.onnx.data", "opset18。置換済み"),
    ("work/qwen/qwen05b_prefill_L2_S16.onnx.data", "2層比較用。結果は記録済み"),
    ("work/qwen/qwen05b_prefill_L2_S16_clamped.onnx", "中間"),
    ("work/qwen/_base2.onnx.data", "2層の fp32 参照。結果は記録済み"),
    ("work/qwen/_probe*.onnx", "診断用の一時モデル"),
    ("work/qwen/_p3.onnx", "診断用"),
    ("work/qwen/_folded.onnx", "等価性確認用"),
    ("work/qwen/_base.onnx", "層1枚の fp32 参照。結果は記録済み"),
    ("work/qwen/_swap.onnx", "等価性確認用"),
    ("work/qwen/_stab.onnx", "等価性確認用"),
    ("work/qwen/_rms6.onnx", "等価性確認用"),
    ("work/qwen/_rms6.onnx.data", "等価性確認用"),
    ("work/ops/qwen_full/qwen_full_a16w8_qdq.onnx*", "C-11 なし版。破綻を確認済み"),
    ("work/ops/qwen_full/qwen_full_final_qdq.onnx", "C-11 なし版"),
    ("work/ops/qwen_full/qwen_full_a16w8_rms_swap_qdq.onnx*", "force 前の中間"),
    ("work/ops/qwen_full/qwen_full_a16w8_mixed_qdq.onnx*", "旧方式"),
    ("work/ops/qwen_full/qwen24_epctx.onnx", "embed_mode=1 版（C-12 で使えないと判明）"),
    ("work/ops/qwen_prefill2/*_qdq.onnx", "2層。結果は記録済み"),
    ("work/ops/qwen_prefill2/ctx*", "2層の context"),
    ("work/ops/qwen_dec/*_qdq.onnx*", "decode。context 生成に失敗した版"),
    ("work/ops/qwen_dec2/*_qdq.onnx", "decode 2層の検証用"),
    ("work/ops/qwen_dec2/ctx*", "decode 2層の検証用"),
    ("work/ops/qwen6/*_qdq.onnx", "6層。結果は記録済み"),
    ("work/ops/qwen6/ctx*", "6層"),
    ("work/ops/qwen_layer1/_*.onnx", "比較用の変種。結果は記録済み"),
    ("work/ops/qwen_layer1/ctx_*", "比較用の context"),
    ("work/qwen21/_base6.onnx", "6層の fp32 参照"),
    ("work/qwen21/_base6.onnx.data", "6層の fp32 参照"),
    ("work/qwen21/qwen05b_prefill_L2_S16.onnx*", "2層 export"),
    ("work/qwen21/qwen05b_prefill_L6_S16.onnx*", "6層 export"),
    ("work/qwen_decode/qwen05b_decode_L24_C128.onnx.data", "decode の重み（1分で再生成）"),
    ("work/qwen_decode/qwen05b_decode_L2_C32.onnx.data", "decode 2層の重み"),
    # --- 2026-09-13 追加: マスク入力版 prefill（qwen_pm） ---
    ("work/qwen_pm/_base_pm.onnx.data", "fp32 参照の重み。元 export が残っているので再生成可"),
    ("work/ops/qwen_pm/qwen_pm_a16w8_rms_swap_qdq.onnx", "force_mixed_matmul 前の中間"),
    ("work/ops/qwen_pm/qwen_pm_a16w8_rms_swap_qdq.onnx.data", "force_mixed_matmul 前の中間"),
    # ★context binary は ctx/ 側（EPContext ONNX の隣）を残し、生成器の出力ディレクトリを消す。
    #   同じ 772MB を2重に持つ意味がない。
    ("work/ops/qwen_pm/ctx/ctx_qwenpm", "生成器の出力。ctx/qwenpm_ctx.SM8350.bin と同一（md5 一致を確認済み）"),
    ("work/devacc_pm_last", "端末出力（結果は results/P4_*_prefillmask.md）"),
    ("work/devacc_pm_valid", "端末出力"),
    ("work/devacc_pm_all", "端末出力"),
    ("work/devacc_nomask", "端末出力（マスク無し版との比較）"),
    ("work/ops/attn_mask_input/ctx/ctx_ami", "単体 op の context。EPContext に埋め込み済み"),
    # --- 2026-09-13 追加: 窓64（qwen_pm64） ---
    ("work/qwen_pm64/*.onnx.data", "fp32 参照の重み。export をやり直せば再生成できる"),
    ("work/ops/qwen_pm64/qwen_pm64_a16w8_rms_swap_qdq.onnx", "force_mixed_matmul 前の中間"),
    ("work/ops/qwen_pm64/qwen_pm64_a16w8_rms_swap_qdq.onnx.data", "force_mixed_matmul 前の中間"),
    # ★context binary は ctx/ 側（EPContext ONNX の隣）を残し、生成器の出力を消す
    ("work/ops/qwen_pm64/ctx/ctx_qwenpm64", "生成器の出力。ctx/qwenpm64_ctx.SM8350.bin と同一（md5 照合済み）"),
    ("work/devacc_pm64", "端末出力（結果は results/P5_*_window64.md）"),
    ("work/devacc_pm16", "端末出力（窓16との比較用）"),
]

# 必ず残すもの（作り直しに 1 時間前後かかる context binary と、それを作り直すための入力・fp32 の参照）。
# PATTERNS に合っても、このパスと同じなら消さない
KEEP = [
    "work/ops/qwen_full/qwen24_ctx.SM8350.bin",
    "work/ops/qwen_full/qwen24ext_epctx.onnx",
    "work/ops/qwen_full/qwen_full_v2_qdq.onnx",
    "work/qwen21/_base24.onnx",
    "work/qwen21/_base24.onnx.data",
    # --- マスク入力版 prefill（生成に約80分。2026-09-13） ---
    "work/ops/qwen_pm/ctx/qwenpm_ctx.SM8350.bin",   # 端末へ配るバイナリ本体
    "work/ops/qwen_pm/ctx/qwen_pm_epctx.onnx",      # 565バイト。これを端末が読む
    "work/ops/qwen_pm/qwen_pm_final_qdq.onnx",      # context を作り直すときの入力
    "work/qwen_pm/qwen05b_prefillmask_L24_S16.onnx",       # fp32 参照
    "work/qwen_pm/qwen05b_prefillmask_L24_S16.onnx.data",
    # --- 窓64（生成に54分。2026-09-13） ---
    "work/ops/qwen_pm64/ctx/qwenpm64_ctx.SM8350.bin",   # 端末へ配るバイナリ本体
    "work/ops/qwen_pm64/ctx/qwen_pm64_epctx.onnx",      # 573バイト。これを端末が読む
    "work/ops/qwen_pm64/qwen_pm64_final_qdq.onnx",      # context を作り直すときの入力
    "work/qwen_pm64/qwen05b_prefillmask_L24_S64_last.onnx",
    "work/qwen_pm64/qwen05b_prefillmask_L24_S64_last.onnx.data",
]


def size_of(p: Path) -> int:
    """ファイルならその大きさ、フォルダなら中のファイルの合計（バイト）。"""
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    """① PATTERNS に合うものを集めて大きさを測る → ② 大きい順に一覧を出す → ③ --execute のときだけ消す。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="付けないと一覧だけ")
    args = ap.parse_args()

    # ① 消す候補を集める（KEEP と同じパスは除く）
    keep = {(ROOT / k).resolve() for k in KEEP}
    rows, total = [], 0
    for pat, why in PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if p.resolve() in keep:
                continue
            sz = size_of(p)
            total += sz
            rows.append((sz, p.relative_to(ROOT).as_posix(), why))

    # ② 大きい順に上位 22 件を出し、残りはまとめて 1 行にする
    rows.sort(reverse=True)
    print(f"{'サイズ':>10s}  対象 / 理由")
    print("-" * 92)
    for sz, rel, why in rows[:22]:
        print(f"{sz / 1e9:8.2f} GB  {rel}")
        print(f"{'':10s}  └ {why}")
    if len(rows) > 22:
        rest = sum(r[0] for r in rows[22:])
        print(f"{rest / 1e9:8.2f} GB  （ほか {len(rows) - 22} 件の小さい中間ファイル）")
    print("-" * 92)
    print(f"{total / 1e9:8.2f} GB  削除対象 合計（{len(rows)} 件）")

    print("\n残すもの:")
    for k in KEEP:
        p = ROOT / k
        if p.exists():
            print(f"{size_of(p) / 1e6:8.1f} MB  {k}")

    if not args.execute:
        print("\n（実行するには --execute を付ける）")
        return

    # ③ 消す。フォルダは中身ごと、ファイルは 1 つずつ。消せなかったものは理由を出して続ける
    freed = 0
    for sz, rel, _ in rows:
        p = ROOT / rel
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            freed += sz
        except OSError as e:
            print(f"  消せなかった: {rel}: {e}")
    print(f"\n{freed / 1e9:.2f} GB を削除した")


if __name__ == "__main__":
    main()
