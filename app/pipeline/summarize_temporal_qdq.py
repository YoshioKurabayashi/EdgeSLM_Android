"""Audit completed P12e evidence and render tables; no model inference."""
# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# 何のための道具か:
#   P12e「先頭・後続の量子化範囲分離」（StudyMemo 11.5 節）の記録が揃っているかを照合し、
#   数値の表（numeric_evidence.md）と集計（summary.json）を作る。推論はしない（JSON を読むだけ）。
#   ※上の英語の docstring は argparse の --help にそのまま出るので、説明はこのコメントに書いた。
# 入力:  --out の下の temporal_qdq_experiment.py・focused_qdq_experiment.py の出力
#        （*_teacher.json・*_focused_*.json・*free_generation.json・calibration_ranges.json・*build_audit.json）と、
#        P12d の対話の記録（results/P12d_20260915_dialogue_activation_ranges/<対話>.json。ROOT は Windows の絶対パス）。
# 出力:  --out/summary.json・--out/numeric_evidence.md（results/P12e_20260915_temporal_qdq/ に保存済み）。
# 動かし方: & $py -B app/pipeline/summarize_temporal_qdq.py --out work/p12e_reproduction
#   （results/P12e_20260915_temporal_qdq.md 7 節の最後の行）
# 照合すること（assert で止まる）: どの条件も同じトークン列を同じ数だけ評価した／観測出力を足した
#   baseline が元の QDQ と 4 地点で完全一致／較正が 8 本・297 トークン／変えてはいけない初期化子が
#   変わっていない／自由生成は全条件 3 ターン・上限 32 トークン。
# ---------------------------------------------------------------------------
from pathlib import Path
import argparse
import hashlib
import json
import statistics

# JSON を UTF-8 で読む。
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

# 平均（statistics.fmean。float で足す）。
def mean(values):
    return statistics.fmean(values)

def main():
    """P12e の記録を照合して表と集計を書く。

    表の行は条件（baseline・shared・split・focused_shared・focused_split）、列は位置1以降の平均:
    予測一致（1位一致）・回答部分の予測一致・add_53 の相対 L2・logits の相対 L2・KL・新しい K/V の誤差。
    「回答部分」は、P12d の記録で次が生成トークンになる位置（＝回答の語を予測している位置）。
    記録では日本語の回答 29 地点で baseline も focused_split も 26/29 で同じだった
    （全体の一致数の増加は入力文を処理する所で起きた。results/P12e_20260915_temporal_qdq.md 3 節）。
    """
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    out=args.out
    # ① 条件の一覧と、自由生成の結果（全層版・168 組版・hybrid があればそれも）をまとめる。
    modes=['baseline','shared','split','focused_shared','focused_split']
    result={'cases':{},'controls':{},'generation':{**read(out/'free_generation.json'),**read(out/'focused_free_generation.json')}}
    if (out/'hybrid_free_generation.json').exists():
        result['generation'].update(read(out/'hybrid_free_generation.json'))
    lines=['# P12e numeric evidence','','All errors are relative L2 against prepped FP32, averaged per token.','']
    for name in ['chat_original','chat_english']:
        record=read(out/f'{name}_teacher.json')
        assert record['status']=='complete'
        # ② 対照: 観測出力を足した baseline と元の QDQ が、同じ入力で logits も KV も完全一致したか
        #    （観測が計算を変えていないことの確認）。
        controls=record['observed_baseline_vs_original_qdq']
        assert len(controls)==4
        assert all(c['logits_rel_l2']==0 and c['max_cache_rel_l2']==0 for c in controls)
        result['controls'][name]={'baseline_observation_exact_matches':len(controls),
            'max_prepped_vs_official_residual_relative_error':max(c[t] for c in record['fp32_prepped_vs_prior_official_reference'] for t in ['add_26','add_53'])}
        case={}
        trace=read(Path('D:/AI/source/n_chap5/EdgeSLM_Android')/f'results/P12d_20260915_dialogue_activation_ranges/{name}.json')
        # ③ 回答を予測している位置の集合: 生成トークンの位置と、その 1 つ手前（次が生成トークン）の位置。
        response_positions={i for i,r in enumerate(trace['rows']) if r['phase']=='generated' or
                            (i+1<len(trace['rows']) and trace['rows'][i+1]['phase']=='generated')}
        lines.extend([f'## {name} ({record["tokens"]} tokens)','','| Variant | Later top1 | Response prediction top1 | Later add53 | Later logits | Later KL | New K | New V |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for mode in modes:
            # ④ 168 組版は別ファイル。入力のトークン列が同じであることを確かめてから使う。
            if mode.startswith('focused_'):
                f=read(out/f'{name}_{mode}.json')
                assert f['status']=='complete' and f['input_ids']==record['input_ids']
                rows=f['rows']
                oracle=f['same_fp32_kv_oracle']
            else:
                rows=record['rows'][mode]
                oracle=record['same_fp32_kv_oracle'][mode]
            assert len(rows)==record['tokens']
            assert [r['token_id'] for r in rows]==record['input_ids']
            # ⑤ 位置0 を除いた平均（位置0 は巨大値の出る特別な位置なので別に見る）。
            #    saturation は量子化範囲の外に出た要素の数を、位置0 と位置1以降に分けて合計する。
            later=rows[1:]
            metrics={k:mean(r['metrics'][k] for r in later) for k in later[0]['metrics']}
            matches=sum(r['metrics']['top1_same'] for r in later)
            response_matches=sum(rows[i]['metrics']['top1_same'] for i in response_positions)
            saturation={phase:{t:{k:sum(r['saturation'][t][k] for r in selected) for k in ['before_q_outside','reference_outside']}
                               for t in ['add_26','add_53']} for phase,selected in [('first',rows[:1]),('later',later)]}
            case[mode]={'later_metrics':metrics,'later_top1_count':matches,'later_tokens':len(later),
                'response_top1_count':response_matches,'response_predictions':len(response_positions),
                'first_metrics':rows[0]['metrics'],'saturation_counts':saturation,'oracle':oracle}
            lines.append(f'| {mode} | {matches}/{len(later)} ({metrics["top1_same"]:.1%}) | {response_matches}/{len(response_positions)} | '
                         f'{metrics["add_53_qdq_rel_l2"]:.2%} | {metrics["logits_rel_l2"]:.2%} | {metrics["kl_ref_to_q"]:.4f} | '
                         f'{metrics["new_k_median_layer_rel_l2"]:.2%} | {metrics["new_v_median_layer_rel_l2"]:.2%} |')
        result['cases'][name]=case
        # focused_split の位置0 は focused_shared と同じモデルなので、位置0 の指標は一致するはず。
        assert case['focused_shared']['first_metrics']==case['focused_split']['first_metrics']
        lines.append('')
    # ⑥ 自由生成の全文を条件ごとに書き出す（合否は数値ではなくこの文を読んで決めた）。
    lines.extend(['## Free generation, original Japanese dialogue','','Temperature -> music -> recall temperature.',''])
    assert {'fp32',*modes}.issubset(result['generation'])
    for mode,turns in result['generation'].items():
        assert len(turns)==3
        assert all(t['max_new']==32 for t in turns)
        lines.extend([f'### {mode}',''])
        for t in turns:
            lines.extend([f'Turn {t["turn"]+1}; tokens={len(t["generated_ids"])}; stop={t["stop_id"]}',
                          '```text',t['text'],'```',''])
    # ⑦ 較正の本数・トークン数と、変更監査（想定外の初期化子の変更が 0）を確かめる。
    result['calibration_sequences']=len(read(out/'calibration_ranges.json')['sequences'])
    result['calibration_tokens']=sum(len(s['processed_ids']) for s in read(out/'calibration_ranges.json')['sequences'])
    assert result['calibration_sequences']==8 and result['calibration_tokens']==297
    for audit_file in ['build_audit.json','focused_build_audit.json']:
        for a in read(out/audit_file).values():
            assert a['unexpected_initializer_changes']==[]
    result['audit']='Complete cases, identical input IDs, original baseline control, and initializer audit passed.'
    # このファイル自身の sha256 を残す（コメントを足した今の版は、当時の記録の hash と一致しない）。
    result['summarizer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'numeric_evidence.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'audit':result['audit'],'cases':{n:{m:{'top1':v['later_metrics']['top1_same'],'add53':v['later_metrics']['add_53_qdq_rel_l2']}
                      for m,v in c.items()} for n,c in result['cases'].items()}},ensure_ascii=False))

if __name__=='__main__':
    main()
