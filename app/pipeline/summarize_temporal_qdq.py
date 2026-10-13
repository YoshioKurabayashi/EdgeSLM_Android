"""Audit completed P12e evidence and render tables; no model inference."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def mean(values):
    return statistics.fmean(values)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    out=args.out
    modes=['baseline','shared','split','focused_shared','focused_split']
    result={'cases':{},'controls':{},'generation':{**read(out/'free_generation.json'),**read(out/'focused_free_generation.json')}}
    if (out/'hybrid_free_generation.json').exists():
        result['generation'].update(read(out/'hybrid_free_generation.json'))
    lines=['# P12e numeric evidence','','All errors are relative L2 against prepped FP32, averaged per token.','']
    for name in ['chat_original','chat_english']:
        record=read(out/f'{name}_teacher.json')
        assert record['status']=='complete'
        controls=record['observed_baseline_vs_original_qdq']
        assert len(controls)==4
        assert all(c['logits_rel_l2']==0 and c['max_cache_rel_l2']==0 for c in controls)
        result['controls'][name]={'baseline_observation_exact_matches':len(controls),
            'max_prepped_vs_official_residual_relative_error':max(c[t] for c in record['fp32_prepped_vs_prior_official_reference'] for t in ['add_26','add_53'])}
        case={}
        trace=read(Path('D:/AI/source/n_chap5/EdgeSLM_Android')/f'results/P12d_20260915_dialogue_activation_ranges/{name}.json')
        response_positions={i for i,r in enumerate(trace['rows']) if r['phase']=='generated' or
                            (i+1<len(trace['rows']) and trace['rows'][i+1]['phase']=='generated')}
        lines.extend([f'## {name} ({record["tokens"]} tokens)','','| Variant | Later top1 | Response prediction top1 | Later add53 | Later logits | Later KL | New K | New V |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for mode in modes:
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
        assert case['focused_shared']['first_metrics']==case['focused_split']['first_metrics']
        lines.append('')
    lines.extend(['## Free generation, original Japanese dialogue','','Temperature -> music -> recall temperature.',''])
    assert {'fp32',*modes}.issubset(result['generation'])
    for mode,turns in result['generation'].items():
        assert len(turns)==3
        assert all(t['max_new']==32 for t in turns)
        lines.extend([f'### {mode}',''])
        for t in turns:
            lines.extend([f'Turn {t["turn"]+1}; tokens={len(t["generated_ids"])}; stop={t["stop_id"]}',
                          '```text',t['text'],'```',''])
    result['calibration_sequences']=len(read(out/'calibration_ranges.json')['sequences'])
    result['calibration_tokens']=sum(len(s['processed_ids']) for s in read(out/'calibration_ranges.json')['sequences'])
    assert result['calibration_sequences']==8 and result['calibration_tokens']==297
    for audit_file in ['build_audit.json','focused_build_audit.json']:
        for a in read(out/audit_file).values():
            assert a['unexpected_initializer_changes']==[]
    result['audit']='Complete cases, identical input IDs, original baseline control, and initializer audit passed.'
    result['summarizer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'numeric_evidence.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'audit':result['audit'],'cases':{n:{m:{'top1':v['later_metrics']['top1_same'],'add53':v['later_metrics']['add_53_qdq_rel_l2']}
                      for m,v in c.items()} for n,c in result['cases'].items()}},ensure_ascii=False))

if __name__=='__main__':
    main()
