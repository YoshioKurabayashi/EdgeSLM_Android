"""Summarize P12d traces with range choices frozen on old calibration data."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np

def quant_stats(a,scale,zp):
    a=a.astype(np.float64)
    lo=-zp*scale
    hi=(65535-zp)*scale
    clipped=(a<lo)|(a>hi)
    reconstructed=(np.clip(np.rint(a/scale)+zp,0,65535)-zp)*scale
    error=reconstructed-a
    per_rel=np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-30)
    return {'scale':scale,'zero_point':zp,'range':[lo,hi],
        'clipped_elements':int(clipped.sum()),'elements':a.size,
        'clipped_tokens':int(clipped.any(axis=1).sum()),'tokens':len(a),
        'mse':float(np.mean(error**2)),'aggregate_rel_l2':float(np.linalg.norm(error)/max(np.linalg.norm(a),1e-30)),
        'per_token_rel_l2_median':float(np.median(per_rel)),
        'per_token_rel_l2_p95':float(np.percentile(per_rel,95)),
        'per_token_rel_l2_max':float(per_rel.max())}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--project-root',type=Path,default=Path('D:/AI/source/n_chap5/EdgeSLM_Android'))
    args=ap.parse_args()
    if not (args.out/'manifest.json').exists():
        names=['calib_'+str(i) for i in range(5)]+['chat_original','chat_english','long_ja','long_code','chat_long']
        records=[]
        calibration={3:[],6:[]}
        for name in names:
            r=json.loads((args.out/(name+'.json')).read_text(encoding='utf-8'))
            assert r['name']==name and r['tokens']==len(r['rows'])
            assert r['window']==256
            z=np.load(args.out/(name+'.npz'))
            for k in [3,6]:
                a=z['layer_'+str(k)]
                assert a.shape==(r['tokens'],1024) and np.isfinite(a).all()
                if name.startswith('calib_'):
                    assert r['tokens']==6
                    calibration[k].append(a[1:])
            records.append(r)
        source=args.project_root/'app/pipeline/export_qwen_decode.py'
        manifest={'note':'Recovered from all ten completed per-scenario JSON/NPZ files after the execution session ended before manifest was available. Source hash captured at recovery.',
            'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'snapshot':str(Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'),
            'args':{'window':256,'long_tokens':640,'max_new':64},'records':records,
            'frozen_later_ranges':{str(k):{'min':float(np.concatenate(v).min()),
                'max':float(np.concatenate(v).max()),'samples':25} for k,v in calibration.items()}}
        (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=json.loads((args.out/'manifest.json').read_text(encoding='utf-8'))
    expected={*('calib_'+str(i) for i in range(5)),
              'chat_original','chat_english','long_ja','long_code','chat_long'}
    assert len(manifest['records'])==len(expected)
    assert {r['name'] for r in manifest['records']}==expected,'Incomplete scenario manifest'
    baseline=json.loads((args.project_root/'results/P12c_20260915_activation_outliers/code_histogram.json').read_text())
    results=[]
    arrays={3:[],6:[]}
    globalrows=[]
    hfchecks=[]
    for record in manifest['records']:
        name=record['name']
        hfchecks.extend(record['checks'])
        for check in record['checks']:
            assert check['logits_rel_l2']<1e-4
            assert all(check['layer_'+str(k)+'_rel_l2']<1e-4 for k in [3,6])
        if name.startswith('calib_'):
            continue
        data=np.load(args.out/(name+'.npz'))
        rows=record['rows']
        assert record['tokens']==len(rows)
        assert [r['position'] for r in rows]==list(range(len(rows)))
        item={'name':name,'tokens':len(rows),'layers':{}}
        for layer in [3,6]:
            a=data['layer_'+str(layer)]
            assert a.shape==(len(rows),1024)
            assert np.isfinite(a).all()
            arrays[layer].append(a)
            events=[]
            for index,channel in zip(*np.where(np.abs(a)>1000)):
                events.append({'position':int(index),'channel':int(channel),'value':float(a[index,channel]),
                               'phase':rows[index]['phase'],'turn':rows[index]['turn']})
            after=a[1:]
            maxpos,maxch=np.unravel_index(np.abs(after).argmax(),after.shape)
            item['layers'][str(layer)]={'min':float(a.min()),'max':float(a.max()),
               'gt1000_events':events,'later_max_abs':float(np.abs(after).max()),
               'later_max_position':int(maxpos+1),'later_max_channel':int(maxch)}
        globalrows.extend([dict(r,scenario=name) for r in rows])
        results.append(item)
    report={'note':'FP32 distribution and local rounding simulation only; no QDQ/HTP end-to-end improvement claim.',
        'scenarios':results,'evaluation_tokens':len(globalrows),'hf_control_checks':len(hfchecks),
        'hf_control_max_logits_rel_l2':max(r['logits_rel_l2'] for r in hfchecks),
        'hf_control_max_hidden_rel_l2':max(r['layer_'+str(k)+'_rel_l2'] for r in hfchecks for k in [3,6]),
        'frozen_later_ranges':manifest['frozen_later_ranges'],'layers':{}}
    groups={'first':np.array([r['position']==0 for r in globalrows]),
            'later_before_eviction':np.array([0<r['position']<manifest['args']['window'] for r in globalrows]),
            'after_eviction':np.array([r['position']>=manifest['args']['window'] for r in globalrows]),
            'later_generated':np.array([r['phase']=='generated' and r['position']>0 for r in globalrows]),
            'all_later':np.array([r['position']>0 for r in globalrows])}
    for layer in [3,6]:
        a=np.concatenate(arrays[layer])
        old=next(r for r in baseline['rows'] if r['tensor']==('add_26' if layer==3 else 'add_53'))
        frozen=manifest['frozen_later_ranges'][str(layer)]
        policies={'existing':(old['scale'],old['zero_point'])}
        for factor in [1,1.5,2,4]:
            lo=min(0,frozen['min']*factor)
            hi=max(0,frozen['max']*factor)
            scale=(hi-lo)/65535
            zp=int(np.clip(np.rint(-lo/scale),0,65535))
            policies['later_x'+str(factor)]=(scale,zp)
        entry={}
        for name,select in groups.items():
            if not select.any():
                continue
            v=a[select]
            cs=np.flatnonzero((np.abs(v)>1000).any(axis=0))
            entry[name]={'tokens':len(v),'min':float(v.min()),'max':float(v.max()),
                'gt1000_elements':int(np.sum(np.abs(v)>1000)), 'gt1000_channels':cs.tolist(),
                'policies':{n:quant_stats(v,*p) for n,p in policies.items()}}
        report['layers'][str(layer)]=entry
    (args.out/'range_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'evaluation_tokens':report['evaluation_tokens'],'scenarios':results,
        'hf_checks':report['hf_control_checks'],
        'layers':{k:{g:{n:v for n,v in d.items() if n!='policies'} for g,d in groups.items()}
                  for k,groups in report['layers'].items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
