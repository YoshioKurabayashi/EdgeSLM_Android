"""Read-only inspection of existing Qwen3 models; writes only to --out."""
from pathlib import Path
import argparse
import json
import os
import hashlib
import copy
import subprocess

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path('D:/AI/source/n_chap5/EdgeSLM_Android')
OPS = ROOT / 'work/ops/qwen3_dec28c256'
DATA = ROOT / 'work/qwen3_dec28c256'
ADB = Path('C:/Users/kurab/AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe')
REMOTE = '/sdcard/Android/data/org.edgeslm.runner/files'

def emit(x):
    print(json.dumps(x, ensure_ascii=False), flush=True)

def graph_info(path):
    m = onnx.load(str(path), load_external_data=False)
    nodes = [dict(name=n.name, op=n.op_type, inputs=list(n.input), outputs=list(n.output))
             for n in m.graph.node if any('add_53' in s for s in [*n.input, *n.output])]
    params = {}
    for t in m.graph.initializer:
        if 'add_53' in t.name and not t.external_data:
            params[t.name] = numpy_helper.to_array(t).tolist()
    return {'path': str(path), 'nodes': nodes, 'params': params,
            'outputs': [o.name for o in m.graph.output]}

def prune_probe(source, dest, targets):
    """Keep unmodified ancestors; resolve original external weights read-only."""
    m = onnx.load(str(source), load_external_data=False)
    needed = set(targets)
    keep = []
    for node in reversed(m.graph.node):
        if needed.intersection(node.output):
            keep.append(copy.deepcopy(node))
            needed.update(node.input)
    inits = [copy.deepcopy(t) for t in m.graph.initializer if t.name in needed]
    for t in inits:
        if t.external_data:
            onnx.external_data_helper.load_external_data_for_tensor(t, str(source.parent))
            t.data_location=onnx.TensorProto.DEFAULT
            del t.external_data[:]
    inputs = [copy.deepcopy(v) for v in m.graph.input if v.name in needed]
    known = {v.name: v for v in [*m.graph.value_info, *m.graph.output]}
    outputs = [copy.deepcopy(known[n]) if n in known else
               onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [1,1,1024])
               for n in targets]
    del m.graph.node[:]
    m.graph.node.extend(reversed(keep))
    del m.graph.initializer[:]
    m.graph.initializer.extend(inits)
    del m.graph.input[:]
    m.graph.input.extend(inputs)
    del m.graph.output[:]
    m.graph.output.extend(outputs)
    # Remove irrelevant value_info and stale function/training references only.
    vis = [copy.deepcopy(v) for v in m.graph.value_info if v.name in needed]
    del m.graph.value_info[:]
    m.graph.value_info.extend(vis)
    onnx.save(m, str(dest))
    return {'source': str(source), 'probe': str(dest), 'nodes': len(keep),
            'sha256_source_graph': hashlib.sha256(source.read_bytes()).hexdigest(),
            'targets': targets}

def compare(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return {'max_abs':float(np.max(np.abs(a-b))),
            'rel_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))}

def device_info(args):
    def adb(*argv):
        r=subprocess.run([str(ADB),*argv],capture_output=True,check=True)
        return r.stdout.decode('utf-8')
    names=adb('shell','ls',REMOTE+'/out').splitlines()
    matches=[n.strip() for n in names if n.startswith(('hid_','l14_','l6v_'))]
    jobs={n:json.loads(adb('shell','cat',REMOTE+'/jobs/'+n+'.json'))
          for n in ['hid','l14','l6v']}
    report={'files':matches,'jobs':jobs}
    (args.out/'device_jobs.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit(report)

def histogram(args):
    import onnxruntime as ort
    jobs=json.loads((args.out/'device_jobs.json').read_text())
    dst=args.out/'saved_device'
    dst.mkdir(exist_ok=True)
    def pull(remote):
        local=dst/Path(remote).name
        subprocess.run([str(ADB),'pull',REMOTE+'/'+remote,str(local)],
                       capture_output=True,check=True)
        return local
    job=jobs['jobs']['hid']
    inputs={k:pull(v) for k,v in job['batches'][0]['inputs'].items()}
    device_paths={name:pull('out/'+name) for name in jobs['files']
                  if name.startswith('hid_') or name=='l14_000_add_53_QuantizeLinear_Output.bin'}
    source=ROOT/'work/q3_hid/hid_qdq.onnx'
    model=onnx.load(str(source),load_external_data=False)
    scales={t.name:numpy_helper.to_array(t) for t in model.graph.initializer
            if t.name.startswith('add_') and t.name.endswith(('_scale','_zero_point'))}
    del model
    so=ort.SessionOptions()
    so.intra_op_num_threads=4
    so.log_severity_level=3
    sess=ort.InferenceSession(str(source),so,providers=['CPUExecutionProvider'])
    dtypes={'tensor(int64)':np.int64,'tensor(float)':np.float32}
    feeds={v.name:np.fromfile(inputs[v.name],dtype=dtypes[v.type]).reshape(v.shape)
           for v in sess.get_inputs()}
    host=dict(zip([o.name for o in sess.get_outputs()],sess.run(None,feeds)))
    rows=[]
    for name,a in host.items():
        if not name.startswith('add_'):
            continue
        b=np.fromfile(device_paths['hid_000_'+name+'.bin'],dtype=np.uint16).reshape(a.shape)
        assert a.dtype==b.dtype==np.uint16
        tensor=name.removesuffix('_QuantizeLinear_Output')
        scale=float(scales[tensor+'_scale'])
        zp=int(scales[tensor+'_zero_point'])
        delta=b.astype(np.int32)-a.astype(np.int32)
        values,counts=np.unique(delta,return_counts=True)
        af=(a.astype(np.float64)-zp)*scale
        bf=(b.astype(np.float64)-zp)*scale
        rows.append({'tensor':tensor,'scale':scale,'zero_point':zp,'elements':a.size,
           'code_histogram':{str(k):int(v) for k,v in zip(values,counts)},
           'exact_fraction':float(np.mean(delta==0)),
           'within_1_fraction':float(np.mean(np.abs(delta)<=1)),
           'within_2_fraction':float(np.mean(np.abs(delta)<=2)),
           'within_5_fraction':float(np.mean(np.abs(delta)<=5)),
           'max_abs_code_diff':int(np.abs(delta).max()),
           'rms_code_diff':float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
           'host_rms':float(np.sqrt(np.mean(af**2))),
           'error_rms':float(np.sqrt(np.mean((bf-af)**2))),
           'dequantized':compare(af,bf),'uncentered_codes_misleading':compare(a,b)})
        np.savez_compressed(dst/(tensor+'_comparison.npz'),host_codes=a,device_codes=b,
                            scale=scale,zero_point=zp)
    same=np.array_equal(np.fromfile(device_paths['hid_000_add_53_QuantizeLinear_Output.bin'],np.uint16),
                        np.fromfile(device_paths['l14_000_add_53_QuantizeLinear_Output.bin'],np.uint16))
    evalz=np.load(DATA/'eval_q3d28.npz')
    candidates=[i for i in range(len(evalz['input_ids']))
                if np.array_equal(evalz['input_ids'][i:i+1],feeds['input_ids'])]
    for name,value in feeds.items():
        arr=evalz[name]
        candidates=[i for i in candidates if np.array_equal(arr[i:i+1],value)]
    fingerprints={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in inputs.items()}
    report={'note':'Comparison against saved device outputs; no device inference rerun.',
        'job':job,'source':str(source), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_sha256':fingerprints,'matching_eval_samples':candidates,
        'hid_vs_l14_add53_exact':same,'rows':rows}
    (args.out/'code_histogram.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    emit(report)

def summarize(args):
    r=json.loads((args.out/'activation_scan.json').read_text(encoding='utf-8'))
    h=json.loads((args.out/'code_histogram.json').read_text(encoding='utf-8'))
    z=np.load(args.out/'activation_scan.npz')
    pos=np.array([row['position'] for row in r['samples']])
    groups={}
    for name,sel in [('first',pos==0),('later',pos>0)]:
        a=z['prepped'][sel]
        groups[name]={'count':int(sel.sum()),'min':float(a.min()),'max':float(a.max()),
            'rms':float(np.sqrt(np.mean(a.astype(np.float64)**2))),
            'hf_vs_export':compare(z['hf'][sel],z['export'][sel]),
            'hf_vs_prepped':compare(z['hf'][sel],z['prepped'][sel])}
    side=[]
    for row in r['samples']:
        side.append({'sample':row['sample'], 'export_rel':row['export_vs_hf']['rel_l2'],
                     'prepped_rel':row['prepped_vs_hf']['rel_l2']})
    a=np.load(args.out/'saved_device/add_53_comparison.npz')
    real=(a['host_codes'].astype(np.float64)-a['zero_point'])*a['scale']
    model_span=65535*float(a['scale'])
    result={'groups':groups,'max_per_sample_export_rel_l2':max(v['export_rel'] for v in side),
            'max_per_sample_prepped_rel_l2':max(v['prepped_rel'] for v in side),
            'device_input':{'token_id':int(np.fromfile(args.out/'saved_device/hid_input_ids.bin',np.int64)[0]),
                'valid_tokens':int(np.count_nonzero(np.fromfile(args.out/'saved_device/hid_attn_mask.bin',np.float32)==0))},
            'host_comparison_range':{'min':float(real.min()),'max':float(real.max()),
                 'unique_codes':int(np.unique(a['host_codes']).size),
                 'code_span':int(a['host_codes'].max())-int(a['host_codes'].min()),
                 'quant_range_span':model_span,
                 'span_ratio':model_span/float(np.ptp(real))},
            'largest_5000_fraction':5/30720}
    assert r['stats']['global_max']['channel']==35
    assert all(np.isfinite(z[k]).all() for k in z.files)
    assert result['max_per_sample_export_rel_l2']<0.01
    assert result['max_per_sample_prepped_rel_l2']<0.01
    assert h['hid_vs_l14_add53_exact']
    row=next(v for v in h['rows'] if v['tensor']=='add_53')
    assert sum(row['code_histogram'].values())==1024
    assert abs(row['rms_code_diff']*row['scale']-row['error_rms'])<1e-12
    (args.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    emit(result)

def scan(args):
    import onnxruntime as ort
    data = dict(np.load(DATA/'calib_q3d28.npz'))
    arrays = {}
    provenance = []
    for label,source in [('export',DATA/'qwen05b_decode_L28_C256.onnx'),
                         ('prepped',OPS/'qwen3_dec28c256_prepped.onnx')]:
        probe = args.out/(label+'_add53.onnx')
        provenance.append(prune_probe(source,probe,['add_53']))
        so=ort.SessionOptions()
        so.intra_op_num_threads=4
        so.log_severity_level=3
        sess=ort.InferenceSession(str(probe),so,providers=['CPUExecutionProvider'])
        names=[i.name for i in sess.get_inputs()]
        outputs=[]
        for i in range(len(data['input_ids'])):
            outputs.append(sess.run(None,{n:data[n][i:i+1] for n in names})[0].reshape(-1))
        arrays[label]=np.stack(outputs)
        emit({'stage':label,'min':float(arrays[label].min()),'max':float(arrays[label].max())})
        del sess
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    snapshot=next((Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots').iterdir())
    hf=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,
                    dtype=torch.float32,attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    rows=[]
    prefix=[]
    hf_h=[]
    with torch.inference_mode():
        for i in range(len(data['input_ids'])):
            pos=int(np.count_nonzero(data['attn_mask'][i]==0))-1
            if pos==0:
                prefix=[]
            assert pos==len(prefix), (i,pos,len(prefix))
            prefix.append(int(data['input_ids'][i,0]))
            pred=hf(input_ids=torch.tensor([prefix]),use_cache=False,output_hidden_states=True)
            h=pred.hidden_states[6][0,-1].float().cpu().numpy().copy()
            hf_h.append(h)
            c=int(np.argmax(arrays['prepped'][i]))
            rows.append({'sample':i,'position':pos,'token_id':prefix[-1],
                         'prefix_ids':prefix.copy(),'prefix_text':tok.decode(prefix),
                         'max_channel':c,'prepped_max':float(arrays['prepped'][i,c]),
                         'export_same_channel':float(arrays['export'][i,c]),
                         'hf_same_channel':float(h[c]),
                         'export_vs_hf':compare(h,arrays['export'][i]),
                         'prepped_vs_hf':compare(h,arrays['prepped'][i])})
            emit({'stage':'hf','sample':i,'position':pos,'channel':c,
                  'prepped':rows[-1]['prepped_max'],'hf':float(h[c])})
    arrays['hf']=np.stack(hf_h)
    a=arrays['prepped']
    i,c=np.unravel_index(a.argmax(),a.shape)
    stats={'global_max':{'sample':int(i),'channel':int(c),'value':float(a[i,c]),
                         'row':rows[i]},
           'shape':list(a.shape),'min':float(a.min()),
           'abs_percentiles':dict(zip(['50','99','99.9','99.99','99.999'],
                                      np.percentile(np.abs(a),[50,99,99.9,99.99,99.999]).tolist())),
           'thresholds':{},'export_vs_prepped':compare(arrays['export'],a),
           'hf_vs_export':compare(arrays['hf'],arrays['export']),
           'hf_vs_prepped':compare(arrays['hf'],a)}
    for threshold in [10,100,1000,5000]:
        si,ch=np.where(np.abs(a)>threshold)
        counts=np.bincount(ch,minlength=1024)
        stats['thresholds'][str(threshold)]={'count':len(si),'total':a.size,
            'samples':np.unique(si).tolist(),
            'channels':{str(k):int(counts[k]) for k in np.flatnonzero(counts)}}
    report={'provenance':provenance,'hf_snapshot':str(snapshot),'stats':stats,'samples':rows,
            'torch':torch.__version__,'ort':ort.__version__}
    np.savez_compressed(args.out/'activation_scan.npz',**arrays)
    (args.out/'activation_scan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    emit({'stage':'done','stats':stats})

def main():
    global ROOT,OPS,DATA
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path,default=ROOT)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase',choices=['inspect','scan','device','histogram','summarize'],default='inspect')
    args = ap.parse_args()
    ROOT=args.project_root.resolve()
    OPS=ROOT/'work/ops/qwen3_dec28c256'
    DATA=ROOT/'work/qwen3_dec28c256'
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase=='summarize':
        summarize(args)
        return
    if args.phase=='histogram':
        histogram(args)
        return
    if args.phase=='device':
        device_info(args)
        return
    if args.phase=='scan':
        scan(args)
        return
    info = [graph_info(p) for p in [DATA/'qwen05b_decode_L28_C256.onnx',
            OPS/'qwen3_dec28c256_prepped.onnx', OPS/'qwen3_dec28c256_final_qdq.onnx']]
    emit(info)
    (args.out/'graph_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    hub = Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots'
    emit({'snapshots': [str(p) for p in hub.iterdir()]})
    with np.load(DATA/'calib_q3d28.npz') as z:
        emit({'calib': {k: {'shape': list(z[k].shape), 'dtype': str(z[k].dtype)}
                        for k in ['input_ids','cos','attn_mask','past_k_0']},
              'token_ids': z['input_ids'].ravel().tolist()})

if __name__ == '__main__':
    main()
