"""P12e: recalibrate existing activation Q/DQ ranges, with frozen weights and KV interfaces.

Independent calibration text; same QDQ topology/dtypes; shared vs first/later
range fit; teacher-forced and free-running Host comparisons. No device deploy.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper,numpy_helper,TensorProto

ROOT=Path('D:/AI/source/n_chap5/EdgeSLM_Android')
OPS=ROOT/'work/ops/qwen3_dec28c256'
SOURCE=OPS/'qwen3_dec28c256_final_qdq.onnx'
FP32=OPS/'qwen3_dec28c256_prepped.onnx'
TARGETS=['add_26','add_53']

def emit(obj):
    print(json.dumps(obj,ensure_ascii=False),flush=True)

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def session(path):
    so=ort.SessionOptions()
    so.intra_op_num_threads=4
    so.inter_op_num_threads=1
    so.log_severity_level=3
    return ort.InferenceSession(str(path),so,providers=['CPUExecutionProvider'])

def qparam_map(out):
    pre=onnx.load(str(FP32),load_external_data=False)
    known={v.name:v.type.tensor_type.elem_type for v in [*pre.graph.input,*pre.graph.value_info,*pre.graph.output]}
    pinits={t.name for t in pre.graph.initializer}
    m=onnx.load(str(SOURCE),load_external_data=False)
    init={t.name:t for t in m.graph.initializer}
    prod={v:n for n in m.graph.node for v in n.output}
    def origin(name,seen=None):
        seen=set() if seen is None else seen
        if name in seen:
            return None
        seen.add(name)
        if name in known or name in pinits:
            return name
        n=prod.get(name)
        if n is not None and n.op_type in ['QuantizeLinear','DequantizeLinear','Identity']:
            return origin(n.input[0],seen)
        return None
    protected=set()
    def freeze_output(name,seen):
        if name in seen:
            return
        seen.add(name)
        n=prod.get(name)
        if n is None:
            return
        if n.op_type in ['QuantizeLinear','DequantizeLinear']:
            protected.add(tuple(n.input[1:3]))
            freeze_output(n.input[0],seen)
        elif n.op_type in ['Identity','Slice','Reshape','Transpose','Squeeze','Unsqueeze']:
            freeze_output(n.input[0],seen)
    for v in m.graph.output:
        if v.name.startswith('present_'):
            freeze_output(v.name,set())
    groups={}
    for n in m.graph.node:
        if n.op_type!='QuantizeLinear':
            continue
        pair=tuple(n.input[1:3])
        if len(pair)!=2 or not all(p in init for p in pair):
            raise ValueError('Dynamic or missing Q/DQ parameter')
        s=numpy_helper.to_array(init[pair[0]])
        z=numpy_helper.to_array(init[pair[1]])
        src=origin(n.input[0])
        g=groups.setdefault(pair,{'scale_name':pair[0],'zero_name':pair[1],
              'old_scale':s.tolist(),'old_zero':z.tolist(),'dtype':str(z.dtype),
              'sources':[],'q_nodes':[],'protected_reasons':[]})
        g['sources'].append(src)
        g['q_nodes'].append(n.name)
        if src is None or src in pinits or known.get(src)!=TensorProto.FLOAT:
            g['protected_reasons'].append('unmapped_or_non_activation')
        if src and (src.startswith('past_') or src in ['cos','sin','attn_mask']):
            g['protected_reasons'].append('cache_or_positional_input')
        if pair in protected:
            g['protected_reasons'].append('cache_output_path')
        if s.size!=1 or z.size!=1 or z.dtype not in (np.uint8,np.uint16):
            g['protected_reasons'].append('non_scalar_or_other_dtype')
    owners=defaultdict(set)
    for pair in groups:
        for p in pair:
            owners[p].add(pair)
    for pair,g in groups.items():
        if any(len(owners[p])>1 for p in pair):
            g['protected_reasons'].append('overlapping_parameter_pairs')
        g['sources']=sorted({s for s in g['sources'] if s is not None})
        g['protected_reasons']=sorted(set(g['protected_reasons']))
    params={'source':str(SOURCE),'source_sha256':digest(SOURCE),
            'groups':list(groups.values()),'nodes':len(m.graph.node),
            'initializer_count':len(init),'external_initializers':sum(bool(t.external_data) for t in init.values())}
    save(out/'qparam_map.json',params)
    emit({'phase':'map','groups':len(groups),'adjustable':sum(not g['protected_reasons'] for g in groups.values()),
          'protected':sum(bool(g['protected_reasons']) for g in groups.values()),
          'external_initializers':params['external_initializers']})
    for t in TARGETS:
        emit({'target':t,'groups':[g for g in groups.values() if t in g['sources']]})
    del m,pre
    gc.collect()
    return params

def prepare_probe(out,params):
    m=onnx.load(str(FP32),load_external_data=False)
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key=='location':
                original=FP32.parent/e.value
                dest=out/('fp32_'+original.name)
                if not dest.exists():
                    shutil.copyfile(original,dest)
                assert dest.stat().st_size==original.stat().st_size
                e.value=dest.name
    existing={v.name for v in m.graph.output}
    for t in TARGETS:
        if t not in existing:
            m.graph.output.append(helper.make_tensor_value_info(t,TensorProto.FLOAT,[1,1,1024]))
    onnx.save(m,str(out/'fp32_reference.onnx'))
    sources=sorted({s for g in params['groups'] if not g['protected_reasons'] for s in g['sources']})
    for i,s in enumerate(sources):
        for operation,suffix in [('ReduceMin','min'),('ReduceMax','max')]:
            name=f'p12e_{i}_{suffix}'
            m.graph.node.append(helper.make_node(operation,[s],[name],name=name,keepdims=0))
            m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[]))
    onnx.save(m,str(out/'fp32_calibration.onnx'))
    save(out/'observed_sources.json',sources)
    emit({'phase':'probe','activation_sources':len(sources)})

class Runner:
    def __init__(self,s,cos,sin):
        self.s,self.cos,self.sin=s,cos,sin
        self.names=[v.name for v in s.get_outputs()]
        self.input_names=[v.name for v in s.get_inputs()]
        self.reset()
    def reset(self):
        self.position=0
        self.cache={v.name:np.zeros(v.shape,np.float32) for v in self.s.get_inputs() if v.name.startswith('past_')}
    def feed(self,tid,s=None):
        s=s or self.s
        pos=self.position
        mask=np.full((1,1,1,256),-60,np.float32)
        mask[...,256-min(pos+1,256):]=0
        inputs={'input_ids':np.array([[tid]],np.int64),'cos':self.cos[pos].reshape(1,1,1,128),
                'sin':self.sin[pos].reshape(1,1,1,128),'attn_mask':mask,**self.cache}
        result=dict(zip(self.names,s.run(None,inputs)))
        self.cache={'past_'+n[len('present_'):]:a for n,a in result.items() if n.startswith('present_')}
        self.position+=1
        return result,inputs

def tokenizer():
    from transformers import AutoTokenizer
    snap=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
    return AutoTokenizer.from_pretrained(str(snap),local_files_only=True)

def tables():
    return tuple(np.fromfile(ROOT/f'work/rope_q3/{n}_table_l2048.bin',np.float32).reshape(2048,128) for n in ['cos','sin'])

def calibrate(out):
    sources=read(out/'observed_sources.json')
    tok=tokenizer()
    plain=['日本の首都は','The capital of France is','1, 2, 3, 4,','人工知能とは','def add(a, b):']
    chats=[('あなたは車載アシスタントです。短く答えてください。','窓を少し開けてください。'),
           ('You are an in-car assistant. Keep your answers short.','Please tell me the distance to the next rest stop.'),
           ('あなたは簡潔に回答するアシスタントです。','雨の日に持っていく物を三つ教えてください。')]
    sequences=[{'name':'plain_'+str(i),'ids':tok.encode(t,add_special_tokens=False),'max_steps':24,'text':t} for i,t in enumerate(plain)]
    for i,(system,user) in enumerate(chats):
        ids=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],
                                   tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert isinstance(ids,list) and all(isinstance(t,int) for t in ids)
        sequences.append({'name':'chat_'+str(i),'ids':ids,'max_steps':min(len(ids)+24,96),'system':system,'user':user})
    r=Runner(session(out/'fp32_calibration.onnx'),*tables())
    ranges={phase:{s:[float('inf'),-float('inf')] for s in sources} for phase in ['first','later']}
    records=[]
    for seq in sequences:
        r.reset()
        ids=seq['ids'].copy()
        processed=[]
        for pos in range(seq['max_steps']):
            if pos>=len(ids):
                next_id=int(outputs['logits'][0,-1].argmax())
                if next_id in (151645,151643):
                    break
                ids.append(next_id)
            tid=ids[pos]
            outputs,_=r.feed(tid)
            phase='first' if pos==0 else 'later'
            for i,s in enumerate(sources):
                lo=float(outputs[f'p12e_{i}_min'])
                hi=float(outputs[f'p12e_{i}_max'])
                assert np.isfinite(lo) and np.isfinite(hi),(s,lo,hi)
                ranges[phase][s][0]=min(ranges[phase][s][0],lo)
                ranges[phase][s][1]=max(ranges[phase][s][1],hi)
            processed.append(tid)
        records.append({**seq,'processed_ids':processed,'processed_text':tok.decode(processed)})
        save(out/'calibration_ranges.json',{'ranges':ranges,'sequences':records})
        emit({'phase':'calibration','sequence':seq['name'],'steps':len(processed),
              'first_add53':ranges['first'].get('add_53'),'later_add53':ranges['later'].get('add_53')})
    del r
    gc.collect()

def fit(lo,hi,qmax,margin):
    lo=min(0.0,lo)*margin
    hi=max(0.0,hi)*margin
    if hi==lo:
        return np.float32(1e-8),0
    scale=(hi-lo)/qmax
    zp=int(np.clip(round(-lo/scale),0,qmax))
    if lo<0 and zp>0:
        scale=max(scale,-lo/zp)
    if hi>0 and zp<qmax:
        scale=max(scale,hi/(qmax-zp))
    scale=np.nextafter(np.float32(scale),np.float32(np.inf))
    return scale,zp

def build(out,margin):
    params=read(out/'qparam_map.json')
    assert digest(SOURCE)==params['source_sha256'],'Source model changed'
    calibration=read(out/'calibration_ranges.json')
    assert len(calibration['sequences'])==8,'Incomplete calibration'
    ranges=calibration['ranges']
    assert all(np.isfinite(v).all() for p in ranges.values() for v in p.values())
    audit={}
    for phase in ['baseline','shared','first','later']:
        m=onnx.load(str(SOURCE),load_external_data=False)
        assert not any(t.external_data for t in m.graph.initializer),'Expected self-contained QDQ'
        original={t.name:hashlib.sha256(t.SerializeToString()).hexdigest() for t in m.graph.initializer}
        original_nodes=hashlib.sha256(b''.join(n.SerializeToString() for n in m.graph.node)).hexdigest()
        init={t.name:t for t in m.graph.initializer}
        changed=[]
        allowed=set()
        if phase!='baseline':
            for g in params['groups']:
                if g['protected_reasons']:
                    continue
                phases=['first','later'] if phase=='shared' else [phase]
                bounds=[ranges[p][s] for p in phases for s in g['sources']]
                lo=min(v[0] for v in bounds)
                hi=max(v[1] for v in bounds)
                dtype=np.dtype(g['dtype'])
                s,z=fit(lo,hi,np.iinfo(dtype).max,margin)
                scale_shape=numpy_helper.to_array(init[g['scale_name']]).shape
                zero_shape=numpy_helper.to_array(init[g['zero_name']]).shape
                init[g['scale_name']].CopyFrom(numpy_helper.from_array(np.asarray(s,np.float32).reshape(scale_shape),g['scale_name']))
                init[g['zero_name']].CopyFrom(numpy_helper.from_array(np.asarray(z,dtype=dtype).reshape(zero_shape),g['zero_name']))
                allowed.update([g['scale_name'],g['zero_name']])
                changed.append({'sources':g['sources'],'scale_name':g['scale_name'],'zero_name':g['zero_name'],
                                'scale':float(s),'zero':z,'dtype':g['dtype'],'observed':[lo,hi],
                                'represented':[-z*float(s),(np.iinfo(dtype).max-z)*float(s)]})
        assert len(m.graph.node)==params['nodes']
        assert hashlib.sha256(b''.join(n.SerializeToString() for n in m.graph.node)).hexdigest()==original_nodes
        unexpected=[t.name for t in m.graph.initializer
                    if t.name not in allowed and hashlib.sha256(t.SerializeToString()).hexdigest()!=original[t.name]]
        assert not unexpected,unexpected
        for target in TARGETS:
            for name in [target,target+'_DequantizeLinear_Output']:
                m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,1,1024]))
        path=out/(phase+'.onnx')
        onnx.save(m,str(path))
        audit[phase]={'changed_groups':len(changed),'changes':changed,'unexpected_initializer_changes':unexpected,
                      'model_sha256':digest(path),'margin':margin,
                      'source_sha256':params['source_sha256'],'node_sha256':original_nodes,
                      'preserved_nodes':len(m.graph.node),'preserved_initializers':len(original)-len(allowed)}
        save(out/'build_audit.json',audit)
        emit({'phase':'build','variant':phase,'changed_groups':len(changed),'path':str(path)})
        del m,init
        gc.collect()

def rel(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def metrics(got,ref):
    a,b=got['logits'].ravel().astype(np.float64),ref['logits'].ravel().astype(np.float64)
    pa=np.exp(a-a.max()); pa/=pa.sum()
    pb=np.exp(b-b.max()); pb/=pb.sum()
    out={'logits_rel_l2':rel(a,b),'kl_ref_to_q':float(np.sum(pb*(np.log(pb+1e-30)-np.log(pa+1e-30)))),
         'top1_same':bool(a.argmax()==b.argmax()),'ref_top1_rank':int(np.sum(a>a[b.argmax()]))+1}
    for t in TARGETS:
        out[t+'_raw_rel_l2']=rel(got[t],ref[t])
        out[t+'_qdq_rel_l2']=rel(got[t+'_DequantizeLinear_Output'],ref[t])
    for kv in ['k','v']:
        per=[rel(got[f'present_{kv}_{i}'][:,:,-1,:],ref[f'present_{kv}_{i}'][:,:,-1,:]) for i in range(28)]
        out['new_'+kv+'_median_layer_rel_l2']=float(np.median(per))
        out['new_'+kv+'_max_layer_rel_l2']=float(max(per))
        out['new_'+kv+'_layer6_rel_l2']=per[6]
    return out

def summarize_rows(rows):
    result={}
    for name,chosen in [('all',rows),('first',rows[:1]),('later',rows[1:])]:
        if not chosen:
            continue
        result[name]={k:float(np.mean([r['metrics'][k] for r in chosen])) for k in chosen[0]['metrics']}
        result[name]['tokens']=len(chosen)
    return result

def evaluate_simultaneous_unused(out):
    cos,sin=tables()
    ss={k:session(out/(k+'.onnx')) for k in ['baseline','shared','first','later']}
    refs=session(out/'fp32_reference.onnx')
    source_control=session(SOURCE)
    source_names=[v.name for v in source_control.get_outputs()]
    audit=read(out/'build_audit.json')
    qmap=read(out/'qparam_map.json')
    bound={}
    for p in ss:
        bound[p]={}
        for t in TARGETS:
            if p=='baseline':
                g=next(g for g in qmap['groups'] if t in g['sources'])
                s,z=float(g['old_scale']),int(g['old_zero'])
                bound[p][t]=[-z*s,(65535-z)*s]
            else:
                bound[p][t]=next(c['represented'] for c in audit[p]['changes'] if t in c['sources'])
    total={}
    for name in ['chat_original','chat_english']:
        trace=read(ROOT/f'results/P12d_20260915_dialogue_activation_ranges/{name}.json')
        independent=np.load(ROOT/f'results/P12d_20260915_dialogue_activation_ranges/{name}.npz')
        reference_checks=[]
        observation_controls=[]
        ids=[r['token_id'] for r in trace['rows']]
        ref=Runner(refs,cos,sin)
        runners={k:Runner(ss['first' if k=='split' else k],cos,sin) for k in ['baseline','shared','split']}
        rows={k:[] for k in runners}
        oracle={k:[] for k in runners}
        for pos,tid in enumerate(ids):
            expected,reference_inputs=ref.feed(tid)
            assert all(np.isfinite(v).all() for v in expected.values())
            reference_checks.append({'position':pos,**{t:rel(expected[t].ravel(),independent[f'layer_{layer}'][pos])
                                    for t,layer in zip(TARGETS,[3,6])}})
            for mode,r in runners.items():
                variant=('first' if pos==0 else 'later') if mode=='split' else mode
                got,_=r.feed(tid,ss[variant])
                assert all(np.isfinite(v).all() for v in got.values()),(name,mode,pos)
                row={'position':pos,'token_id':tid,'metrics':metrics(got,expected),'saturation':{}}
                for t in TARGETS:
                    lo,hi=bound[variant][t]
                    row['saturation'][t]={'before_q_outside':int(np.sum((got[t]<lo)|(got[t]>hi))),
                                           'reference_outside':int(np.sum((expected[t]<lo)|(expected[t]>hi))),
                                           'range':[lo,hi]}
                rows[mode].append(row)
                if pos in [0,1,40,len(ids)-1]:
                    same=dict(zip(r.names,ss[variant].run(None,reference_inputs)))
                    oracle[mode].append({'position':pos,'metrics':metrics(same,expected)})
                    if mode=='baseline':
                        unobserved=dict(zip(source_names,source_control.run(None,reference_inputs)))
                        observation_controls.append({'position':pos,
                            'logits_rel_l2':rel(same['logits'],unobserved['logits']),
                            'max_cache_rel_l2':max(rel(same[n],v) for n,v in unobserved.items() if n.startswith('present_'))})
            if pos%24==0:
                emit({'phase':'evaluate','case':name,'position':pos,
                     'add53':{k:rows[k][-1]['metrics']['add_53_qdq_rel_l2'] for k in runners}})
        result={'case':name,'tokens':len(ids),'rows':rows,'same_fp32_kv_oracle':oracle,
                'summary':{k:summarize_rows(v) for k,v in rows.items()},'input_ids':ids,
                'fp32_prepped_vs_prior_official_reference':reference_checks,
                'observed_baseline_vs_original_qdq':observation_controls}
        save(out/(name+'_teacher.json'),result)
        total[name]=result['summary']
        save(out/'teacher_summary.json',total)
        emit({'phase':'evaluated','case':name,'summary':total[name]})
    del ss,refs
    gc.collect()

def compact(result):
    return {n:(a[:,:,-1:,:].copy() if n.startswith('present_') else a.copy()) for n,a in result.items()}

def evaluate(out):
    """Run one variant at a time to avoid swapping several large ORT sessions."""
    cos,sin=tables()
    cases=['chat_original','chat_english']
    audit=read(out/'build_audit.json')
    qmap=read(out/'qparam_map.json')
    bounds={}
    for p in ['baseline','shared','first','later']:
        bounds[p]={}
        for t in TARGETS:
            if p=='baseline':
                g=next(g for g in qmap['groups'] if t in g['sources'])
                s,z=float(g['old_scale']),int(g['old_zero'])
                bounds[p][t]=[-z*s,(65535-z)*s]
            else:
                bounds[p][t]=next(c['represented'] for c in audit[p]['changes'] if t in c['sources'])
    records={}
    ref_session=session(out/'fp32_reference.onnx')
    for name in cases:
        trace=read(ROOT/f'results/P12d_20260915_dialogue_activation_ranges/{name}.json')
        ids=[r['token_id'] for r in trace['rows']]
        independent=np.load(ROOT/f'results/P12d_20260915_dialogue_activation_ranges/{name}.npz')
        r=Runner(ref_session,cos,sin)
        vectors=[]
        checks=[]
        for pos,tid in enumerate(ids):
            got,inputs=r.feed(tid)
            assert all(np.isfinite(a).all() for a in got.values())
            vectors.append(compact(got))
            checks.append({'position':pos,**{t:rel(got[t].ravel(),independent[f'layer_{layer}'][pos])
                           for t,layer in zip(TARGETS,[3,6])}})
            if pos in [0,1,40,len(ids)-1]:
                np.savez(out/f'{name}_oracle_{pos}.npz',**inputs)
        np.savez(out/f'{name}_reference.npz',**{n:np.stack([v[n] for v in vectors]) for n in vectors[0]})
        records[name]={'case':name,'tokens':len(ids),'input_ids':ids,'rows':{},'summary':{},
                       'same_fp32_kv_oracle':{},'fp32_prepped_vs_prior_official_reference':checks,
                       'observed_baseline_vs_original_qdq':[],'status':'incomplete'}
        save(out/(name+'_teacher.json'),records[name])
        emit({'phase':'reference','case':name,'tokens':len(ids),'max_reference_relative_error':
              {t:max(c[t] for c in checks) for t in TARGETS}})
        del r,vectors,got,inputs,independent
        gc.collect()
    del ref_session
    gc.collect()
    for mode in ['baseline','shared','split']:
        first=session(out/('first.onnx' if mode=='split' else mode+'.onnx'))
        later=session(out/'later.onnx') if mode=='split' else first
        source=session(SOURCE) if mode=='baseline' else None
        source_names=[v.name for v in source.get_outputs()] if source else []
        for name in cases:
            record=records[name]
            reference=np.load(out/f'{name}_reference.npz')
            ref_arrays={n:reference[n] for n in reference.files}
            reference.close()
            r=Runner(first,cos,sin)
            rows=[]
            oracle=[]
            for pos,tid in enumerate(record['input_ids']):
                variant=('first' if pos==0 else 'later') if mode=='split' else mode
                s=first if pos==0 else later
                got,_=r.feed(tid,s)
                expected={n:a[pos] for n,a in ref_arrays.items()}
                assert all(np.isfinite(a).all() for a in got.values()),(name,mode,pos)
                row={'position':pos,'token_id':tid,'metrics':metrics(got,expected),'saturation':{}}
                for t in TARGETS:
                    lo,hi=bounds[variant][t]
                    row['saturation'][t]={'before_q_outside':int(np.sum((got[t]<lo)|(got[t]>hi))),
                        'reference_outside':int(np.sum((expected[t]<lo)|(expected[t]>hi))),'range':[lo,hi]}
                rows.append(row)
                if pos in [0,1,40,len(record['input_ids'])-1]:
                    with np.load(out/f'{name}_oracle_{pos}.npz') as f:
                        inputs={n:f[n] for n in f.files}
                    same=dict(zip(r.names,s.run(None,inputs)))
                    oracle.append({'position':pos,'metrics':metrics(same,expected)})
                    if source:
                        original=dict(zip(source_names,source.run(None,inputs)))
                        record['observed_baseline_vs_original_qdq'].append({'position':pos,
                            'logits_rel_l2':rel(same['logits'],original['logits']),
                            'max_cache_rel_l2':max(rel(same[n],a) for n,a in original.items() if n.startswith('present_'))})
                        del original
                    del inputs,same
                if pos%24==0:
                    record['rows'][mode]=rows
                    save(out/(name+'_teacher.json'),record)
                    emit({'phase':'evaluate','case':name,'variant':mode,'position':pos,
                          'add53':row['metrics']['add_53_qdq_rel_l2'],'logits':row['metrics']['logits_rel_l2']})
            record['rows'][mode]=rows
            record['summary'][mode]=summarize_rows(rows)
            record['same_fp32_kv_oracle'][mode]=oracle
            record['status']='complete' if mode=='split' else 'incomplete'
            save(out/(name+'_teacher.json'),record)
            save(out/'teacher_summary.json',{n:rec['summary'] for n,rec in records.items()})
            emit({'phase':'evaluated','case':name,'variant':mode,'later':record['summary'][mode]['later']})
            del r,got,expected,ref_arrays
            gc.collect()
        del first,later,source,s
        gc.collect()

def generate(out,max_new):
    tok=tokenizer()
    cos,sin=tables()
    record=read(ROOT/'results/P12d_20260915_dialogue_activation_ranges/chat_original.json')
    chunks=[[r['token_id'] for r in record['rows'] if r['turn']==i and r['phase']=='prompt'] for i in range(3)]
    results={}
    for mode in ['fp32','baseline','shared','split']:
        first=session(out/('fp32_reference.onnx' if mode=='fp32' else ('first.onnx' if mode=='split' else mode+'.onnx')))
        later=session(out/'later.onnx') if mode=='split' else first
        r=Runner(first,cos,sin)
        turns=[]
        for turn,ids in enumerate(chunks):
            for tid in ids:
                got,_=r.feed(tid,first if r.position==0 else later)
            output=[]
            stop=None
            for _ in range(max_new):
                tid=int(got['logits'].ravel().argmax())
                if tid in [151645,151643]:
                    stop=tid
                    break
                output.append(tid)
                got,_=r.feed(tid,first if r.position==0 else later)
            turns.append({'turn':turn,'text':tok.decode(output),'generated_ids':output,'stop_id':stop,
                          'total_consumed':r.position,'max_new':max_new})
            results[mode]=turns
            save(out/'free_generation.json',results)
            emit({'phase':'generate','variant':mode,'turn':turn,'text':turns[-1]['text'],'stop':stop})
        del r,first,later
        gc.collect()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','calibrate','build','evaluate','generate'],required=True)
    ap.add_argument('--margin',type=float,default=1.5)
    ap.add_argument('--max-new',type=int,default=64)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/('run_'+args.phase+'.json'),{'phase':args.phase,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'script_sha256':digest(Path(__file__)),'python':sys.version,'numpy':np.__version__,
         'onnx':onnx.__version__,'onnxruntime':ort.__version__,'providers':['CPUExecutionProvider'],
         'intra_op_threads':4,'margin':args.margin,'max_new':args.max_new})
    if args.phase=='prepare':
        params=qparam_map(args.out)
        prepare_probe(args.out,params)
    elif args.phase=='calibrate':
        calibrate(args.out)
    elif args.phase=='build':
        build(args.out,args.margin)
    elif args.phase=='evaluate':
        evaluate(args.out)
    elif args.phase=='generate':
        generate(args.out,args.max_new)

if __name__=='__main__':
    main()
