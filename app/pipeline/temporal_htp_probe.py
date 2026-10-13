"""P12e: controlled six-layer HTP probe using the exact saved P12c input.

Not a full-model dialogue test. Baseline and candidate each run three times.
Device files use unique prefixes; no previous results or models are deleted.
"""
from temporal_qdq_experiment import *
import subprocess

SDK=Path('D:/QAIRT/v2.48.0.260626/qairt/2.48.0.260626')
QPY=Path('D:/QAIRT/qairtenv/Scripts/python.exe')
ADB=Path('C:/Users/kurab/AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe')
SERIAL='QV780Y6DA3'
REMOTE='/sdcard/Android/data/org.edgeslm.runner/files'
PREVIOUS=ROOT/'results/P12c_20260915_activation_outliers'

def prepare(out):
    dest=out/'htp_probe'
    dest.mkdir(exist_ok=True)
    original=ROOT/'work/q3_hid/hid_qdq.onnx'
    previous=read(PREVIOUS/'code_histogram.json')
    assert digest(original)==previous['source_sha256']
    fullmap={g['scale_name']:g for g in read(out/'qparam_map.json')['groups']}
    m=onnx.load(str(original),load_external_data=False)
    init={t.name:t for t in m.graph.initializer}
    before={n:hashlib.sha256(t.SerializeToString()).hexdigest() for n,t in init.items()}
    allowed=set()
    changed=[]
    for c in read(out/'focused_build_audit.json')['later']['changes']:
        if c['scale_name'] not in init:
            continue
        g=fullmap[c['scale_name']]
        assert not g['protected_reasons']
        assert float(numpy_helper.to_array(init[c['scale_name']]))==g['old_scale']
        assert int(numpy_helper.to_array(init[c['zero_name']]))==g['old_zero']
        for n,v in [(c['scale_name'],c['scale']),(c['zero_name'],c['zero'])]:
            a=numpy_helper.to_array(init[n])
            init[n].CopyFrom(numpy_helper.from_array(np.asarray(v,dtype=a.dtype).reshape(a.shape),n))
            allowed.add(n)
        changed.append(c)
    unexpected=[n for n,t in init.items() if n not in allowed and hashlib.sha256(t.SerializeToString()).hexdigest()!=before[n]]
    assert not unexpected
    targets=[v.name.removesuffix('_QuantizeLinear_Output') for v in m.graph.output]
    assert 'add_53' in targets
    qparams={t:{'scale':float(numpy_helper.to_array(init[t+'_scale'])),
                'zero':int(numpy_helper.to_array(init[t+'_zero_point']))} for t in targets}
    model=dest/'p12e_hid_qdq.onnx'
    onnx.save(m,str(model))
    del m,init
    gc.collect()
    input_paths={n:PREVIOUS/'saved_device'/Path(p).name for n,p in previous['job']['batches'][0]['inputs'].items()}
    assert {n:digest(p) for n,p in input_paths.items()}==previous['input_sha256']
    report={'source_sha256':digest(original),'candidate_sha256':digest(model),'changes':changed,
            'unexpected_initializer_changes':unexpected,'qparams':qparams,
            'input_sha256':previous['input_sha256'],'input_paths':{n:str(p) for n,p in input_paths.items()}}
    for mode,path in [('baseline',original),('candidate',model)]:
        s=session(path)
        feeds={i.name:np.fromfile(input_paths[i.name],np.int64 if i.type=='tensor(int64)' else np.float32).reshape(i.shape)
               for i in s.get_inputs()}
        values=dict(zip([v.name for v in s.get_outputs()],s.run(None,feeds)))
        np.savez(dest/f'{mode}_host.npz',**values)
        if mode=='baseline':
            for t in targets:
                prior=np.load(PREVIOUS/f'saved_device/{t}_comparison.npz')
                assert np.array_equal(values[t+'_QuantizeLinear_Output'],prior['host_codes'])
        del s,values
        gc.collect()
    # Prune the existing prepped observer while retaining its local external-data location.
    m=onnx.load(str(out/'fp32_reference.onnx'),load_external_data=False)
    prod={v:n for n in m.graph.node for v in n.output}
    needed=set()
    def visit(n):
        if not n or n in needed:
            return
        needed.add(n)
        if n in prod:
            for x in prod[n].input:
                visit(x)
    visit('add_53')
    nodes=[n for n in m.graph.node if any(v in needed for v in n.output)]
    inputs=[v for v in m.graph.input if v.name in needed]
    inits=[t for t in m.graph.initializer if t.name in needed]
    infos=[v for v in m.graph.value_info if v.name in needed]
    for field,values in [('node',nodes),('input',inputs),('initializer',inits),('value_info',infos)]:
        del getattr(m.graph,field)[:]
        getattr(m.graph,field).extend(values)
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info('add_53',TensorProto.FLOAT,[1,1,1024]))
    refpath=out/'fp32_hid_probe.onnx'
    onnx.save(m,str(refpath))
    del m
    s=session(refpath)
    reference=s.run(None,{i.name:feeds[i.name] for i in s.get_inputs()})[0]
    np.save(dest/'fp32_add53.npy',reference)
    for mode in ['baseline','candidate']:
        codes=np.load(dest/f'{mode}_host.npz')['add_53_QuantizeLinear_Output']
        p=qparams['add_53'] if mode=='candidate' else {'scale':fullmap['add_53_scale']['old_scale'],'zero':fullmap['add_53_scale']['old_zero']}
        report[mode+'_host_vs_fp32']=rel((codes.astype(np.float64)-p['zero'])*p['scale'],reference)
    save(dest/'prepare.json',report)
    emit({'phase':'htp_prepare','changed_groups':len(changed),'host_vs_fp32':
          {k:report[k+'_host_vs_fp32'] for k in ['baseline','candidate']}})

def convert(out):
    sys.path.insert(0,str(ROOT))
    from app.pipeline import verify_artifacts,wrap_epctx
    dest=out/'htp_probe'
    model=dest/'p12e_hid_qdq.onnx'
    c1=verify_artifacts.check_c1_no_float_segments(model)
    assert c1.ok,c1
    env=dict(os.environ)
    env.update(VIRTUAL_ENV=str(QPY.parent.parent),PYTHONPATH=str(SDK/'lib/python'),PYTHONUTF8='1')
    env['PATH']=str(SDK/'bin/x86_64-windows-msvc')+os.pathsep+str(SDK/'lib/x86_64-windows-msvc')+os.pathsep+env.get('PATH','')
    binary_dir=dest/'context'
    binary_dir.mkdir(exist_ok=True)
    commands=[('convert',[str(QPY),str(SDK/'bin/x86_64-windows-msvc/qairt-converter'),'--input_network',str(model),
               '--preserve_io_datatype','--output_path',str(dest/'p12e_hid.dlc')]),
              ('context',[str(SDK/'bin/x86_64-windows-msvc/qnn-context-binary-generator.exe'),'--dlc_path',str(dest/'p12e_hid.dlc'),
               '--backend',str(SDK/'lib/x86_64-windows-msvc/QnnHtp.dll'),'--binary_file','p12e_hid_ctx',
               '--output_dir',str(binary_dir),'--htp_socs','sm8350'])]
    for label,cmd in commands:
        with (dest/f'{label}.log').open('w',encoding='utf-8') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
        emit({'phase':'htp_convert','stage':label,'complete':True})
    binaries=list(binary_dir.glob('*.bin'))
    assert len(binaries)==1
    meta=dest/'context_meta.json'
    cmd=[str(SDK/'bin/x86_64-windows-msvc/qnn-context-binary-utility.exe'),'--context_binary',str(binaries[0]),'--json_file',str(meta)]
    with (dest/'meta.log').open('w',encoding='utf-8') as f:
        subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
    commands.append(('metadata',cmd))
    c2=verify_artifacts.check_c2_socmodel_zero(meta)
    assert c2.ok,c2
    wrapped=wrap_epctx.run(binaries[0],model,dest/'p12e_hid_epctx.onnx','p12e_hid',embed=False)
    save(dest/'convert.json',{'commands':commands,'C1':c1,'C2':c2,'wrapper':wrapped,
                             'context_sha256':digest(binaries[0]),'epctx_sha256':digest(dest/'p12e_hid_epctx.onnx')})

def adb(*args,check=True):
    return subprocess.run([str(ADB),'-s',SERIAL,*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace',check=check,timeout=120)

def device(out):
    dest=out/'htp_probe'
    prepared=read(dest/'prepare.json')
    converted=read(dest/'convert.json')
    assert adb('get-state').stdout.strip()=='device'
    # Check the existing baseline on the phone before using it; never overwrite it.
    local_base=ROOT/'work/q3_hid/hid_epctx.onnx'
    remote_base='models/ops/hid_epctx.onnx'
    assert adb('shell','sha256sum',REMOTE+'/'+remote_base).stdout.split()[0]==digest(local_base)
    for name in ['p12e_hid_epctx.onnx',converted['wrapper']['context_file']]:
        adb('push',dest/name,REMOTE+'/models/ops/'+name)
        assert adb('shell','sha256sum',REMOTE+'/models/ops/'+name).stdout.split()[0]==digest(dest/name)
    inputs={n:'inputs/p12e_'+Path(p).name for n,p in prepared['input_paths'].items()}
    for n,p in prepared['input_paths'].items():
        adb('push',p,REMOTE+'/'+inputs[n])
        assert adb('shell','sha256sum',REMOTE+'/'+inputs[n]).stdout.split()[0]==prepared['input_sha256'][n]
    result={}
    reference=np.load(dest/'fp32_add53.npy')
    original_qmap={g['scale_name']:g for g in read(out/'qparam_map.json')['groups']}
    for mode,model in [('baseline',remote_base),('candidate','models/ops/p12e_hid_epctx.onnx')]:
        job=copy.deepcopy(read(PREVIOUS/'device_jobs.json')['jobs']['hid'])
        job['job_id']='p12e_'+mode+'_'+str(time.time_ns())
        job['model']=model
        job['batches']=[{'id':f'{i:03d}','inputs':inputs} for i in range(3)]
        local=dest/f'{mode}_job.json'
        save(local,job)
        adb('push',local,REMOTE+'/jobs/'+local.name)
        adb('shell','am','start','-n','org.edgeslm.runner/.RunnerActivity','-a','org.edgeslm.runner.RUN','--es','job','jobs/'+local.name)
        res=None
        for _ in range(90):
            raw=adb('shell','cat',REMOTE+'/out/last_result.json',check=False)
            try:
                candidate=json.loads(raw.stdout)
                if candidate.get('job_id')==job['job_id']:
                    res=candidate
                    break
            except json.JSONDecodeError:
                pass
            time.sleep(2)
        assert res is not None,'Timed out waiting for matching job_id'
        save(dest/f'{mode}_device_result.json',res)
        assert res.get('ok') and res.get('completed')==3,res
        host=np.load(dest/f'{mode}_host.npz')
        rows=[]
        all_codes={}
        for rec in res['records']:
            assert 'error' not in rec and len(rec['outputs'])==6,rec
            for item in rec['outputs']:
                name=item['name']
                p=dest/Path(item['path']).name
                adb('pull',REMOTE+'/'+item['path'],p)
                assert item['type']=='uint16' and p.stat().st_size==item['bytes']
                codes=np.fromfile(p,np.uint16).reshape(item['shape'])
                all_codes.setdefault(name,[]).append(codes)
                t=name.removesuffix('_QuantizeLinear_Output')
                q=prepared['qparams'][t] if mode=='candidate' else {'scale':original_qmap[t+'_scale']['old_scale'],'zero':original_qmap[t+'_scale']['old_zero']}
                a=(host[name].astype(np.float64)-q['zero'])*q['scale']
                b=(codes.astype(np.float64)-q['zero'])*q['scale']
                delta=codes.astype(np.int32)-host[name].astype(np.int32)
                row={'repeat':rec['id'],'tensor':t,'scale':q['scale'],'zero':q['zero'],
                     'htp_vs_host_rel_l2':rel(b,a),'error_rms':float(np.sqrt(np.mean((b-a)**2))),
                     'host_rms':float(np.sqrt(np.mean(a*a))), 'max_abs_code_diff':int(np.abs(delta).max())}
                if t=='add_53':
                    row['host_vs_fp32_rel_l2']=rel(a,reference)
                    row['htp_vs_fp32_rel_l2']=rel(b,reference)
                rows.append(row)
        repeated=all(all(np.array_equal(values[0],a) for a in values[1:]) for values in all_codes.values())
        result[mode]={'rows':rows,'three_repeats_bit_exact':repeated,'job_id':job['job_id']}
        save(dest/'summary.json',result)
        emit({'phase':'htp_device','variant':mode,'repeats_bit_exact':repeated,
              'add53':[r for r in rows if r['tensor']=='add_53']})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','convert','device','all'],required=True)
    args=ap.parse_args()
    if args.phase in ['prepare','all']:
        prepare(args.out)
    if args.phase in ['convert','all']:
        convert(args.out)
    if args.phase in ['device','all']:
        device(args.out)
