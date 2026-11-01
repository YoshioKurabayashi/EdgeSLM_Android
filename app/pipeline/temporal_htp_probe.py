"""P12e: controlled six-layer HTP probe using the exact saved P12c input.

Not a full-model dialogue test. Baseline and candidate each run three times.
Device files use unique prefixes; no previous results or models are deleted.
"""
# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# 何のための道具か:
#   P12e の範囲分離の候補（focused_qdq_experiment.py の後続用の範囲）を、P12c で使った
#   「6 層の診断モデル」（work/q3_hid/hid_qdq.onnx。層6 に入る残差 add_53 までを切り出したもの）に当てはめ、
#   Host（ORT の CPU）と実機（HTP）で整数コードを比べるための準備・変換・実機実行。
#   ※上の英語の docstring は argparse の --help にそのまま出るので、説明はこのコメントに書いた。
# どの段か: StudyMemo 11.5 節。記録は results/P12e_20260915_temporal_qdq.md 5 節。
#   実行したのは prepare だけ。Host 対 FP32 では add_53 の誤差が 27.84% → 10.53% に下がったが、
#   全層の自由生成で指示追従が悪化したため、convert・device は**未実行・未検証**
#   （P12c の「Host 対 HTP 31.53%」が改善したという結果ではない）。
# 入力:  work/q3_hid/hid_qdq.onnx（元の 6 層診断モデル）、results/P12c_20260915_activation_outliers/ の
#        code_histogram.json・device_jobs.json・saved_device/（実機に与えた入力 16 本と当時の出力）、
#        --out の下の qparam_map.json・focused_build_audit.json・fp32_reference.onnx（temporal_qdq_experiment.py などの出力）。
# 出力:  --out/htp_probe/ に p12e_hid_qdq.onnx・<条件>_host.npz・fp32_add53.npy・prepare.json
#        （convert・device を流せば DLC・context binary・EPContext・実機の結果も）。
#        prepare の記録は results/P12e_20260915_temporal_qdq/htp_probe/。
# 動かし方: & $py -B app/pipeline/temporal_htp_probe.py --out work/p12e_reproduction --phase prepare
#   （--phase convert / device / all もある。device は端末をつないで adb が通ること）
# 注意: 1 行目の import * で temporal_qdq_experiment.py の定数・関数（ROOT・read・digest・session・rel など）を
#   そのまま使う。このファイルと同じフォルダから読むので、スクリプトのパスを指定して起動する。
#   SDK・adb・端末のシリアルは当時の PC の値で固定。
# ---------------------------------------------------------------------------
from temporal_qdq_experiment import *
import subprocess

# 当時の環境: QAIRT 2.48 の SDK と、その Python（3.10）、adb、端末（SD888、シリアル固定）、アプリの置き場。
SDK=Path('D:/QAIRT/v2.48.0.260626/qairt/2.48.0.260626')
QPY=Path('D:/QAIRT/qairtenv/Scripts/python.exe')
ADB=Path('C:/Users/kurab/AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe')
SERIAL='QV780Y6DA3'
REMOTE='/sdcard/Android/data/org.edgeslm.runner/files'
PREVIOUS=ROOT/'results/P12c_20260915_activation_outliers'

def prepare(out):
    """6 層診断モデルの候補を作り、Host で動かして元の版と比べる（実機は使わない）。

    出力: htp_probe/p12e_hid_qdq.onnx・baseline_host.npz・candidate_host.npz・fp32_add53.npy・prepare.json。
    prepare.json の <条件>_host_vs_fp32 が「同じ入力での Host 対 FP32 の add_53 の相対 L2」
    （記録では baseline 27.84% → candidate 10.53%）。
    """
    dest=out/'htp_probe'
    dest.mkdir(exist_ok=True)
    original=ROOT/'work/q3_hid/hid_qdq.onnx'
    previous=read(PREVIOUS/'code_histogram.json')
    # ① 元の 6 層モデルが P12c の時と同じもの（sha256）であることを確かめる。
    assert digest(original)==previous['source_sha256']
    fullmap={g['scale_name']:g for g in read(out/'qparam_map.json')['groups']}
    m=onnx.load(str(original),load_external_data=False)
    init={t.name:t for t in m.graph.initializer}
    before={n:hashlib.sha256(t.SerializeToString()).hexdigest() for n,t in init.items()}
    allowed=set()
    changed=[]
    # ② 168 組版の「後続用の範囲」のうち、6 層モデルにもある組（記録では 27 組）だけを書き換える。
    #    書き換える前の値が全層モデルの元の値と同じであることも確かめる（同じ組を指しているか）。
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
    # ③ 診断モデルの出力は量子化後の整数コード（<名前>_QuantizeLinear_Output）。その名前と scale/zero を控える。
    #    観測を整数コードで取るのは、float を出力にすると HTP の変換が落ちるため（StudyMemo 11.8 (2)）。
    targets=[v.name.removesuffix('_QuantizeLinear_Output') for v in m.graph.output]
    assert 'add_53' in targets
    qparams={t:{'scale':float(numpy_helper.to_array(init[t+'_scale'])),
                'zero':int(numpy_helper.to_array(init[t+'_zero_point']))} for t in targets}
    model=dest/'p12e_hid_qdq.onnx'
    onnx.save(m,str(model))
    del m,init
    gc.collect()
    # ④ P12c で実機に与えたのと同じ入力ファイルを使う（sha256 で照合）。
    input_paths={n:PREVIOUS/'saved_device'/Path(p).name for n,p in previous['job']['batches'][0]['inputs'].items()}
    assert {n:digest(p) for n,p in input_paths.items()}==previous['input_sha256']
    report={'source_sha256':digest(original),'candidate_sha256':digest(model),'changes':changed,
            'unexpected_initializer_changes':unexpected,'qparams':qparams,
            'input_sha256':previous['input_sha256'],'input_paths':{n:str(p) for n,p in input_paths.items()}}
    # ⑤ 元と候補を Host で動かす。元の出力が P12c で保存した Host の整数コードと完全一致するかを確かめる。
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
    # ⑥ FP32 の基準モデルから add_53 の祖先だけを残し、同じ入力で FP32 の add_53 を計算する。
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
    # ⑦ 整数コードを実数に戻し（(code − zero) × scale）、FP32 との相対 L2 を出す。
    for mode in ['baseline','candidate']:
        codes=np.load(dest/f'{mode}_host.npz')['add_53_QuantizeLinear_Output']
        p=qparams['add_53'] if mode=='candidate' else {'scale':fullmap['add_53_scale']['old_scale'],'zero':fullmap['add_53_scale']['old_zero']}
        report[mode+'_host_vs_fp32']=rel((codes.astype(np.float64)-p['zero'])*p['scale'],reference)
    save(dest/'prepare.json',report)
    emit({'phase':'htp_prepare','changed_groups':len(changed),'host_vs_fp32':
          {k:report[k+'_host_vs_fp32'] for k in ['baseline','candidate']}})

def convert(out):
    """候補の QDQ を DLC → context binary（sm8350 = SD888）→ EPContext の ONNX にする（未実行・未検証）。

    途中で C-1（float の区間が残っていない）と C-2（SoC model が 0）を verify_artifacts で確かめる。
    EPContext は embed=False（context binary を別ファイルにする。C-12）。結果は convert.json。
    """
    sys.path.insert(0,str(ROOT))
    from app.pipeline import verify_artifacts,wrap_epctx
    dest=out/'htp_probe'
    model=dest/'p12e_hid_qdq.onnx'
    c1=verify_artifacts.check_c1_no_float_segments(model)
    assert c1.ok,c1
    # QAIRT の Python と SDK のライブラリを使う環境を作る（プロジェクトの venv は 3.12、QAIRT は 3.10 が要る）。
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

# 端末を 1 台に決めて adb を呼ぶ（出力は文字列、2 分で打ち切る）。
def adb(*args,check=True):
    return subprocess.run([str(ADB),'-s',SERIAL,*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace',check=check,timeout=120)

def device(out):
    """元の 6 層モデル（端末に既にあるもの）と候補を実機で 3 回ずつ動かし、Host の整数コードと比べる（未実行・未検証）。

    出力: htp_probe/<条件>_device_result.json・summary.json（Host 対 HTP の相対 L2・コードの最大差・
    3 回の結果がビット単位で同じか、add_53 は FP32 との誤差も）。
    """
    dest=out/'htp_probe'
    prepared=read(dest/'prepare.json')
    converted=read(dest/'convert.json')
    assert adb('get-state').stdout.strip()=='device'
    # Check the existing baseline on the phone before using it; never overwrite it.
    local_base=ROOT/'work/q3_hid/hid_epctx.onnx'
    remote_base='models/ops/hid_epctx.onnx'
    assert adb('shell','sha256sum',REMOTE+'/'+remote_base).stdout.split()[0]==digest(local_base)
    # ① 候補のモデルと入力を、既存のファイルと重ならない名前（p12e_ 付き）で push し、sha256 で照合する。
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
    # ② P12c の job を写し、job_id とモデルだけ替えて 3 回分のバッチで起動する。
    #    結果は out/last_result.json を 2 秒ごとに見て、job_id が一致するまで待つ（最大 180 秒）。
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
                # ③ Host と実機の整数コードを、同じ scale/zero で実数に戻して比べる（分母は Host の実数値。
                #    コード値のまま比べると zero point で誤差が薄まる。StudyMemo 11.8 (1)）。
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

# 引数: --out（temporal_qdq_experiment.py と同じ作業先）・--phase（prepare / convert / device / all）。
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
