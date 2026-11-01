"""Read-only inspection of existing Qwen3 models; writes only to --out.

【日本語の説明（学習用、2026-10-18 追記）】
何のための道具か:
    P12c「Qwen3 の残差の最大値 7257 を公式 FP32 で追跡」の診断。既存のモデル・量子化パラメータ・
    アプリは一切変えず（読むだけ）、--out にだけ書く。
    層6 に入る残差 add_53 で見えた巨大な値（7257）が、較正や前処理が作った異常値なのか、
    公式モデルでも出る値なのかを確かめ、さらに保存済みの実機出力と Host の整数コードの差を数える。
どの段か: StudyMemo 11.4 節「なぜ scale がこんなに粗いのか — 位置0 の巨大活性」と 11.8 節 (1)「分母」。
結果（results/P12c_20260915_activation_outliers.md）:
    巨大値は公式 HF の FP32 でも出る（7258.10）。較正 30 入力のうち 1000 を超える 5 要素はすべて
    位置0・チャンネル35。add_53 の Host↔HTP は ±2 段以内が 93.16% でも、逆量子化後の相対 L2 は 31.53%
    （整数コードのまま測ると 0.093% に見える）。scale は層3 の入口（add_26）で 0.113 に跳ねる。
段（--phase）と出力（すべて --out の下。保存先は results/P12c_20260915_activation_outliers/）:
    inspect   … add_53 のまわりのノードと量子化パラメータ → graph_info.json
    scan      … 較正データ全 30 入力で、export 直後・前処理後の ONNX と公式 HF の add_53 を比べる
                → activation_scan.json / .npz
    device    … 端末の out/ と jobs/ の一覧を読む（新しい推論はしない） → device_jobs.json
    histogram … 保存済みの実機出力を pull し、同じ入力で Host を再実行して整数コードの差を数える
                → code_histogram.json・saved_device/
    summarize … 先頭／後続に分けた集計と、測り方の訂正値 → summary.json
動かし方（プロジェクトのルートで。results/P12c_….md 7 節）:
    & $py -B -u app/pipeline/trace_qwen3_activation.py --phase inspect --out <出力先>
    （scan → device → histogram → summarize の順）
"""
from pathlib import Path
import argparse
import json
import os
import hashlib
import copy
import subprocess

# HF をオフラインに固定（公式モデルはローカルのスナップショットだけを使う。追加のダウンロードなし）。
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import onnx
from onnx import numpy_helper

# 既定のパス（--project-root で変えられる）。DATA の export 直後の ONNX は名前が qwen05b_… だが中身は Qwen3 の 28 層
# （Qwen2.5 の時の名前の付け方をそのまま使った。推定）。
ROOT = Path('D:/AI/source/n_chap5/EdgeSLM_Android')
OPS = ROOT / 'work/ops/qwen3_dec28c256'
DATA = ROOT / 'work/qwen3_dec28c256'
ADB = Path('C:/Users/kurab/AppData/Local/Microsoft/WinGet/Packages/Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe/platform-tools/adb.exe')
REMOTE = '/sdcard/Android/data/org.edgeslm.runner/files'

# 途中経過を 1 行の JSON で出す。
def emit(x):
    print(json.dumps(x, ensure_ascii=False), flush=True)

def graph_info(path):
    """モデルのうち add_53 を入力か出力に持つノードと、add_53 の量子化パラメータ（scale・zero point）を集める。

    引数 path: ONNX のパス（重みは読まない）。戻り値: {'path','nodes','params','outputs'} の dict。
    """
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
    # 【説明】targets（例 ['add_53']）を作るのに要るノードだけを、出力側から遡って残した観測用モデルを dest に保存する。
    #   ノードは書き換えずに写すので、値は元のモデルと同じになる。外部データの重みは読み込んでモデルの中に入れる
    #   （元の .data を書き換えないため）。戻り値: 何を元に何ノード残したかの記録。
    m = onnx.load(str(source), load_external_data=False)
    needed = set(targets)
    keep = []
    # ① ノードを後ろから見て、必要な出力を作るノードを残し、その入力を「必要」に足していく。
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
    # ② 出力は観測点だけにする。型の情報が無ければ [1,1,1024] の float として宣言する。
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

# a を基準に b を比べる: 最大の絶対差と相対 L2（||a−b|| / ||a||）。分母は基準（a）側。
def compare(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return {'max_abs':float(np.max(np.abs(a-b))),
            'rel_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))}

def device_info(args):
    """端末の out/ にある hid_・l14_・l6v_ の出力ファイル名と、その 3 つの job の JSON を読んで device_jobs.json に残す。

    実機での推論はしない（前回保存した出力を使う）。
    """
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
    """保存済みの実機出力（hid の各層の残差、uint16 の整数コード）と、同じ入力で Host を再実行した整数コードを比べる。

    出力: code_histogram.json（テンソルごとに、コード差のヒストグラム・±1/±2/±5 段以内の割合・
    逆量子化後の誤差）と saved_device/<テンソル>_comparison.npz。
    """
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
    # ① 端末から入力 16 本と hid の出力を pull する。l14 の add_53 も、hid と同じかの照合に使う。
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
    # ② 同じ 6 層診断モデル（work/q3_hid/hid_qdq.onnx）を Host（CPU）で、同じ入力で動かす。
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
        # ③ 差は int32 に直してから引く（uint16 のまま引くと負の差が 65535 側に回ってしまう）。
        #    af/bf は (code − zero) × scale で実数に戻した値。誤差はこの実数で測る。
        #    'uncentered_codes_misleading' は、わざとコード値のまま測った値（zero point 1496 を含む大きな数が
        #    分母になり誤差が薄まる。0.093% 対 31.53%。StudyMemo 11.8 (1)）。
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
    # ④ 端末に与えた入力が評価データのどの sample と同じかを探す（記録では完全一致する sample は無かった）。
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
    """scan と histogram の結果を、位置0（first）と位置1以降（later）に分けて集計し summary.json に書く。

    assert で、最大値がチャンネル35 であること・export/前処理後と公式 HF の差が 1% 未満であること・
    hid と l14 の add_53 が完全一致すること・コード差 RMS × scale が実数の誤差 RMS と一致することを確かめる。
    """
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
    # 量子化の全幅（65535 段 × scale）と、実際の Host 出力の幅の比（記録では 508.02 倍）。
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
    """較正データ（work/qwen3_dec28c256/calib_q3d28.npz、5 文 × 6 歩 = 30 入力）で add_53 を 3 系統で測る。

    3 系統: export 直後の ONNX・前処理後の ONNX（量子化に渡す prepped）・公式 HF の FP32。
    出力: activation_scan.npz（[30, 1024] × 3）と activation_scan.json（最大の位置・閾値ごとの数など）。
    """
    import onnxruntime as ort
    data = dict(np.load(DATA/'calib_q3d28.npz'))
    arrays = {}
    provenance = []
    # ① ONNX の 2 系統は add_53 の祖先だけを切り出したモデルで、30 入力を 1 つずつ流す。
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
    # ② 公式 HF（FP32・CPU・eager attention・キャッシュ無し）。
    snapshot=next((Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots').iterdir())
    hf=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,
                    dtype=torch.float32,attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    rows=[]
    prefix=[]
    hf_h=[]
    with torch.inference_mode():
        for i in range(len(data['input_ids'])):
            # 各入力の位置は「マスクで有効なスロットの数 − 1」。位置0 で新しい文が始まるので prefix を空に戻す。
            # 較正時と同じトークン列を作り、HF には prefix 全体を毎回渡す（use_cache=False）。
            pos=int(np.count_nonzero(data['attn_mask'][i]==0))-1
            if pos==0:
                prefix=[]
            assert pos==len(prefix), (i,pos,len(prefix))
            prefix.append(int(data['input_ids'][i,0]))
            pred=hf(input_ids=torch.tensor([prefix]),use_cache=False,output_hidden_states=True)
            # hidden_states[6] = 6 層を通った後 = 層6 に入る残差（ONNX の add_53 に当たる）の最後の位置。
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
    # ③ 絶対値が 10・100・1000・5000 を超えた要素が、どの sample・どのチャンネルにあるかを数える。
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
    """引数を読み、--phase の段だけを走らせる。既定（inspect）はグラフと較正データの形を見るだけ。"""
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
