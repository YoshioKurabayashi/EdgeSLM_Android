"""Exploratory P12e control: change only ranges dominated by the first token.

Selection uses calibration only: shared/later scale ratio >= 32. Both variants
use the identical shared-range model for their first token. Weights, cache I/O,
and the remaining activation parameters retain the original model settings.
"""
# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# 何のための道具か:
#   P12e の続きの対照実験。temporal_qdq_experiment.py の全面的な較正し直し（shared/split）は大きく悪化したので、
#   「位置0（先頭）の巨大値に引っ張られている組」だけ、つまり較正で
#   「先頭も含めた共通の範囲の scale ÷ 後続だけの範囲の scale」が 32 以上の 168 組だけを変える。
#   変種: focused_shared（全位置で共通の範囲）・focused_split（位置0 は focused_shared、位置1 以降は後続の範囲）・
#   hybrid（自由生成だけ。位置0 は baseline、位置1 以降は focused_split の後続モデル）。
#   ※上の英語の docstring は argparse の --help に出るので、説明はこのコメントに書いた。
# 結果（results/P12e_20260915_temporal_qdq.md の表と 5 節）:
#   focused_split は日本語の予測一致 86/106 = 81.1%（baseline と同等以上）だが、自由生成で「音楽をかけて」に
#   直前と同じエアコンの説明を返した。hybrid でも同じ → 採用せず。この崩れの解剖が P12f（divergence_probe.py）以降。
#   StudyMemo 11.5 節・11.8 節「集計指標は生成の健全性を予測しない」の根拠。
# 入力:  --out のフォルダ（temporal_qdq_experiment.py の prepare〜evaluate を先に流して、build_audit.json・
#        qparam_map.json・<対話>_teacher.json・<対話>_reference.npz・<対話>_oracle_<位置>.npz があること）。
#        hybrid は同じフォルダの baseline.onnx も使う。
# 出力:  --out の下に focused_shared.onnx・focused_later.onnx・focused_build_audit.json・
#        <対話>_<変種>.json・focused_teacher_summary.json・focused_free_generation.json（hybrid は hybrid_free_generation.json）・
#        run_focused_<段>.json。
# 動かし方（results/P12e_20260915_temporal_qdq.md の 7 節。ファイルを直接起動する。
#   下の import * は同じフォルダの temporal_qdq_experiment を探すので、python -m では動かない）:
#   & $py -B app/pipeline/focused_qdq_experiment.py --out $out --phase all --max-new 32
#   … --phase generate-hybrid で hybrid の自由生成だけ。
# 注意: run_focused_<段>.json にはこのファイルの sha256 を記録するので、コメントを足した今の版の hash は
#   当時の記録と一致しない（仕様）。
# ---------------------------------------------------------------------------
# 共通部品（read・save・digest・session・Runner・tables・metrics・summarize_rows・emit・ROOT・SOURCE・TARGETS など）を全部借りる
from temporal_qdq_experiment import *

def build_focused(out,threshold=32):
    """168 組だけ範囲を変えた 2 つのモデル（focused_shared.onnx・focused_later.onnx）を作り、監査の記録を残す。

    out: 作業フォルダ。threshold: 選ぶ基準（共通の scale ÷ 後続の scale の下限。既定 32）。
    ① 較正の記録から 168 組を選ぶ（観測点 add_26・add_53 が必ず入ることを確かめる）
    ② 相（shared・later）ごとに、元の QDQ のその組の scale・zero point だけを置き換える
    ③ 置き換えを許した初期化子以外が 1 バイトも変わっていないこと・ノードが元と同じことを hash で確かめてから保存する
    """
    audit=read(out/'build_audit.json')
    params=read(out/'qparam_map.json')
    assert digest(SOURCE)==params['source_sha256']
    later={c['scale_name']:c for c in audit['later']['changes']}
    selected=[c['scale_name'] for c in audit['shared']['changes']
              if c['scale']/later[c['scale_name']]['scale']>=threshold]
    assert all(t+'_scale' in selected for t in TARGETS)
    result={}
    for phase in ['shared','later']:
        m=onnx.load(str(SOURCE),load_external_data=False)
        init={t.name:t for t in m.graph.initializer}
        original={t.name:hashlib.sha256(t.SerializeToString()).hexdigest() for t in m.graph.initializer}
        node_hash=hashlib.sha256(b''.join(n.SerializeToString() for n in m.graph.node)).hexdigest()
        allowed=set()
        changes=[]
        for c in audit[phase]['changes']:
            if c['scale_name'] not in selected:
                continue
            for name,value in [(c['scale_name'],c['scale']),(c['zero_name'],c['zero'])]:
                before=numpy_helper.to_array(init[name])
                init[name].CopyFrom(numpy_helper.from_array(np.asarray(value,dtype=before.dtype).reshape(before.shape),name))
                allowed.add(name)
            changes.append(c)
        unexpected=[t.name for t in m.graph.initializer if t.name not in allowed and
                    hashlib.sha256(t.SerializeToString()).hexdigest()!=original[t.name]]
        assert not unexpected
        assert node_hash==audit['baseline']['node_sha256']
        # 観測点（add_26・add_53 と、その逆量子化の出力）を出力に足す。形 [1, 1, 1024]（Qwen3-0.6B の hidden 1024）
        for t in TARGETS:
            for name in [t,t+'_DequantizeLinear_Output']:
                m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,1,1024]))
        path=out/f'focused_{phase}.onnx'
        onnx.save(m,str(path))
        result[phase]={'changes':changes,'changed_groups':len(changes),'threshold':threshold,
            'selection':'shared/later scale ratio on calibration only','source_sha256':params['source_sha256'],
            'model_sha256':digest(path),'node_sha256':node_hash,'unexpected_initializer_changes':unexpected}
        save(out/'focused_build_audit.json',result)
        emit({'phase':'focused_build','variant':phase,'changed_groups':len(changes)})
        del m,init
        gc.collect()

def evaluate_focused(out):
    """2 つの変種を、P12d の日英の対話で teacher forcing して FP32 と比べる（自前の KV を積む）。

    位置ごとに観測点・logits・新しい KV の誤差（metrics）と、観測点が表現範囲の外に出た数を記録する。
    位置 0・1・40・最後では、FP32 と同じ KV を与えた 1 ステップ比較（oracle）も行う。
    出力: <対話>_<変種>.json と focused_teacher_summary.json（24 位置ごとに途中の記録も保存する）
    """
    audit=read(out/'focused_build_audit.json')
    cos,sin=tables()
    results={}
    for mode in ['focused_shared','focused_split']:
        first=session(out/'focused_shared.onnx')
        later=session(out/'focused_later.onnx') if mode=='focused_split' else first
        for name in ['chat_original','chat_english']:
            original=read(out/(name+'_teacher.json'))
            trace=read(ROOT/f'results/P12d_20260915_dialogue_activation_ranges/{name}.json')
            assert original['status']=='complete'
            assert original['input_ids']==[r['token_id'] for r in trace['rows']]
            with np.load(out/f'{name}_reference.npz') as f:
                reference={n:f[n] for n in f.files}
            r=Runner(first,cos,sin)
            rows=[]
            oracle=[]
            for pos,tid in enumerate(original['input_ids']):
                phase='shared' if pos==0 or mode=='focused_shared' else 'later'
                s=first if pos==0 else later
                got,_=r.feed(tid,s)
                expected={n:a[pos] for n,a in reference.items()}
                assert all(np.isfinite(a).all() for a in got.values())
                row={'position':pos,'token_id':tid,'metrics':metrics(got,expected),'saturation':{}}
                for t in TARGETS:
                    lo,hi=next(c['represented'] for c in audit[phase]['changes'] if t in c['sources'])
                    row['saturation'][t]={'before_q_outside':int(np.sum((got[t]<lo)|(got[t]>hi))),
                        'reference_outside':int(np.sum((expected[t]<lo)|(expected[t]>hi))),'range':[lo,hi]}
                rows.append(row)
                if pos in [0,1,40,len(original['input_ids'])-1]:
                    with np.load(out/f'{name}_oracle_{pos}.npz') as f:
                        inputs={n:f[n] for n in f.files}
                    same=dict(zip(r.names,s.run(None,inputs)))
                    oracle.append({'position':pos,'metrics':metrics(same,expected)})
                    del inputs,same
                if pos%24==0:
                    save(out/f'{name}_{mode}.json',{'rows':rows,'status':'incomplete'})
                    emit({'phase':'focused_evaluate','case':name,'variant':mode,'position':pos,
                          'add53':row['metrics']['add_53_qdq_rel_l2'],'logits':row['metrics']['logits_rel_l2']})
            record={'case':name,'variant':mode,'tokens':len(rows),'input_ids':original['input_ids'],
                    'rows':rows,'same_fp32_kv_oracle':oracle,'summary':summarize_rows(rows),'status':'complete'}
            save(out/f'{name}_{mode}.json',record)
            results.setdefault(name,{})[mode]=record['summary']
            save(out/'focused_teacher_summary.json',results)
            emit({'phase':'focused_evaluated','case':name,'variant':mode,'later':record['summary']['later']})
            del r,got,expected,reference
            gc.collect()
        del first,later,s
        gc.collect()

def generate_focused(out,max_new,modes=None):
    """元の日本語 3 ターンを、各変種で greedy に自由生成する（各回答 最大 max_new 語。終わりのトークンで止める）。

    modes: 変種の名前のリスト（既定は focused_shared と focused_split。["hybrid"] で hybrid だけ）。
    生成した語は次のターンの文脈にそのまま残る（自分の回答を履歴にする）
    """
    tok=tokenizer()
    cos,sin=tables()
    record=read(ROOT/'results/P12d_20260915_dialogue_activation_ranges/chat_original.json')
    chunks=[[r['token_id'] for r in record['rows'] if r['turn']==i and r['phase']=='prompt'] for i in range(3)]
    results={}
    for mode in (modes or ['focused_shared','focused_split']):
        first=session(out/('baseline.onnx' if mode=='hybrid' else 'focused_shared.onnx'))
        later=session(out/'focused_later.onnx') if mode in ['focused_split','hybrid'] else first
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
            save(out/('hybrid_free_generation.json' if modes==['hybrid'] else 'focused_free_generation.json'),results)
            emit({'phase':'focused_generate','variant':mode,'turn':turn,'text':turns[-1]['text'],'stop':stop})
        del r,first,later
        gc.collect()

# コマンドの入口: 実行の記録（開始時刻・このファイルと共通部品の sha256・段・しきい値）を残してから、--phase の段を流す
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['all','build','evaluate','generate','generate-hybrid'],default='all')
    ap.add_argument('--max-new',type=int,default=64)
    args=ap.parse_args()
    save(args.out/f'run_focused_{args.phase}.json',{'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'script_sha256':digest(Path(__file__)),'common_script_sha256':digest(Path(__file__).with_name('temporal_qdq_experiment.py')),
         'phase':args.phase,'threshold':32,'max_new':args.max_new,'providers':['CPUExecutionProvider']})
    if args.phase in ['all','build']:
        build_focused(args.out)
    if args.phase in ['all','evaluate']:
        evaluate_focused(args.out)
    if args.phase in ['all','generate']:
        generate_focused(args.out,args.max_new)
    if args.phase=='generate-hybrid':
        generate_focused(args.out,args.max_new,modes=['hybrid'])
