"""P12e: recalibrate existing activation Q/DQ ranges, with frozen weights and KV interfaces.

Independent calibration text; same QDQ topology/dtypes; shared vs first/later
range fit; teacher-forced and free-running Host comparisons. No device deploy.
"""
# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# 何のための道具か:
#   P12e「先頭・後続の量子化範囲分離」の実験の本体。Qwen3-0.6B（28層・KV256）の
#   既存 QDQ（qwen3_dec28c256_final_qdq.onnx）を複製し、活性値（中間の値）の
#   QuantizeLinear/DequantizeLinear の scale・zero point **だけ**を較正し直した変種を作って、
#   ホスト（ONNX Runtime の CPU）で FP32 と比べる。重み・ノード・KV の入出力は 1 バイトも変えない。
#   ※上の英語の docstring は argparse の --help にそのまま出るので、説明はこのコメントに書いた。
# なぜ作ったか:
#   P12c/P12d で「位置0（先頭トークン）だけが 8142 級の巨大な値を出し、量子化の 1 段（step）が
#   0.113 まで粗くなっている」と分かった（StudyMemo 11.4 節）。そこで「先頭用の範囲」と
#   「後続用の範囲」を分けて量子化すれば後続の誤差が減るのでは、という仮説を確かめた。
# 結果（results/P12e_20260915_temporal_qdq.md）:
#   全活性 2,226 組を較正し直す shared/split は大きく悪化（日本語の予測一致率 76.4% → 0〜2.8%）。
#   範囲を絞った focused_split（focused_qdq_experiment.py）は一致率 81.1% と数字は改善したが、
#   自由生成で「音楽をかけて」にエアコンの説明を返して指示を無視した → 採用せず、実機へは進めていない。
#   StudyMemo 11.5 節「直そうとして起きたこと」・11.8 節 (3)(4)「teacher forcing と自由生成は別物」
#   「集計指標は生成の健全性を予測しない」の根拠の 1 つ。
# 入力:  work/ops/qwen3_dec28c256/ の final_qdq.onnx（変える元）と prepped.onnx（FP32 の基準）、
#        work/rope_q3/ の RoPE 表、P12d の対話トークン列（results/P12d_…/chat_*.json・.npz）、
#        Qwen3-0.6B のトークナイザ（HF のキャッシュ。オフライン）。
# 出力:  --out の下に qparam_map.json・calibration_ranges.json・build_audit.json・
#        <変種>.onnx・*_teacher.json・teacher_summary.json・free_generation.json・run_<段>.json。
#        記録の保存先は results/P12e_20260915_temporal_qdq/。
# 動かし方（段ごとに別プロセスで流す。results/P12e_20260915_temporal_qdq.md の 7 節）:
#   & $py -B app/pipeline/temporal_qdq_experiment.py --out work/p12e_reproduction --phase prepare
#   … --phase calibrate → --phase build --margin 1.5 → --phase evaluate → --phase generate --max-new 32
#   続きは focused_qdq_experiment.py → temporal_htp_probe.py → summarize_temporal_qdq.py。
# 注意: パス（ROOT）は Windows の絶対パスで固定。run_<段>.json にはこのファイル自身の sha256 を
#   記録するので、コメントを足した今の版の hash は当時の記録（results/）と一致しない（推定ではなく仕様）。
# ---------------------------------------------------------------------------
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

# ① 標準出力を UTF-8 にし、HF をオフラインに固定、BLAS のスレッドを 1 本にする
#    （ORT 自身のスレッド数を session() で 4 本に決めるので、二重に並列させない）。
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

# ② 変える元の QDQ（SOURCE）と FP32 の基準（FP32 = 前処理後の prepped.onnx）。
#    TARGETS は観測点: add_26 = 層3 に入る残差、add_53 = 層6 に入る残差（層番号は 0 始まり）。
ROOT=Path('D:/AI/source/n_chap5/EdgeSLM_Android')
OPS=ROOT/'work/ops/qwen3_dec28c256'
SOURCE=OPS/'qwen3_dec28c256_final_qdq.onnx'
FP32=OPS/'qwen3_dec28c256_prepped.onnx'
TARGETS=['add_26','add_53']

# 進み具合を 1 行の JSON で出す（長い段の途中経過を後から追えるように）。
def emit(obj):
    print(json.dumps(obj,ensure_ascii=False),flush=True)

# JSON を UTF-8・字下げ付きで書く。
def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

# JSON を読む。
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

# ファイルの sha256 を 8MB ずつ読んで計算する（2GB 近いモデルでもメモリに載せない）。
# 「元のモデルが変わっていないか」「作った変種がどれか」を記録で照合するために使う。
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

# CPU の ORT セッションを作る。スレッド数を固定して、条件の間で速さ・丸めの順が揃うようにする。
def session(path):
    so=ort.SessionOptions()
    so.intra_op_num_threads=4
    so.inter_op_num_threads=1
    so.log_severity_level=3
    return ort.InferenceSession(str(path),so,providers=['CPUExecutionProvider'])

def qparam_map(out):
    """既存 QDQ の Q/DQ の組（scale と zero point の初期化子の組）を全部数え、どれを変えてよいかを決める。

    引数 out: 出力先フォルダ（Path）。戻り値: qparam_map.json と同じ dict
    （groups = 組ごとに scale/zero の名前・元の値・dtype・元になった FP32 テンソル名 sources・
    変えてはいけない理由 protected_reasons）。
    なぜ: 範囲を変えるのは「活性値の組」だけにしたい。KV の入出力（past_/present_）や
    cos/sin/mask、重み、他の組と初期化子を共有している組まで変えると、比べる変数が 1 つでなくなる
    （共有初期化子の巻き添えは app/pipeline/README.md の落とし穴の表にもある）。
    記録上、活性 2,226 組が変えてよい組、116 組が固定（results/P12e_20260915_temporal_qdq.md 1 節）。
    """
    pre=onnx.load(str(FP32),load_external_data=False)
    # ① FP32 側で名前と型が分かっているテンソル（入力・中間・出力）と、FP32 の初期化子（重み）の名前。
    known={v.name:v.type.tensor_type.elem_type for v in [*pre.graph.input,*pre.graph.value_info,*pre.graph.output]}
    pinits={t.name for t in pre.graph.initializer}
    m=onnx.load(str(SOURCE),load_external_data=False)
    init={t.name:t for t in m.graph.initializer}
    prod={v:n for n in m.graph.node for v in n.output}
    # ② Q/DQ/Identity を遡って、その値が FP32 グラフのどのテンソルから来たかを返す（分からなければ None）。
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
    # ③ present_*（KV の出力）から遡れる Q/DQ の組は「キャッシュの出口」なので固定する。
    #    形を変えるだけの op（Slice・Reshape など）は値を変えないので通り抜けて遡る。
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
    # ④ QuantizeLinear ごとに (scale 名, zero 名) の組でまとめ、変えてはいけない理由を集める。
    #    scale/zero が初期化子でない（動的）なら想定外なので止める。
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
    # ⑤ 1 つの初期化子が 2 つ以上の組に使われていたら、片方だけ変えられないので固定する。
    owners=defaultdict(set)
    for pair in groups:
        for p in pair:
            owners[p].add(pair)
    for pair,g in groups.items():
        if any(len(owners[p])>1 for p in pair):
            g['protected_reasons'].append('overlapping_parameter_pairs')
        g['sources']=sorted({s for s in g['sources'] if s is not None})
        g['protected_reasons']=sorted(set(g['protected_reasons']))
    # ⑥ 元モデルの sha256 も記録し、build の段で「元が変わっていない」ことを確かめる。
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
    """較正用の FP32 モデル（fp32_calibration.onnx）と、評価の基準モデル（fp32_reference.onnx）を作る。

    引数 out: 出力先、params: qparam_map() の戻り値。戻り値なし。
    較正用は、変えてよい組の元テンソルごとに ReduceMin/ReduceMax の出力（p12e_<番号>_min/max、スカラー）を
    足したもの。1 トークンずつ流すたびにその位置の最小・最大が取れる。
    なぜ: ORT の較正器を使わず自分で測るのは、「位置0 と位置1以降」を分けて最小・最大を取りたいから。
    """
    m=onnx.load(str(FP32),load_external_data=False)
    # ① prepped.onnx は重みを外部データ（.data）に持つので、隣にコピーして参照先の名前を書き換える。
    #    コピーの大きさを元と照合する（壊れたコピーで測らないため）。
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key=='location':
                original=FP32.parent/e.value
                dest=out/('fp32_'+original.name)
                if not dest.exists():
                    shutil.copyfile(original,dest)
                assert dest.stat().st_size==original.stat().st_size
                e.value=dest.name
    # ② 観測点（add_26・add_53、形 [1,1,1024]）を出力に足した版を「基準」として保存。
    existing={v.name for v in m.graph.output}
    for t in TARGETS:
        if t not in existing:
            m.graph.output.append(helper.make_tensor_value_info(t,TensorProto.FLOAT,[1,1,1024]))
    onnx.save(m,str(out/'fp32_reference.onnx'))
    # ③ 変えてよい組の元テンソルすべてに min/max を足した版を「較正用」として保存。
    sources=sorted({s for g in params['groups'] if not g['protected_reasons'] for s in g['sources']})
    for i,s in enumerate(sources):
        for operation,suffix in [('ReduceMin','min'),('ReduceMax','max')]:
            name=f'p12e_{i}_{suffix}'
            m.graph.node.append(helper.make_node(operation,[s],[name],name=name,keepdims=0))
            m.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[]))
    onnx.save(m,str(out/'fp32_calibration.onnx'))
    save(out/'observed_sources.json',sources)
    emit({'phase':'probe','activation_sources':len(sources)})

# 1 トークンずつ decode する小さな実行器。端末と同じ「固定長 256 の KV キャッシュ（左シフト）＋マスク −60」
# の形で入力を作り、出力の present_* を次の past_* に回す（StudyMemo 10.1・10.2 節）。
class Runner:
    def __init__(self,s,cos,sin):
        """s: ORT セッション、cos/sin: RoPE 表 [2048, 128]（Qwen3 は head_dim 128）。"""
        self.s,self.cos,self.sin=s,cos,sin
        self.names=[v.name for v in s.get_outputs()]
        self.input_names=[v.name for v in s.get_inputs()]
        self.reset()
    def reset(self):
        """位置を 0 に戻し、KV キャッシュを全部 0（空）にする。形はモデルの入力の形そのまま。"""
        self.position=0
        self.cache={v.name:np.zeros(v.shape,np.float32) for v in self.s.get_inputs() if v.name.startswith('past_')}
    def feed(self,tid,s=None):
        """トークン tid を 1 つ流す。s を渡すとそのセッションで走らせる（キャッシュは共通）。

        戻り値: (出力名→配列の dict, 入力の dict)。入力は「同じ KV を与える 1 ステップ比較」に使う。
        """
        s=s or self.s
        pos=self.position
        # マスク [1,1,1,256]: 右詰めで有効な所（直近 pos+1 個、最大 256）だけ 0、ほかは −60（C-9 の値）。
        mask=np.full((1,1,1,256),-60,np.float32)
        mask[...,256-min(pos+1,256):]=0
        inputs={'input_ids':np.array([[tid]],np.int64),'cos':self.cos[pos].reshape(1,1,1,128),
                'sin':self.sin[pos].reshape(1,1,1,128),'attn_mask':mask,**self.cache}
        result=dict(zip(self.names,s.run(None,inputs)))
        # 出力 present_k_i/present_v_i（左シフト済みの 256 スロット）を、そのまま次の past_ にする。
        self.cache={'past_'+n[len('present_'):]:a for n,a in result.items() if n.startswith('present_')}
        self.position+=1
        return result,inputs

# Qwen3-0.6B のトークナイザを HF のキャッシュ（固定のスナップショット）から読む。ネットには出ない。
def tokenizer():
    from transformers import AutoTokenizer
    snap=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
    return AutoTokenizer.from_pretrained(str(snap),local_files_only=True)

# RoPE の cos/sin 表（work/rope_q3/、make_rope_table.py で作ったもの）を [2048, 128] で読む。
def tables():
    return tuple(np.fromfile(ROOT/f'work/rope_q3/{n}_table_l2048.bin',np.float32).reshape(2048,128) for n in ['cos','sin'])

def calibrate(out):
    """較正用 FP32 モデルにトークンを流し、変えてよい組ごとに「位置0」と「位置1以降」の min/max を集める。

    出力: calibration_ranges.json（ranges[phase][テンソル名] = [min, max]、phase は first/later）。
    なぜ: 評価用の対話（エアコン・音楽）とは別の文で較正し、評価データに合わせ込まないため。
    記録では 8 本・297 トークン（先頭 8・後続 289）。チャットの開始トークンは 3 本とも同じ ID
    （results/P12e_20260915_temporal_qdq.md 2 節）。
    """
    sources=read(out/'observed_sources.json')
    tok=tokenizer()
    # ① 較正の文: 既存の較正で使った短文 5 本（各 24 歩）＋ 評価とは別の車載チャット 3 本（最大 96 歩）。
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
    # ② 位置0（first）と位置1以降（later）を分けて最小・最大を更新していく。
    ranges={phase:{s:[float('inf'),-float('inf')] for s in sources} for phase in ['first','later']}
    records=[]
    for seq in sequences:
        r.reset()
        ids=seq['ids'].copy()
        processed=[]
        for pos in range(seq['max_steps']):
            # 入力が尽きたら FP32 の予測（greedy）を足して続ける。終わりのトークンで止める。
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
        # 1 本終わるごとに保存（途中で落ちても、それまでの範囲が残る）。
        save(out/'calibration_ranges.json',{'ranges':ranges,'sequences':records})
        emit({'phase':'calibration','sequence':seq['name'],'steps':len(processed),
              'first_add53':ranges['first'].get('add_53'),'later_add53':ranges['later'].get('add_53')})
    del r
    gc.collect()

def fit(lo,hi,qmax,margin):
    """観測した [lo, hi] から非対称量子化の (scale, zero point) を作る。

    引数: lo/hi = 観測した最小・最大（float）、qmax = 整数の最大（uint16 なら 65535）、
    margin = 両端を何倍に広げるか（既定 1.5）。戻り値: (np.float32 の scale, int の zero point)。
    0 を必ず範囲に含める（0 をちょうど表せるようにする、量子化の決まり。StudyMemo 2 章）。
    zero point を整数に丸めた後、両端が表せるように scale を少し広げ直し、最後に float32 で
    1 つ上の数に寄せて、端の値がはみ出さないようにしている。
    """
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
    """変種のモデルを 4 つ作る: baseline（変更なし）・shared（先頭と後続をまとめた範囲）・
    first（先頭の範囲）・later（後続の範囲）。split は「位置0 を first、以降を later で走らせる」組み合わせ。

    出力: <変種>.onnx と build_audit.json（変えた組・新旧の範囲・sha256）。
    なぜ監査するか: 変えてよい scale/zero 以外の初期化子とノードが 1 バイトも変わっていないことを
    sha256 で確かめ、「変えた変数は範囲だけ」と言えるようにする。
    """
    params=read(out/'qparam_map.json')
    # ① prepare の時と同じ元モデルか・較正が 8 本そろっているか・値が有限かを先に確かめる。
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
                # ② shared は first と later の範囲の和、それ以外はその phase の範囲。
                #    1 つの組に元テンソルが複数あれば、その全部を含む範囲にする。
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
        # ③ ノードの数と並び（sha256）が元と同じ、変えた初期化子は許した組だけ、を確かめる。
        assert len(m.graph.node)==params['nodes']
        assert hashlib.sha256(b''.join(n.SerializeToString() for n in m.graph.node)).hexdigest()==original_nodes
        unexpected=[t.name for t in m.graph.initializer
                    if t.name not in allowed and hashlib.sha256(t.SerializeToString()).hexdigest()!=original[t.name]]
        assert not unexpected,unexpected
        # ④ 観測のため、add_26・add_53 の量子化前と逆量子化後（_DequantizeLinear_Output）を出力に足す。
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

# 相対 L2 誤差 ||a−b|| / ||b||。分母は基準（FP32）側。整数コードではなく実数で測る（StudyMemo 4.5・11.8 (1)）。
def rel(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def metrics(got,ref):
    """1 トークン分の量子化モデルの出力 got を、FP32 の出力 ref と比べた指標の dict を返す。

    logits の相対 L2・KL（FP32→量子化）・1 位一致・FP32 の 1 位が何位か、
    観測点の量子化前（raw）と逆量子化後（qdq）の相対 L2、
    新しく書いた KV スロット（各層の最後の 1 つ。[1, KV, 256, D] の [:,:,-1,:]）の層ごとの誤差の中央値・最大・層6。
    なぜ最後の 1 スロットか: 空のスロットを含めると誤差が薄まるため（results/P12e_….md 3 節）。
    """
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

# トークンごとの指標を平均する。all（全位置）・first（位置0）・later（位置1以降）に分ける。
def summarize_rows(rows):
    result={}
    for name,chosen in [('all',rows),('first',rows[:1]),('later',rows[1:])]:
        if not chosen:
            continue
        result[name]={k:float(np.mean([r['metrics'][k] for r in chosen])) for k in chosen[0]['metrics']}
        result[name]['tokens']=len(chosen)
    return result

def evaluate_simultaneous_unused(out):
    """（使っていない）全変種のセッションを同時に開いて評価する最初の版。

    メモリ負荷が大きく途中で止めたため、下の evaluate()（1 変種ずつ）に置き換えた。
    途中の値は最終集計に混ぜていない（results/P12e_20260915_temporal_qdq.md 7 節）。記録として残している。
    """
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

# 出力のうち present_*（KV、[1, KV, 256, D]）は最後の 1 スロットだけに縮めて持つ（基準の保存を軽くする）。
def compact(result):
    return {n:(a[:,:,-1:,:].copy() if n.startswith('present_') else a.copy()) for n,a in result.items()}

def evaluate(out):
    """Run one variant at a time to avoid swapping several large ORT sessions."""
    # 【説明】固定したトークン列（P12d の日本語 107・英語 109 トークン）を各変種に同じ順で与え、
    #   FP32 と 1 トークンずつ比べる（teacher forcing。StudyMemo 8.2 節）。各変種は自分の KV を使うので、
    #   誤差の積み重なりも含まれる。出力: <対話>_teacher.json・teacher_summary.json。
    # ① まず FP32 の基準を 1 回だけ流して <対話>_reference.npz に保存する（以降は読むだけ）。
    #    位置 0・1・40・最後では入力（FP32 の KV 込み）も保存し、後で「同じ KV を与える 1 ステップ比較」に使う。
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
    # ② 変種を 1 つずつ開いて評価する（同時に開くとメモリが足りなかった）。
    #    split は位置0 を first.onnx、以降を later.onnx で走らせる。KV は 2 つのモデルで受け渡す。
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
                # 観測点で、量子化の範囲の外に出た要素の数（自分の値・FP32 の値の両方）を数える。
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
                        # 対照: 観測出力を足した baseline と、足していない元の QDQ が同じ値を出すか
                        #       （観測が計算を変えていないか。記録では 8 地点で完全一致）。
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
    """元の日本語 3 ターン（エアコン 22 度 → 音楽 → さっき何度？）を、各変種が選んだトークンで自由生成する。

    greedy・各回答最大 max_new トークン。出力: free_generation.json。
    なぜ: 固定トークン列の一致率が上がっても、自由生成で指示に従うとは限らない。
    合否はこの自由生成で決める（StudyMemo 11.8 (4)、結果は results/P12e_….md 4 節）。
    """
    tok=tokenizer()
    cos,sin=tables()
    record=read(ROOT/'results/P12d_20260915_dialogue_activation_ranges/chat_original.json')
    # P12d の記録から、3 ターンそれぞれの「入力（prompt）」部分のトークンだけを取り出す（回答はモデルに作らせる）。
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
    """引数を読み、--phase の段だけを走らせる。段の始めに実行環境と自分の sha256 を run_<段>.json に残す。"""
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
