"""P12d: trace FP32 residual ranges under the project's fixed-window decoder.

No quantized model modification or device inference. Narrow-range estimates
are local rounding diagnostics, not measured end-to-end accuracy.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

LAYERS=(3,6)
STOP_IDS={151645,151643}
EMPTY_THINK='<think>\n\n</think>\n\n'

def emit(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)

def rel(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Observer:
    def __init__(self,hf,module,window,out):
        self.hf,self.window,self.out=hf,window,out
        self.step=module.QwenDecodeStep(hf,window).eval()
        self.capture=True
        self.current={}
        self.hooks=[]
        for layer in LAYERS:
            def hook(_module,inputs,layer=layer):
                if self.capture:
                    self.current[layer]=inputs[0][0,-1].detach().numpy().copy()
            self.hooks.append(hf.model.layers[layer].input_layernorm.register_forward_pre_hook(hook))
        self.cos,self.sin=module.rope_cos_sin(hf,list(range(2048)))
        self.reset()

    def reset(self):
        self.caches=[torch.zeros(1,self.step.nkv,self.window,self.step.hd)
                     for _ in range(2*len(self.step.layers))]
        self.ids=[]
        self.rows=[]
        self.values={n:[] for n in LAYERS}
        self.logits=None
        self.checks=[]

    def feed(self,tid,phase,turn=0,check=False):
        pos=len(self.ids)
        mask=torch.full((1,1,1,self.window),-60.0)
        mask[...,self.window-min(pos+1,self.window):]=0
        self.current={}
        output=self.step(torch.tensor([[tid]]),
                    torch.from_numpy(self.cos[:,:,pos:pos+1,:]),
                    torch.from_numpy(self.sin[:,:,pos:pos+1,:]),mask,*self.caches)
        self.logits=output[0][0,-1].numpy().copy()
        self.caches=list(output[1:])
        self.ids.append(tid)
        row={'position':pos,'token_id':tid,'phase':phase,'turn':turn,
             'valid_tokens':min(pos+1,self.window),'evicted':max(pos+1-self.window,0)}
        for layer in LAYERS:
            v=self.current[layer]
            assert np.isfinite(v).all()
            self.values[layer].append(v)
            row['layer_'+str(layer)]={'min':float(v.min()),'max':float(v.max()),
                'max_abs_channel':int(np.argmax(np.abs(v))),
                'channel_35':float(v[35]),'abs_gt_1000':int(np.sum(np.abs(v)>1000))}
        self.rows.append(row)
        if check:
            # Before eviction, independently compare with official HF full-prefix forward.
            assert len(self.ids)<=self.window
            self.capture=False
            try:
                ref=self.hf(input_ids=torch.tensor([self.ids]),use_cache=False,output_hidden_states=True)
                record={'position':pos,'logits_rel_l2':rel(ref.logits[0,-1].numpy(),self.logits),
                        'top1_same':int(ref.logits[0,-1].argmax())==int(self.logits.argmax())}
                for layer in LAYERS:
                    record['layer_'+str(layer)+'_rel_l2']=rel(
                        ref.hidden_states[layer][0,-1].numpy(),self.values[layer][-1])
                self.checks.append(record)
                assert record['logits_rel_l2']<1e-4,record
                assert all(record['layer_'+str(n)+'_rel_l2']<1e-4 for n in LAYERS),record
            finally:
                self.capture=True
        return self.logits

    def save(self,name,meta):
        np.savez_compressed(self.out/(name+'.npz'),
                           **{'layer_'+str(k):np.stack(v) for k,v in self.values.items()})
        record={'name':name,'tokens':len(self.rows),'window':self.window,
                'meta':meta,'checks':self.checks,'rows':self.rows}
        (self.out/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        emit({'finished':name,'tokens':len(self.rows),
              'outliers':{str(k):[r['position'] for r in self.rows if r['layer_'+str(k)]['abs_gt_1000']]
                          for k in LAYERS},'kind':meta['kind']})
        return record

def make_dialogue(tok,system,users):
    chunks=[]
    for i,user in enumerate(users):
        text=('<|im_start|>system\n'+system+'<|im_end|>\n' if i==0 else '<|im_end|>\n')
        text+='<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'+EMPTY_THINK
        chunks.append(tok.encode(text,add_special_tokens=False))
    return chunks

def run_dialogue(obs,tok,name,chunks,max_new):
    obs.reset()
    turns=[]
    for turn,ids in enumerate(chunks):
        start=len(obs.ids)
        for tid in ids:
            obs.feed(tid,'prompt',turn,check=len(obs.ids) in [0,40,100,255])
            if len(obs.ids)%128==0:
                emit({'running':name,'processed_tokens':len(obs.ids),'turn':turn})
        generated=[]
        stopped=None
        for _ in range(max_new):
            nxt=int(obs.logits.argmax())
            if nxt in STOP_IDS:
                stopped=nxt
                break
            generated.append(nxt)
            obs.feed(nxt,'generated',turn,check=len(obs.ids) in [100,255])
        turns.append({'turn':turn,'prompt_start':start,'new_prompt_tokens':len(ids),
                      'generated_ids':generated,'text':tok.decode(generated),
                      'stop_id_not_consumed':stopped,'max_new':max_new})
    return obs.save(name,{'kind':'greedy_dialogue','turns':turns,
        'note':'Each emitted non-stop token is consumed; stop token is consumed once as next turn prefix.'})

def build_stress_texts():
    ja=[]
    en=[]
    for i in range(1,51):
        ja.append(f'第{i}区間の記録。車内の温度は{18+i%8}度で、外気温は{12+i%15}度です。'
                  f'目的地まで残り{120-i}キロメートルです。運転者は静かな経路を希望し、'
                  '助手席の同乗者は次の休憩地点を確認しました。設定を変える前に現在の状態を説明します。')
        en.append(f'Section {i}: Read the current sensor value and validate the request.\n'
                  f'def normalize_{i}(value, lower=18, upper=26):\n'
                  '    if value is None:\n        return {"status": "missing"}\n'
                  '    return {"temperature": max(lower, min(upper, value))}\n'
                  'The function retains the original bounds and records each change for later review.\n')
    return [('long_ja','\n'.join(ja)),('long_code','\n'.join(en))]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path('D:/AI/source/n_chap5/EdgeSLM_Android'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--window',type=int,choices=[256],default=256)
    ap.add_argument('--long-tokens',type=int,default=640)
    ap.add_argument('--max-new',type=int,default=64)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root=args.project_root
    source=root/'app/pipeline/export_qwen_decode.py'
    spec=importlib.util.spec_from_file_location('existing_decode',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
    hf=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,
                            dtype=torch.float32,attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    observer=Observer(hf,module,args.window,args.out)
    manifest={'source':str(source),'source_sha256':sha(source),'snapshot':str(snapshot),
        'trace_script_sha256':sha(__file__),
        'torch':torch.__version__,'threads':4,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'records':[]}
    def checkpoint():
        (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    started=time.time()
    with torch.inference_mode():
        # Freeze narrow ranges using only the old 30 calibration token IDs, not evaluation traces.
        with np.load(root/'work/qwen3_dec28c256/calib_q3d28.npz') as z:
            ids=z['input_ids'].ravel().tolist()
        calibration={k:[] for k in LAYERS}
        for group in range(5):
            observer.reset()
            for tid in ids[group*6:(group+1)*6]:
                observer.feed(tid,'calibration',group,check=True)
            manifest['records'].append(observer.save('calib_'+str(group),{'kind':'calibration'}))
            checkpoint()
            for k in LAYERS:
                calibration[k].extend(observer.values[k][1:])
        ranges={str(k):{'min':float(np.min(calibration[k])),'max':float(np.max(calibration[k])),
                       'samples':25} for k in LAYERS}
        manifest['frozen_later_ranges']=ranges
        checkpoint()
        emit({'frozen_later_ranges':ranges})
        saved_turns=json.loads((root/'work/q3chat.json').read_text())['turns']
        saved_users=[tok.decode(t).split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
                     for t in saved_turns]
        jp=make_dialogue(tok,'あなたは車載アシスタントです。短く答えてください。',saved_users)
        assert jp[0]==saved_turns[0],'Original 41-token prompt mismatch'
        assert [len(x) for x in jp]==[41,18,22],[len(x) for x in jp]
        manifest['records'].append(run_dialogue(observer,tok,'chat_original',jp,args.max_new))
        checkpoint()
        en=make_dialogue(tok,'You are an in-car assistant. Keep your answers short.',
                         ['Set the temperature to 22 degrees.',
                          'Please turn down the fan without changing the temperature.',
                          'What temperature did I ask for?'])
        manifest['records'].append(run_dialogue(observer,tok,'chat_english',en,args.max_new))
        checkpoint()
        for name,text in build_stress_texts():
            observer.reset()
            tids=tok.encode(text,add_special_tokens=False)[:args.long_tokens]
            assert len(tids)==args.long_tokens
            for pos,tid in enumerate(tids):
                observer.feed(tid,'teacher_forced_long',check=pos in [0,40,100,255])
                if pos%128==0:
                    emit({'running':name,'position':pos,'layer_3_max':observer.rows[-1]['layer_3']['max'],
                          'layer_6_max':observer.rows[-1]['layer_6']['max']})
            manifest['records'].append(observer.save(name,{'kind':'synthetic_long_text','text':tok.decode(tids),
                                                         'all_input_ids':tids}))
            checkpoint()
        # One long conversational prompt exercises special tokens and eviction before generation.
        long_user='次の記録を読んで、最後の区間の設定温度を答えてください。\n'+build_stress_texts()[0][1]
        long_ids=make_dialogue(tok,'あなたは車載アシスタントです。短く答えてください。',[long_user])[0]
        # Preserve assistant suffix after trimming the user body to a bounded test size.
        suffix=tok.encode('<|im_end|>\n<|im_start|>assistant\n'+EMPTY_THINK,add_special_tokens=False)
        long_ids=long_ids[:550]+suffix
        manifest['records'].append(run_dialogue(observer,tok,'chat_long',[long_ids],args.max_new))
        checkpoint()
    manifest['elapsed_seconds']=time.time()-started
    checkpoint()
    emit({'complete':True,'elapsed_seconds':manifest['elapsed_seconds'],
          'total_tokens':sum(r['tokens'] for r in manifest['records'])})

if __name__=='__main__':
    main()
