"""P12d: trace FP32 residual ranges under the project's fixed-window decoder.

No quantized model modification or device inference. Narrow-range estimates
are local rounding diagnostics, not measured end-to-end accuracy.
"""
# ---------------------------------------------------------------------------
# 【日本語の説明（学習用、2026-10-18 追記）】
# 何のための道具か:
#   P12d「対話・生成・KV256 入替後の残差の分布」の観測。量子化モデルも実機も使わず、FP32（PyTorch・CPU）で
#   端末と同じ固定長 256 の KV キャッシュ（左シフト＋マスク −60）の decoder を動かし、層3・層6 に入る残差
#   （ONNX の add_26・add_53 に当たる）の値をトークンごとに記録する。
#   ※上の英語の docstring は argparse の --help にそのまま出るので、説明はこのコメントに書いた。
# なぜ: P12c は較正の 5 文 × 6 歩しか見ていなかった。長い対話・生成中・キャッシュから古いトークンを
#   捨てた後にも巨大値（1000 超）が出るかを確かめ、「先頭用と後続用で量子化範囲を分ける」案の根拠にする。
# 結果（results/P12d_20260915_dialogue_activation_ranges.md）: 評価 2,119 トークンで、絶対値 1000 超は
#   各入力列の先頭（位置0・チャンネル35）だけ。チャット開始トークン <|im_start|> の最大は約 8,140 で、
#   既存の量子化の上限（約 7,257）を超えていた。StudyMemo 11.4 節の表（9 通りのプロンプト）の続き。
# 入力:  app/pipeline/export_qwen_decode.py（QwenDecodeStep をそのまま読み込んで使う）、
#        Qwen3-0.6B の公式スナップショット（HF のキャッシュ、オフライン）、
#        work/qwen3_dec28c256/calib_q3d28.npz（P12c と同じ較正のトークン）、work/q3chat.json（元の対話の入力）。
# 出力:  --out の下にケースごとの <名前>.json（トークンごとの値域・フェーズ）と <名前>.npz（[トークン数, 1024] × 2 層）、
#        manifest.json。保存先は results/P12d_20260915_dialogue_activation_ranges/。
# 動かし方: & $py -B -u app/pipeline/trace_dialogue_ranges.py --out <出力先>
#   続けて summarize_dialogue_ranges.py --out <同じ出力先> で集計する。
# ---------------------------------------------------------------------------
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

# 観測する層（0 始まり）。層3 の入口 = add_26、層6 の入口 = add_53。
# STOP_IDS は <|im_end|>（151645）と <|endoftext|>（151643）。EMPTY_THINK は Qwen3 の思考を空にする決まり文句。
LAYERS=(3,6)
STOP_IDS={151645,151643}
EMPTY_THINK='<think>\n\n</think>\n\n'

# 途中経過を 1 行の JSON で出す。
def emit(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)

# 相対 L2: ||a−b|| / ||a||。分母は第 1 引数（基準＝公式 HF 側）。
def rel(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))

# ファイルの sha256（使ったソース・このスクリプトを記録に残すため）。
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# FP32 の固定長 decoder を 1 トークンずつ動かし、層3・層6 に入る残差を記録する観測器。
class Observer:
    def __init__(self,hf,module,window,out):
        """hf: 公式 HF のモデル、module: export_qwen_decode.py を読み込んだもの、window: キャッシュ長（256）、out: 出力先。

        層の input_layernorm の直前に forward_pre_hook を付けて、その入力（＝層に入る残差の最後の位置、[1024]）を写す。
        値は変えない。capture を False にすると写さない（公式 HF での照合の時に上書きしないため）。
        """
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
        """KV キャッシュ（層ごとに K と V、各 [1, KV ヘッド数, window, head_dim] のゼロ）と記録を空にする。"""
        self.caches=[torch.zeros(1,self.step.nkv,self.window,self.step.hd)
                     for _ in range(2*len(self.step.layers))]
        self.ids=[]
        self.rows=[]
        self.values={n:[] for n in LAYERS}
        self.logits=None
        self.checks=[]

    def feed(self,tid,phase,turn=0,check=False):
        """トークン tid を 1 つ流し、残差の値域などを 1 行記録する。戻り値: その位置の logits（[語彙数]）。

        phase は 'prompt'・'generated'・'calibration'・'teacher_forced_long' のどれか（集計で分けるため）。
        check=True なら、キャッシュから捨てる前（256 以下）に限り、公式 HF に全 prefix を与えて logits と残差を照合する
        （相対 L2 1e-4 未満でなければ止める。記録では最大 0.0013567%）。
        """
        pos=len(self.ids)
        # ① マスク: 右詰めで有効な所（直近 pos+1 個、最大 window）だけ 0、他は −60。端末と同じ形。
        #    RoPE の位置 pos はキャッシュ長でリセットしない（cos/sin は 2048 位置ぶん用意してある）。
        mask=torch.full((1,1,1,self.window),-60.0)
        mask[...,self.window-min(pos+1,self.window):]=0
        self.current={}
        output=self.step(torch.tensor([[tid]]),
                    torch.from_numpy(self.cos[:,:,pos:pos+1,:]),
                    torch.from_numpy(self.sin[:,:,pos:pos+1,:]),mask,*self.caches)
        self.logits=output[0][0,-1].numpy().copy()
        self.caches=list(output[1:])
        self.ids.append(tid)
        # ② 記録: evicted = キャッシュから押し出された古いトークンの数。層ごとに最小・最大・最大のチャンネル・
        #    チャンネル35 の値・絶対値 1000 超の数。
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
        """このケースの値を <name>.npz（層ごとに [トークン数, 1024]）、記録を <name>.json に書き、記録の dict を返す。"""
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
    """チャットの入力をターンごとのトークン列に組み立てる。

    1 ターン目は system から、2 ターン目以降は前の回答を閉じる <|im_end|> から始める。
    各ターンの最後に assistant の見出しと空の思考ブロック（EMPTY_THINK）を付ける。戻り値: [ターンごとの ID のリスト]。
    """
    chunks=[]
    for i,user in enumerate(users):
        text=('<|im_start|>system\n'+system+'<|im_end|>\n' if i==0 else '<|im_end|>\n')
        text+='<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'+EMPTY_THINK
        chunks.append(tok.encode(text,add_special_tokens=False))
    return chunks

def run_dialogue(obs,tok,name,chunks,max_new):
    """ターンごとに、入力を流してから greedy で最大 max_new トークン生成する（生成したトークンも流す）。

    停止トークンはその場では流さず、次のターンの先頭（<|im_end|>）として 1 回だけ流す。
    照合（check）は、位置 0・40・100・255 で行う。戻り値: obs.save() の記録。
    """
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
    """長文のストレス用の合成文を 2 つ作る（日本語の走行記録 50 区間・英語の説明と Python コード 50 区間）。"""
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
    """較正 → 元の日本語 3 ターン → 英語 3 ターン → 長文 2 本 → 長い対話入力 1 本、の順に観測し manifest.json を書く。"""
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
    # ① 既存の export_qwen_decode.py をファイルから直接読み込み、その QwenDecodeStep（端末と同じ配線）を使う。
    #    書き写さないのは、実機に載せたモデルと同じ計算であることを保つため。sha256 も記録する。
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
        # ② 後続用の範囲を、評価の値を使わずに先に固定する: 古い較正の 30 トークン（5 文 × 6 歩）を流し、
        #    各文の最初を除いた 25 トークンの最小・最大を frozen_later_ranges に残す。
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
        # ③ 元の日本語の対話（エアコン 22 度 → 音楽 → さっき何度？）。保存してある入力と 1 ターン目の 41 トークンが
        #    完全に一致すること、各ターンの入力が 41 / 18 / 22 トークンであることを確かめる。
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
        # ④ 長文 2 本を 640 トークンずつ流す（生成はしない）。256 を超えるのでキャッシュの入替後も観測できる。
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
        # 入力を 550 トークンで切り、assistant の見出し（suffix）を付け直す（記録では 559 トークンの入力＋64 生成）。
        long_ids=long_ids[:550]+suffix
        manifest['records'].append(run_dialogue(observer,tok,'chat_long',[long_ids],args.max_new))
        checkpoint()
    manifest['elapsed_seconds']=time.time()-started
    checkpoint()
    emit({'complete':True,'elapsed_seconds':manifest['elapsed_seconds'],
          'total_tokens':sum(r['tokens'] for r in manifest['records'])})

if __name__=='__main__':
    main()
