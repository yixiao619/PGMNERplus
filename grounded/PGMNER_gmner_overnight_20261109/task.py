import sys,json,os,time,argparse,importlib.util,traceback,hashlib,subprocess
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0,'/home/jyx/PGMNER_gmner_teacher_20261109');import experiment as base
m=base.m;ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def save(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def read(p):return base.legacy.read(p)
def key(x):return x.text,x.etype
def triple(j):return m.EntityTriple(j['mention'],j['type'],j['boxes'],True)
def prepare(split):
 data,idx,p,t=base.inputs(split);context=base.old.context_cache(split,data);jobs=[]
 for i in idx:
  s=data[0][i];pd={key(x):x for x in p[i]};td={key(x):x for x in t[i]}
  for k in sorted(pd.keys()|td.keys()):
   options=[None];teacher=td.get(k);pg=pd.get(k)
   # Preserve multi-box teacher output as a complete option.
   proposals=[x.regions for x in [teacher,pg] if x is not None and x.region_valid]
   for cache in [data[1],context]:
    r=cache.get(f'{s["sample_index"]}\t{k[0]}\t{k[1]}')
    if r and r['parse_valid']:proposals.append(r['pred_boxes'])
   for boxes in proposals:
    boxes=None if boxes is None else [list(b) for b in boxes]
    if boxes not in options:options.append(boxes)
   ent=next((e for e in s['entities'] if (e['mention'],e['official_type'])==k),{})
   for oi,boxes in enumerate(options):
    tb=None if teacher is None or teacher.regions is None else [list(b) for b in teacher.regions]
    jobs.append(dict(id=f'{i}\t{k[0]}\t{k[1]}\t{oi}',group=f'{i}\t{k[0]}\t{k[1]}',sample_index=i,text=s['text'],image_path=s['image_path'],mention=k[0],type=k[1],boxes=boxes,teacher_entity=teacher is not None,pg_entity=pg is not None,teacher_choice=teacher is not None and teacher.region_valid and boxes==tb,entity_confidence=ent.get('entity_confidence',0),groundability=ent.get('groundability_probability',0)))
 with (OUT/f'{split}_jobs.jsonl').open('w') as f:
  for j in jobs:f.write(json.dumps(j,ensure_ascii=False)+'\n')
 print('prepared',split,len(jobs),flush=True)
def worker(split,shard):
 import torch
 from PIL import Image
 from e2egmner.evaluation.geometry import smart_resize
 spec=importlib.util.spec_from_file_location('loader',base.legacy.ROOT/'kais_extension/experiment14_gmner_integration/scripts/run_qwen_q2_cache.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
 jobs=[j for j in read(OUT/f'{split}_jobs.jsonl') if j['sample_index']%3==shard];path=OUT/f'{split}_worker{shard}.jsonl';done={j['id'] for j in read(path)} if path.exists() else set();jobs=[j for j in jobs if j['id'] not in done]
 print('pending',len(jobs),flush=True)
 if not jobs:return
 torch.set_num_threads(4);model,processor=q.load_model('cuda:0');ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'AB'];assert all(len(x)==1 for x in ids);ids=[x[0] for x in ids];entity_cache={};start=time.time()
 @torch.no_grad()
 def score(prompt,images,positive):
  content=[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]
  messages=[{'role':'system','content':'Verify the given hypothesis using available evidence. Select only A or B.'},{'role':'user','content':content}]
  txt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);batch=processor(text=[txt],**({'images':images} if images else {}),return_tensors='pt',padding=True);batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
  logits=model(**batch,logits_to_keep=1).logits[0,-1,ids].float();return float(torch.softmax(logits,0)[positive])
 with path.open('a') as f:
  for n,j in enumerate(jobs,1):
   with Image.open(j['image_path']) as im:image=im.convert('RGB')
   head='Post: '+j['text']+'\nMention: '+j['mention']+'\nType: '+j['type']+'\n'
   if j['group'] not in entity_cache:
    prompt=head+'Is this exact text span a named entity of the stated type? Types: PER, LOC, ORG, OTHER. Visual presence is not required.\n'
    entity_cache[j['group']]=[score(prompt+'A: Correct. B: Incorrect.\nAnswer:',[],0),score(prompt+'A: Incorrect. B: Correct.\nAnswer:',[],1)]
   images=[image]
   if j['boxes'] is None:question='Is this entity visually absent from the image? A textual mention alone is not proof of visibility.'
   else:
    w,h=image.size;nh,nw=smart_resize(h,w,factor=28,min_pixels=256*28*28,max_pixels=1024*28*28)
    for box in j['boxes']:
     a,b,c,d=box;a=max(0,min(w-1,int(a*w/nw)));c=max(a+1,min(w,int(c*w/nw)));b=max(0,min(h-1,int(b*h/nh)));d=max(b+1,min(h,int(d*h/nh)));crop=image.crop((a,b,c,d));side=max(crop.size);canvas=Image.new('RGB',(side,side),(127,127,127));canvas.paste(crop,((side-crop.width)//2,(side-crop.height)//2));images.append(canvas)
    question='Does at least one proposed region correctly localize this specific entity? The first image is the whole scene; subsequent images are candidate crops. Do not accept a merely related object. Full resized-image boxes: '+str(j['boxes'])+f'; image size {nw}x{nh}.'
   g=[score(head+question+'\nA: Yes. B: No.\nAnswer:',images,0),score(head+question+'\nA: No. B: Yes.\nAnswer:',images,1)]
   f.write(json.dumps(dict(id=j['id'],entity_scores=entity_cache[j['group']],ground_scores=g))+'\n');f.flush()
   if n%25==0 or n==len(jobs):print(f'{n}/{len(jobs)} seconds/job={(time.time()-start)/n:.2f}',flush=True)
def bundle(split):
 data,idx,p,t=base.inputs(split);jobs=read(OUT/f'{split}_jobs.jsonl');r={x['id']:x for f in OUT.glob(f'{split}_worker*.jsonl') for x in read(f)};assert {j['id'] for j in jobs}<=r.keys();X=[];y=[];groups=[]
 for j in jobs:
  a,b=r[j['id']]['entity_scores'];c,d=r[j['id']]['ground_scores'];X.append([a,b,c,d,(a+b)/2,(c+d)/2,abs(a-b),abs(c-d),j['entity_confidence'],j['groundability'],float(j['teacher_entity']),float(j['pg_entity']),float(j['teacher_choice']),float(j['boxes'] is not None)]+[float(j['type']==z) for z in ['PER','LOC','ORG','OTHER']])
  groups.append(base.legacy.norm(j['text']))
  if split=='dev':y.append(m.count_correct_gmner([triple(j)],data[2][j['sample_index']],.5))
 return data,idx,p,t,jobs,np.array(X),np.array(y),np.array(groups)
def predict(b,probs,threshold,mode):
 data,idx,p,t,jobs,*_=b;out={i:[] for i in idx};by={}
 for j,v in zip(jobs,probs):
  if mode=='teacher_entities' and not j['teacher_entity']:continue
  if mode=='teacher_boxes' and not j['teacher_choice']:continue
  by.setdefault(j['group'],[]).append((float(v),j))
 entries=[]
 for rows in by.values():
  v,j=max(rows,key=lambda z:(z[0],z[1]['teacher_choice']));
  if v>=threshold:entries.append((v,j))
 # Deterministic confidence ordering handles incompatible overlapping candidates.
 for v,j in sorted(entries,key=lambda z:(-z[0],not z[1]['teacher_entity'],z[1]['id'])):
  i=j['sample_index'];x=triple(j)
  if not any(x.text in z.text or z.text in x.text for z in out[i]):out[i].append(x)
 return out
def estimator(kind):
 if kind=='tree':return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=20,learning_rate=.05,early_stopping=False,random_state=42)
 return make_pipeline(StandardScaler(),LogisticRegression(C=float(kind),max_iter=2500,random_state=42))
def fit():
 b=bundle('dev');data,idx,p,t,jobs,X,y,groups=b;folds=list(GroupKFold(n_splits=5).split(X,y,groups));save('folds.json',[dict(train=a.tolist(),valid=z.tolist()) for a,z in folds]);trials=[dict(model='teacher',f1=base.metrics(data,idx,t)['gmner_f1'])];models={}
 for kind in ['0.01','0.1','1.0','tree']:
  v=np.zeros(len(X))
  for a,z in folds:
   model=estimator(kind);model.fit(X[a],y[a]);v[z]=model.predict_proba(X[z])[:,1]
  models[kind]=v
  for mode in ['teacher_boxes','teacher_entities','union']:
   for threshold in [.2,.3,.4,.5,.6,.7]:trials.append(dict(model=kind,mode=mode,threshold=threshold,f1=base.metrics(data,idx,predict(b,v,threshold,mode))['gmner_f1']))
 win=max(trials,key=lambda z:z['f1']);save('locked.json',dict(winner=win,trials=trials,scope='Grouped five-fold DEV OOF selection; independent teacher first wins ties. Modes isolate filtering, region decisions and PG candidate additions. No TEST-based follow-up.'))
 if win['model']!='teacher':
  model=estimator(win['model']);model.fit(X,y);joblib.dump(model,OUT/'model.joblib')
 print('winner',win,flush=True)
def evaluate():
 win=json.loads((OUT/'locked.json').read_text())['winner'];b=bundle('test');data,idx,p,t,jobs,X,*_=b
 pred=t if win['model']=='teacher' else predict(b,joblib.load(OUT/'model.joblib').predict_proba(X)[:,1],win['threshold'],win['mode'])
 cc=base.counts(data,idx,pred);tt=base.counts(data,idx,t);assert abs(base.old.score(tt)-.6465234801601747)<1e-10
 rng=np.random.default_rng(42);ds=[]
 for _ in range(2000):
  ii=rng.integers(0,len(idx),len(idx));ds.append(100*(base.old.score(cc[ii])-base.old.score(tt[ii])))
 save('final_results.json',dict(selected=base.metrics(data,idx,pred),teacher=base.metrics(data,idx,t),delta_vs_teacher_pp=100*(base.old.score(cc)-base.old.score(tt)),ci95_pp=np.quantile(ds,[.025,.975]).tolist(),winner=win,scope='Previously examined TEST835; exploratory. If teacher wins this is fallback, not hybrid gain.'))
 np.savez(OUT/'counts.npz',selected=cc,teacher=tt)
 with (OUT/'predictions.jsonl').open('w') as f:
  for i in idx:f.write(json.dumps(dict(sample_index=i,prediction=m.triples_to_canon_text(pred[i])),ensure_ascii=False)+'\n')
def run():
 def status(stage):save('status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z')));print(stage,flush=True)
 try:
  save('source_hash.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  for split in ['dev','test']:
   status('prepare_'+split);prepare(split);status('gpu_'+split);procs=[]
   for shard in range(3):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(shard),PYTHONPATH='/data/cyc/LLARA/peft/src',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    f=(OUT/f'{split}_worker{shard}.log').open('a');p=subprocess.Popen([sys.executable,'-u',__file__,'worker',split,str(shard)],env=env,stdout=f,stderr=subprocess.STDOUT);procs.append((p,f))
   codes=[]
   for p,f in procs:codes.append(p.wait());f.close()
   if any(codes):raise RuntimeError(str(codes))
   if split=='dev':status('fit');fit()
   else:status('evaluate');evaluate()
  status('complete')
 except Exception:save('status.json',dict(stage='failed',error=traceback.format_exc()));raise
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
 else:run()
