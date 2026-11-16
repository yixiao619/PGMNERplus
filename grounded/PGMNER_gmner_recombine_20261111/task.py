import os,sys,json,time,subprocess,traceback,hashlib,importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/jyx/PGMNER_gmner_teacher_20261109');import experiment as base
m=base.m;ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);ADAPT=Path('/home/jyx/PGMNER_gmner_adapt_20261110/results')
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def key(x):return x.text,x.etype
def load(split):
 data,idx,pg,teacher=base.inputs(split);path=ADAPT/('answer/dev_pred_epoch2.jsonl' if split=='dev' else 'selected_test/test_pred_selected.jsonl');rows=base.legacy.read(path);assert len(rows)==len(idx)
 new={}
 for i,r in zip(idx,rows):
  assert m.parse_triples(r['labels_answer_scaled'],strict=True,where='alignment')==data[2][i]
  new[i]=m.parse_triples(r['response_answer_scaled'],strict=False,where='adapted')
 return data,idx,pg,teacher,new

def prepare(split):
 data,idx,pg,teacher,new=load(split);jobs={}
 for i in idx:
  oldkeys={key(x) for x in teacher[i]}
  for x in new[i]:
   if key(x) not in oldkeys:
    jid=f'{i}\t{x.text}\t{x.etype}';jobs[jid]=dict(id=jid,sample_index=i,sample=data[0][i],entity=dict(mention=x.text,official_type=x.etype))
 with (OUT/f'{split}_jobs.jsonl').open('w') as f:
  for j in jobs.values():f.write(json.dumps(j,ensure_ascii=False)+'\n')
 print('PREPARED',split,len(jobs),flush=True)

def worker(split,shard):
 import torch
 torch.set_num_threads(4)
 spec=importlib.util.spec_from_file_location('qwen_old',base.legacy.ROOT/'kais_extension/experiment14_gmner_integration/scripts/run_qwen_q2_cache.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
 jobs=[j for j in base.legacy.read(OUT/f'{split}_jobs.jsonl') if j['sample_index']%3==shard];path=OUT/f'{split}_worker{shard}.jsonl';done={r['id'] for r in base.legacy.read(path)} if path.exists() else set();jobs=[j for j in jobs if j['id'] not in done]
 print('pending',len(jobs),flush=True)
 if not jobs:return
 model,processor=q.load_model('cuda:0');prompt=q.Q2;args=type('Args',(),dict(device='cuda:0',max_new_tokens=64))();start=time.time()
 with path.open('a') as f:
  for n,j in enumerate(jobs,1):
   # Escape braces because legacy generate formats entity and type placeholders.
   context=j['sample']['text'].replace('{','{{').replace('}','}}')
   q.Q2='Post context: '+context+'\nUse the post to disambiguate the named entity; return a box only if visually present.\n\n'+prompt
   full,raw,parsed,seconds=q.generate(model,processor,args,j['sample'],j['entity'])
   f.write(json.dumps(dict(id=j['id'],prompt=full,raw=raw,boxes=parsed['boxes'],valid=parsed['valid'],seconds=seconds),ensure_ascii=False)+'\n');f.flush()
   if n%25==0 or n==len(jobs):print(f'{n}/{len(jobs)} elapsed={time.time()-start:.1f}s',flush=True)

MODES=['teacher','new','new_old','new_old_query','new_old_pg_extras','new_old_query_pg_extras','teacher_new_ground','consensus_old','union_old','union_old_query']
def predictions(bundle,mode,query):
 data,idx,pg,teacher,new=bundle;out={}
 for i in idx:
  td={key(x):x for x in teacher[i]};nd={key(x):x for x in new[i]};pk={key(x) for x in pg[i]}
  if mode=='teacher':out[i]=teacher[i];continue
  if mode=='new':out[i]=new[i];continue
  if mode=='teacher_new_ground':out[i]=[nd.get(key(x),x) for x in teacher[i]];continue
  if mode=='consensus_old':out[i]=[x for x in teacher[i] if key(x) in nd];continue
  out[i]=[]
  for x in new[i]:
   k=key(x)
   if k in td:y=td[k]
   else:
    if 'pg_extras' in mode and k not in pk:continue
    if 'query' in mode:
     r=query[f'{i}\t{x.text}\t{x.etype}'];y=m.EntityTriple(x.text,x.etype,r['boxes'],r['valid'])
    else:y=x
   out[i].append(y)
  if mode.startswith('union'):
   # Add old-only, nonoverlapping entities; teacher/new common entities keep old grounding.
   for x in teacher[i]:
    if key(x) not in nd and not any(x.text in y.text or y.text in x.text for y in out[i]):out[i].append(x)
 return out

def score(split):
 b=load(split);data,idx,pg,t,new=b;query={r['id']:r for p in OUT.glob(f'{split}_worker*.jsonl') for r in base.legacy.read(p)};expected={r['id'] for r in base.legacy.read(OUT/f'{split}_jobs.jsonl')};assert expected<=query.keys()
 if split=='dev':
  trials=[dict(mode=mode,**base.metrics(data,idx,predictions(b,mode,query))) for mode in MODES]
  winner=max(trials,key=lambda x:x['gmner_f1']);save('locked.json',dict(winner=winner,trials=trials,rule='Fixed ten combinations, DEV joint F1, teacher wins ties; no TEST-driven search.'))
  print('WINNER',winner,flush=True)
 else:
  lock=json.loads((OUT/'locked.json').read_text());pred=predictions(b,lock['winner']['mode'],query);cc=base.counts(data,idx,pred);tt=base.counts(data,idx,t);nn=base.counts(data,idx,new);assert abs(base.old.score(tt)-.6465234801601747)<1e-10
  rng=np.random.default_rng(42);delta=[]
  for _ in range(2000):
   ii=rng.integers(0,len(idx),len(idx));delta.append(100*(base.old.score(cc[ii])-base.old.score(tt[ii])))
  save('final_results.json',dict(selected=base.metrics(data,idx,pred),teacher=base.metrics(data,idx,t),adapted=base.metrics(data,idx,new),delta_vs_teacher_pp=100*(base.old.score(cc)-base.old.score(tt)),ci95_pp=np.quantile(delta,[.025,.975]).tolist(),lock=lock['winner'],scope='Same835 previously examined TEST. Combines adapted entity predictions and original grounding; teacher fallback is not improvement. Only pg_extras modes explicitly require PGMNER support for new entities.'))
  np.savez(OUT/'counts.npz',selected=cc,teacher=tt,adapted=nn)
  with (OUT/'predictions.jsonl').open('w') as f:
   for i in idx:f.write(json.dumps(dict(sample_index=i,prediction=m.triples_to_canon_text(pred[i])),ensure_ascii=False)+'\n')

def run():
 def status(stage):save('status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z')));print(stage,flush=True)
 try:
  save('source_hash.json',dict(sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  for split in ['dev','test']:
   status('prepare_'+split);prepare(split);status('ground_'+split);jobs=[]
   for shard in range(3):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(shard),PYTHONPATH='/data/cyc/LLARA/peft/src',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    f=(OUT/f'{split}_worker{shard}.log').open('a');p=subprocess.Popen([sys.executable,'-u',__file__,'worker',split,str(shard)],env=e,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f))
   codes=[]
   for p,f in jobs:codes.append(p.wait());f.close()
   if any(codes):raise RuntimeError(f'Workers failed: {codes}')
   status('score_'+split);score(split)
  status('complete')
 except Exception:save('status.json',dict(stage='failed',error=traceback.format_exc()));raise
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
 else:run()
