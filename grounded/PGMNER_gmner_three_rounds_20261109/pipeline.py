import os,sys,json,re,hashlib,importlib.util,argparse,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,'/home/jyx/PGMNER_gmner_teacher_20261109')
import experiment as base
m=base.m;legacy=base.legacy

def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def key(x):return x.text,x.etype
def encode(x):return dict(text=x.text,etype=x.etype,regions=x.regions,region_valid=x.region_valid)
def decode(x):return m.EntityTriple(**x)
def norm(x):return legacy.norm(x)
def load(split):return base.inputs(split)
def results(round,split):
 return {r['id']:r for p in OUT.glob(f'r{round}_{split}_worker*.jsonl') for r in legacy.read(p)}

def training_pool():
 excluded={norm(x['text']) for split in ['dev','test'] for x in legacy.read(legacy.R/f'pgmner_predicted_entities_{split}.jsonl')}
 pool=[]
 for row in legacy.read(m.E2E/'data/sft/train_sft.jsonl'):
  user=next(x['content'] for x in row['messages'] if x['role']=='user');text=user.split('Text:',1)[-1].strip()
  if norm(text) in excluded:continue
  answer=next(x['content'] for x in row['messages'] if x['role']=='assistant')
  entities=m.parse_triples(m.extract_answer_text(answer),strict=False,where='train_demo')
  pool.append(dict(id=row['id'],text=text,tokens=set(re.findall(r'\w+',text.lower())),entities=[dict(mention=x.text,type=x.etype) for x in entities]))
 return pool

def prepare(round,split):
 data,idx,p,t=load(split);jobs=[];pool=training_pool() if round==3 else None
 raw=data[1];context=base.old.context_cache(split,data)
 for i in idx:
  sample=data[0][i];pd={key(x):x for x in p[i]};td={key(x):x for x in t[i]}
  if round in [1,3]:
   candidates=[(k,td.get(k,pd.get(k))) for k in sorted(pd.keys()^td.keys())]
   for k,x in candidates:
    demos='';demoids=[]
    if round==3:
     tokens=set(re.findall(r'\w+',sample['text'].lower()))-set(re.findall(r'\w+',x.text.lower()))
     ranked=sorted(pool,key=lambda r:(-len(tokens&r['tokens'])/max(math.sqrt(len(r['tokens'])),1),r['id']))[:2]
     demoids=[r['id'] for r in ranked]
     demos='Annotated TRAIN examples for mention boundaries and types:\n'+json.dumps([dict(text=r['text'],entities=r['entities']) for r in ranked],ensure_ascii=False)+'\n\n'
    prompt=demos+'Verify this proposed named entity using the post. Is the EXACT mention span and proposed type correct? Do not reject a real text entity just because it is absent from the image. Types: PER, LOC, ORG, OTHER.\nPost: '+sample['text']+'\nMention: '+x.text+'\nType: '+x.etype+'\nA: Correct mention and type.\nB: Incorrect mention boundary, non-entity, or incorrect type.\nChoose A or B only.'
    jobs.append(dict(id=f'{i}\t{x.text}\t{x.etype}',sample_index=i,image_path=sample['image_path'],prompt=prompt,letters=['A','B'],entity=encode(x),source='teacher' if k in td else 'pg',train_demo_ids=demoids))
  else:
   anchored=base.hybrid(data,i,p,t,'pg_relocalized',0)
   for x in anchored:
    options=[None];k=key(x)
    for qr in [raw.get(f'{sample["sample_index"]}\t{x.text}\t{x.etype}'),context.get(f'{sample["sample_index"]}\t{x.text}\t{x.etype}')]:
     if qr and qr['parse_valid'] and qr['pred_boxes']:
      for b in qr['pred_boxes']:
       box=[list(b)]
       if box not in options:options.append(box)
    if k in td and td[k].region_valid and td[k].regions:
     for b in td[k].regions:
      box=[list(b)]
      if box not in options:options.append(box)
    if len(options)==1:continue
    assert len(options)<=26
    letters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)])
    prompt='Choose the image region that actually depicts the named entity in this post, or None if it is not visually present. Boxes use the resized full-image coordinate system. Additional cropped images, if supplied, follow non-None options in order and only provide close-up evidence.\nPost: '+sample['text']+'\nEntity: '+x.text+'\nType: '+x.etype+'\n'+'\n'.join(f'{a}: '+('None' if b is None else str(b[0])) for a,b in zip(letters,options))+'\nReturn only the option letter.'
    jobs.append(dict(id=f'{i}\t{x.text}\t{x.etype}',sample_index=i,image_path=sample['image_path'],prompt=prompt,letters=letters,entity=encode(x),options=options))
 path=OUT/f'r{round}_{split}_jobs.jsonl'
 with path.open('w') as f:
  for j in jobs:f.write(json.dumps(j,ensure_ascii=False)+'\n')
 if round==2 and split=='dev':
  # Candidate oracle is diagnostic only; never read by worker or selector.
  byid={j['id']:j for j in jobs};cc=[]
  for i in idx:
   pred=base.hybrid(data,i,p,t,'pg_relocalized',0);allp=[]
   for x in pred:
    j=byid.get(f'{i}\t{x.text}\t{x.etype}')
    allp.extend([m.EntityTriple(x.text,x.etype,b,True) for b in j['options']] if j else [x])
   cc.append([m.count_correct_gmner(allp,data[2][i],.5),len(pred),len(data[2][i])])
  save('r2_dev_oracle_diagnostic.json',dict(candidate_set_upper_bound_f1=base.old.score(cc),note='Relaxed matching upper bound with fixed entity denominator, not deployable accuracy.'))
 print('PREPARED',round,split,len(jobs),flush=True)

def predictions(round,split,config,data,idx,p,t):
 jobs={j['id']:j for j in legacy.read(OUT/f'r{round}_{split}_jobs.jsonl')};rr=results(round,split);assert jobs.keys()<=rr.keys()
 out={}
 for i in idx:
  if round in [1,3]:
   pk={key(x) for x in p[i]};tk={key(x) for x in t[i]};out[i]=[x for x in t[i] if key(x) in pk]
   additions=[]
   for x,source in [(x,'teacher') for x in t[i] if key(x) not in pk]+[(x,'pg') for x in p[i] if key(x) not in tk]:
    jid=f'{i}\t{x.text}\t{x.etype}';prob=rr[jid]['probs'][0]
    if prob>=config[source]:additions.append((prob,x))
   for _,x in sorted(additions,key=lambda y:-y[0]):
    if not any(x.text in y.text or y.text in x.text for y in out[i]):out[i].append(x)
  else:
   out[i]=[]
   for x in base.hybrid(data,i,p,t,'pg_relocalized',0):
    jid=f'{i}\t{x.text}\t{x.etype}'
    if jid in jobs:
     probs=rr[jid]['probs'];a=int(np.argmax(probs))
     if probs[a]>=config['confidence']:x=m.EntityTriple(x.text,x.etype,jobs[jid]['options'][a],True)
    out[i].append(x)
 return out

def evaluate(round,split):
 data,idx,p,t=load(split)
 configs=([dict(teacher=a,pg=b) for a in [0.5,0.65,0.8,1.01] for b in [0.5,0.65,0.8,1.01]] if round in [1,3] else [dict(confidence=c) for c in [0.,.5,.7,.9,1.01]])
 if split=='dev':
  trials=[dict(config=c,**base.metrics(data,idx,predictions(round,split,c,data,idx,p,t))) for c in configs]
  winner=max(trials,key=lambda r:r['gmner_f1']);save(f'r{round}_locked.json',dict(winner=winner,trials=trials,criterion='DEV joint F1 only; fixed enumerated grid; lock before TEST job preparation'))
 else:
  lock=json.loads((OUT/f'r{round}_locked.json').read_text());pred=predictions(round,split,lock['winner']['config'],data,idx,p,t)
  consensus={i:base.hybrid(data,i,p,t,'consensus',0) for i in idx}
  anchored={i:base.hybrid(data,i,p,t,'pg_relocalized',0) for i in idx}
  save(f'r{round}_test_results.json',dict(selected=base.metrics(data,idx,pred),consensus=base.metrics(data,idx,consensus),anchored=base.metrics(data,idx,anchored),teacher=base.metrics(data,idx,t),lock=lock,scope='Exploratory previously examined 835-post TEST. No TEST-based search or selection between rounds.'))
  np.savez(OUT/f'r{round}_test_counts.npz',selected=base.counts(data,idx,pred),consensus=base.counts(data,idx,consensus),teacher=base.counts(data,idx,t))
  with (OUT/f'r{round}_test_predictions.jsonl').open('w') as f:
   for i in idx:f.write(json.dumps(dict(sample_index=i,prediction=m.triples_to_canon_text(pred[i])),ensure_ascii=False)+'\n')
 print('EVALUATED',round,split,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','evaluate']);a.add_argument('round',type=int);a.add_argument('split',choices=['dev','test']);args=a.parse_args();globals()[args.mode](args.round,args.split)
