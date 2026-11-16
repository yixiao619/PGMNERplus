import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
import joblib
sys.path.insert(0,'/home/jyx/PGMNER_gmner_three_rounds_20261109')
import pipeline as prev
base=prev.base;m=prev.m;OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
def save(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));t.replace(p)
def dataset(split):
 data,idx,p,t=prev.load(split);jobs=prev.legacy.read(prev.OUT/f'r1_{split}_jobs.jsonl');old=prev.results(1,split)
 new={x['id']:x for f in OUT.glob(f'{split}_worker*.jsonl') for x in prev.legacy.read(f)};assert {j['id'] for j in jobs}<=new.keys()
 X=[];yn=[];yj=[];groups=[];sources=[];scores=[]
 for j in jobs:
  i=j['sample_index'];e=j['entity'];s=new[j['id']]['scores'];orig=old[j['id']]['probs'][0];v=(orig+s['visual_swapped'])/2;tx=(s['text']+s['text_swapped'])/2
  source=j['source']=='teacher';conf=max([x['entity_confidence'] for x in data[0][i]['entities'] if (x['mention'],x['official_type'])==(e['text'],e['etype'])]+[0])
  h=max([x['groundability_probability'] for x in data[0][i]['entities'] if (x['mention'],x['official_type'])==(e['text'],e['etype'])]+[0])
  X.append([orig,s['visual_swapped'],s['text'],s['text_swapped'],v,tx,abs(orig-s['visual_swapped']),abs(s['text']-s['text_swapped']),float(source),conf,h,float(e['regions'] is not None),min(len(e['text'].split()),10)/10]+[float(e['etype']==typ) for typ in ['PER','LOC','ORG','OTHER']])
  scores.append([orig,v,tx,(v+tx)/2]);sources.append(source);groups.append(prev.norm(data[0][i]['text']))
  if split=='dev':
   candidate=prev.decode(e);yn.append(m.count_correct_mner([candidate],data[2][i]));yj.append(m.count_correct_gmner([candidate],data[2][i],.5))
 return data,idx,p,t,jobs,np.array(X),np.array(yn),np.array(yj),np.array(groups),np.array(sources),np.array(scores)
def predict(bundle,values,cfg):
 data,idx,p,t,jobs,*_=bundle;out={i:base.hybrid(data,i,p,t,'consensus',0) for i in idx};add={i:[] for i in idx}
 for j,v in zip(jobs,values):
  if v>=cfg[j['source']]:add[j['sample_index']].append((float(v),prev.decode(j['entity'])))
 for i in idx:
  # Preserve original round1 teacher-first candidate order for equal scores.
  order={}; seq=[x for x in t[i] if base.key(x) not in {base.key(z) for z in p[i]}]+[x for x in p[i] if base.key(x) not in {base.key(z) for z in t[i]}]
  for rank,x in enumerate(seq):order.setdefault(base.key(x),rank)
  for _,x in sorted(add[i],key=lambda z:(-z[0],order[base.key(z[1])])):
   if not any(x.text in y.text or y.text in x.text for y in out[i]):out[i].append(x)
 return out
def fit():
 b=dataset('dev');data,idx,p,t,jobs,X,yn,yj,groups,src,scores=b
 models={'original':scores[:,0],'visual_balanced':scores[:,1],'text_balanced':scores[:,2],'combined':scores[:,3]};specs={}
 folds=list(GroupKFold(n_splits=5).split(X,yn,groups));save('fold_assignments.json',[dict(train_jobs=a.tolist(),validation_jobs=z.tolist()) for a,z in folds])
 for label,y in [('entity',yn),('joint',yj)]:
  for c in [.01,.1,1.]:
   name=f'lr_{label}_{c}';oof=np.zeros(len(jobs))
   for a,z in folds:
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42));model.fit(X[a],y[a]);oof[z]=model.predict_proba(X[z])[:,1]
   models[name]=oof;specs[name]=(label,c)
 baseline_cfg=dict(teacher=.5,pg=.8);baseline_pred=predict(b,scores[:,0],baseline_cfg)
 trials=[dict(model='original',config=baseline_cfg,**base.metrics(data,idx,baseline_pred))]
 for name,values in models.items():
  for a in [.25,.4,.5,.65,.8,1.01]:
   for z in [.4,.65,.8,1.01]:
    cfg=dict(teacher=a,pg=z);trials.append(dict(model=name,config=cfg,**base.metrics(data,idx,predict(b,values,cfg))))
 winner=max(trials,key=lambda r:r['gmner_f1']);save('locked.json',dict(winner=winner,trials=trials,scope='Learned calibrators trained on grouped five-fold DEV out-of-fold labels; source and thresholds selected on those OOF predictions. Selected model refit on full DEV before TEST. Reused DEV remains exploratory. Fixed original R1 baseline wins ties.'))
 if winner['model'] in specs:
  label,c=specs[winner['model']];model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42));model.fit(X,yn if label=='entity' else yj);joblib.dump(model,OUT/'selected_calibrator.joblib')
 print('LOCKED',json.dumps(winner),flush=True)
def evaluate():
 b=dataset('test');data,idx,p,t,jobs,X,yn,yj,groups,src,scores=b;lock=json.loads((OUT/'locked.json').read_text());win=lock['winner'];name=win['model']
 values=joblib.load(OUT/'selected_calibrator.joblib').predict_proba(X)[:,1] if name.startswith('lr_') else scores[:,['original','visual_balanced','text_balanced','combined'].index(name)]
 pred=predict(b,values,win['config']);baseline=predict(b,scores[:,0],dict(teacher=.5,pg=.8))
 cc=base.counts(data,idx,pred);bb=base.counts(data,idx,baseline);tt=base.counts(data,idx,t);rng=np.random.default_rng(42);d=[]
 assert abs(base.old.score(bb)-.6374045801526718)<1e-10
 for _ in range(2000):
  ii=rng.integers(0,len(idx),len(idx));d.append([100*(base.old.score(cc[ii])-base.old.score(z[ii])) for z in [bb,tt]])
 save('final_results.json',dict(selected=base.metrics(data,idx,pred),round1=base.metrics(data,idx,baseline),teacher=base.metrics(data,idx,t),delta_vs_round1_pp=100*(base.old.score(cc)-base.old.score(bb)),delta_vs_teacher_pp=100*(base.old.score(cc)-base.old.score(tt)),paired_bootstrap_95ci_pp=np.quantile(d,[.025,.975],axis=0).T.tolist(),lock=win,scope='Exploratory same835 TEST. No PGMNER or Qwen weight training; only optional logistic calibration learned on DEV with grouped cross-fitting.'))
 np.savez(OUT/'paired_counts.npz',selected=cc,round1=bb,teacher=tt)
 with (OUT/'predictions.jsonl').open('w') as f:
  for i in idx:f.write(json.dumps(dict(sample_index=i,prediction=m.triples_to_canon_text(pred[i])),ensure_ascii=False)+'\n')
 print((OUT/'final_results.json').read_text(),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['fit','evaluate']);args=a.parse_args();globals()[args.stage]()
