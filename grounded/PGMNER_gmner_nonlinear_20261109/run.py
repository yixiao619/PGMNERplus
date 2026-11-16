import sys,json,time,os,traceback,hashlib
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
sys.path.insert(0,'/home/jyx/PGMNER_gmner_refine_20261109')
import train as old
OUT=Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
def save(n,x):
 p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def status(stage,**kw):
 x=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z'),**kw);save('status.json',x);print(json.dumps(x),flush=True)
def make(kind):
 if kind=='baseline':return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=42))
 if kind=='interaction':return make_pipeline(PolynomialFeatures(degree=2,interaction_only=True,include_bias=False),StandardScaler(),LogisticRegression(C=.05,max_iter=3000,random_state=42))
 if kind=='forest':return RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=30,max_features=.8,n_jobs=4,random_state=42)
 return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,min_samples_leaf=30,learning_rate=.05,l2_regularization=10 if kind=='boost10' else 30,early_stopping=False,random_state=42)
def main():
 status('training');save('source_sha256.json',dict(run=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()))
 b=old.dataset('dev');data,idx,p,t,jobs,X,yn,yj,groups,src,scores=b
 folds=list(GroupKFold(n_splits=5).split(X,yn,groups));save('folds.json',[dict(train=a.tolist(),validation=z.tolist()) for a,z in folds])
 predictions={};specs={}
 for label,y in [('entity',yn),('joint',yj)]:
  for kind in ['baseline','interaction','boost10','boost30','forest']:
   name=f'{kind}_{label}';status('training',model=name);v=np.zeros(len(X))
   for a,z in folds:
    model=make(kind);model.fit(X[a],y[a]);v[z]=model.predict_proba(X[z])[:,1]
   predictions[name]=v;specs[name]=dict(kind=kind,label=label)
 # Fixed 50:50 blends reduce variance; no optimizing mixture weights on TEST.
 for name,v in list(predictions.items()):
  if not name.startswith('baseline'):
   predictions['blend_'+name]=.5*predictions['baseline_entity']+.5*v
 status('selecting')
 baseline_cfg=dict(teacher=.4,pg=.4);baseline=old.predict(b,predictions['baseline_entity'],baseline_cfg)
 trials=[dict(model='baseline_entity',config=baseline_cfg,**old.base.metrics(data,idx,baseline))]
 for name,v in predictions.items():
  for th in [.25,.4,.5,.65,.8,1.01]:
   for pg in [.4,.65,.8,1.01]:
    cfg=dict(teacher=th,pg=pg);pred=old.predict(b,v,cfg);trials.append(dict(model=name,config=cfg,**old.base.metrics(data,idx,pred)))
 winner=max(trials,key=lambda r:r['gmner_f1']);save('locked.json',dict(winner=winner,trials=trials,scope='Fixed candidates, five-fold grouped DEV cross-fitting, select OOF micro-F1; baseline wins ties. Refit winner on full DEV before TEST. Reused DEV/TEST exploratory.'))
 status('refitting',winner=winner)
 names=[winner['model']] if not winner['model'].startswith('blend_') else ['baseline_entity',winner['model'][6:]]
 models=[]
 for name in names:
  spec=specs[name];model=make(spec['kind']);model.fit(X,yn if spec['label']=='entity' else yj);models.append(model)
 joblib.dump(models,OUT/'selected_models.joblib')
 status('evaluating')
 bt=old.dataset('test');d,it,pp,tt,jj,xt,*_=bt;v=np.mean([model.predict_proba(xt)[:,1] for model in models],axis=0);pred=old.predict(bt,v,winner['config'])
 parent=joblib.load(old.OUT/'selected_calibrator.joblib');parentpred=old.predict(bt,parent.predict_proba(xt)[:,1],baseline_cfg)
 cc=old.base.counts(d,it,pred);bb=old.base.counts(d,it,parentpred);tc=old.base.counts(d,it,tt)
 assert abs(old.base.old.score(bb)-.6405178979436406)<1e-10
 rng=np.random.default_rng(42);deltas=[]
 for _ in range(2000):
  ii=rng.integers(0,len(it),len(it));deltas.append([100*(old.base.old.score(cc[ii])-old.base.old.score(z[ii])) for z in [bb,tc]])
 save('final_results.json',dict(selected=old.base.metrics(d,it,pred),baseline64052=old.base.metrics(d,it,parentpred),teacher=old.base.metrics(d,it,tt),delta_vs_baseline_pp=100*(old.base.old.score(cc)-old.base.old.score(bb)),delta_vs_teacher_pp=100*(old.base.old.score(cc)-old.base.old.score(tc)),paired_bootstrap_95ci_pp=np.quantile(deltas,[.025,.975],axis=0).T.tolist(),winner=winner,scope='Decision-layer calibration; no PGMNER/Qwen fine-tuning. Same previously examined 835 TEST.'))
 np.savez(OUT/'paired_counts.npz',selected=cc,baseline=bb,teacher=tc)
 with (OUT/'predictions.jsonl').open('w') as f:
  for i in it:f.write(json.dumps(dict(sample_index=i,prediction=old.m.triples_to_canon_text(pred[i])),ensure_ascii=False)+'\n')
 status('complete')
if __name__=='__main__':
 try:main()
 except Exception:status('failed',error=traceback.format_exc());raise
