import sys,json,itertools,hashlib,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/jyx/PGMNER_gmner_improve_20261108')
# Different module name avoids importing this file recursively.
import importlib.util
spec=importlib.util.spec_from_file_location('previous_experiment','/home/jyx/PGMNER_gmner_improve_20261108/experiment.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
legacy=old.legacy; m=legacy.m
OUT=Path(__file__).parent/'results'; OUT.mkdir(exist_ok=True)
TEACHER=Path('/data/hfy/VG-Think/E2E-GMNER/outputs/exp_cot_paper')
FIXED={t:(.98,'head',.5) for t in old.TYPES}

def save(name,obj):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)

def inputs(split):
    data=legacy.load(split)
    cq=old.context_cache(split,data)
    indices=list(range(len(data[0]))) if split=='dev' else old.test_indices()
    pg={i:old.predict(data[0][i],cq,FIXED) for i in indices}
    path=TEACHER/('dev_pred_epoch3.jsonl' if split=='dev' else 'test_pred_best_ep3.jsonl')
    rows=legacy.read(path);assert len(rows)==len(data[0])
    teacher={}
    for i in indices:
        # Gold fields are used only to verify cached row/coordinate alignment.
        assert m.parse_triples(rows[i]['labels_answer_scaled'],strict=True,where='cache_gold_alignment')==data[2][i],(split,i)
        teacher[i]=m.parse_triples(rows[i]['response_answer_scaled'],strict=False,where='teacher_prediction')
    return data,indices,pg,teacher

def key(p):return p.text,p.etype

def hybrid(data,i,pg,teacher,mode,threshold):
    p=pg[i];t=teacher[i]
    pk={key(x) for x in p};tk={key(x) for x in t}
    tdict={key(x):x for x in t}
    if mode=='teacher':return t
    if mode=='pg':return p
    if mode=='consensus':return [x for x in t if key(x) in pk]
    if mode=='pg_relocalized':return [tdict.get(key(x),x) for x in p]
    conf={}
    for e in data[0][i]['entities']:
        k=(e['mention'],e['official_type']);conf[k]=max(conf.get(k,0),e['entity_confidence'])
    if mode=='teacher_plus_pg':
        # Keep teacher decisions, add confident PGMNER-only nonoverlapping mentions.
        return t+[x for x in p if key(x) not in tk and conf.get(key(x),0)>=threshold and not any(x.text in y.text or y.text in x.text for y in t)]
    if mode=='pg_plus_teacher':
        base=[tdict.get(key(x),x) for x in p if conf.get(key(x),0)>=threshold]
        return base+[x for x in t if key(x) not in {key(y) for y in base} and not any(x.text in y.text or y.text in x.text for y in base)]
    raise ValueError(mode)

def counts(data,idx,pred):
    return np.asarray([[m.count_correct_gmner(pred[i],data[2][i],.5),len(pred[i]),len(data[2][i])] for i in idx])

def metrics(data,idx,pred):
    c=counts(data,idx,pred);v=c.sum(0)
    ner=sum(m.count_correct_mner(pred[i],data[2][i]) for i in idx)
    return dict(gmner_f1=old.score(c),mner_f1=float(2*ner/max(v[1]+v[2],1)),correct=int(v[0]),predicted=int(v[1]),gold=int(v[2]))

def cached():
    data,idx,p,t=inputs('dev')
    configs=[dict(mode=mode,threshold=thr) for mode in ['pg_relocalized','consensus','teacher_plus_pg','pg_plus_teacher'] for thr in ([.98,.995,.999] if mode in ['teacher_plus_pg','pg_plus_teacher'] else [0])]
    trials=[]
    for cfg in configs:
        pred={i:hybrid(data,i,p,t,**cfg) for i in idx}
        trials.append(dict(config=cfg,**metrics(data,idx,pred)))
    winner=max(trials,key=lambda x:x['gmner_f1'])
    lock=dict(winner=winner,trials=trials,dev_baseline=metrics(data,idx,p),dev_teacher=metrics(data,idx,t),selection='Fixed finite candidates selected by DEV only; saved before loading TEST. Hybrid winner does not replace independent teacher control.')
    save('cached_locked.json',lock)
    data,idx,p,t=inputs('test')
    h={i:hybrid(data,i,p,t,**winner['config']) for i in idx}
    anchored={i:hybrid(data,i,p,t,'pg_relocalized',0) for i in idx}
    verified={i:hybrid(data,i,p,t,'consensus',0) for i in idx}
    result=dict(pg_context=metrics(data,idx,p),teacher_independent=metrics(data,idx,t),hybrid=metrics(data,idx,h),pg_fixed_entities_teacher_grounding=metrics(data,idx,anchored),pg_teacher_consensus=metrics(data,idx,verified),hybrid_exactly_equals_teacher=all(h[i]==t[i] for i in idx),lock=lock,scope='Historical cached predictions; same 835 posts and official scorer. Teacher is externally trained end-to-end GMNER adapter, not PGMNER. TEST previously examined. Anchored and consensus controls are fixed descriptive comparisons, not TEST-selected candidates.')
    save('cached_results.json',result)
    print(json.dumps(result),flush=True)

def review_cache(split,idx):
    rows={r['sample_index']:r for p in OUT.glob(f'review_{split}_shard*.jsonl') for r in legacy.read(p)}
    assert set(idx)<=rows.keys(),f'Missing {len(set(idx)-rows.keys())} review predictions'
    return {i:m.parse_triples(m.extract_answer_text(rows[i]['raw_generation']),strict=False,where='review_prediction') for i in idx}

def select():
    data,idx,p,t=inputs('dev');review=review_cache('dev',idx)
    cfg=json.loads((OUT/'cached_locked.json').read_text())['winner']['config']
    h={i:hybrid(data,i,p,t,**cfg) for i in idx}
    anchored={i:hybrid(data,i,p,t,'pg_relocalized',0) for i in idx}
    verified={i:hybrid(data,i,p,t,'consensus',0) for i in idx}
    choices={name:metrics(data,idx,pred) for name,pred in [('pg_context',p),('anchored_grounding',anchored),('consensus',verified),('review',review)]}
    # Exclude degenerate cached hybrids identical to the independent teacher.
    if any(h[i]!=t[i] for i in idx): choices['cached_hybrid']=metrics(data,idx,h)
    winner=max(choices,key=lambda x:choices[x]['gmner_f1'])
    save('locked_final.json',dict(winner=winner,dev_candidates=choices,teacher_control=metrics(data,idx,t),cached_hybrid_config=cfg,selection='Best DEV F1 among PG baseline, fixed cached hybrid and candidate-assisted review. No TEST-dependent tuning. Teacher independent is always reported as a control.'))
    print(json.dumps(choices),flush=True)

def final():
    lock=json.loads((OUT/'locked_final.json').read_text());data,idx,p,t=inputs('test')
    name=lock['winner']
    if name=='review':chosen=review_cache('test',idx)
    elif name=='cached_hybrid':chosen={i:hybrid(data,i,p,t,**lock['cached_hybrid_config']) for i in idx}
    elif name=='anchored_grounding':chosen={i:hybrid(data,i,p,t,'pg_relocalized',0) for i in idx}
    elif name=='consensus':chosen={i:hybrid(data,i,p,t,'consensus',0) for i in idx}
    else:chosen=p
    cc=counts(data,idx,chosen);tt=counts(data,idx,t);pp=counts(data,idx,p)
    rng=np.random.default_rng(42);diffs=[]
    for _ in range(2000):
        ii=rng.integers(0,len(idx),len(idx));diffs.append([100*(old.score(cc[ii])-old.score(b[ii])) for b in [pp,tt]])
    np.savez(OUT/'paired_counts.npz',chosen=cc,pg_context=pp,teacher=tt)
    save('final_results.json',dict(selected=metrics(data,idx,chosen),pg_context=metrics(data,idx,p),teacher_independent=metrics(data,idx,t),delta_vs_pg_pp=100*(old.score(cc)-old.score(pp)),delta_vs_teacher_pp=100*(old.score(cc)-old.score(tt)),paired_bootstrap_95ci_pp=np.quantile(diffs,[.025,.975],axis=0).T.tolist(),lock=lock,scope='Exploratory same-835 comparison. Gains over PG may derive from externally supervised teacher. Only gain over teacher control supports added hybrid value; neither is evidence of improving PPIB/GOTA alone.'))
    with (OUT/'selected_predictions.jsonl').open('w') as f:
        for i in idx:f.write(json.dumps(dict(sample_index=i,prediction=m.triples_to_canon_text(chosen[i])),ensure_ascii=False)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['cached','select','final']);args=a.parse_args()
    globals()[args.stage]()
