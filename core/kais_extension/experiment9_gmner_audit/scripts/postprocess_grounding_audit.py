#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path
from collections import defaultdict,Counter
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu=None
ROOT=Path('/home/jyx/yixiao/MINER-My')
EXP=ROOT/'kais_extension/experiment9_gmner_audit'
RES=EXP/'results'

def f(x):
    try:
        if x=='' or x is None: return None
        v=float(x)
        if math.isnan(v): return None
        return v
    except Exception: return None

def read_csv(p):
    if not p.exists(): return []
    with p.open(newline='',encoding='utf-8') as fh: return list(csv.DictReader(fh))

def write_csv(p,rows,keys=None):
    if keys is None:
        keys=[]
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)

def write_json(p,o): p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')

def mean(xs): return sum(xs)/len(xs) if xs else None
def stdev(xs): return statistics.stdev(xs) if len(xs)>1 else 0.0 if xs else None
def median(xs): return statistics.median(xs) if xs else None
def quant(xs,q):
    if not xs: return None
    xs=sorted(xs); pos=(len(xs)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo] if lo==hi else xs[lo]*(hi-pos)+xs[hi]*(pos-lo)
def stat(xs):
    return {'n':len(xs),'mean':mean(xs),'std':stdev(xs),'median':median(xs),'q25':quant(xs,0.25),'q75':quant(xs,0.75)}

def ranks(vals):
    order=sorted(range(len(vals)), key=lambda i: vals[i])
    r=[0]*len(vals); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and vals[order[j+1]]==vals[order[i]]: j+=1
        avg=(i+j+2)/2.0
        for k in range(i,j+1): r[order[k]]=avg
        i=j+1
    return r

def auroc(y,score):
    pairs=[(int(a),float(b)) for a,b in zip(y,score) if b is not None]
    npos=sum(a for a,_ in pairs); nneg=len(pairs)-npos
    if npos==0 or nneg==0: return None
    rs=ranks([b for _,b in pairs])
    sum_pos=sum(r for r,(a,_) in zip(rs,pairs) if a==1)
    return (sum_pos-npos*(npos+1)/2)/(npos*nneg)

def aupr(y,score):
    pairs=sorted([(int(a),float(b)) for a,b in zip(y,score) if b is not None], key=lambda x:x[1], reverse=True)
    pos=sum(a for a,_ in pairs)
    if pos==0: return None
    tp=0; fp=0; last_recall=0; area=0
    for a,_ in pairs:
        if a: tp+=1
        else: fp+=1
        recall=tp/pos; prec=tp/(tp+fp)
        area += (recall-last_recall)*prec
        last_recall=recall
    return area

def mwu(x0,x1):
    if not x0 or not x1: return None
    if mannwhitneyu:
        try: return float(mannwhitneyu(x1,x0,alternative='two-sided').pvalue)
        except Exception: return None
    return None

def group_rows(rows, key):
    d=defaultdict(list)
    for r in rows: d[r.get(key,'')].append(r)
    return d

def main():
    rows=[]
    for split in ['train','dev','test']:
        part=read_csv(RES/f'grounding_entity_level_{split}.csv')
        rows.extend(part)
    write_csv(RES/'grounding_entity_level.csv', rows)
    ground=[r for r in rows if r.get('groundable')=='1']
    summary={'overall':{},'by_split':{},'by_type':{}}
    for name,subset in [('overall',rows)]:
        gr=[r for r in subset if r.get('groundable')=='1']
        summary[name]={
            'entities':len(subset),'groundable':len(gr),'ungroundable':len(subset)-len(gr),
            'pointing_accuracy':mean([f(r.get('pointing_hit')) for r in gr if f(r.get('pointing_hit')) is not None]),
            'mean_inside_mass':mean([f(r.get('inside_mass')) for r in gr if f(r.get('inside_mass')) is not None]),
            'median_inside_mass':median([f(r.get('inside_mass')) for r in gr if f(r.get('inside_mass')) is not None]),
            'mean_box_area_ratio':mean([f(r.get('box_area_ratio')) for r in gr if f(r.get('box_area_ratio')) is not None]),
            'mean_mass_lift':mean([f(r.get('mass_lift')) for r in gr if f(r.get('mass_lift')) is not None]),
            'median_mass_lift':median([f(r.get('mass_lift')) for r in gr if f(r.get('mass_lift')) is not None]),
            'mean_iou_top10':mean([f(r.get('iou_top10')) for r in gr if f(r.get('iou_top10')) is not None]),
            'mean_iou_top20':mean([f(r.get('iou_top20')) for r in gr if f(r.get('iou_top20')) is not None]),
            'mean_iou_top30':mean([f(r.get('iou_top30')) for r in gr if f(r.get('iou_top30')) is not None]),
            'acc_iou30_ge_05':mean([1.0 if (f(r.get('iou_top30')) or 0)>=0.5 else 0.0 for r in gr if f(r.get('iou_top30')) is not None]),
        }
    for split,subset in group_rows(rows,'split').items():
        gr=[r for r in subset if r.get('groundable')=='1']
        summary['by_split'][split]={'entities':len(subset),'groundable':len(gr),'ungroundable':len(subset)-len(gr),'pointing_accuracy':mean([f(r.get('pointing_hit')) for r in gr if f(r.get('pointing_hit')) is not None]),'mean_mass_lift':mean([f(r.get('mass_lift')) for r in gr if f(r.get('mass_lift')) is not None]),'mean_iou_top30':mean([f(r.get('iou_top30')) for r in gr if f(r.get('iou_top30')) is not None])}
    for typ,subset in group_rows(ground,'type').items():
        summary['by_type'][typ]={'groundable':len(subset),'pointing_accuracy':mean([f(r.get('pointing_hit')) for r in subset if f(r.get('pointing_hit')) is not None]),'mean_mass_lift':mean([f(r.get('mass_lift')) for r in subset if f(r.get('mass_lift')) is not None]),'mean_iou_top30':mean([f(r.get('iou_top30')) for r in subset if f(r.get('iou_top30')) is not None])}
    # random pointing baseline from area ratio, not a Monte Carlo draw.
    summary['overall']['random_pointing_baseline_area'] = summary['overall']['mean_box_area_ratio']
    write_json(RES/'grounding_summary.json', summary)
    signals=['dustbin_mass_est','transport_mass','max_transport','transport_entropy','norm_transport_entropy']
    sig_rows=[]
    y=[1 if r.get('groundable')=='1' else 0 for r in rows]
    for sig in signals:
        vals=[f(r.get(sig)) for r in rows]
        gvals=[v for r,v in zip(rows,vals) if r.get('groundable')=='1' and v is not None]
        uvals=[v for r,v in zip(rows,vals) if r.get('groundable')!='1' and v is not None]
        a=auroc(y,vals); ap=aupr(y,vals)
        aflip=1-a if a is not None else None
        orient='higher_predicts_groundable'
        used=a; used_ap=ap
        if a is not None and aflip>a:
            orient='lower_predicts_groundable'; used=aflip; used_ap=aupr(y,[-v if v is not None else None for v in vals])
        sig_rows.append({'signal':sig,'groundable_n':len(gvals),'ungroundable_n':len(uvals),'groundable_mean':mean(gvals),'ungroundable_mean':mean(uvals),'groundable_median':median(gvals),'ungroundable_median':median(uvals),'groundable_iqr':(quant(gvals,0.75)-quant(gvals,0.25)) if gvals else None,'ungroundable_iqr':(quant(uvals,0.75)-quant(uvals,0.25)) if uvals else None,'auroc_raw_higher':a,'aupr_raw_higher':ap,'orientation':orient,'auroc_oriented':used,'aupr_oriented':used_ap,'mann_whitney_p':mwu(uvals,gvals)})
    write_csv(RES/'groundability_summary.csv', sig_rows)
    write_csv(RES/'groundability_signals.csv', rows)
    # MNER relationship
    rel=[]
    for key,subset in group_rows(rows,'groundable').items():
        for ok,sub2 in group_rows(subset,'mner_correct').items():
            rel.append({'groundable':key,'mner_correct':ok,'n':len(sub2),'mean_mass_lift':mean([f(r.get('mass_lift')) for r in sub2 if f(r.get('mass_lift')) is not None]),'mean_dustbin_mass_est':mean([f(r.get('dustbin_mass_est')) for r in sub2 if f(r.get('dustbin_mass_est')) is not None]),'mean_max_transport':mean([f(r.get('max_transport')) for r in sub2 if f(r.get('max_transport')) is not None])})
    write_csv(RES/'mner_grounding_relationship.csv', rel)
    # Optional logistic probe, use sklearn if available and all splits exist.
    probe={'status':'NOT_EXECUTED','reason':'sklearn unavailable or insufficient split data'}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
        feat=[s for s in signals if all(f(r.get(s)) is not None for r in rows[:10])]
        train=[r for r in rows if r.get('split')=='train']; dev=[r for r in rows if r.get('split')=='dev']; test=[r for r in rows if r.get('split')=='test']
        if train and dev and test and feat:
            def xy(sub): return [[f(r.get(k)) or 0.0 for k in feat] for r in sub],[1 if r.get('groundable')=='1' else 0 for r in sub]
            Xtr,ytr=xy(train); Xdv,ydv=xy(dev); Xte,yte=xy(test)
            sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xdv=sc.transform(Xdv); Xte=sc.transform(Xte)
            best=None
            for C in [0.1,1,10]:
                clf=LogisticRegression(C=C,max_iter=1000,solver='liblinear').fit(Xtr,ytr)
                pv=clf.predict_proba(Xdv)[:,1]
                score=roc_auc_score(ydv,pv)
                if best is None or score>best[0]: best=(score,C,clf)
            _,C,clf=best; pt=clf.predict_proba(Xte)[:,1]; pred=[1 if x>=0.5 else 0 for x in pt]
            probe={'status':'EXECUTED','features':feat,'selected_C':C,'dev_auroc':best[0],'test_auroc':float(roc_auc_score(yte,pt)),'test_aupr':float(average_precision_score(yte,pt)),'test_accuracy':float(accuracy_score(yte,pred)),'test_f1':float(f1_score(yte,pred)),'coefficients':dict(zip(feat,[float(x) for x in clf.coef_[0]])),'intercept':float(clf.intercept_[0])}
    except Exception as e:
        probe={'status':'NOT_EXECUTED','reason':repr(e)}
    write_json(RES/'groundability_logistic_probe.json', probe)
    print(json.dumps({'rows':len(rows),'groundable':len(ground),'best_signal':max(sig_rows,key=lambda r: r['auroc_oriented'] if r['auroc_oriented'] is not None else -1),'probe':probe},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
