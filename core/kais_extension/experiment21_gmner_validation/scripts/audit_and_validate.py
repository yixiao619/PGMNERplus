#!/usr/bin/env python3
"""Experiment 21 read-only overlap audit and cache-only validation."""
from __future__ import annotations
import csv, html, json, re, sys
from collections import Counter
from pathlib import Path

ROOT=Path('/home/jyx/yixiao/MINER-My')
EXP=ROOT/'kais_extension/experiment21_gmner_validation'
EXP14=ROOT/'kais_extension/experiment14_gmner_integration/results'
SIDE=ROOT/'kais_extension/experiment9_gmner_audit/cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl'
SFT=Path('/data/hfy/VG-Think/E2E-GMNER/data/sft')
E2E=Path('/data/hfy/VG-Think/E2E-GMNER')
sys.path.insert(0,str(E2E/'src'))
from e2egmner.evaluation.geometry import max_iou
from e2egmner.evaluation.metrics import count_correct_eeg,count_correct_gmner,count_correct_mner,prf
from e2egmner.evaluation.parsing import parse_triples

def read_jsonl(path): return [json.loads(x) for x in Path(path).open(encoding='utf-8') if x.strip()]
def norm(s):
    s=html.unescape(s).lower().replace('&amp;','&')
    s=re.sub(r'\s+',' ',s).strip()
    return s
def parse_bio(path,dataset,split):
    rows=[]; image=None; tokens=[]
    for raw in Path(path).open(encoding='utf-8'):
        line=raw.rstrip('\n')
        if line.startswith('IMGID:'):
            if image is not None: rows.append({'dataset':dataset,'split':split,'image_id':image,'text':' '.join(tokens)})
            image=line.split(':',1)[1].strip();tokens=[]
        elif line.strip(): tokens.append(line.split('\t',1)[0])
    if image is not None: rows.append({'dataset':dataset,'split':split,'image_id':image,'text':' '.join(tokens)})
    return rows
def overlap(a,b):
    exact={x['text'] for x in a}; normalized={norm(x['text']) for x in a}; images={x['image_id'] for x in a}; pairs={(x['image_id'],norm(x['text'])) for x in a}
    return {'eval_n':len(b),'exact_text':sum(x['text'] in exact for x in b),'normalized_text':sum(norm(x['text']) in normalized for x in b),
      'image_id_or_path':sum(x['image_id'] in images for x in b),'confirmed_text_image_pair':sum((x['image_id'],norm(x['text'])) in pairs for x in b)}
def write_csv(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def score(rows,method,indices):
    out={}
    for name,fn in [('mner',count_correct_mner),('eeg',count_correct_eeg),('gmner',count_correct_gmner)]:
        c=p=g=0
        for i in indices:
            pred=parse_triples(rows[i][method],strict=False,where=f'{method}:{i}');gold=parse_triples(rows[i]['gold'],strict=True,where=f'gold:{i}')
            c+=fn(pred,gold) if name=='mner' else fn(pred,gold,.5);p+=len(pred);g+=len(gold)
        precision,recall,f1=prf(c,p,g);out.update({f'{name}_precision':precision,f'{name}_recall':recall,f'{name}_f1':f1,f'{name}_correct':c,f'{name}_predicted':p,f'{name}_gold':g})
    return out
def match(pred,gold):
    used=set();pairs=[]
    for pi,p in enumerate(pred):
        for gi,g in enumerate(gold):
            if gi not in used and p.text==g.text and p.etype==g.etype:used.add(gi);pairs.append((pi,gi));break
    return pairs
def correct_count(text,gold): return count_correct_gmner(parse_triples(text,strict=False,where='pred'),gold,.5)

def main():
    for d in ['results','scripts','figures']: (EXP/d).mkdir(parents=True,exist_ok=True)
    side=read_jsonl(SIDE); gm={s:[x for x in side if x['split']==s] for s in ['train','dev','test']}
    original={}
    for ds in ['twitter2015','twitter2017']:
        for sp in ['train','dev','test']: original[(ds,sp)]=parse_bio(ROOT/f'data/{ds}/{sp}.txt',ds,sp)
    audit=[]
    for ev in ['dev','test']:
        for ds in ['twitter2015','twitter2017']:
            subset=[x for x in gm[ev] if ds in x['image_rel'].lower()]
            for sp in ['train','dev','test']:
                audit.append({'model_or_resource':'PGMNER paper-full seed42','training_or_selection_source':f'{ds}_{sp}',
                  'role':'train' if (ds,sp)==('twitter2017','train') else ('checkpoint_selection' if (ds,sp)==('twitter2017','test') else 'not_used_by_this_checkpoint'),
                  'evaluation_split':f'gmner_{ev}_{ds}',**overlap(original[(ds,sp)],subset)})
    for source_role,source in [('qwen_lora_train',gm['train']),('qwen_lora_dev_selection',gm['dev'])]:
        for ev in ['dev','test']:
            audit.append({'model_or_resource':'Qwen2.5-VL GMNER LoRA','training_or_selection_source':source_role,'role':'train' if 'train' in source_role else 'checkpoint_selection',
              'evaluation_split':f'gmner_{ev}',**overlap(source,gm[ev])})
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        audit.append({'model_or_resource':'GMNER internal split audit','training_or_selection_source':f'gmner_{a}','role':'split_integrity',
          'evaluation_split':f'gmner_{b}',**overlap(gm[a],gm[b])})
    write_csv(EXP/'results/dataset_overlap_audit.csv',audit)

    train_dev_norm={norm(x['text']) for x in gm['train']+gm['dev']}
    clean=[i for i,x in enumerate(gm['test']) if 'twitter2015' in x['image_rel'].lower() and norm(x['text']) not in train_dev_norm]
    full=list(range(len(gm['test'])))
    rows=read_jsonl(EXP14/'full_predictions_test.jsonl');filtered=read_jsonl(EXP14/'confidence_filtered_test_predictions.jsonl')
    if len(rows)!=len(gm['test']) or len(filtered)!=len(rows):raise RuntimeError('Exp14 cache alignment failed')
    for i,x in enumerate(filtered): rows[i]['confidence_calibrated']=x['prediction']
    methods=[('PGMNER+Qwen','direct_qwen'),('PGMNER+Groundability+Qwen','groundability_aware'),('PGMNER+Groundability+Qwen+DEVConfidence','confidence_calibrated')]
    metric_rows=[]
    for population,idx,credibility in [('full_mixed_test',full,'protocol_limited_overlap'),('source_disjoint_twitter2015',clean,'credible_cross_dataset_external_validation')]:
        base=[rows[i]['direct_qwen'] for i in idx]
        for label,key in methods:
            pred=[rows[i][key] for i in idx]; changed=sum(a!=b for a,b in zip(base,pred))
            metric_rows.append({'population':population,'protocol_status':credibility,'samples':len(idx),'method':label,'prediction_changed_samples_vs_direct':changed,**score(rows,key,idx)})
    write_csv(EXP/'results/official_gmner_metrics.csv',metric_rows)

    breakdown=[]
    for population,idx in [('full_mixed_test',full),('source_disjoint_twitter2015',clean)]:
        for label,key,details_key in [('PGMNER+Qwen','direct_qwen','entities_direct'),('PGMNER+Groundability+Qwen','groundability_aware','entities_groundability_aware')]:
            stats={g:{'n':0,'pred_visual':0,'correct_decision':0,'iou_hit':0} for g in ['groundable','ungroundable']}
            for i in idx:
                gold=parse_triples(rows[i]['gold'],strict=True,where=f'g:{i}');pred=parse_triples(rows[i][key],strict=False,where=f'p:{i}');details=rows[i][details_key]
                for pi,gi in match(pred,gold):
                    gt=gold[gi];pd=pred[pi];group='groundable' if gt.regions is not None else 'ungroundable';z=stats[group];z['n']+=1
                    visual=pd.regions is not None;z['pred_visual']+=int(visual);z['correct_decision']+=int(visual==(gt.regions is not None))
                    z['iou_hit']+=int(group=='groundable' and visual and max_iou(pd.regions,gt.regions)>=.5)
            for group,z in stats.items():
                breakdown.append({'population':population,'method':label,'gold_group':group,**z,
                  'groundability_decision_accuracy':z['correct_decision']/max(z['n'],1),
                  'false_grounding_rate':z['pred_visual']/max(z['n'],1) if group=='ungroundable' else '',
                  'groundable_acc_iou_05':z['iou_hit']/max(z['n'],1) if group=='groundable' else ''})
    write_csv(EXP/'results/groundability_breakdown.csv',breakdown)

    categories={'gate_correct_reject':[],'gate_correct_retain':[],'gate_false_reject':[],'gate_false_accept':[]}
    for i in clean:
        gold=parse_triples(rows[i]['gold'],strict=True,where=f'g:{i}');direct=parse_triples(rows[i]['direct_qwen'],strict=False,where=f'd:{i}');gated=parse_triples(rows[i]['groundability_aware'],strict=False,where=f'a:{i}')
        for dpi,gi in match(direct,gold):
            gt=gold[gi];dp=direct[dpi];gp=next((x for x in gated if x.text==dp.text and x.etype==dp.etype),None)
            if gp is None:continue
            if gt.regions is None and dp.regions is not None and gp.regions is None:cat='gate_correct_reject'
            elif gt.regions is not None and dp.regions is not None and gp.regions is not None and max_iou(gp.regions,gt.regions)>=.5:cat='gate_correct_retain'
            elif gt.regions is not None and dp.regions is not None and max_iou(dp.regions,gt.regions)>=.5 and gp.regions is None:cat='gate_false_reject'
            elif gt.regions is None and gp.regions is not None:cat='gate_false_accept'
            else:continue
            detail=rows[i]['entities_groundability_aware'][dpi] if dpi<len(rows[i]['entities_groundability_aware']) else {}
            categories[cat].append({'category':cat,'sample_index':i,'sample_id':rows[i]['sample_id'],'image_id':rows[i]['image_id'],'text':rows[i]['text'],
              'entity':gt.text,'type':gt.etype,'groundability_probability':detail.get('groundability_probability'),'gold':rows[i]['gold'],'direct':rows[i]['direct_qwen'],'gated':rows[i]['groundability_aware']})
    cases=[]
    for cat,vals in categories.items():
        if vals:cases.append(vals[0])
    write_csv(EXP/'results/qualitative_cases.csv',cases)
    dist=[]
    for population,idx in [('full_mixed_test',full),('source_disjoint_twitter2015',clean)]:
        entities=[e for i in idx for e in gm['test'][i]['entities']]
        c=Counter(e['type'] for e in entities);g=sum(e['groundable'] for e in entities)
        dist.append({'population':population,'samples':len(idx),'entities':len(entities),'groundable':g,'ungroundable':len(entities)-g,**{f'type_{k}':c[k] for k in ['PER','LOC','ORG','MISC']}})
    write_csv(EXP/'results/evaluation_population_statistics.csv',dist)
    summary={'clean_indices_count':len(clean),'clean_definition':'Twitter-2015 source and normalized text absent from GMNER TRAIN+DEV','qualitative_category_counts':{k:len(v) for k,v in categories.items()}}
    (EXP/'results/protocol_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
