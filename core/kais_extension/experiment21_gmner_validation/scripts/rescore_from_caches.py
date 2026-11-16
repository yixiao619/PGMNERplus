#!/usr/bin/env python3
"""Re-score Exp21 directly from Exp14 entity/Qwen caches, avoiding canonical round trips."""
import csv,html,importlib.util,json,re,sys
from pathlib import Path
ROOT=Path('/home/jyx/yixiao/MINER-My');EXP=ROOT/'kais_extension/experiment21_gmner_validation';R14=ROOT/'kais_extension/experiment14_gmner_integration/results'
E2E=Path('/data/hfy/VG-Think/E2E-GMNER');sys.path.insert(0,str(E2E/'src'))
from e2egmner.evaluation.metrics import count_correct_eeg,count_correct_gmner,count_correct_mner,prf
from e2egmner.evaluation.parsing import parse_triples
spec=importlib.util.spec_from_file_location('exp14eval',ROOT/'kais_extension/experiment14_gmner_integration/scripts/evaluate_integration.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def read(path):return [json.loads(x) for x in Path(path).open(encoding='utf-8') if x.strip()]
def norm(s):return re.sub(r'\s+',' ',html.unescape(s).lower().replace('&amp;','&')).strip()
def score(preds,golds,idx):
 out={}
 for name,fn in [('mner',count_correct_mner),('eeg',count_correct_eeg),('gmner',count_correct_gmner)]:
  c=p=g=0
  for i in idx:c+=fn(preds[i],golds[i]) if name=='mner' else fn(preds[i],golds[i],.5);p+=len(preds[i]);g+=len(golds[i])
  pp,rr,ff=prf(c,p,g);out.update({f'{name}_precision':pp,f'{name}_recall':rr,f'{name}_f1':ff,f'{name}_correct':c,f'{name}_predicted':p,f'{name}_gold':g})
 return out
def write(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
pg=read(R14/'pgmner_predicted_entities_test.jsonl');q={x['key']:x for x in read(R14/'qwen_q2_cache_test.jsonl')};sft=read(E2E/'data/sft/test_sft.jsonl')
golds=[m.scaled_gold(x,i) for i,x in enumerate(sft)];direct=[];gated=[]
for sample in pg:
 direct.append(m.predictions(sample,q,None)[0]);gated.append(m.predictions(sample,q,.5)[0])
canon=read(R14/'full_predictions_test.jsonl');filt=read(R14/'confidence_filtered_test_predictions.jsonl');confidence=[parse_triples(x['prediction'],strict=False,where=f'confidence:{i}') for i,x in enumerate(filt)]
side=read(ROOT/'kais_extension/experiment9_gmner_audit/cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl');train_dev={norm(x['text']) for x in side if x['split'] in ('train','dev')};test=[x for x in side if x['split']=='test']
clean=[i for i,x in enumerate(test) if 'twitter2015' in x['image_rel'].lower() and norm(x['text']) not in train_dev];full=list(range(1500))
predsets=[('PGMNER+Qwen',direct),('PGMNER+Groundability+Qwen',gated),('PGMNER+Groundability+Qwen+DEVConfidence',confidence)]
rows=[]
for pop,idx,status in [('full_mixed_test',full,'protocol_limited_overlap'),('source_disjoint_twitter2015',clean,'credible_cross_dataset_external_validation')]:
 for label,preds in predsets:
  changed=sum(m.triples_to_canon_text(preds[i])!=m.triples_to_canon_text(direct[i]) for i in idx)
  rows.append({'population':pop,'protocol_status':status,'samples':len(idx),'method':label,'prediction_changed_samples_vs_direct':changed,**score(preds,golds,idx)})
write(EXP/'results/official_gmner_metrics.csv',rows)

breakdown=[]
for pop,idx in [('full_mixed_test',full),('source_disjoint_twitter2015',clean)]:
 for label,preds in predsets[:2]:
  stats={g:{'n':0,'pred_visual':0,'correct_decision':0,'iou_hit':0} for g in ('groundable','ungroundable')}
  for i in idx:
   for pi,gi in m.match_pred_to_gold(preds[i],golds[i]):
    gt=golds[i][gi];pd=preds[i][pi];group='groundable' if gt.regions is not None else 'ungroundable';z=stats[group];z['n']+=1;visual=pd.regions is not None;z['pred_visual']+=int(visual);z['correct_decision']+=int(visual==(gt.regions is not None));z['iou_hit']+=int(group=='groundable' and visual and m.max_iou(pd.regions,gt.regions)>=.5)
  for group,z in stats.items():breakdown.append({'population':pop,'method':label,'gold_group':group,**z,'groundability_decision_accuracy':z['correct_decision']/max(z['n'],1),'false_grounding_rate':z['pred_visual']/max(z['n'],1) if group=='ungroundable' else '','groundable_acc_iou_05':z['iou_hit']/max(z['n'],1) if group=='groundable' else ''})
write(EXP/'results/groundability_breakdown.csv',breakdown)
print(json.dumps({'full_direct':score(direct,golds,full),'full_gated':score(gated,golds,full),'clean_n':len(clean)},indent=2))
