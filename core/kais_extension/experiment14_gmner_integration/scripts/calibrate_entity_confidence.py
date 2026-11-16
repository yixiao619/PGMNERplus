#!/usr/bin/env python3
"""Small DEV-only entity-confidence calibration for the Exp14 routed output."""
import csv, json, sys
from pathlib import Path

ROOT=Path('/home/jyx/yixiao/MINER-My')
EXP=ROOT/'kais_extension/experiment14_gmner_integration'
E2E=Path('/data/hfy/VG-Think/E2E-GMNER')
sys.path.insert(0,str(E2E/'src'))
from e2egmner.evaluation.metrics import count_correct_eeg,count_correct_gmner,count_correct_mner,prf
from e2egmner.evaluation.parsing import parse_triples,triples_to_canon_text

CANDIDATES=[0.0,0.80,0.90,0.95,0.98,0.99,0.995,0.998]

def load(split):
    return [json.loads(x) for x in (EXP/f'results/full_predictions_{split}.jsonl').open(encoding='utf-8')]

def filtered(row,tau):
    triples=parse_triples(row['groundability_aware'],strict=False,where='confidence_filter')
    details=row['entities_groundability_aware']
    if len(triples)!=len(details): raise RuntimeError('triple/detail length mismatch')
    return [t for t,d in zip(triples,details) if float(d['entity_confidence'])>=tau]

def score(rows,tau):
    counts={k:[0,0,0] for k in ['mner','eeg','gmner']}
    outputs=[]
    for row in rows:
        pred=filtered(row,tau); gold=parse_triples(row['gold'],strict=True,where='gold')
        outputs.append(pred)
        counts['mner'][0]+=count_correct_mner(pred,gold)
        counts['eeg'][0]+=count_correct_eeg(pred,gold,.5)
        counts['gmner'][0]+=count_correct_gmner(pred,gold,.5)
        for k in counts: counts[k][1]+=len(pred); counts[k][2]+=len(gold)
    out={'entity_confidence_threshold':tau}
    for k,(correct,predicted,gold) in counts.items():
        p,r,f=prf(correct,predicted,gold)
        out.update({f'{k}_precision':p,f'{k}_recall':r,f'{k}_f1':f,f'{k}_correct':correct,f'{k}_predicted':predicted,f'{k}_gold':gold})
    return out,outputs

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    dev=load('dev'); dev_rows=[score(dev,t)[0] for t in CANDIDATES]
    # Primary DEV GMNER F1; ties prefer lower filtering threshold.
    best=max(dev_rows,key=lambda r:(r['gmner_f1'],-r['entity_confidence_threshold']))
    write_csv(EXP/'results/entity_confidence_dev_calibration.csv',dev_rows)
    locked={'threshold':best['entity_confidence_threshold'],'selection':'max DEV official GMNER F1','candidates':CANDIDATES,'dev':best}
    (EXP/'results/locked_entity_confidence_threshold.json').write_text(json.dumps(locked,indent=2)+'\n')
    test=load('test'); test_row,outputs=score(test,best['entity_confidence_threshold'])
    write_csv(EXP/'results/confidence_filtered_test_metrics.csv',[test_row])
    with (EXP/'results/confidence_filtered_test_predictions.jsonl').open('w',encoding='utf-8') as f:
        for row,pred in zip(test,outputs):
            f.write(json.dumps({'sample_index':row['sample_index'],'sample_id':row['sample_id'],'prediction':triples_to_canon_text(pred)},ensure_ascii=False)+'\n')
    print(json.dumps({'locked':locked,'test':test_row},indent=2))

if __name__=='__main__':main()
