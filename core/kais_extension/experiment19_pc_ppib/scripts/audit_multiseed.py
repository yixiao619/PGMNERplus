#!/usr/bin/env python3
import csv,json,re
from pathlib import Path
ROOT=Path('/home/jyx/yixiao/MINER-My'); EXP=ROOT/'kais_extension/experiment19_pc_ppib'; RES=EXP/'results';RES.mkdir(parents=True,exist_ok=True)
base={'dataset':'Twitter-2017','configuration':'dc10_m05 paper-full','epochs':'40','batch_size':'16','optimizer':'AdamW','lr':'3e-5','bert_lr':'1e-5','scheduler':'linear warmup_ratio=0.1','ema':'True (0.999)','data_model_match':'YES'}
rows=[]
for seed in [42,123,456]:
 d=Path(f'/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed{seed}')
 row={'seed':seed,**base,'best_dev_f1':'','test_precision':'','test_recall':'','test_f1':'','checkpoint_selection':'historical TEST span F1','checkpoint_available':str((d/'best_model.pth').exists()).upper(),'artifact_status':'AVAILABLE' if d.exists() else 'DELETED','evidence_path':str(d)}
 if seed==42:
  row.update(best_dev_f1='0.89462',test_precision='0.89685',test_recall='0.91629',test_f1='0.90647',artifact_status='COMPLETE; strict evaluation reproduced')
 else:
  row['artifact_status']='RUN CONFIRMED BY launch scripts and cleanup log; metrics/checkpoint unavailable after 2026-05-31 deletion'
 rows.append(row)
fields=list(rows[0]);
with (RES/'original_multiseed_audit.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
summary={'recoverable_numeric_seeds':[42],'confirmed_but_deleted_seeds':[123,456],'mean_test_f1':0.90647,'std_test_f1':None,'status':'MULTISEED STATISTICS UNAVAILABLE','limitation':'Only seed42 numeric artifact survives and checkpoint was selected using TEST F1; it is not an unbiased generalization estimate.','sources':['artifacts/ablation_twitter2017/paper/pgmner_full_dc10_m05.json','scripts/run_d1_dc10_m05_seed123.sh','scripts/run_d1_dc10_m05_seed456.sh','logs/disk_cleanup_20260531_103752.log']}
(RES/'original_multiseed_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
