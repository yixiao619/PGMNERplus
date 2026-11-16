#!/usr/bin/env python3
from __future__ import annotations
import json,time
from datetime import datetime
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory";S=F/"stage3"
def main():
 cfg=json.loads((S/"configs/stage3_matrix.json").read_text());stamp=datetime.now().strftime("stage3_%Y%m%d_%H%M%S");sub=S/"runs"/stamp
 for d in (sub/"jobs",sub/"runs",sub/"logs"):d.mkdir(parents=True,exist_ok=True)
 jobs=[{"id":"B_FULL_S123","group":"B","runner":"factory","seed":123,"route":"G1","variant":"full_ot","ot_backend":"full","stage":2,"expected_peak_mib":18432,"timeout_seconds":21600,"max_attempts":2}]
 for seed in cfg["group_a"]["seeds"]:
  for variant,w in (("FULL",cfg["group_a"]["original_kl_weight"]),("KL_HALF",cfg["group_a"]["half_kl_weight"])):jobs.append({"id":f"A_{variant}_S{seed}","group":"A","runner":"formal","seed":seed,"variant":variant,"ib_kl_weight":w,"expected_peak_mib":18432,"timeout_seconds":43200,"max_attempts":1})
 for j in jobs:(sub/"jobs"/f"{j['id']}.json").write_text(json.dumps(j,indent=2)+"\n")
 (sub/"gpu_policy.json").write_text((S/"configs/gpu_policy.json").read_text())
 snap={"schema_version":1,"submission_id":stamp,"created_at":time.time(),"requested_stage":3,"custom_summary":True,"baseline_dev_f1":.89462,"seed":42,"checkpoint":"/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth","stage2":{"max_epochs":5,"patience":2,"timeout_seconds":21600},"optimizer":{"new_lr":1e-4,"module_lr":1e-5,"head_lr":5e-6,"backbone_lr":1e-6,"weight_decay":1e-4,"warmup_ratio":.1},"test_evaluation":False,"formal_protocol":{"epochs":40,"batch_size":16,"lr":3e-5,"bert_lr":1e-5,"weight_decay":.01,"warmup_ratio":.1,"ema_decay":.999,"selection":"dev","do_predict":False,"post_paper_ppib_repairs_disabled":True},"jobs":[j["id"] for j in jobs]};(sub/"submission.json").write_text(json.dumps(snap,indent=2)+"\n")
 audits=[]
 for seed in cfg["group_a"]["seeds"]:
  a=next(x for x in jobs if x["id"]==f"A_FULL_S{seed}");b=next(x for x in jobs if x["id"]==f"A_KL_HALF_S{seed}");diff=[k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)];audits.append({"seed":seed,"different_fields":diff,"valid":diff==["ib_kl_weight","id","variant"]})
 (sub/"pair_config_audit.json").write_text(json.dumps(audits,indent=2)+"\n");(S/"latest_submission.txt").write_text(str(sub)+"\n");print(sub)
if __name__=="__main__":main()
