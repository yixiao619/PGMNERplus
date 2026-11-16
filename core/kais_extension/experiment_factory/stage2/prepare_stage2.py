#!/usr/bin/env python3
from __future__ import annotations
import datetime,hashlib,json,shutil
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory";S=F/"stage2"
def main():
 matrix=json.loads((S/"configs/stage2_matrix.yaml").read_text());policy=json.loads((F/"configs/gpu_policy.json").read_text());policy["global_max_concurrency"]=2
 payload={"matrix":matrix,"policy":policy};digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:10];stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S");sub=S/"runs"/f"stage2_{stamp}_{digest}"
 for p in (sub/"jobs",sub/"runs",sub/"logs"):p.mkdir(parents=True,exist_ok=False)
 snapshot={"schema_version":2,"submission_id":sub.name,"submitted_at":datetime.datetime.now().isoformat(),"requested_stage":2,"custom_summary":True,"baseline_dev_f1":0.89482,"seed":42,"dataset":"twitter2017","checkpoint":matrix["checkpoint"],"stage1":matrix["training"],"stage2":matrix["training"],"stage3":{"enable_full_training":False},"optimizer":matrix["optimizer"],"selected_jobs":[j["id"] for j in matrix["jobs"]],"matrix_sha256":digest,"test_evaluation":False}
 (sub/"submission.json").write_text(json.dumps(snapshot,indent=2)+"\n");(sub/"gpu_policy.json").write_text(json.dumps(policy,indent=2)+"\n");shutil.copy2(S/"configs/stage2_matrix.yaml",sub/"stage2_matrix.yaml")
 for j in matrix["jobs"]:(sub/"jobs"/f"{j['id']}.json").write_text(json.dumps(j,indent=2)+"\n")
 (S/"latest_submission.txt").write_text(str(sub)+"\n");print(sub)
if __name__=="__main__":main()
