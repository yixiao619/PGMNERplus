#!/usr/bin/env python3
from __future__ import annotations
import argparse,datetime,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory"
def args():
 p=argparse.ArgumentParser();p.add_argument("--stage",type=int,choices=[0,1,2,3],default=1);p.add_argument("--routes",nargs="*",default=[]);p.add_argument("--source-submission");p.add_argument("--dry-run",action="store_true");p.add_argument("--allow-overlap-exp22",action="store_true");return p.parse_args()
def main():
 a=args();routes=json.loads((F/"configs/routes.json").read_text());gpu=json.loads((F/"configs/gpu_policy.json").read_text())
 if a.stage==3 and not routes["stage3"]["enable_full_training"]:raise SystemExit("Stage 3 disabled in immutable source config")
 jobs=[j for j in routes["jobs"] if j.get("enabled",True) and (j["stage"]==a.stage or (a.stage==0 and j["stage"]==1)) and (not a.routes or j["route"] in a.routes or j["id"] in a.routes)]
 if a.stage==2:
  source=Path(a.source_submission) if a.source_submission else None
  if source is None:
   candidates=sorted((F/"runs/submissions").glob("stage1_*"))
   source=candidates[-1] if candidates else None
  if source is None:raise SystemExit("Stage 2 requires a completed Stage-1 submission")
  eligible=[]
  for sp in (source/"runs").glob("*/status.json"):
   s=json.loads(sp.read_text())
   if s.get("status")=="complete" and (float(s.get("best_dev_f1",0))>float(routes["baseline_dev_f1"]) or (int(s.get("prediction_changes",0))>0 and float(s.get("best_dev_f1",0))>=float(routes["baseline_dev_f1"]))):eligible.append(s)
  eligible=sorted(eligible,key=lambda s:float(s.get("best_dev_f1",0)),reverse=True)[:int(routes["stage2"]["max_candidates"])]
  ids={s["id"] for s in eligible};jobs=[j for j in jobs if ids.intersection(j.get("control_for",[]))]
  if not jobs:raise SystemExit(f"No Stage-1 candidate in {source} met the predefined Stage-2 promotion rule")
 exp22=subprocess.run(["pgrep","-f","experiment22_module_analysis.*(main-mner|run_kl_short_screen)"],capture_output=True,text=True).stdout.strip()
 if a.stage>0 and exp22 and not a.allow_overlap_exp22 and not a.dry_run:raise SystemExit(f"Exp22 still running (PID(s) {exp22}); submission not started")
 stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S");digest=hashlib.sha256(json.dumps({"routes":routes,"gpu":gpu,"stage":a.stage,"jobs":jobs},sort_keys=True).encode()).hexdigest()[:10];sid=f"stage{a.stage}_{stamp}_{digest}";sub=F/"runs"/"submissions"/sid
 payload={**routes,"submission_id":sid,"submitted_at":datetime.datetime.now().isoformat(),"requested_stage":a.stage,"selected_jobs":[j["id"] for j in jobs],"source_config_sha256":digest,"exp22_running_at_submit":bool(exp22)}
 print(json.dumps({"submission":str(sub),"jobs":[j["id"] for j in jobs],"gpu_policy":gpu},indent=2))
 if a.dry_run:return
 (sub/"jobs").mkdir(parents=True);(sub/"runs").mkdir();(sub/"logs").mkdir();(sub/"submission.json").write_text(json.dumps(payload,indent=2)+"\n");(sub/"gpu_policy.json").write_text(json.dumps(gpu,indent=2)+"\n")
 for j in jobs:(sub/"jobs"/f"{j['id']}.json").write_text(json.dumps(j,indent=2)+"\n")
 (F/"runs/latest_submission.txt").write_text(str(sub)+"\n");print(f"Created immutable submission: {sub}")
if __name__=="__main__":main()
