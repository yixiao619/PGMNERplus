#!/usr/bin/env python3
from __future__ import annotations
import json,os,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment23_prediction_coupling";M=json.loads((EXP/"configs/experiment_matrix.json").read_text())
def submission(name,jobs):
 sub=EXP/"runs"/f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 for d in (sub/"jobs",sub/"runs",sub/"logs"):d.mkdir(parents=True,exist_ok=True)
 for j in jobs:(sub/"jobs"/f"{j['id']}.json").write_text(json.dumps(j,indent=2)+"\n")
 return sub
def run_scheduler(sub):return subprocess.run([sys.executable,"-u",str(EXP/"scripts/scheduler.py"),"--submission",str(sub)]).returncode
def statuses(sub):
 return [json.loads(p.read_text()) for p in (sub/"runs").glob("*/status.json")]
def main():
 state={"status":"running","started_at":time.time(),"submissions":[],"selected_route":None};(EXP/"runner.pid").write_text(str(os.getpid())+"\n");(EXP/"run_state.json").write_text(json.dumps(state,indent=2)+"\n")
 try:
  jobs=[{**j,"stage":1} for j in M["jobs"]];s1=submission("stage1",jobs);state["submissions"].append(str(s1));(EXP/"run_state.json").write_text(json.dumps(state,indent=2)+"\n");rc=run_scheduler(s1)
  if rc:raise RuntimeError(f"stage1 scheduler rc={rc}")
  xs=statuses(s1);b=next((x for x in xs if x.get("route")=="baseline" and x.get("status")=="complete"),None);cands=[]
  if b:
   for x in xs:
    if x.get("route") in ("ot_fusion","entity_routing") and x.get("status")=="complete":
     delta=float(x["best_dev_f1"])-float(b["best_dev_f1"]);mech=float(x.get("task_gradient_norm",0))>0 and int(x.get("prediction_changes",0))>0 and .005<=float(x.get("residual_ratio",0))<=.15
     if delta>=float(M["stage2"]["qualification_delta"]) and mech:cands.append((delta,x))
  if cands:
   _,best=max(cands,key=lambda z:z[0]);route=best["route"];routes=["baseline",route] if route=="ot_fusion" else ["baseline","ot_fusion","entity_routing"];jobs=[]
   for seed in M["stage2"]["seeds"]:
    for r in routes:jobs.append({"id":f"E23_{r.upper()}_S{seed}","route":r,"seed":seed,"stage":2})
   s2=submission("stage2",jobs);state["selected_route"]=route;state["submissions"].append(str(s2));(EXP/"run_state.json").write_text(json.dumps(state,indent=2)+"\n");rc=run_scheduler(s2)
   if rc:raise RuntimeError(f"stage2 scheduler rc={rc}")
  state["status"]="complete";state["finished_at"]=time.time();(EXP/"run_state.json").write_text(json.dumps(state,indent=2)+"\n");subprocess.run([sys.executable,str(EXP/"scripts/summarize.py")],check=False)
 except BaseException as e:
  state["status"]="failed";state["error"]=repr(e);state["traceback"]=traceback.format_exc();state["finished_at"]=time.time();(EXP/"run_state.json").write_text(json.dumps(state,indent=2)+"\n");subprocess.run([sys.executable,str(EXP/"scripts/summarize.py")],check=False);raise
if __name__=="__main__":main()
