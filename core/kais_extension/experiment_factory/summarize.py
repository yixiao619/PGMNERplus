#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
F=Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory")
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);snap=json.loads((sub/"submission.json").read_text());rows=[]
 for jp in sorted((sub/"jobs").glob("*.json")):
  j=json.loads(jp.read_text());sp=sub/"runs"/j["id"]/"status.json";s=json.loads(sp.read_text()) if sp.exists() else {"status":"pending"};mech=s.get("mechanism",{});controls=j.get("control_for",[]);status=s.get("status");dev=s.get("best_dev_f1","");changes=s.get("prediction_changes","");collapse=s.get("gate_collapse",False)
  decision=status
  if status=="complete" and dev!="":
   delta=float(dev)-float(snap["baseline_dev_f1"]);changed=int(changes or 0)
   if collapse:decision="ELIMINATE_GATE_COLLAPSE"
   elif delta<0 and changed==0:decision="ELIMINATE_NO_MECHANISM_OR_DEV_GAIN"
   elif abs(delta)<1e-8 and changed==0:decision="ELIMINATE_INVARIANT"
   elif delta>0:decision="CANDIDATE_REQUIRES_MATCHED_CONTROL" if not controls else "CONTROL_RESULT"
   else:decision="MECHANISM_ONLY_CANDIDATE"
  rows.append({"submission":snap["submission_id"],"id":j["id"],"route":j["route"],"variant":j["variant"],"stage":j["stage"],"status":status,"screening_decision":decision,"best_dev_f1":dev,"delta_dev":s.get("delta_dev_vs_baseline",""),"best_epoch":s.get("best_epoch",""),"prediction_changes":changes,"task_gradient_norm":s.get("task_gradient_norm",""),"gate_collapse":collapse,"gate_std":mech.get("gate_std",""),"entity_gate_std":mech.get("entity_gate_std",""),"mass_std":mech.get("mass_std",""),"reliability_std":mech.get("reliability_std",""),"residual_ratio":mech.get("residual_ratio",""),"training_seconds":s.get("training_seconds",""),"peak_gpu_memory_mb":s.get("peak_gpu_memory_mb",""),"control_for":";".join(controls),"error":s.get("error",""),"test_evaluated":s.get("test_evaluated",False)})
 out=F/"results";out.mkdir(exist_ok=True);fields=list(rows[0]) if rows else []
 with (out/"summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 complete=[r.copy() for r in rows if r["status"]=="complete" and r["best_dev_f1"]!=""];rank=sorted(complete,key=lambda r:float(r["best_dev_f1"]),reverse=True)
 for i,r in enumerate(rank,1):r["rank"]=i
 if rank:
  with (out/"route_ranking.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=["rank"]+fields);w.writeheader();w.writerows(rank)
 lines=["# Factory V2 DEV Summary","","Single-seed short runs are screening evidence only. TEST is never evaluated.","","| ID | Route | Status | Decision | Best DEV F1 | Delta | Changes | Control |","|---|---|---|---|---:|---:|---:|---|"]
 for r in rows:lines.append(f"| {r['id']} | {r['route']} | {r['status']} | {r['screening_decision']} | {r['best_dev_f1']} | {r['delta_dev']} | {r['prediction_changes']} | {r['control_for']} |")
 (out/"summary.md").write_text("\n".join(lines)+"\n")
 if rank:
  fig,ax=plt.subplots(figsize=(max(7,len(rank)*.55),4));ax.bar([r["id"] for r in rank],[float(r["best_dev_f1"]) for r in rank],color="#4472C4");ax.axhline(float(snap["baseline_dev_f1"]),ls="--",color="#C44E52",label="Original DEV baseline");ax.set_ylabel("Best DEV micro-F1");ax.tick_params(axis="x",rotation=55);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"dev_f1_comparison.pdf");fig.savefig(out/"dev_f1_comparison.png",dpi=300);plt.close(fig)
 print(f"wrote {out/'summary.csv'}")
if __name__=="__main__":main()
