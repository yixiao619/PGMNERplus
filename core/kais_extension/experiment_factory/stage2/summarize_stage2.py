#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
import torch
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory";S=F/"stage2"
MECH=["gate_mean","gate_std","entity_gate_std","mass_mean","mass_std","transported_mass","unmatched_mass","matching_entropy","transport_concentration","reliability_std","residual_ratio"]
def write_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True);keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def load_status(path):return json.loads(path.read_text()) if path.exists() else {"status":"pending"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);matrix=json.loads((sub/"stage2_matrix.yaml").read_text());source=Path(matrix["source_stage1"]);rows=[];by_id={}
 for e in matrix["reused"]:
  status=load_status(source/"runs"/e["source_id"]/"status.json");item={**e,"source":"stage1_reused","source_id":e["source_id"],"status":status.get("status","missing")}
  item.update({k:status.get(k,"") for k in ["best_dev_f1","best_epoch","epochs","prediction_changes","task_gradient_norm","training_seconds","peak_gpu_memory_mb"]});mech=status.get("mechanism",{});item.update({k:mech.get(k,"") for k in MECH});rows.append(item);by_id[e["id"]]=item
 for j in matrix["jobs"]:
  status=load_status(sub/"runs"/j["id"]/"status.json");item={"id":j["id"],"group":j["group"],"seed":j["seed"],"role":j["role"],"source":"stage2_new","source_id":j["id"],"route":j["route"],"variant":j["variant"],"status":status.get("status","pending")}
  item.update({k:status.get(k,"") for k in ["best_dev_f1","best_epoch","epochs","prediction_changes","task_gradient_norm","training_seconds","peak_gpu_memory_mb"]});mech=status.get("mechanism",{});item.update({k:mech.get(k,"") for k in MECH});rows.append(item);by_id[j["id"]]=item
 pairs=[]
 for pair in matrix["pairs"]:
  t=by_id[pair["treatment"]];c=by_id[pair["control"]];done=t["status"]==c["status"]=="complete" and t["best_dev_f1"]!="" and c["best_dev_f1"]!="";pr={**pair,"status":"complete" if done else "pending","treatment_dev_f1":t["best_dev_f1"],"control_dev_f1":c["best_dev_f1"],"paired_delta":float(t["best_dev_f1"])-float(c["best_dev_f1"]) if done else "","treatment_prediction_changes":t["prediction_changes"],"control_prediction_changes":c["prediction_changes"],"prediction_difference":""}
  tp=sub/"runs"/t["source_id"]/"best_dev_predictions.pt";cp=sub/"runs"/c["source_id"]/"best_dev_predictions.pt"
  if tp.exists() and cp.exists():
   a1=torch.load(tp,map_location="cpu",weights_only=True)["predictions"];a2=torch.load(cp,map_location="cpu",weights_only=True)["predictions"];pr["prediction_difference"]=sum(x!=y for x,y in zip(a1,a2))
  pairs.append(pr)
 for r in rows:
  relevant=[x for x in pairs if r["id"] in (x["treatment"],x["control"]) and x["status"]=="complete"]
  r["matched_baseline_dev_f1"]=relevant[0]["control_dev_f1"] if relevant else "";r["paired_delta"]=relevant[0]["paired_delta"] if relevant and r["role"]=="treatment" else (0.0 if relevant else "")
 out=S/"results";write_csv(out/"stage2_summary.csv",rows);write_csv(out/"paired_comparisons.csv",pairs)
 complete_pairs=[x for x in pairs if x["status"]=="complete"];group_stats={}
 for g in "ABCD":
  vals=[float(x["paired_delta"]) for x in complete_pairs if x["group"]==g];group_stats[g]={"n":len(vals),"mean":statistics.mean(vals) if vals else None,"std":statistics.stdev(vals) if len(vals)>1 else (0.0 if vals else None),"positive":sum(v>0 for v in vals)}
 eligible=[]
 if group_stats["A"]["n"]==3 and group_stats["A"]["mean"]>0 and group_stats["A"]["positive"]>=2:eligible.append((group_stats["A"]["mean"],"P1B_KL_HALF"))
 if group_stats["B"]["n"]==2 and group_stats["B"]["mean"]>0 and group_stats["B"]["positive"]==2:eligible.append((group_stats["B"]["mean"],"G1_PARTIAL_OT"))
 if group_stats["C"]["n"]==1:
  t=by_id["C_RELIABILITY_S42"];p0=[x for x in complete_pairs if x["group"]=="C"][0]
  if group_stats["C"]["mean"]>0 and float(t.get("reliability_std") or 0)>1e-4 and float(t.get("task_gradient_norm") or 0)>0 and int(p0.get("prediction_difference") or 0)>0:eligible.append((group_stats["C"]["mean"],"G3_RELIABILITY_OT"))
 if group_stats["D"]["n"]==1:
  t=by_id["D_STABLE_S42"];p0=[x for x in complete_pairs if x["group"]=="D"][0]
  if group_stats["D"]["mean"]>0 and float(t.get("entity_gate_std") or 0)>1e-3 and float(t.get("residual_ratio") or 0)>1e-4 and int(p0.get("prediction_difference") or 0)>0:eligible.append((group_stats["D"]["mean"],"P3_STABLE_GATE"))
 recommended=[name for _,name in sorted(eligible,reverse=True)[:2]]
 protocol=[
  "Stage1 Full/P1/G1 used four maximum epochs and patience 2; G3/P3 used the same epoch budget but different trainable parameter scopes required by their route definitions.",
  "All Stage1 jobs used Twitter-2017 TRAIN/DEV, AdamW, module LR 1e-5, weight decay 1e-4, linear warmup/decay, no EMA, and the same paper checkpoint.",
  "Stage1 summary deltas used historical DEV 0.89462. This report discards those deltas and computes only within-seed matched deltas.",
  "Seed42 Full/P1B and Partial/Full are reused because each pair is internally matched. New controls use five maximum epochs and patience 2.",
  "G3 and P3 learned variants are rerun under the five-epoch Stage2 budget so their controls are exact training-budget matches.",
  "No TEST evaluation is performed."
 ]
 lines=["# Stage 2 Parallel Validation","","## Protocol Audit",*['- '+x for x in protocol],"","## Execution Status","",f"Completed pairs: {len(complete_pairs)}/{len(pairs)}.","","| Group | Seed | Contrast | Treatment DEV | Control DEV | Paired delta | Prediction difference | Status |","|---|---:|---|---:|---:|---:|---:|---|"]
 for x in pairs:lines.append(f"| {x['group']} | {x['seed']} | {x['contrast']} | {x['treatment_dev_f1']} | {x['control_dev_f1']} | {x['paired_delta']} | {x['prediction_difference']} | {x['status']} |")
 lines.extend(["","## Group Summary","","| Group | Complete N | Mean paired delta | Std | Positive |","|---|---:|---:|---:|---:|"])
 for g,s in group_stats.items():lines.append(f"| {g} | {s['n']} | {s['mean'] if s['mean'] is not None else ''} | {s['std'] if s['std'] is not None else ''} | {s['positive']} |")
 lines.extend(["","## Stage 3 Recommendation","",("Recommended candidates: "+", ".join(recommended)+"." if recommended else "No route is recommended until all required pairs complete and satisfy the predefined rules."),"","At most two candidates are listed; Stage 3 is not started automatically.","","## Limitations","","- Seed42 reused runs had a four-epoch cap, while new Stage2 runs have a five-epoch cap; every reported delta remains paired within an identical budget.","- Exact learned-vs-control prediction differences are available for newly matched G3/P3 runs. Reused Stage1 runs did not cache complete DEV prediction vectors.","- Short single-run effects are screening evidence, not final performance claims."])
 (S/"STAGE2_REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps({"completed_pairs":len(complete_pairs),"recommended":recommended},indent=2))
if __name__=="__main__":main()
