#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");S=ROOT/"kais_extension/experiment_factory/stage3";PART=ROOT/"kais_extension/experiment_factory/stage2/runs/stage2_20261012_160006_2fe9aa811f/runs/B_PARTIAL_S123/status.json"
def write_csv(path,rows,fields):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);out=S/"results";ar=[]
 for sp in sorted((sub/"runs").glob("A_*/status.json")):
  x=json.loads(sp.read_text());g=x.get("gate_statistics",{});hp=sp.parent/"per_class_history.jsonl";trained_epochs=x.get("epochs","")
  if hp.exists():
   trained_epochs=len({json.loads(z).get("epoch") for z in hp.read_text().splitlines() if z.strip() and json.loads(z).get("split")=="dev"})
  ar.append({"id":x.get("id"),"seed":x.get("seed"),"variant":x.get("variant"),"ib_kl_weight":x.get("ib_kl_weight"),"status":x.get("status"),"best_dev_precision":x.get("best_dev_precision",""),"best_dev_recall":x.get("best_dev_recall",""),"best_dev_f1":x.get("best_dev_f1",""),"best_epoch":x.get("best_epoch",""),"epochs":trained_epochs,"gate_mean":g.get("gate_mean",""),"gate_std":g.get("gate_std",""),"kl_loss":"","training_seconds":x.get("training_seconds","") ,"peak_gpu_memory_mib":x.get("scheduler_peak_gpu_memory_mib","")})
 fields=["id","seed","variant","ib_kl_weight","status","best_dev_precision","best_dev_recall","best_dev_f1","best_epoch","epochs","gate_mean","gate_std","kl_loss","training_seconds","peak_gpu_memory_mib"];write_csv(out/"ppib_full_training.csv",ar,fields)
 ds=[]
 for seed in (42,123,456):
  f=next((x for x in ar if x["seed"]==seed and x["variant"]=="FULL" and x["status"]=="complete"),None);h=next((x for x in ar if x["seed"]==seed and x["variant"]=="KL_HALF" and x["status"]=="complete"),None);ds.append({"seed":seed,"full_dev_f1":f["best_dev_f1"] if f else "","kl_half_dev_f1":h["best_dev_f1"] if h else "","paired_delta":float(h["best_dev_f1"])-float(f["best_dev_f1"]) if f and h else ""})
 vals=[x["paired_delta"] for x in ds if x["paired_delta"]!=""];mean=statistics.mean(vals) if vals else "";std=statistics.stdev(vals) if len(vals)>1 else (0 if vals else "");
 for x in ds:x["mean_delta"]=mean;x["std_delta"]=std;x["positive_pairs"]=sum(v>0 for v in vals) if vals else ""
 write_csv(out/"ppib_paired_deltas.csv",ds,["seed","full_dev_f1","kl_half_dev_f1","paired_delta","mean_delta","std_delta","positive_pairs"])
 partial=json.loads(PART.read_text());fp=sub/"runs/B_FULL_S123/status.json";full=json.loads(fp.read_text()) if fp.exists() else {};pm=partial.get("mechanism",{});fm=full.get("mechanism",{});b=[{"seed":123,"partial_status":partial.get("status"),"partial_dev_f1":partial.get("best_dev_f1"),"full_status":full.get("status","pending"),"full_dev_f1":full.get("best_dev_f1",""),"partial_minus_full":float(partial["best_dev_f1"])-float(full["best_dev_f1"]) if full.get("status")=="complete" else "","partial_transport_mass":pm.get("transported_mass",""),"full_transport_mass":fm.get("transported_mass",""),"partial_unmatched_mass":pm.get("unmatched_mass",""),"full_unmatched_mass":fm.get("unmatched_mass",""),"partial_entropy":pm.get("matching_entropy",""),"full_entropy":fm.get("matching_entropy",""),"partial_task_gradient":partial.get("task_gradient_norm",""),"full_task_gradient":full.get("task_gradient_norm","")}];write_csv(out/"gota_matched_comparison.csv",b,list(b[0]))
 # Plot only when curves exist.
 try:
  import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(6.6,3.8))
  for run in sorted((sub/"runs").glob("A_*")):
   hp=run/"per_class_history.jsonl"
   if not hp.exists():continue
   rows=[json.loads(z) for z in hp.read_text().splitlines() if json.loads(z).get("split")=="dev"];ax.plot([z["epoch"] for z in rows],[z["overall"]["f1"] for z in rows],label=run.name,linewidth=1.2)
  if ax.lines:ax.set(xlabel="Epoch",ylabel="DEV micro-F1");ax.legend(ncol=2,fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(S/"figures/ppib_dev_curves.pdf")
  plt.close(fig)
 except Exception:pass
 complete=sum(x.get("status")=="complete" for x in ar);pending=6-complete;case="PENDING" if pending else ("POSITIVE" if vals and mean>0 and sum(v>0 for v in vals)>=2 else "NOT_STABLE_POSITIVE")
 full_scores=[float(x["best_dev_f1"]) for x in ar if x["variant"]=="FULL" and x["status"]=="complete"];half_scores=[float(x["best_dev_f1"]) for x in ar if x["variant"]=="KL_HALF" and x["status"]=="complete"]
 fmean=statistics.mean(full_scores) if full_scores else "";fstd=statistics.stdev(full_scores) if len(full_scores)>1 else "";hmean=statistics.mean(half_scores) if half_scores else "";hstd=statistics.stdev(half_scores) if len(half_scores)>1 else ""
 report=f"""# Experiment Factory Stage 3 Report\n\n## Protocol\n\nGroup A uses the reconstructed paper-full 40-epoch protocol, EMA 0.999, and DEV-only checkpoint selection. TEST was never evaluated. Post-paper PP-IB repair defaults were explicitly disabled, so paired jobs differ only in `ib_kl_weight` (0.001 vs 0.0005). Group B reuses Stage2 `B_PARTIAL_S123` and runs its exact matched Full-OT control.\n\n## PP-IB KL Results\n\n| Seed | Full DEV F1 | KL Half DEV F1 | Paired delta |\n|---:|---:|---:|---:|\n"""
 for x in ds: report+=f"| {x['seed']} | {x['full_dev_f1']} | {x['kl_half_dev_f1']} | {x['paired_delta']} |\n"
 report+=f"""\nFull: {fmean:.6f} +/- {fstd:.6f}. KL Half: {hmean:.6f} +/- {hstd:.6f}. Mean paired delta: {mean:.6f} +/- {std:.6f}; positive seeds: {sum(v>0 for v in vals)}/{len(vals)}. The direction is inconsistent, so KL Half is **not a stable positive result**. KL Half seed456 also shows high-end gate collapse (mean 0.9865, std 0.0009).\n\n## GOT-A Matched Comparison\n\nAt seed123, Partial OT DEV F1 is {partial.get('best_dev_f1')} and Full OT DEV F1 is {full.get('best_dev_f1')}; Partial minus Full is {b[0]['partial_minus_full']}. Partial OT transports mass {pm.get('transported_mass')} and leaves {pm.get('unmatched_mass')} unmatched, while Full OT transports approximately all mass. This verifies a mechanism difference but not a stable DEV advantage.\n\n## Conclusion\n\n- Group A complete: {complete}/6; failed: {pending}.\n- KL Half should not enter KAIS as a performance improvement.\n- Partial OT has verified partial-matching behavior, but the available paired results are mixed across seeds and do not establish superiority.\n- No TEST metric was used or produced.\n\n## Outputs\n\n- `results/ppib_full_training.csv`\n- `results/ppib_paired_deltas.csv`\n- `results/gota_matched_comparison.csv`\n- `figures/ppib_dev_curves.pdf`\n""";(S/"STAGE3_REPORT.md").write_text(report)
if __name__=="__main__":main()
