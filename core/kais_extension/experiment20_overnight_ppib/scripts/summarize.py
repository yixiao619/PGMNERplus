#!/usr/bin/env python3
"""Summarize every completed, failed, skipped, or pending Experiment 20 run."""
from __future__ import annotations
import csv, json
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]
if not (EXP/"configs").exists(): EXP=Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment20_overnight_ppib")
RES=EXP/"results"; FIG=EXP/"figures"; FIG.mkdir(exist_ok=True)
matrix=json.loads((EXP/"configs/experiment_matrix.json").read_text()); rows=[]
ids=[]
for c in matrix["experiments"]: ids.extend([c["id"],c["id"]+"_fixed"])
for run_id in ids:
    path=RES/"runs"/run_id/"status.json"
    if path.exists():
        d=json.loads(path.read_text()); cfg=d.get("config",{})
        rows.append({"id":run_id,"status":d.get("status"),"kind":cfg.get("kind"),"gate_mode":cfg.get("gate_mode"),"alpha":cfg.get("alpha"),
          "unfreeze_head":cfg.get("unfreeze_head"),"epochs":d.get("epochs_completed"),"best_dev_f1":d.get("best_dev_f1"),
          "delta_dev":d.get("delta_dev_vs_original"),"gate_mean":d.get("gate_mean"),"gate_std":d.get("gate_std"),
          "span_gate_variance":d.get("span_gate_variance"),"residual_ratio":d.get("residual_ratio"),
          "changed_predictions":d.get("changed_valid_span_predictions"),"gate_collapse":d.get("gate_collapse"),
          "matched_fixed_control":(RES/"runs"/(run_id.removesuffix("_fixed")+"_fixed")/"status.json").exists() if not run_id.endswith("_fixed") else True,
          "training_seconds":d.get("training_seconds"),"peak_gpu_memory_mb":d.get("peak_gpu_memory_mb"),"exit_reason":d.get("exit_reason",d.get("error"))})
completed=sorted([r for r in rows if r["status"]=="complete"],key=lambda x:x["best_dev_f1"],reverse=True)
others=[r for r in rows if r["status"]!="complete"]
rows=completed+others
fields=["id","status","kind","gate_mode","alpha","unfreeze_head","epochs","best_dev_f1","delta_dev","gate_mean","gate_std","span_gate_variance","residual_ratio","changed_predictions","gate_collapse","matched_fixed_control","training_seconds","peak_gpu_memory_mb","exit_reason"]
with (RES/"overnight_summary.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def fmt(v,n=5):
    try:return f"{float(v):.{n}f}"
    except:return "-"
lines=["# Experiment 20: Automated PP-IB Improvement Screening","",
 "Generated automatically from local artifacts. TEST was not evaluated. No online model or API was used.","",
 "## Protocol","",
 "- Initialization: historical paper-full seed42 checkpoint","- Selection: Twitter-2017 DEV only",
 "- Baseline DEV F1: 0.89462","- Maximum 5 epochs per run; patience 2","- Seed: 42","",
 "## Results","", "| ID | Status | Best DEV F1 | Delta | Gate SD | Span variance | Residual ratio | Changed spans | Collapse | Exit |",
 "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
for r in rows:
    lines.append(f"| {r['id']} | {r['status']} | {fmt(r['best_dev_f1'])} | {fmt(r['delta_dev'])} | {fmt(r['gate_std'],7)} | {fmt(r['span_gate_variance'],7)} | {fmt(r['residual_ratio'],6)} | {r['changed_predictions'] if r['changed_predictions'] is not None else '-'} | {r['gate_collapse'] or '-'} | {r['exit_reason'] or '-'} |")
lines += ["","## Automatic Screening Interpretation",""]
for r in completed:
    if r["best_dev_f1"]<.89462: verdict="ELIMINATE: DEV below baseline."
    elif abs(r["best_dev_f1"]-.89462)<1e-12 and (r["changed_predictions"] or 0)==0 and (r["gate_std"] or 0)<1e-4: verdict="ELIMINATE: tied, constant gate, unchanged predictions."
    elif abs(r["best_dev_f1"]-.89462)<1e-12: verdict="CANDIDATE_DIAGNOSTIC_ONLY: tied but mechanism/predictions changed."
    elif r["best_dev_f1"]>.89462: verdict="CANDIDATE: DEV improvement; inspect matched fixed control before gate attribution."
    else: verdict="REVIEW."
    lines.append(f"- **{r['id']}**: {verdict}")
lines += ["","## Integrity Notes","","All complete and failed runs are retained. Fixed controls are newly trained controls. No TEST metric was used for selection. Original code and checkpoint were not modified.",""]
(EXP/"OVERNIGHT_REPORT.md").write_text("\n".join(lines))

try:
    import matplotlib.pyplot as plt
    valid=completed[::-1]
    if valid:
        plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
        fig,ax=plt.subplots(figsize=(7,4)); ax.barh([r["id"] for r in valid],[100*r["best_dev_f1"] for r in valid],color="#2364AA")
        ax.axvline(89.462,color="#D1495B",ls="--",label="Original DEV"); ax.set_xlabel("DEV F1 (%)"); ax.legend(frameon=False); fig.tight_layout()
        for ext in ("pdf","png"): fig.savefig(FIG/f"fig20_dev_screening.{ext}",dpi=300,bbox_inches="tight")
        plt.close(fig)
except Exception as exc:
    (RES/"figure_generation_warning.txt").write_text(repr(exc)+"\n")
