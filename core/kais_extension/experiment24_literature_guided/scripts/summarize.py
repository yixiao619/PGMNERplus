#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment24_literature_guided";RES=EXP/"results"

def write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    matrix=json.loads((EXP/"configs/experiment_matrix.json").read_text());statuses=[]
    for j in matrix["jobs"]:
        p=EXP/"runs"/j["id"]/"status.json"
        x=json.loads(p.read_text()) if p.exists() else {"status":"pending","route":j["route"]}
        x["id"]=j["id"];statuses.append(x)
    summary=[];visual=[];align=[]
    for x in statuses:
        route=x.get("route")
        summary.append({"id":x["id"],"route":route,"dataset":x.get("dataset",""),"status":x.get("status"),"best_epoch":x.get("best_epoch",""),
                        "dev_f1":x.get("best_dev_f1","N/A"),"dev_grounding_proxy":x.get("best_dev_proxy","N/A"),"gradient_connectivity":x.get("gradient_norm",""),
                        "trainable_parameters":x.get("trainable_parameters",""),"gpu_time_seconds":x.get("gpu_time_seconds",""),"peak_gpu_memory_mb":x.get("peak_gpu_memory_mb",""),"test_evaluated":x.get("test_evaluated",False)})
        visual.append({"id":x["id"],"route":route,"clean_random_prediction_changes":x.get("clean_random_changed","N/A"),"clean_to_wrong":x.get("clean_random_c_to_w","N/A"),"wrong_to_correct":x.get("clean_random_w_to_c","N/A"),"note":"Batch-roll counterfactual on DEV" if route in ("A","C") else "N/A: separate GMNER grounding task"})
        align.append({"id":x["id"],"route":route,"entity_patch_metric_scope":"GMNER DEV" if route=="B" else "No bbox labels; N/A",
                      "patch_auroc":x.get("patch_auroc","N/A"),"patch_aupr":x.get("patch_aupr","N/A"),"pointing_accuracy":x.get("pointing_accuracy","N/A"),
                      "inside_mass":x.get("inside_mass","N/A"),"area_expectation":x.get("area_expectation","N/A"),"mass_lift":x.get("mass_lift","N/A"),
                      "dustbin_rejection_accuracy":x.get("dustbin_rejection_accuracy","N/A"),"transport_entropy":x.get("transport_entropy",x.get("normalized_entropy","N/A")),"transport_max":x.get("transport_max","N/A")})
    write(RES/"route_summary.csv",summary,list(summary[0]));write(RES/"visual_dependency.csv",visual,list(visual[0]));write(RES/"alignment_quality.csv",align,list(align[0]))
    rows="\n".join(f"| {r['id']} | {r['dataset']} | {r['status']} | {r['dev_f1']} | {r['dev_grounding_proxy']} | {r['gradient_connectivity']} | {r['gpu_time_seconds']} |" for r in summary)
    completed=sum(r["status"]=="complete" for r in summary)
    report=f"""# Experiment 24: Literature-Guided PP-IB and Partial OT Research

## Scope

This is a DEV-only, seed-42, at-most-five-epoch feasibility screen. No TEST labels or TEST metrics are accessed. Route B uses a different dataset and task and is not numerically compared with Routes A/C.

## Literature Basis

The exact formulas and transfer boundaries are documented in `LITERATURE_TO_CODE.md`, based on the official publications: AAAI 2024 PPOT, CVPR 2024 L2RM, Findings of EMNLP 2024 GEM, and LREC-COLING 2024 SMNER.

## Implemented Routes

- **A, Task-Guided OT Polarization:** entity labels supervise transport-derived visual evidence; a scale-aware PPOT-style penalty acts on normalized transport entries for gold entity spans.
- **B, Grounding-Supervised Partial OT:** GMNER bbox overlap supervises patch mass and bbox=None supervises a dustbin. This is a one-entity entropic partial-matching prototype, not a GEM reproduction.
- **C, Counterfactual Visual Alignment:** correct and deterministic within-batch wrong images are contrasted through the gold-class log-probability margin.

## Results

| Run | Dataset | Status | DEV MNER F1 | DEV grounding proxy | Signal gradient | GPU seconds |
|---|---|---|---:|---:|---:|---:|
{rows}

Completed routes: {completed}/3. Detailed mechanism metrics are in `results/visual_dependency.csv` and `results/alignment_quality.csv`.

## Interpretation Rules

For A/C, a route is useful only if it retains or improves official Twitter-2017 DEV F1 and exhibits a nonzero training gradient plus measurable clean/random behavior. Twitter-2017 has no bbox labels, so no grounding accuracy is reported. For B, useful evidence requires non-chance patch metrics and dustbin rejection on GMNER DEV; its proxy is not an MNER F1.

## Limitations

This screen is single-seed and short. Route C's wrong-image mapping is deterministic within each minibatch rather than a global dataset permutation. Route B uses frozen features cached by Experiment 10 and reduces entity-to-patch partial OT to a one-source entropic distribution with an explicit dustbin. Neither result is a complete 40-epoch performance claim.
"""
    (EXP/"EXPERIMENT_24_REPORT.md").write_text(report)
    print(json.dumps({"completed":completed,"statuses":{x["id"]:x.get("status") for x in statuses}},indent=2))

if __name__=="__main__":main()
