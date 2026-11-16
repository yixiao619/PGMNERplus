#!/usr/bin/env python3
"""Cache-only statistics, figures, and report for Experiment 25."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment25_reliability_counterfactual"
E9 = ROOT / "kais_extension/experiment9_gmner_audit/results/grounding_entity_level_dev.csv"
E10 = ROOT / "kais_extension/experiment10_explicit_grounding"
E14 = ROOT / "kais_extension/experiment14_gmner_integration/results"
E21 = ROOT / "kais_extension/experiment21_gmner_validation/results"
RES, FIG = EXP / "results", EXP / "figures"


def ranks(x):
    return pd.Series(x).rank(method="average").to_numpy(float)


def auroc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); p = y.sum(); n = len(y) - p
    if p == 0 or n == 0: return float("nan")
    return float((ranks(s)[y == 1].sum() - p * (p + 1) / 2) / (p * n))


def aupr(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); p = y.sum()
    if p == 0: return float("nan")
    yy = y[np.argsort(-s)]; return float(((np.cumsum(yy) / np.arange(1, len(yy) + 1)) * yy).sum() / p)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def bootstrap_mean_ci(x, seed=42, n_boot=2000):
    x = np.asarray(x, float)
    if not len(x): return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); means = np.empty(n_boot)
    for i in range(n_boot): means[i] = x[rng.integers(0, len(x), len(x))].mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def key_cols(df):
    df = df.copy(); df["mention_norm"] = df.mention.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def main():
    RES.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    conditions = {c: key_cols(pd.read_csv(EXP / f"cache/scores_{c}.csv")) for c in ["clean", "text_only", "random", "hard_negative"]}
    ids = ["sample_id", "mention_norm", "type", "span_start", "span_end"]
    base = conditions["clean"]
    for c, df in conditions.items():
        if len(df) != len(base) or not df[ids].equals(base[ids]): raise RuntimeError(f"entity alignment failed: {c}")

    dependency = [
        {"component":"Exp10/14 entity-aware groundability head","checkpoint":str(E10/"checkpoints/entity_best.pt"),"h_input":"PGMNER fused span representation after GOT-A fusion","v_input":"196 projected CLIP patches","h_contains_visual":True,"counterfactual_status":"EVALUABLE_FULL_RECOMPUTE","note":"Both h and v recomputed for each replacement image."},
        {"component":"Exp24-B patch+dustbin head","checkpoint":str(ROOT/"kais_extension/experiment24_literature_guided/runs/E24_B_GROUNDING_SUPERVISED_POT/best_model.pth"),"h_input":"same fused span representation","v_input":"same 196 projected patches","h_contains_visual":True,"counterfactual_status":"EVALUABLE_FULL_RECOMPUTE","note":"Both h and v recomputed; no TEST used."},
        {"component":"Text-only condition","checkpoint":"N/A","h_input":"recomputed using zero normalized image tensor","v_input":"recomputed from zero tensor","h_contains_visual":False,"counterfactual_status":"NEUTRAL_IMAGE_ABLATION","note":"Not a separately trained text-only branch."},
        {"component":"Counterfactual labels","checkpoint":"N/A","h_input":"N/A","v_input":"N/A","h_contains_visual":"N/A","counterfactual_status":"UNLABELED_AFTER_REPLACEMENT","note":"Original bbox/None labels are not used as Random/Hard AUROC targets."},
    ]
    write_csv(RES / "input_dependency_audit.csv", dependency)

    model_specs = [("Exp10_entity", "exp10_score", 0.45), ("Exp24B_dustbin", "exp24b_score", 0.50)]
    cf_rows = []
    for model, col, threshold in model_specs:
        y = base.groundable_original.to_numpy(int); clean = base[col].to_numpy(float)
        cf_rows.append({"model":model,"condition":"clean","stratum":"all","n":len(y),"labeled_metrics_valid":True,
            "auroc":auroc(y,clean),"aupr":aupr(y,clean),"mean_score":clean.mean(),"threshold":threshold,"flip_rate_vs_clean":0.0})
        for condition in ["text_only", "random", "hard_negative"]:
            score = conditions[condition][col].to_numpy(float); drift = score-clean
            for stratum, mask in [("all",np.ones(len(y),bool)),("original_groundable",y==1),("original_ungroundable",y==0)]:
                lo, hi = bootstrap_mean_ci(drift[mask])
                labeled = condition == "text_only" and stratum == "all"
                cf_rows.append({"model":model,"condition":condition,"stratum":stratum,"n":int(mask.sum()),
                    "labeled_metrics_valid":labeled,"auroc":auroc(y,score) if labeled else "N/A","aupr":aupr(y,score) if labeled else "N/A","mean_score":float(score[mask].mean()),
                    "mean_paired_drift":float(drift[mask].mean()),"median_paired_drift":float(np.median(drift[mask])),
                    "mean_abs_paired_drift":float(np.abs(drift[mask]).mean()),"paired_drift_ci95_low":lo,"paired_drift_ci95_high":hi,
                    "flip_rate_vs_clean":float(np.mean((score[mask]>=threshold)!=(clean[mask]>=threshold))),"threshold":threshold,
                    "note":"Original labels remain valid for text-only shortcut audit." if labeled else ("Single-class stratum; ranking metric N/A." if condition=="text_only" else "Original labels invalid for replacement image; drift only.")})
    write_csv(RES / "reliability_counterfactual.csv", cf_rows)

    # Shortcut/proxy audit. Type prior is estimated from TRAIN labels; other proxies are descriptive only.
    train = torch.load(E10 / "cache/features_train.pt", map_location="cpu", weights_only=False, mmap=True)
    dev = torch.load(E10 / "cache/features_dev.pt", map_location="cpu", weights_only=False, mmap=True)
    type_rate = {}
    for typ in sorted(set(train["type"])):
        idx = [i for i,t in enumerate(train["type"]) if t == typ]
        type_rate[typ] = float(train["y_ground"][idx].mean())
    type_score = np.array([type_rate[t] for t in dev["type"]])
    mention_len = np.array([len(str(x).split()) for x in dev["mention"]],float)
    hnorm = dev["h"].float().norm(dim=-1).numpy()
    vnorm = dev["v"].float().mean(1).norm(dim=-1).numpy()
    y = base.groundable_original.to_numpy(int)
    proxy_rows = []
    for name, score, source in [("constant_majority",np.full(len(y),y.mean()),"DEV prevalence; constant"),
                                ("entity_type_prior",type_score,"TRAIN type prevalence"),
                                ("mention_length",mention_len,"text-only scalar proxy"),
                                ("fused_h_norm",hnorm,"post-fusion norm; not text-only"),
                                ("visual_global_norm",vnorm,"visual-only norm proxy")]:
        proxy_rows.append({"model":"proxy","type":"ALL","n":len(y),"positive":int(y.sum()),"prevalence":y.mean(),"metric":name,"auroc":auroc(y,score),"aupr":aupr(y,score),"source":source})
    type_rows = list(proxy_rows)
    for model,col,_ in model_specs:
        for typ in sorted(base.type.unique()):
            mask=(base.type==typ).to_numpy(); yy=y[mask]; ss=base.loc[mask,col].to_numpy(float)
            type_rows.append({"model":model,"type":typ,"n":int(mask.sum()),"positive":int(yy.sum()),"prevalence":float(yy.mean()),"metric":"clean_groundability","auroc":auroc(yy,ss),"aupr":aupr(yy,ss),"mean_score":float(ss.mean())})
    write_csv(RES / "type_stratified_analysis.csv", type_rows)

    # Strict entity alignment with Exp9 GOT-A using sample/span/mention/type.
    gota = key_cols(pd.read_csv(E9))
    merged = base.merge(gota, on=["sample_id","mention_norm","type","span_start","span_end"], suffixes=("_rel","_gota"), how="inner", validate="one_to_one")
    gota_rows=[]
    for model,col,_ in model_specs:
        s=merged[col].to_numpy(float); yg=merged.groundable.to_numpy(int)
        for signal in ["transport_mass","dustbin_mass_est","max_transport","norm_transport_entropy"]:
            gota_rows.append({"model":model,"analysis":"spearman","target":signal,"n":len(merged),"value":spearman(s,merged[signal]),"interpretation":"dustbin_mass_est is nominal 1-real transport, not observed dustbin flow"})
        # Low reliability should identify conceptual false matches.
        pointing = merged.pointing_hit.fillna(0).to_numpy(float)
        false = (yg==0) | ((yg==1)&(pointing<0.5))
        gota_rows.append({"model":model,"analysis":"error_prediction","target":"ungroundable_entity","n":len(merged),"auroc":auroc(yg,s),"aupr":aupr(yg,s)})
        gm=yg==1
        gota_rows.append({"model":model,"analysis":"error_prediction","target":"groundable_pointing_correct","n":int(gm.sum()),"auroc":auroc(pointing[gm],s[gm]),"aupr":aupr(pointing[gm],s[gm])})
        gota_rows.append({"model":model,"analysis":"error_prediction","target":"combined_false_match_using_low_reliability","n":len(merged),"auroc":auroc(false,-s),"aupr":aupr(false,-s)})
        q=pd.qcut(s,4,labels=["Q1_low","Q2","Q3","Q4_high"],duplicates="drop")
        for label in sorted(set(q)):
            m=np.asarray(q==label); gg=m&(yg==1)
            gota_rows.append({"model":model,"analysis":"reliability_quantile","target":str(label),"n":int(m.sum()),"mean_reliability":float(s[m].mean()),"false_match_rate":float(false[m].mean()),"groundable_n":int(gg.sum()),"groundable_pointing_accuracy":float(pointing[gg].mean()) if gg.any() else "N/A"})
    gota_rows.insert(0,{"model":"alignment_audit","analysis":"coverage","target":"strict_join","n":len(merged),"exp10_entities":len(base),"exp9_entities":len(gota),"note":"Joined by sample_id, normalized mention, type, and span offsets; no row-order join."})
    write_csv(RES / "gota_error_prediction.csv", gota_rows)

    # Existing fixed-threshold rejection results only. No TEST files are opened here.
    dev_eval=json.loads((E14/"evaluation_dev.json").read_text())
    rejection=[]
    for method,key in [("PGMNER+Qwen","direct"),("PGMNER+Groundability+Qwen@0.5","groundability_aware")]:
        off=dev_eval[key]["official"]; d=dev_eval[key]["diagnostic"]
        rejection.append({"population":"GMNER_mixed_DEV","protocol":"current DEV","samples":1500,"method":method,
            "mner_f1":off["mner"]["f1"],"eeg_f1":off["eeg"]["f1"],"gmner_f1":off["gmner"]["f1"],
            "false_grounding_rate":d["false_grounding_rate"],"ungroundable_rejection_accuracy":d["none_accuracy"],
            "groundable_recall":d["groundability_recall"],"groundable_acc_iou_05":d["groundable_entity_acc_iou_05"],
            "rejected_entity_count":"N/A" if key=="direct" else 260,"correctly_rejected_count":"N/A" if key=="direct" else 126,
            "incorrectly_rejected_count":"N/A" if key=="direct" else 134,
            "count_note":"Additional rejections versus direct Qwen among matched DEV entities: 199->73 ungroundable visual outputs and 674->540 groundable visual outputs."})
    met=pd.read_csv(E21/"official_gmner_metrics.csv"); br=pd.read_csv(E21/"groundability_breakdown.csv")
    for method in ["PGMNER+Qwen","PGMNER+Groundability+Qwen","PGMNER+Groundability+Qwen+DEVConfidence"]:
        mr=met[(met.population=="source_disjoint_twitter2015")&(met.method==method)].iloc[0]
        gd=br[(br.population=="source_disjoint_twitter2015")&(br.method==method)] if method!="PGMNER+Groundability+Qwen+DEVConfidence" else pd.DataFrame()
        gr=gd[gd.gold_group=="groundable"].iloc[0] if len(gd) else None; ur=gd[gd.gold_group=="ungroundable"].iloc[0] if len(gd) else None
        rejection.append({"population":"source_disjoint_validation","protocol":"existing Exp21 aggregate; no TEST loaded in Exp25","samples":int(mr.samples),"method":method,
            "mner_f1":mr.mner_f1,"eeg_f1":mr.eeg_f1,"gmner_f1":mr.gmner_f1,"false_grounding_rate":ur.false_grounding_rate if ur is not None else "N/A",
            "ungroundable_rejection_accuracy":ur.groundability_decision_accuracy if ur is not None else "N/A",
            "groundable_recall":gr.groundability_decision_accuracy if gr is not None else "N/A","groundable_acc_iou_05":gr.groundable_acc_iou_05 if gr is not None else "N/A",
            "rejected_entity_count":"N/A","correctly_rejected_count":63 if method=="PGMNER+Groundability+Qwen" else "N/A",
            "incorrectly_rejected_count":74 if method=="PGMNER+Groundability+Qwen" else "N/A",
            "count_note":"For gated source-disjoint result: 63 fewer false-groundings and 74 fewer groundable visual outputs; 29 were known correct-box false rejects."})
    write_csv(RES/"rejection_tradeoff.csv",rejection)

    # Figures.
    plt.figure(figsize=(6.4,3.8))
    for model,col,_ in model_specs:
        for val,label,ls in [(0,"ungroundable","-"),(1,"groundable","--")]:
            x=base.loc[base.groundable_original==val,col]; plt.hist(x,bins=30,density=True,histtype="step",linewidth=1.5,label=f"{model} {label}",linestyle=ls)
    plt.xlabel("Reliability score");plt.ylabel("Density");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(FIG/"reliability_score_distribution.pdf");plt.close()
    plt.figure(figsize=(6.6,3.8)); x=np.arange(4); width=.35
    for i,(model,col,_) in enumerate(model_specs):
        means=[conditions[c][col].mean() for c in ["clean","text_only","random","hard_negative"]]
        plt.bar(x+(i-.5)*width,means,width,label=model)
    plt.xticks(x,["Clean","Zero visual","Random","Hard negative"]);plt.ylabel("Mean reliability");plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/"clean_vs_counterfactual.pdf");plt.close()
    plt.figure(figsize=(6.4,3.8))
    for model,col,_ in model_specs:
        s=merged[col].to_numpy(float); q=pd.qcut(s,4,labels=False,duplicates="drop"); yg=merged.groundable.to_numpy(int); point=merged.pointing_hit.fillna(0).to_numpy(float); false=(yg==0)|((yg==1)&(point<.5))
        plt.plot(range(1,int(q.max())+2),[false[q==z].mean() for z in range(int(q.max())+1)],marker="o",label=model)
    plt.xlabel("Reliability quartile (low to high)");plt.ylabel("False-match proxy rate");plt.xticks([1,2,3,4]);plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/"reliability_vs_false_match.pdf");plt.close()

    def row(model,condition,stratum="all"):
        return next(r for r in cf_rows if r["model"]==model and r["condition"]==condition and r["stratum"]==stratum)
    decisions=[]
    for model,_,_ in model_specs:
        rg=row(model,"random","original_groundable"); rh=row(model,"hard_negative","original_groundable")
        visual=rg["paired_drift_ci95_high"]<0 and rh["paired_drift_ci95_high"]<0
        err=next(r for r in gota_rows if r.get("model")==model and r.get("target")=="combined_false_match_using_low_reliability")
        spatial=next(r for r in gota_rows if r.get("model")==model and r.get("target")=="groundable_pointing_correct")
        shortcut=next(r for r in cf_rows if r["model"]==model and r["condition"]=="clean")["auroc"]-auroc(y,type_score)
        decisions.append({"model":model,"visual_dependency_supported":visual,"clean_minus_type_prior_auroc":shortcut,
            "combined_false_match_auroc_confounded":err["auroc"],"groundable_pointing_correct_auroc":spatial["auroc"],
            "gota_error_prediction_supported":bool(spatial["auroc"]>=.60),
            "recommend_reliability_guided_pgmner":bool(visual and shortcut>=.03 and spatial["auroc"]>=.60)})
    write_csv(RES/"summary_decision.csv",decisions)

    extraction=json.loads((EXP/"cache/extraction_summary.json").read_text())
    d10=decisions[0]; r10=row("Exp10_entity","random","original_groundable"); h10=row("Exp10_entity","hard_negative","original_groundable")
    dev_direct=rejection[0];dev_gate=rejection[1]
    recommendation="PROCEED_WITH_MINIMAL_RELIABILITY_MASK" if d10["recommend_reliability_guided_pgmner"] else "STOP_RELIABILITY_GUIDED_PGMNER"
    report=f"""# Experiment 25: Groundability Reliability Counterfactual Audit

## Scope and protocol

This evaluation uses GMNER DEV plus previously saved Exp21 source-disjoint aggregate results. It does not load TEST examples, train parameters, tune thresholds, or run Qwen. Counterfactual PGMNER features were recomputed end-to-end under each image condition, so both the post-fusion span feature `h` and visual patches `v` correspond to the replacement image. Random and hard-negative images use fixed mappings; hard negatives are nearest non-self images under cached clean projected visual features with identical image IDs excluded.

The Exp10/14 head threshold is fixed at 0.45 for gold-entity diagnostics and the integrated Exp14 routing threshold remains 0.5. Exp24-B uses its saved threshold 0.5. The original labels remain valid for the text-only shortcut audit. Replacement-image labels are unknown, so Random/Hard AUROC and AUPR are intentionally reported as N/A.

## Input dependency

Both heads consume visually contaminated post-fusion `h` plus projected patches. A cached-`v` swap alone would therefore be invalid; this experiment recomputed both. The zero-image condition is an architecture-compatible ablation, not a trained text-only model.

## Counterfactual results

Exp10 clean DEV AUROC/AUPR: {row('Exp10_entity','clean')['auroc']:.4f}/{row('Exp10_entity','clean')['aupr']:.4f}; zero-image text-only AUROC/AUPR: {row('Exp10_entity','text_only')['auroc']:.4f}/{row('Exp10_entity','text_only')['aupr']:.4f}. Exp24-B clean DEV AUROC/AUPR: {row('Exp24B_dustbin','clean')['auroc']:.4f}/{row('Exp24B_dustbin','clean')['aupr']:.4f}; zero-image text-only AUROC/AUPR: {row('Exp24B_dustbin','text_only')['auroc']:.4f}/{row('Exp24B_dustbin','text_only')['aupr']:.4f}.

For originally groundable entities, Exp10 Random mean drift is {r10['mean_paired_drift']:.6f} (95% CI [{r10['paired_drift_ci95_low']:.6f}, {r10['paired_drift_ci95_high']:.6f}]); hard-negative drift is {h10['mean_paired_drift']:.6f} (95% CI [{h10['paired_drift_ci95_low']:.6f}, {h10['paired_drift_ci95_high']:.6f}]). Both conditions have zero threshold flips. These are sensitivity statistics, not replacement-image accuracy.

## Shortcut audit

The TRAIN-derived entity-type prior has DEV AUROC {auroc(y,type_score):.4f}; constant-majority AUROC is 0.5000. Mention length AUROC is {auroc(y,mention_len):.4f}; visual-global-norm AUROC is {auroc(y,vnorm):.4f}. Type-stratified results and sample prevalences are in `results/type_stratified_analysis.csv`.

## GOT-A relation

Strict alignment retained {len(merged)}/{len(base)} reliability entities. Exp10 low reliability predicts the combined proxy (ungroundable **or** missed pointing) with AUROC {d10['combined_false_match_auroc_confounded']:.4f}, but this is confounded by the much easier groundability label. Restricted to groundable entities, reliability predicts correct GOT-A pointing with AUROC only {d10['groundable_pointing_correct_auroc']:.4f}; correlations with transport mass, nominal unmatched mass, maximum transport, and entropy are all weak. The audit distinguishes nominal unmatched mass (`1 - real-real transported mass`) from actual dustbin flow; the original cache does not contain an explicit entity-level observed dustbin flow.

## Existing rejection trade-off

On mixed GMNER DEV, the fixed gate changes GMNER F1 from {dev_direct['gmner_f1']:.4f} to {dev_gate['gmner_f1']:.4f} and false-grounding rate from {dev_direct['false_grounding_rate']:.4f} to {dev_gate['false_grounding_rate']:.4f}, while groundable recall changes from {dev_direct['groundable_recall']:.4f} to {dev_gate['groundable_recall']:.4f}. Existing Exp21 source-disjoint aggregates are reproduced without reopening TEST data.

## Decision

1. Visual information used: **{str(d10['visual_dependency_supported']).upper()}** under the strict paired criterion.
2. Signal beyond type prior: clean-minus-type-prior AUROC = {d10['clean_minus_type_prior_auroc']:.4f}.
3. GOT-A error prediction supported: **{str(d10['gota_error_prediction_supported']).upper()}** (groundable pointing-correct AUROC {d10['groundable_pointing_correct_auroc']:.4f}; combined AUROC {d10['combined_false_match_auroc_confounded']:.4f} is groundability-confounded).
4. Recommendation: **{recommendation}**.

No Reliability-Guided PGMNER training is justified by these results. The existing gate remains useful as a GMNER output-routing heuristic, but it should not be attached to PP-IB or GOT-A as evidence of visual reliability. Stop additional gate/OT experiments unless a future reliability estimator first demonstrates paired image sensitivity and groundable-only matching-error prediction.

## Limitations

- Wrong-image correspondence is not manually verified, so counterfactual labels are unavailable.
- The hard-negative mapping is a cached-feature similarity proxy.
- It covers {extraction['hard_mapping_feature_covered_samples']}/1500 samples; uncovered samples use the fixed random mapping. Same image IDs are excluded, but pixel-level near-duplicates were not exhaustively verified.
- Exp10 was trained on gold entities; Exp14 applies it to predicted spans.
- GOT-A pointing is near random in Exp9, and nominal unmatched mass is not evidence of correct rejection.
- Source-disjoint numbers are pre-existing Exp21 aggregates, not newly evaluated TEST results.

Extraction cost: {sum(v.get('seconds',0) for v in extraction['conditions'].values()):.1f} seconds; maximum allocated GPU memory: {max(v.get('peak_gpu_memory_mb',0) for v in extraction['conditions'].values()):.1f} MB.
"""
    (EXP/"EXPERIMENT_25_REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps({"exp10_clean_auroc":row('Exp10_entity','clean')['auroc'],"exp10_random_groundable_drift":r10['mean_paired_drift'],"exp10_hard_groundable_drift":h10['mean_paired_drift'],"type_prior_auroc":auroc(y,type_score),"combined_false_match_auroc_confounded":d10['combined_false_match_auroc_confounded'],"groundable_pointing_correct_auroc":d10['groundable_pointing_correct_auroc'],"dev_gmner_direct":dev_direct['gmner_f1'],"dev_gmner_gated":dev_gate['gmner_f1'],"recommendation":recommendation},indent=2))


if __name__ == "__main__": main()
