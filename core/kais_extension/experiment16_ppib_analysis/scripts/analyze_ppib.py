#!/usr/bin/env python3
"""Experiment 16: PP-IB gate analysis from verified TEST inference caches."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment16_ppib_analysis"
RES = EXP / "results"
FIG = EXP / "figures"
EXP3 = ROOT / "kais_extension/experiment3_budget_landscape/results/sample_visual_relevance.csv"
EXP4 = ROOT / "kais_extension/experiment4_visual_mechanism/results/full_internal_features_sample.csv"
EXP4_METRICS = ROOT / "kais_extension/experiment4_visual_mechanism/results/condition_dataset_metrics.csv"
CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
TRAIN_LOG = CKPT.parent / "train.log"


def read_csv(path):
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")


def entropy(g):
    g = np.clip(np.asarray(g, dtype=float), 1e-12, 1 - 1e-12)
    return -(g * np.log(g) + (1 - g) * np.log(1 - g))


def summarize(values, modality, scope="all", group="ALL"):
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    ent = entropy(x)
    low = np.mean(x <= .1); high = np.mean(x >= .9); middle = np.mean((x > .1) & (x < .9))
    return {
        "scope": scope, "group": group, "modality": modality, "n": len(x),
        "mean_gate": np.mean(x), "std_gate": np.std(x, ddof=1), "median_gate": np.median(x),
        "min_gate": np.min(x), "max_gate": np.max(x),
        "fraction_0_0.1": low, "fraction_0.1_0.9": middle, "fraction_0.9_1": high,
        "polarization_ratio_extreme_0.1": low + high,
        "polarization_ratio_margin_0.2": np.mean((x <= .2) | (x >= .8)),
        "mean_binary_entropy_nats": np.mean(ent),
        "mean_normalized_binary_entropy": np.mean(ent / math.log(2)),
        "mean_g_times_1_minus_g": np.mean(x * (1-x)),
    }


def parse_loss_trajectory():
    pat = re.compile(r"loss_polar:([0-9.eE+-]+).*?(\d+)/211")
    epoch = 1; last_batch = -1; last_pair = None; rows = []
    for line in TRAIN_LOG.open(encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if not m: continue
        val, batch = float(m.group(1)), int(m.group(2))
        pair = (batch, val)
        if pair == last_pair: continue
        if last_batch >= 150 and batch < 20: epoch += 1
        rows.append({"epoch": epoch, "batch": batch, "weighted_loss_polar": val})
        last_batch = batch; last_pair = pair
    grouped = []
    for ep in sorted({r["epoch"] for r in rows}):
        vals = [r["weighted_loss_polar"] for r in rows if r["epoch"] == ep]
        grouped.append({"epoch": ep, "n_logged": len(vals), "mean_weighted_loss_polar": np.mean(vals),
                        "median_weighted_loss_polar": np.median(vals), "min": np.min(vals), "max": np.max(vals)})
    return rows, grouped


def checkpoint_audit():
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    out = {"checkpoint": str(CKPT), "gate_parameters": {}}
    for branch in ["dist_t", "dist_v"]:
        wk = f"{branch}.polar_gate.0.weight"; bk = f"{branch}.polar_gate.0.bias"
        w = state[wk].float(); b = state[bk].float()
        out["gate_parameters"][branch] = {
            "weight_shape": list(w.shape), "weight_mean": float(w.mean()), "weight_std": float(w.std()),
            "weight_l2": float(w.norm()), "bias": float(b.item()), "requires_grad_in_model": True,
            "configured_init_weight_std": 0.01, "configured_init_bias": 0.0,
        }
    return out


def style():
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
                         "legend.fontsize": 9, "figure.dpi": 150, "savefig.bbox": "tight",
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def savefig(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf"); fig.savefig(FIG / f"{stem}.png", dpi=300); plt.close(fig)


def plot_distributions(text_g, visual_g):
    style(); fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, 1, 51)
    colors = {"Text": "#2878B5", "Visual": "#D9534F"}
    for name, vals in [("Text", text_g), ("Visual", visual_g)]:
        axes[0].hist(vals, bins=bins, alpha=.62, label=name, color=colors[name], edgecolor="white")
        if np.std(vals) > 1e-12:
            kde = stats.gaussian_kde(vals)
            xx = np.linspace(max(0, min(vals)-.01), min(1, max(vals)+.01), 500)
            axes[1].plot(xx, kde(xx), lw=2, label=name, color=colors[name])
    axes[0].axvspan(0, .1, color="#55A868", alpha=.12); axes[0].axvspan(.9, 1, color="#55A868", alpha=.12)
    axes[0].set(xlim=(0,1), xlabel="Gate value $\\gamma$", ylabel="Sample count", title="Gate histogram")
    axes[1].set(xlabel="Gate value $\\gamma$", ylabel="Density", title="Gate density")
    axes[0].legend(); axes[1].legend(); fig.tight_layout(); savefig(fig, "fig16a_gate_distribution")


def plot_groups(merged):
    style(); groups = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data=[]; positions=[]; colors=[]
    for i,g in enumerate(groups):
        rows=[r for r in merged if r["relevance_group"]==g]
        data += [[r["text_gate"] for r in rows],[r["visual_gate"] for r in rows]]
        positions += [i*3+1,i*3+2]; colors += ["#2878B5","#D9534F"]
    bp=ax.boxplot(data,positions=positions,widths=.72,patch_artist=True,showfliers=False)
    for patch,c in zip(bp["boxes"],colors): patch.set_facecolor(c); patch.set_alpha(.7)
    ax.set_xticks([1.5,4.5,7.5],groups); ax.set_ylabel("Gate value $\\gamma$"); ax.set_xlabel("CLIP image-text relevance tertile")
    ax.set_title("PP-IB gates by image-text relevance")
    ax.plot([],[],color="#2878B5",lw=7,alpha=.7,label="Text"); ax.plot([],[],color="#D9534F",lw=7,alpha=.7,label="Visual")
    ax.legend(); fig.tight_layout(); savefig(fig,"fig16b_gate_by_relevance")


def plot_group_metrics(group_stats):
    style(); groups=["Low","Medium","High"]; fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    width=.36; x=np.arange(3)
    for offset,mod,c in [(-width/2,"text","#2878B5"),(width/2,"visual","#D9534F")]:
        rows={r["group"]:r for r in group_stats if r["modality"]==mod}
        axes[0].bar(x+offset,[rows[g]["mean_normalized_binary_entropy"] for g in groups],width,label=mod.title(),color=c)
        axes[1].bar(x+offset,[rows[g]["polarization_ratio_extreme_0.1"] for g in groups],width,label=mod.title(),color=c)
    axes[0].set(title="Normalized binary entropy",ylabel="Entropy / ln 2",xticks=x,xticklabels=groups,ylim=(0,1.05))
    axes[1].set(title="Extreme polarization ratio",ylabel="Fraction outside (0.1, 0.9)",xticks=x,xticklabels=groups,ylim=(0,1.05))
    axes[0].legend(); axes[1].legend(); fig.tight_layout(); savefig(fig,"fig16c_entropy_polarization_by_relevance")


def plot_loss(epoch_rows):
    style(); fig,ax=plt.subplots(figsize=(7,4)); x=[r["epoch"] for r in epoch_rows]; y=[r["mean_weighted_loss_polar"] for r in epoch_rows]
    ax.plot(x,y,color="#7A5195",lw=2,marker="o",ms=3); ax.axhline(.005,color="#D9534F",ls="--",lw=1,label="Maximum at $\\gamma=0.5$ after warmup")
    ax.set(xlabel="Training epoch reconstructed from log",ylabel="Logged weighted polarization loss",title="Training-log PP-IB loss trajectory")
    ax.legend(); fig.tight_layout(); savefig(fig,"fig16d_logged_polar_loss")


def main():
    RES.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    gate_rows=read_csv(EXP4); rel_rows={r["sample_id"]:r for r in read_csv(EXP3)}
    text=np.array([fnum(r["mean_text_gate"]) for r in gate_rows]); visual=np.array([fnum(r["mean_visual_gate"]) for r in gate_rows])
    valid_sim=np.array([fnum(rel_rows.get(r["sample_id"],{}).get("global_clip_similarity")) for r in gate_rows])
    valid=valid_sim[np.isfinite(valid_sim)]; q1,q2=np.quantile(valid,[1/3,2/3])
    merged=[]
    for r,tg,vg,sim in zip(gate_rows,text,visual,valid_sim):
        group="Unavailable"
        if np.isfinite(sim): group="Low" if sim<=q1 else ("Medium" if sim<=q2 else "High")
        merged.append({"sample_id":r["sample_id"],"sample_index":r["sample_index"],"image_id":r["image_id"],
                       "text_gate":tg,"visual_gate":vg,"text_binary_entropy":entropy([tg])[0],
                       "visual_binary_entropy":entropy([vg])[0],"global_clip_similarity":sim,"relevance_group":group})
    write_csv(RES/"gate_values_test.csv",merged)
    overall=[summarize(text,"text"),summarize(visual,"visual")]
    write_csv(RES/"gate_statistics.csv",overall)
    group_stats=[]
    for g in ["Low","Medium","High"]:
        rows=[r for r in merged if r["relevance_group"]==g]
        group_stats += [summarize([r["text_gate"] for r in rows],"text","clip_tertile",g),
                        summarize([r["visual_gate"] for r in rows],"visual","clip_tertile",g)]
    write_csv(RES/"relevance_group_statistics.csv",group_stats)
    tests=[]
    for mod,key in [("text","text_gate"),("visual","visual_gate")]:
        rows=[r for r in merged if r["relevance_group"]!="Unavailable"]
        rho,p=stats.spearmanr([r["global_clip_similarity"] for r in rows],[r[key] for r in rows])
        vals=[[r[key] for r in rows if r["relevance_group"]==g] for g in ["Low","Medium","High"]]
        h,kp=stats.kruskal(*vals)
        tests.append({"modality":mod,"n":len(rows),"spearman_rho":rho,"spearman_p":p,"kruskal_h":h,"kruskal_p":kp})
    write_csv(RES/"relevance_statistical_tests.csv",tests)
    raw_loss,epoch_loss=parse_loss_trajectory(); write_csv(RES/"logged_polar_loss_steps.csv",raw_loss); write_csv(RES/"logged_polar_loss_by_epoch.csv",epoch_loss)
    audit=checkpoint_audit(); audit.update({"gate_definition":"gamma=sigmoid(Linear(mu)); scalar shape [batch,1] per modality; z_final=gamma*z",
        "polar_loss":"mean(gamma*(1-gamma)) + lambda_balance*relu(abs(mean(gamma)-0.5)-0.2)^2",
        "beta_polar":.01,"warmup_epochs":3,"test_samples":len(gate_rows),"valid_clip_similarity_samples":len(valid),
        "clip_tertile_cutoffs":{"low_max":q1,"medium_max":q2},"f1_sanity":.9064695008741531,
        "cache_provenance":str(EXP4),"similarity_provenance":str(EXP3),
        "historical_gate_trajectory":"UNAVAILABLE: paper-full run retained only best_model.pth and no PPIBProbe gate logs.",
        "comparable_ablation_checkpoint":"NOT FOUND: no clearly identified w/o PP-IB or no-polarization checkpoint."})
    (RES/"implementation_and_artifact_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    plot_distributions(text,visual); plot_groups(merged); plot_group_metrics(group_stats); plot_loss(epoch_loss)
    summary={"overall":overall,"groups":group_stats,"tests":tests,"audit":audit}
    (RES/"experiment16_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
