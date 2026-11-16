#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import EXP,FIGURES,RESULTS

def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def save(fig,name):
 fig.tight_layout();fig.savefig(FIGURES/f"{name}.pdf",bbox_inches="tight");fig.savefig(FIGURES/f"{name}.png",dpi=300,bbox_inches="tight");plt.close(fig)
def main():
 FIGURES.mkdir(exist_ok=True);rows=read(RESULTS/"partial_vs_full_ot.csv");groups=["no_groundable","mixed_groundability","all_groundable"]; methods=["partial_checkpoint","partial_fixed_0.6","full_ot"]
 fig,ax=plt.subplots(figsize=(6.6,3.6));x=np.arange(3);w=.24
 for i,m in enumerate(methods):ax.bar(x+(i-1)*w,[np.mean([float(r["dust_edge_mass"]) for r in rows if r["method"]==m and r["groundability_group"]==g]) for g in groups],w,label=m.replace("_"," "))
 ax.set_xticks(x,labels=["None","Mixed","All"]);ax.set_xlabel("Annotated visual groundability in sample");ax.set_ylabel("Dust-edge mass");ax.legend(frameon=False,fontsize=8);save(fig,"fig22a_unmatched_mass")
 fig,axs=plt.subplots(1,2,figsize=(7.0,3.1))
 for m in methods:
  axs[0].hist([float(r["normalized_entropy"]) for r in rows if r["method"]==m],bins=35,histtype="step",density=True,label=m.replace("_"," "))
  axs[1].hist([float(r["transport_concentration"]) for r in rows if r["method"]==m],bins=35,histtype="step",density=True,label=m.replace("_"," "))
 axs[0].set_xlabel("Normalized matching entropy");axs[1].set_xlabel("Transport concentration");axs[0].set_ylabel("Density");axs[1].legend(frameon=False,fontsize=7);save(fig,"fig22b_entropy_concentration")
 cases=json.loads((RESULTS/"ot_heatmap_cases.json").read_text());
 if cases:
  n=len(cases);fig,axs=plt.subplots(n,2,figsize=(7,2.1*n),squeeze=False)
  for i,c in enumerate(cases):
   for j,key in enumerate(["partial","full"]):
    a=np.asarray(c[key]);mesh=axs[i,j].pcolormesh(a,shading="nearest",cmap="viridis");axs[i,j].set_title(f"{c['group']} | {key}",fontsize=9);axs[i,j].set_xlabel("Visual node");axs[i,j].set_ylabel("Span");fig.colorbar(mesh,ax=axs[i,j],fraction=.035)
  save(fig,"fig22c_partial_full_heatmaps")
 # MNER-correct relation: samples with/without fully correct annotated entities.
 rel=[]
 for m in methods:
  q=[r for r in rows if r["method"]==m and r["mner_gold_entity_accuracy"] not in ("","nan")]
  for label,test in [("all_correct",lambda x:float(x)==1),("has_error",lambda x:float(x)<1)]:
   z=[r for r in q if test(r["mner_gold_entity_accuracy"])]
   if z:
    for metric in ["transported_mass","dust_edge_mass","normalized_entropy","transport_concentration","mass_normalized_cost"]:rel.append({"method":m,"mner_group":label,"metric":metric,"n":len(z),"mean":np.mean([float(r[metric]) for r in z]),"median":np.median([float(r[metric]) for r in z])})
 with (RESULTS/"ot_mner_correctness.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rel[0]));w.writeheader();w.writerows(rel)
if __name__=="__main__":main()
