#!/usr/bin/env python3
from __future__ import annotations
import csv,json,re
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment22_module_analysis";RES=EXP/"results";FIG=EXP/"figures"
def read_csv(p):
 with p.open() as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields=None):
 fields=fields or list(rows[0]);
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def hist(run):
 p=EXP/"checkpoints"/run/"per_class_history.jsonl"; return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def run_row(run,kl):
 h=hist(run); best=max(h,key=lambda x:x["overall"]["f1"]) if h else None; log=EXP/"logs"/f"{run}.log"; text=log.read_text(errors="ignore") if log.exists() else ""; tm=re.search(r"平均时间为:\s*([0-9.]+)",text)
 return {"configuration":run,"ib_kl_weight":kl,"status":"complete" if len(h)==5 else "incomplete","epochs_completed":len(h),"best_dev_epoch":best["epoch"] if best else "","best_dev_precision":best["overall"]["p"] if best else "","best_dev_recall":best["overall"]["r"] if best else "","best_dev_f1":best["overall"]["f1"] if best else "","final_epoch_dev_f1":h[-1]["overall"]["f1"] if h else "","mean_epoch_seconds":float(tm.group(1)) if tm else "","gate_statistics":"not persisted: stock PPIBProbe was instantiated but trainer did not call log_step","selection":"DEV only","test_evaluated":False}
def mean(rows,method,metric):return float(np.mean([float(r[metric]) for r in rows if r["method"]==method]))
def grad(rows,loss,group):
 q=[r for r in rows if r["batch"]=="mean" and r["loss_component"]==loss and r["parameter_group"]==group];return float(q[0]["grad_l2"])
def main():
 RES.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True); kl=[run_row("full_kl_short",.001),run_row("no_kl_short",0.0)];write_csv(RES/"ppib_kl_comparison.csv",kl)
 write_csv(RES/"gota_loss_comparison.csv",[{"status":"NOT_EXECUTED","reason":"Stage B failed the preregistered mechanism condition: near-uniform plans, negligible real-dust flow, no monotonic groundability response.","test_evaluated":False}])
 # DEV curves.
 fig,ax=plt.subplots(figsize=(5.8,3.5))
 for run,label in [("full_kl_short","Full, KL=0.001"),("no_kl_short","w/o KL")]:
  h=hist(run)
  if h:ax.plot([x["epoch"] for x in h],[x["overall"]["f1"] for x in h],marker="o",label=label)
 ax.set(xlabel="Epoch",ylabel="DEV micro-F1",title="Short-run KL screening (DEV only)");ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/"fig22d_kl_dev_curve.pdf");fig.savefig(FIG/"fig22d_kl_dev_curve.png",dpi=300);plt.close(fig)
 ot=read_csv(RES/"partial_vs_full_ot.csv");gr=read_csv(RES/"gradient_path_audit.csv");
 part={k:mean(ot,"partial_checkpoint",k) for k in ["transported_mass","dust_edge_mass","dust_to_dust","normalized_entropy","effective_matched_region_count","mass_normalized_cost","transport_concentration"]};full={k:mean(ot,"full_ot",k) for k in ["transported_mass","normalized_entropy","effective_matched_region_count","mass_normalized_cost","transport_concentration"]}
 groups={}
 for g in ["no_groundable","mixed_groundability","all_groundable"]:
  q=[r for r in ot if r["method"]=="partial_checkpoint" and r["groundability_group"]==g];groups[g]={"n":len(q),"dust_edge_mass":float(np.mean([float(x["dust_edge_mass"]) for x in q])),"normalized_entropy":float(np.mean([float(x["normalized_entropy"]) for x in q]))}
 architecture=f"""# Experiment 22 Architecture and Gradient Audit

## Verified forward paths

- PP-IB creates sample-level text and visual latent variables through `mu_net`, `logvar_net`, and `polar_gate`. The sampled gated latents are not passed to GOT-A or the span classifier.
- The weighted KL term (`ib_kl_weight=0.001`) is computed from PP-IB `mu/logvar`. Autograd confirms that it reaches both shared encoders: text gradient L2 `{grad(gr,'ppib_kl_weighted','shared_text_encoder'):.6g}`, visual `{grad(gr,'ppib_kl_weighted','shared_visual_encoder'):.6g}`.
- The polarization term reaches both shared encoders and PP-IB gate/mu branches, but at much smaller scale (text-encoder `{grad(gr,'ppib_polarization_weighted','shared_text_encoder'):.6g}`, visual-encoder `{grad(gr,'ppib_polarization_weighted','shared_visual_encoder'):.6g}`).
- GOT-A projects span and patch features, builds `C=0.5*(1-cosine)`, computes a transport plan, and injects `align_text=T@visual_proj` through `fusion_text`, LayerNorm residual, and `span_classifier`.
- The weighted GOT-A loss reaches shared text (`{grad(gr,'gota_transport_weighted','shared_text_encoder'):.6g}`) and visual (`{grad(gr,'gota_transport_weighted','shared_visual_encoder'):.6g}`) encoders, both OT projections, and `s_logit`.
- The MNER classification loss reaches GOT-A projections and `s_logit`, but has zero gradient to PP-IB gate/mu/logvar branches.

## OT constraints

Partial OT and Full OT were recomputed from identical frozen projections and identical cost matrices. Full OT uses uniform valid marginals totaling 1 and no dust. Partial OT uses real marginals totaling `s={json.loads((RES/'ot_analysis_metadata.json').read_text())['checkpoint_s']:.6f}` plus one dust marginal of `1-s` on each side. Raw objectives are not compared across unequal mass; the analysis reports cost divided by actual real-real mass.

The augmented dust plan was reconstructed explicitly. Consequently, `real_to_dust`, `dust_to_real`, and `dust_to_dust` are not conflated with the earlier approximation `1-sum(T_real-real)`.
""";(RES/"architecture_audit.md").write_text(architecture)
 short_ok=all(x["status"]=="complete" for x in kl);best_full=kl[0]["best_dev_f1"];best_no=kl[1]["best_dev_f1"]
 report=f"""# Experiment 22: PP-IB Regularization and GOT-A Partial-Matching Analysis

## Scope and protocol

The audit used the original seed-42 checkpoint, two real Twitter-2017 TRAIN batches for autograd, and all 1,500 GMNER TEST samples only for frozen mechanism statistics. The GMNER groundability annotation is used as an analysis label, not as a training target. The KL comparison is a five-epoch, seed-42 DEV-only screening run from identical pretrained initialization; TEST was not evaluated.

## A. Gradient-path audit

The PP-IB KL term has a real shared-encoder path (text gradient L2 `{grad(gr,'ppib_kl_weighted','shared_text_encoder'):.6g}`, visual `{grad(gr,'ppib_kl_weighted','shared_visual_encoder'):.6g}`). The GOT-A loss also reaches both shared encoders (text `{grad(gr,'gota_transport_weighted','shared_text_encoder'):.6g}`, visual `{grad(gr,'gota_transport_weighted','shared_visual_encoder'):.6g}`), OT projections, and transported-mass scalar. The original MNER loss does not reach PP-IB-specific gate/mu/logvar parameters, while it does reach the GOT-A path. Full details are in `gradient_path_audit.csv` and `architecture_audit.md`.

## B. Partial versus Full OT

On 1,500 samples, checkpoint Partial OT transported real-real mass `{part['transported_mass']:.6f}` versus Full OT `{full['transported_mass']:.6f}`. However, the actual Partial dust-edge flow was only `{part['dust_edge_mass']:.6f}`; nominal unmatched capacity mostly became dust-to-dust mass (`{part['dust_to_dust']:.6f}`). The Partial plan was effectively uniform: normalized entropy `{part['normalized_entropy']:.7f}` and effective matched visual nodes `{part['effective_matched_region_count']:.3f}` of 197. Full OT was similarly uniform (entropy `{full['normalized_entropy']:.7f}`). Mass-normalized costs were `{part['mass_normalized_cost']:.6f}` (Partial) and `{full['mass_normalized_cost']:.6f}` (Full).

Annotation groups did not show the expected monotonic rejection behavior: dust-edge mass was `{groups['no_groundable']['dust_edge_mass']:.6f}` for no-groundable samples, `{groups['mixed_groundability']['dust_edge_mass']:.6f}` for mixed, and `{groups['all_groundable']['dust_edge_mass']:.6f}` for all-groundable samples. This does not support a claim that the learned dust mechanism detects visual groundability. It is consistent with Experiments 4 and 9.

## C. PP-IB KL screening

Short-run status: `{short_ok}`. Full KL best DEV F1: `{best_full}`; w/o KL best DEV F1: `{best_no}`. This five-epoch screen is not a complete performance ablation and no TEST result was consulted. The stock probe did not persist gate trajectories because the trainer instantiated `PPIBProbe` without calling `log_step`; this limitation is recorded rather than backfilled.

## D. GOT-A training contribution

Not executed. Although the gradient path exists, Stage B did not show the required mechanism signal: matching remained almost uniform, real dust-edge routing was negligible, and annotation groups were non-monotonic. Under the stopping rule, another loss-level training pair would not justify its cost.

## Conclusions

1. PP-IB KL is a genuine auxiliary regularizer of the shared encoders, but the original PP-IB gated samples remain disconnected from the prediction path.
2. GOT-A is computationally connected to both task predictions and shared training representations.
3. The evaluated checkpoint's Partial OT mainly rescales an almost-uniform real-real plan; it does not provide empirical evidence of selective unmatched visual routing or bbox-quality alignment.
4. Any KAIS claim should distinguish verified gradient coupling from unverified semantic matching quality.

## Limitations

The OT comparison is post-hoc on one checkpoint; GMNER annotations differ from the Twitter-2017 training distribution; sample-level groundability groups are not direct supervision of every candidate span; and the KL comparison is only a five-epoch screen. No TEST-based model selection was performed.
""";(EXP/"EXPERIMENT_22_REPORT.md").write_text(report)
 draft=f"""# KAIS Module Analysis Draft

## Motivation
We examined whether the PP-IB and GOT-A objectives are connected to the task representation and whether partial transport realizes selective matching in the trained PGMNER checkpoint.

## Experimental setup
Autograd paths were measured on real Twitter-2017 training batches. Partial and full entropy-regularized OT were recomputed from identical frozen span/patch projections and cosine-cost matrices on 1,500 annotated multimodal samples. We report mass-normalized costs because the feasible total masses differ.

## Partial versus full OT
Partial OT assigned `{part['transported_mass']:.4f}` mass to real-real pairs, compared with 1.0 for Full OT. Its normalized plan entropy was `{part['normalized_entropy']:.4f}`, and its effective visual support was approximately all 197 nodes. Explicit reconstruction of the augmented coupling showed only `{part['dust_edge_mass']:.4f}` mass crossing real-dust edges; most dust capacity (`{part['dust_to_dust']:.4f}`) matched dust to dust. Moreover, dust-edge flow did not increase monotonically as annotated visual groundability decreased. These observations verify reduced transported mass but do not establish improved semantic correspondence or groundability estimation.

## PP-IB regularization
The weighted KL loss produced non-zero gradients in both shared encoders, confirming a training-time regularization pathway. In contrast, the MNER loss had zero gradient to the PP-IB-specific gate and distribution heads. The five-epoch DEV-only screen obtained `{best_full}` F1 with KL and `{best_no}` without KL; this is screening evidence rather than a full ablation.

## Discussion
GOT-A is structurally and differentiably coupled to PGMNER prediction, whereas the original PP-IB gate is only auxiliary. The current checkpoint does not support interpreting its nearly uniform transport as entity-region grounding. We therefore describe partial matching as a constrained optimization mechanism and reserve semantic grounding claims for the separately supervised GMNER extension.
""";(EXP/"KAIS_MODULE_ANALYSIS_DRAFT.md").write_text(draft)
 print(json.dumps({"full_kl":kl[0],"no_kl":kl[1],"report":str(EXP/"EXPERIMENT_22_REPORT.md")},indent=2))
if __name__=="__main__":main()
