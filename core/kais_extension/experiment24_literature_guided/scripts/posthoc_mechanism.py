#!/usr/bin/env python3
"""Offline DEV-only mechanism audit after all Experiment 24 training jobs finish."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,SequentialSampler

ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment24_literature_guided";RES=EXP/"results"
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP/"scripts"))
import run_twitter_route as rt
from models.partial_ot import NodeSinkhorn

def write(path,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def status(route):
    name={"A":"E24_A_TASK_GUIDED_OT","B":"E24_B_GROUNDING_SUPERVISED_POT","C":"E24_C_COUNTERFACTUAL_ALIGNMENT"}[route]
    return json.loads((EXP/"runs"/name/"status.json").read_text()),EXP/"runs"/name

def load_model(route):
    st,run=status(route);m,model,tok,labels,_=rt.setup(route,run,42)
    model.load_state_dict(torch.load(run/"best_model.pth",map_location=m.args.device,weights_only=True),strict=True);model.eval()
    from utils.datasets import SpanNerDataset,collate_fn
    ex=m.load_examples(m.args.data_dir,"dev",tok,max_seq_len=128)
    ds=SpanNerDataset(ex,args=m.args,tokenizer=tok,labels=labels,dev=True)
    loader=DataLoader(ds,sampler=SequentialSampler(ds),batch_size=8,collate_fn=collate_fn)
    return st,m,model,loader,collate_fn

def make_image_bank(model,loader,device):
    imgs=[];emb=[];model.eval()
    with torch.no_grad():
        for bc in loader:
            im=bc["image"];imgs.append(im.cpu());g,_=model.image_model(im.to(device));emb.append(F.normalize(g.float(),dim=-1).cpu())
    images=torch.cat(imgs);z=torch.cat(emb);sim=z@z.T;sim.fill_diagonal_(-float("inf"));order=sim.argsort(dim=1,descending=True)
    hard=[]
    for i in range(len(images)):
        chosen=None
        for j in order[i,:32].tolist():
            if not torch.equal(images[i],images[j]):chosen=j;break
        hard.append(chosen if chosen is not None else int((i+1)%len(images)))
    return images,torch.tensor(hard),torch.roll(torch.arange(len(images)),1)

def scan(model,loader,device,images,mapping=None,ot_mode="partial"):
    preds=[];gold=[];probs=[];goldprob=[];sample_idx=[];gates_t=[];gates_v=[];rel=[];plans=[];offset=0;current={"mask":None};full=NodeSinkhorn(eps=.1,n_iters=30,cost_half=True).to(device)
    def hook(module,inputs,output):
        del module,inputs
        text,vis,at,av,plan,loss,comp=output
        if ot_mode=="uniform":
            rp=plan.sum(-1,keepdim=True);new=rp*torch.ones_like(plan)/plan.size(-1)
        elif ot_mode=="full":new,_=full(text,vis,mask_t=current["mask"],mask_v=None)
        else:new=plan
        plans.append(new.detach().cpu());at=torch.bmm(new,vis);av=torch.bmm(new.transpose(1,2),text)
        return text,vis,at,av,new,loss,comp
    h=model.semantic_ot_aligner.register_forward_hook(hook)
    torch.manual_seed(24042);model.eval()
    with torch.no_grad():
        for bc in loader:
            b={k:v.to(device) for k,v in bc.items()};bs=b["image"].size(0);current["mask"]=b["span_masks"].float()
            if mapping is not None:b["image"]=images[mapping[offset:offset+bs]].to(device)
            out=model(b)[0][0];mask=b["span_masks"].bool();lab=b["span_labels"]
            preds.extend(out.argmax(-1)[mask].cpu().tolist());gold.extend(lab[mask].cpu().tolist());probs.append(out[mask].cpu())
            gp=out.gather(-1,lab.unsqueeze(-1)).squeeze(-1);goldprob.extend(gp[mask].cpu().tolist())
            for j in range(bs):sample_idx.extend([offset+j]*int(mask[j].sum()))
            pt=model.dist_t.last_probe;pv=model.dist_v.last_probe;gt=pt["gate"].flatten().cpu();gv=pv["gate"].flatten().cpu();gates_t.extend(gt.tolist());gates_v.extend(gv.tolist())
            rel.extend(F.cosine_similarity(pt["mu"],pv["mu"],dim=-1).cpu().tolist());offset+=bs
    h.remove();p=torch.cat(plans);valid=[]
    # Valid rows are identified by positive row mass because masks are already enforced by OT.
    rm=p.sum(-1);valid=rm>1e-12;pv=p[valid];pn=pv/pv.sum(-1,keepdim=True).clamp_min(1e-12)
    return {"pred":preds,"gold":gold,"prob":torch.cat(probs).numpy(),"goldprob":np.asarray(goldprob),"sample_idx":np.asarray(sample_idx),"relevance":np.asarray(rel),
            "text_gate":np.asarray(gates_t),"visual_gate":np.asarray(gates_v),"transported_mass":float(p.sum((1,2)).mean()),"unmatched_mass":float((1-p.sum((1,2))).mean()),
            "entropy":float((-(pn.clamp_min(1e-12)*pn.clamp_min(1e-12).log()).sum(-1)/math.log(pn.size(-1))).mean()),"concentration":float(pn.max(-1).values.mean())}

def compare(clean,other):
    a=np.asarray(clean["pred"]);b=np.asarray(other["pred"]);g=np.asarray(clean["gold"]);idx=(a!=0)|(b!=0)|(g!=0)
    return {"prediction_changes":int((a[idx]!=b[idx]).sum()),"correct_to_wrong":int(((a[idx]==g[idx])&(b[idx]!=g[idx])).sum()),"wrong_to_correct":int(((a[idx]!=g[idx])&(b[idx]==g[idx])).sum()),
            "mean_abs_probability_drift":float(np.abs(clean["prob"]-other["prob"]).mean()),"gold_probability_change":float((other["goldprob"]-clean["goldprob"]).mean())}

def ppib_rows():
    # These values are archival measurements from Exp16/19/20/23, not recomputed or attributed to Exp24 A/C.
    return [
      {"version":"Original ICDM PGMNER","prediction_coupled":False,"task_gradient_to_gate":0.0,"gate_scope":"sample","text_gate_mean":0.49972,"text_gate_std":0.00018,"visual_gate_mean":0.50000,"visual_gate_std":0.00009,"polarization_ratio":0.0,"intervention_changes_predictions":0,"modality_relevance_relation":"No material tertile difference","evidence":"Exp16/16C"},
      {"version":"Exp19 PC-PP-IB","prediction_coupled":True,"task_gradient_to_gate":3.8975e-08,"gate_scope":"sample","text_gate_mean":0.50230,"text_gate_std":0.00018,"visual_gate_mean":0.50226,"visual_gate_std":0.00008,"polarization_ratio":0.0,"intervention_changes_predictions":0,"modality_relevance_relation":"Not established","evidence":"Exp19"},
      {"version":"Exp20 entity-conditioned B2","prediction_coupled":True,"task_gradient_to_gate":"nonzero","gate_scope":"span","text_gate_mean":0.96872,"text_gate_std":0.08019,"visual_gate_mean":"N/A","visual_gate_std":"N/A","polarization_ratio":"N/A","intervention_changes_predictions":17,"modality_relevance_relation":"Not established; single-end collapse","evidence":"Exp20"},
      {"version":"Exp23 entity routing","prediction_coupled":True,"task_gradient_to_gate":0.00063423,"gate_scope":"span","text_gate_mean":0.48382,"text_gate_std":0.00264,"visual_gate_mean":"N/A","visual_gate_std":"N/A","polarization_ratio":"N/A","intervention_changes_predictions":2,"modality_relevance_relation":"Not established; negligible span variance","evidence":"Exp23"},
      {"version":"Exp24 A/C","prediction_coupled":False,"task_gradient_to_gate":0.0,"gate_scope":"unchanged original PP-IB","text_gate_mean":"see original","text_gate_std":"see original","visual_gate_mean":"see original","visual_gate_std":"see original","polarization_ratio":"see original","intervention_changes_predictions":"N/A","modality_relevance_relation":"Not modified","evidence":"Exp24 modifies GOT-A supervision, not PP-IB"}
    ]

def main():
    visual=[];gota=[]
    for route in ("A","C"):
        st,m,model,loader,_=load_model(route);images,hard,rnd=make_image_bank(model,loader,m.args.device)
        clean=scan(model,loader,m.args.device,images);random=scan(model,loader,m.args.device,images,rnd);hardm=scan(model,loader,m.args.device,images,hard)
        for name,obj in (("clean",clean),("random",random),("hard_negative",hardm)):
            row={"route":route,"condition":name,"dev_f1_constraint":st["best_dev_f1"],"mapping":"original" if name=="clean" else ("global_shift" if name=="random" else "nearest nonduplicate visual embedding")}
            if name!="clean":row.update(compare(clean,obj))
            else:row.update({"prediction_changes":0,"correct_to_wrong":0,"wrong_to_correct":0,"mean_abs_probability_drift":0.0,"gold_probability_change":0.0})
            # High relevance is an IB-latent cosine proxy, not groundability.
            q=np.quantile(clean["relevance"],[1/3,2/3]);high_samples=np.where(clean["relevance"]>q[1])[0];emask=np.isin(clean["sample_idx"],high_samples)
            row["high_relevance_proxy_gold_probability_change"]=0.0 if name=="clean" else float((obj["goldprob"][emask]-clean["goldprob"][emask]).mean())
            visual.append(row)
        for mode in ("partial","uniform","full"):
            obj=clean if mode=="partial" else scan(model,loader,m.args.device,images,ot_mode=mode)
            d=compare(clean,obj)
            gota.append({"version":f"Exp24-{route}","dataset":"Twitter-2017 DEV","ot_condition":mode,"transported_mass":obj["transported_mass"],"unmatched_mass":obj["unmatched_mass"],"matching_entropy":obj["entropy"],"transport_concentration":obj["concentration"],
                         "groundability_auroc":"N/A","pointing_accuracy":"N/A","bbox_iou":"N/A","false_match_rate":"N/A","logit_probability_drift":d["mean_abs_probability_drift"],"prediction_changes":d["prediction_changes"],"supervision":"MNER labels only; no bbox"})
    bst,_=status("B")
    gota.append({"version":"Exp24-B","dataset":"GMNER DEV","ot_condition":"bbox/None-supervised patch+dustbin","transported_mass":bst.get("matched_transport_mass"),
                 "unmatched_mass":bst.get("unmatched_dustbin_mass"),"matching_entropy":bst.get("normalized_entropy"),"transport_concentration":"N/A","groundability_auroc":bst.get("groundability_auroc"),
                 "groundability_aupr":bst.get("groundability_aupr"),"dustbin_rejection_accuracy":bst.get("dustbin_rejection_accuracy"),"pointing_accuracy":bst.get("pointing_accuracy"),"bbox_iou":bst.get("mean_iou"),"acc_iou_05":bst.get("acc_iou_05"),
                 "false_match_rate":bst.get("false_match_rate"),"logit_probability_drift":"N/A","prediction_changes":"N/A","supervision":"Additional GMNER bbox/None labels"})
    write(RES/"ppib_mechanism.csv",ppib_rows());write(RES/"gota_mechanism.csv",gota);write(RES/"visual_counterfactual.csv",visual)
    ranks=[]
    for route in ("A","C"):
        st,_=status(route);vr=[x for x in visual if x["route"]==route and x["condition"]!="clean"]
        causal=any(x["prediction_changes"]>0 or x["mean_abs_probability_drift"]>1e-6 for x in vr);constraint=float(st["best_dev_f1"])+.005>=float(st["initial_dev_f1"])
        ranks.append({"route":route,"version":"PGMNER mechanism refinement","performance_constraint_pass":constraint,"mechanism_evidence":causal,"uses_extra_bbox_supervision":False,"candidate":bool(constraint and causal),"reason":"Clean/counterfactual and OT interventions; no true bbox labels."})
    constraint=True;mechanism=(float(bst.get("groundability_auroc",0))>.5 and float(bst.get("pointing_accuracy",0))>float(bst.get("area_expectation",1)))
    ranks.append({"route":"B","version":"bbox-supervised extension","performance_constraint_pass":"separate task","mechanism_evidence":mechanism,"uses_extra_bbox_supervision":True,"candidate":bool(mechanism),"reason":"Requires real GMNER bbox/None gains; not comparable to Twitter F1."})
    write(RES/"mechanism_candidate_ranking.csv",ranks)
    report=f"""# Experiment 24 Mechanism Validation Report

## Evaluation Position

The primary question is whether information selection and partial matching have a measurable, causally relevant role. DEV MNER F1 is retained only as a non-degradation constraint (within 0.005 absolute of the route's checkpoint-initialized DEV result). No TEST data are used.

## PP-IB

Experiment 24 does not modify PP-IB. Therefore all PP-IB statements are archival diagnoses from Experiments 16C/19/20/23 and are not attributed to Routes A-C. The original gate is sample-level, nearly constant at 0.5, receives zero MNER gradient, and has no established prediction or modality-relevance effect. Prediction-coupled variants established a gradient path, but matched fixed-gate controls or gate collapse prevented a selection-mechanism claim. See `results/ppib_mechanism.csv`.

## GOT-A and Supervision Regimes

- Routes A/C are PGMNER refinements trained with Twitter-2017 MNER labels only. Because Twitter-2017 has no bbox/groundability annotation, true Groundability AUROC, Pointing, IoU, and false-match rate are **N/A**.
- Route B is a separate extension trained with additional GMNER bbox/None labels. Its grounding metrics cannot be presented as evidence for the original unsupervised GOT-A protocol.
- Partial, uniform, and full transport interventions for A/C quantify changes in probabilities and discrete predictions. Lower entropy alone is not treated as success.

## Counterfactual Images

Random images use a deterministic global shift. Hard negatives are nearest non-self, pixel-nonduplicate images under the route's frozen visual global embedding. This is a visual-similarity proxy, not ground-truth semantic mismatch. Results are in `results/visual_counterfactual.csv`.

## Candidate Decision

Candidate rows are in `results/mechanism_candidate_ranking.csv`. A route qualifies only when its performance constraint is met and at least one causal/alignment metric changes meaningfully. Variance, entropy, or prescribed mass alone never qualifies a route.

## Limitations

This is a short seed-42 DEV screen. Route B's bbox decoder is selected on DEV and reported on DEV, so it is diagnostic rather than an unbiased final estimate. A/C cannot validate spatial correctness without bbox labels. No further training is launched automatically.
"""
    (EXP/"MECHANISM_VALIDATION_REPORT.md").write_text(report)
    print(json.dumps({"status":"complete","outputs":["ppib_mechanism.csv","gota_mechanism.csv","visual_counterfactual.csv","mechanism_candidate_ranking.csv"]},indent=2))

if __name__=="__main__":main()
