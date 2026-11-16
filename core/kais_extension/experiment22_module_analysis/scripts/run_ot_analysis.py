#!/usr/bin/env python3
"""Fair Partial-vs-Full OT comparison on identical frozen projections and costs."""
from __future__ import annotations
import csv,json,math,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,SequentialSampler
from common import EXP,FIGURES,RESULTS,build_model,load_main

DATA=EXP.parent/"experiment9_gmner_audit/cache/gmner_pgmner_format"
SIDECAR=DATA/"gmner_entity_sidecar.jsonl"

def sidecar():
    out={}; counts=defaultdict(int)
    for line in SIDECAR.open():
        r=json.loads(line); sp=r["split"]; counts[sp]+=1; out[(sp,str(r["sample_id"]))]=r; out[(sp,f"{sp}-{counts[sp]}")]=r
    return out
def sinkhorn(C,a,b,iters=30,eps=.1):
    la=torch.log(a.clamp_min(1e-12)); lb=torch.log(b.clamp_min(1e-12)); K=-C/eps; u=torch.zeros_like(la);v=torch.zeros_like(lb)
    for _ in range(iters):v=lb-torch.logsumexp(K+u.unsqueeze(2),dim=1);u=la-torch.logsumexp(K+v.unsqueeze(1),dim=2)
    return torch.exp(K+u.unsqueeze(2)+v.unsqueeze(1))
def solve(C,mt,mv,method,s=.6,dust_cost=1.):
    B,n,m=C.shape;dt=C.dtype;dev=C.device; nt=mt.sum(1,keepdim=True).clamp_min(1e-12);nv=mv.sum(1,keepdim=True).clamp_min(1e-12)
    if method=="full":
        T=sinkhorn(C+(1-mt[:,:,None]*mv[:,None,:])*1e4,mt/nt,mv/nv); z=torch.zeros(B,device=dev,dtype=dt); return T,z,z,z
    Caug=torch.full((B,n+1,m+1),dust_cost,device=dev,dtype=dt);Caug[:,:n,:m]=C+(1-mt[:,:,None]*mv[:,None,:])*1e4;Caug[:,n,m]=0
    a=torch.zeros(B,n+1,device=dev,dtype=dt);b=torch.zeros(B,m+1,device=dev,dtype=dt);a[:,:n]=s*mt/nt;a[:,n]=1-s;b[:,:m]=s*mv/nv;b[:,m]=1-s
    A=sinkhorn(Caug,a,b);return A[:,:n,:m],A[:,:n,m].sum(1),A[:,n,:m].sum(1),A[:,n,m]
def metrics(T,C,mt,mv,rd,dr,dd):
    mask=mt[:,:,None]*mv[:,None,:]; X=T*mask; mass=X.sum((1,2)); q=X/(mass[:,None,None]+1e-12); ent=-(q.clamp_min(1e-12)*q.clamp_min(1e-12).log()*mask).sum((1,2)); valid=(mt.sum(1)*mv.sum(1)).clamp_min(2); nent=ent/valid.log(); col=X.sum(1);cq=col/(mass[:,None]+1e-12); eff=torch.exp(-(cq.clamp_min(1e-12)*cq.clamp_min(1e-12).log()*mv).sum(1)); raw=(X*C).sum((1,2)); conc=(q*q).sum((1,2)); top=q.amax((1,2)); return mass,ent,nent,eff,top,raw,raw/(mass+1e-12),conc,rd,dr,dd
def write(path,rows):
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    RESULTS.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True);mod=load_main(DATA,DATA/"labels.txt",batch_size=8);model,tok,labels=build_model(mod);model.eval()
    ex=mod.load_examples(str(DATA),"test",tok,max_seq_len=128);from utils.datasets import SpanNerDataset,collate_fn
    ds=SpanNerDataset(ex,args=mod.args,tokenizer=tok,labels=labels,dev=True);dl=DataLoader(ds,sampler=SequentialSampler(ds),batch_size=8,collate_fn=collate_fn);meta=sidecar();rows=[];cases=[];offset=0;t0=time.time();sck=float(model.semantic_ot_aligner._partial_ot.get_s().detach())
    with torch.no_grad():
      for bi,bc in enumerate(dl):
        b={k:v.to(mod.args.device) for k,v in bc.items()}; spans,tg,attn=model.span_encoding(**b);ig,iv=model.image_model(b["image"]);out=model.semantic_ot_aligner(spans,iv,mask_t=b["span_masks"].float(),mask_v=None,attn_t=attn,span_token_idxes=b.get("span_token_idxes"));tp,vp=out[0],out[1];C=.5*(1-torch.einsum("btd,bvd->btv",F.normalize(tp,dim=-1),F.normalize(vp,dim=-1)));mt=b["span_masks"].float();mv=torch.ones(C.shape[0],C.shape[2],device=C.device)
        pred=model.span_classifier(model.ln_text(model.fusion_text(torch.cat([tp,out[2]],-1))+out[2])).argmax(-1); gold=b["span_labels"]
        methods=[("partial_checkpoint",sck),("partial_fixed_0.6",.6),("full_ot",1.)]
        solved={name:(solve(C,mt,mv,"full" if name=="full_ot" else "partial",s) if name!="partial_checkpoint" else solve(C,mt,mv,"partial",sck)) for name,s in methods}
        for k in range(C.shape[0]):
          sample=ex[offset+k]; sm=meta.get(("test",str(getattr(sample,"guid",None)))) or {}; ents=[e for e in sm.get("entities",[]) if e.get("matched_in_text")]; g=sum(bool(e.get("bbox")) for e in ents); group="no_groundable" if g==0 else "all_groundable" if g==len(ents) else "mixed_groundability"; valid=mt[k].bool(); entity_mask=(gold[k]>0)&valid; mcorrect=float((pred[k][entity_mask]==gold[k][entity_mask]).float().mean()) if entity_mask.any() else float("nan")
          for name,s in methods:
            T,rd,dr,dd=solved[name]; vals=metrics(T,C,mt,mv,rd,dr,dd); keys=["transported_mass","matching_entropy","normalized_entropy","effective_matched_region_count","top_matching_score","cost_objective","mass_normalized_cost","transport_concentration","real_to_dust","dust_to_real","dust_to_dust"]
            row={"sample_index":offset+k,"sample_id":getattr(sample,"guid",offset+k),"image_id":getattr(sample,"imgid",""),"method":name,"configured_mass":s,"groundability_group":group,"groundable_fraction":g/len(ents) if ents else "","annotated_entities":len(ents),"mner_gold_entity_accuracy":mcorrect,"valid_spans":int(valid.sum()),"visual_nodes":int(mv[k].sum())}
            row.update({key:float(val[k]) for key,val in zip(keys,vals)});row["dust_edge_mass"]=row["real_to_dust"]+row["dust_to_real"];rows.append(row)
          if len(cases)<6 and group not in [c["group"] for c in cases]: cases.append({"sample_id":str(getattr(sample,"guid",offset+k)),"group":group,"partial":solved["partial_checkpoint"][0][k,valid].cpu().numpy().tolist(),"full":solved["full_ot"][0][k,valid].cpu().numpy().tolist()})
        offset+=C.shape[0]; print(f"batch {bi+1}/{len(dl)} samples={offset}",flush=True)
    write(RESULTS/"partial_vs_full_ot.csv",rows);(RESULTS/"ot_heatmap_cases.json").write_text(json.dumps(cases)+"\n")
    # Grouped aggregation is deliberately exhaustive: all methods x all predefined annotation groups.
    grouped=[]
    for method in sorted({r["method"] for r in rows}):
      for group in ["no_groundable","mixed_groundability","all_groundable"]:
        q=[r for r in rows if r["method"]==method and r["groundability_group"]==group]
        if not q:continue
        for metric in ["transported_mass","dust_edge_mass","normalized_entropy","effective_matched_region_count","top_matching_score","mass_normalized_cost","transport_concentration"]:
          a=np.array([r[metric] for r in q]);grouped.append({"method":method,"relevance_definition":"GMNER annotation-derived sample groundability group","relevance_group":group,"metric":metric,"n":len(a),"mean":a.mean(),"std":a.std(),"median":np.median(a),"q25":np.quantile(a,.25),"q75":np.quantile(a,.75)})
    write(RESULTS/"ot_visual_relevance.csv",grouped); metaout={"samples":offset,"rows":len(rows),"checkpoint_s":sck,"elapsed_seconds":time.time()-t0,"cost":"0.5*(1-cosine) shared by all methods","full_constraints":"uniform valid marginals, total mass 1, no dust","partial_constraints":"uniform real marginals summing s plus one dust marginal of 1-s on each side","fixed_control_mass":.6,"visual_nodes_include_clip_cls":True};(RESULTS/"ot_analysis_metadata.json").write_text(json.dumps(metaout,indent=2)+"\n");print(json.dumps(metaout,indent=2))
if __name__=="__main__":main()
