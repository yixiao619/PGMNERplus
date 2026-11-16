#!/usr/bin/env python3
"""Exp16C representation robustness audit (inference only)."""
from __future__ import annotations

import argparse, csv, importlib.util, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension/experiment16c_ppib_audit"
RES = EXP / "results"
CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
CONDITIONS = ["clean", "mild_blur", "moderate_blur", "random_image"]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def args_parser():
    p=argparse.ArgumentParser(); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--gpu-id",default="0"); return p.parse_args()

def load_main(cli):
    argv=["main-mner-robert.py","--data_dir","data/twitter2017","--output_dir",str(EXP/"cache"),
      "--model_name_or_path","pretrain/roberta-large","--tokenizer_name","pretrain/roberta-large","--vit_name","pretrain/clip-vit-base-patch16",
      "--gpu_id",cli.gpu_id,"--batch_size",str(cli.batch_size),"--max_seq_len","128","--max_span_len","4","--seed","42",
      "--use_pp_ib","--beta_polar","0.01","--delta_minus","0.2","--delta_plus","0.8","--pp_ib_lambda_balance","1.0",
      "--use_got","--got_mode","partial_sinkhorn","--ot_alpha","0.05","--entropy_eps","0.1","--ot_sinkhorn_iter","30",
      "--partial_mass_init","0.5","--dust_cost","1.0","--auto_img_dims","true","--use_hf_clip_vision","true",
      "--do_train","false","--do_eval","false","--do_predict","false"]
    spec=importlib.util.spec_from_file_location("main_exp16c_rep",ROOT/"main-mner-robert.py"); mod=importlib.util.module_from_spec(spec)
    old=sys.argv; sys.argv=argv
    try: spec.loader.exec_module(mod)
    finally: sys.argv=old
    mod.args.labels=str((ROOT/"kais_extension/configs/labels_twitter2017.txt").resolve()); return mod

def remap(state):
    old="image_model._hf.encoder.vision_model."; new="image_model._hf.encoder."
    return {new+k[len(old):] if k.startswith(old) else k:v for k,v in state.items()}

def random_examples(examples):
    rng=random.Random(42); perm=list(range(len(examples)))
    while True:
        rng.shuffle(perm)
        if all(i!=j for i,j in enumerate(perm)): break
    out=[]
    for i,j in enumerate(perm):
        x=type("Obj",(),{})(); x.__dict__.update(examples[i].__dict__.copy()); x.imgid=examples[j].imgid; out.append(x)
    return out

def gaussian(x,k,sigma):
    q=torch.arange(k,device=x.device,dtype=x.dtype)-(k-1)/2; g=torch.exp(-(q*q)/(2*sigma*sigma)); g/=g.sum(); ker=(g[:,None]*g[None,:]).expand(x.shape[1],1,k,k)
    return F.conv2d(F.pad(x,(k//2,)*4,mode="reflect"),ker,groups=x.shape[1])

def entity_sets(idx,probs,labels,masks):
    raw=probs.argmax(-1); cand=[]
    for i in range(len(raw)):
        if masks[i]<=0: continue
        ws,we=int(idx[i,0]),int(idx[i,1]); r=int(raw[i]); cand.append((ws,we,r,float(probs[i,r]),int(labels[i])))
    chosen=[]
    for c in sorted([x for x in cand if x[2]!=0],key=lambda x:-x[3]):
        if not any(not(c[1]<z[0] or z[1]<c[0]) for z in chosen): chosen.append(c)
    return {(x[0],x[1],x[2]) for x in chosen},{(x[0],x[1],x[4]) for x in cand if x[4]!=0}

def sim(a,b):
    a=a.float().reshape(a.shape[0],-1); b=b.float().reshape(b.shape[0],-1)
    return F.cosine_similarity(a,b,dim=1), (a-b).norm(dim=1)/(((a.norm(dim=1)+b.norm(dim=1))/2).clamp_min(1e-12))

def reduce_spans(x,mask):
    m=mask.float().unsqueeze(-1); return (x*m).sum(1)/m.sum(1).clamp_min(1)

def run():
    cli=args_parser(); os.chdir(ROOT); RES.mkdir(parents=True,exist_ok=True); mod=load_main(cli); labels=mod.get_labels(mod.args.labels)
    cfg=mod.AutoConfig.from_pretrained(mod.args.model_name_or_path,id2label={str(i):v for i,v in enumerate(labels)},label2id={v:i for i,v in enumerate(labels)})
    tok=mod.RobertaTokenizer.from_pretrained(mod.args.tokenizer_name); model=mod.BertSpanNerBN(cfg,args=mod.args,num_labels=len(labels))
    model.load_state_dict(remap(torch.load(CKPT,map_location="cpu",weights_only=True)),strict=True); model.to(mod.args.device).eval(); model.set_partial_ot_warmup(False)
    ex=mod.load_examples(mod.args.data_dir,"test",tok,max_seq_len=mod.args.max_seq_len); rex=random_examples(ex)
    from utils.datasets import SpanNerDataset,collate_fn
    ds=SpanNerDataset(ex,args=mod.args,tokenizer=tok,labels=labels,dev=True); dr=SpanNerDataset(rex,args=mod.args,tokenizer=tok,labels=labels,dev=True)
    dl=DataLoader(ds,sampler=SequentialSampler(ds),batch_size=cli.batch_size,collate_fn=collate_fn); rl=DataLoader(dr,sampler=SequentialSampler(dr),batch_size=cli.batch_size,collate_fn=collate_fn)
    capture={}
    def prehook(name): return lambda m,a: capture.__setitem__(name+"_pre",a[0].detach())
    def posthook(name):
        def f(m,a,o):
            capture[name+"_mu"]=o[0].detach(); capture[name+"_gated"]=o[2].detach(); capture[name+"_gate"]=m.last_probe["gate"].detach()
        return f
    hs=[model.dist_t.register_forward_pre_hook(prehook("text")),model.dist_t.register_forward_hook(posthook("text")),model.dist_v.register_forward_pre_hook(prehook("visual")),model.dist_v.register_forward_hook(posthook("visual")),model.ln_text.register_forward_hook(lambda m,a,o:capture.__setitem__("fused",o.detach()))]
    sample_rows=[]; counts={c:[0,0,0] for c in CONDITIONS}; t0=time.time(); gi=0
    torch.cuda.reset_peak_memory_stats(mod.args.device)
    with torch.no_grad():
      for bi,(bc,rc) in enumerate(zip(dl,rl)):
        base={k:v.to(mod.args.device) for k,v in bc.items()}; rand={k:v.to(mod.args.device) for k,v in rc.items()}; cleanrep=None
        for ci,cond in enumerate(CONDITIONS):
            b=dict(base)
            if cond=="mild_blur": b["image"]=gaussian(base["image"],5,1.0)
            elif cond=="moderate_blur": b["image"]=gaussian(base["image"],9,2.0)
            elif cond=="random_image": b["image"]=rand["image"]
            torch.manual_seed(160000+bi)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(160000+bi)
            capture.clear(); out,_=model(b); probs=out[0]
            reps={"text_pre":capture["text_pre"],"text_mu":capture["text_mu"],"text_gated":capture["text_gated"],
              "visual_pre":capture["visual_pre"],"visual_mu":capture["visual_mu"],"visual_gated":capture["visual_gated"],
              "fused_classifier_input":reduce_spans(capture["fused"],base["span_masks"])}
            for branch in ("text","visual"):
                g=capture[branch+"_gate"].clamp_min(1e-8); z=capture[branch+"_gated"]/g; reps[branch+"_sampled_z"]=z; reps[branch+"_fixed_gate_0.5"]=.5*z
            if cond=="clean": cleanrep={k:v.clone() for k,v in reps.items()}
            bsz=base["input_ids"].shape[0]
            for j in range(bsz):
                pred,gold=entity_sets(base["span_word_idxes"][j],probs[j],base["span_labels"][j],base["span_masks"][j]); counts[cond][0]+=len(pred&gold); counts[cond][1]+=len(pred-gold); counts[cond][2]+=len(gold-pred)
            for name,val in reps.items():
                co,nd=sim(cleanrep[name],val)
                for j in range(bsz): sample_rows.append({"sample_index":gi+j,"sample_id":ex[gi+j].guid,"condition":cond,"representation":name,"cosine_similarity_to_clean":float(co[j]),"normalized_l2_drift":float(nd[j]),"text_gate":float(capture["text_gate"][j]),"visual_gate":float(capture["visual_gate"][j])})
        gi+=base["input_ids"].shape[0]
        if (bi+1)%25==0: print(f"batch {bi+1}/{len(dl)}",flush=True)
    for h in hs:h.remove()
    fields=list(sample_rows[0]);
    with (RES/"representation_robustness_samples.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sample_rows)
    agg=[]
    for cond in CONDITIONS:
      tp,fp,fn=counts[cond]; p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0;f1=2*p*r/(p+r) if p+r else 0
      for name in sorted({x["representation"] for x in sample_rows}):
        q=[x for x in sample_rows if x["condition"]==cond and x["representation"]==name]; c=np.array([x["cosine_similarity_to_clean"] for x in q]); d=np.array([x["normalized_l2_drift"] for x in q])
        agg.append({"condition":cond,"representation":name,"n":len(q),"mean_cosine":c.mean(),"median_cosine":np.median(c),"std_cosine":c.std(),"mean_normalized_l2":d.mean(),"median_normalized_l2":np.median(d),"std_normalized_l2":d.std(),"precision":p,"recall":r,"f1":f1,"tp":tp,"fp":fp,"fn":fn})
    with (RES/"representation_robustness.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    meta={"checkpoint":str(CKPT),"n_samples":len(ex),"conditions":{"clean":"unaltered preprocessed tensor","mild_blur":"5x5 Gaussian sigma=1.0 after normalization","moderate_blur":"9x9 Gaussian sigma=2.0 after normalization","random_image":"seed42 derangement"},"matched_vib_noise":True,"fixed_gate_control":"0.5 * recovered sampled z; LayerNorm/fusion classifier path is separately captured","seconds":time.time()-t0,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated(mod.args.device)/1024**2,"gpu":torch.cuda.get_device_name(0)}
    (RES/"representation_robustness_metadata.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2))

if __name__=="__main__": run()
