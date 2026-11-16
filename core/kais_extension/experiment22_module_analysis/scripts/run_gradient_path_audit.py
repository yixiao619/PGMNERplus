#!/usr/bin/env python3
from __future__ import annotations
import csv,json,time
import numpy as np
import torch
from torch.utils.data import DataLoader,SequentialSampler,Subset
from common import RESULTS,ROOT,build_model,load_main

def groups(m):
    g={"shared_text_encoder":list(m.bert.parameters()),"shared_visual_encoder":list(m.image_model.parameters()),"span_encoder":list(m.span_layer.parameters()),"gota_text_projection":list(m.semantic_ot_aligner.text_proj.parameters()),"gota_visual_projection":list(m.semantic_ot_aligner.visual_proj.parameters()),"fusion_and_classifier":list(m.fusion_text.parameters())+list(m.ln_text.parameters())+list(m.span_classifier.parameters()),"ppib_text_gate":list(m.dist_t.polar_gate.parameters()),"ppib_text_mu_logvar":list(m.dist_t.mu_net.parameters())+list(m.dist_t.logvar_net.parameters()),"ppib_visual_gate":list(m.dist_v.polar_gate.parameters()),"ppib_visual_mu_logvar":list(m.dist_v.mu_net.parameters())+list(m.dist_v.logvar_net.parameters()),"partial_mass":[m.semantic_ot_aligner._partial_ot.s_logit]}
    return {k:[p for p in v if p.requires_grad] for k,v in g.items()}
def stats(loss,ps):
    gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True); gs=[g for g in gs if g is not None]
    if not gs:return 0,0.,0.,True
    return len(gs),float(torch.sqrt(sum((g.float()**2).sum() for g in gs))),float(max(g.abs().max() for g in gs)),bool(all(torch.isfinite(g).all() for g in gs))
def main():
    RESULTS.mkdir(parents=True,exist_ok=True); mod=load_main("data/twitter2017",ROOT/"kais_extension/configs/labels_twitter2017.txt"); model,tok,labels=build_model(mod); model.train()
    ex=mod.load_examples(mod.args.data_dir,"train",tok,max_seq_len=mod.args.max_seq_len); from utils.datasets import SpanNerDataset,collate_fn
    ds=SpanNerDataset(ex,args=mod.args,tokenizer=tok,labels=labels); sub=Subset(ds,list(range(4))); dl=DataLoader(sub,sampler=SequentialSampler(sub),batch_size=2,collate_fn=collate_fn)
    rows=[]; values=[]; t0=time.time(); torch.cuda.reset_peak_memory_stats(mod.args.device)
    for bi,bc in enumerate(dl):
        b={k:v.to(mod.args.device) for k,v in bc.items()}; torch.manual_seed(22000+bi); _,L=model(b)
        cs={"mner_classification":L["c"],"mner_task_with_scl":L["c"]+L["scl"],"ppib_kl_weighted":L["loss_ib_tv"],"ppib_polarization_weighted":L["loss_polar"],"gota_transport_weighted":L["ot_total"],"total_training_loss":L["loss"]}; values.append({"batch":bi,**{k:float(v.detach()) for k,v in cs.items()}})
        for ln,loss in cs.items():
            for gn,ps in groups(model).items():
                n,norm,mx,finite=stats(loss,ps); rows.append({"batch":bi,"loss_component":ln,"parameter_group":gn,"trainable_parameter_tensors":len(ps),"connected_parameter_tensors":n,"grad_l2":norm,"grad_abs_max":mx,"finite":finite})
    for ln in sorted({r["loss_component"] for r in rows}):
        for gn in groups(model):
            q=[r for r in rows if r["loss_component"]==ln and r["parameter_group"]==gn]; rows.append({"batch":"mean","loss_component":ln,"parameter_group":gn,"trainable_parameter_tensors":q[0]["trainable_parameter_tensors"],"connected_parameter_tensors":np.mean([r["connected_parameter_tensors"] for r in q]),"grad_l2":np.mean([r["grad_l2"] for r in q]),"grad_abs_max":np.mean([r["grad_abs_max"] for r in q]),"finite":all(r["finite"] for r in q)})
    for path,data in [(RESULTS/"gradient_path_audit.csv",rows),(RESULTS/"gradient_loss_values.csv",values)]:
        with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    meta={"train_batches":2,"batch_size":2,"optimizer_steps":0,"elapsed_seconds":time.time()-t0,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated(mod.args.device)/1024**2,"gpu":torch.cuda.get_device_name(0)}; (RESULTS/"gradient_path_audit_metadata.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
