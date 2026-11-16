#!/usr/bin/env python3
"""Exp16C per-loss PP-IB gradient contribution audit; no optimizer step."""
from __future__ import annotations
import csv, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset

ROOT=Path(__file__).resolve().parents[3]; EXP=ROOT/"kais_extension/experiment16c_ppib_audit"; RES=EXP/"results"
CKPT=Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
ARGS=["main-mner-robert.py","--data_dir","data/twitter2017","--output_dir",str(EXP/"cache/grad"),"--model_name_or_path","pretrain/roberta-large","--tokenizer_name","pretrain/roberta-large","--vit_name","pretrain/clip-vit-base-patch16","--use_hf_clip_vision","true","--auto_img_dims","true","--gpu_id","0","--batch_size","2","--max_seq_len","128","--max_span_len","4","--seed","42","--use_pp_ib","--beta_polar","0.01","--delta_minus","0.2","--delta_plus","0.8","--pp_ib_lambda_balance","1.0","--beta_polar_warmup_epochs","3","--use_got","--got_mode","partial_sinkhorn","--ot_alpha","0.05","--entropy_eps","0.1","--ot_sinkhorn_iter","30","--partial_mass_init","0.5","--dust_cost","1.0","--do_train","false","--do_eval","false","--do_predict","false"]

def load_main():
 spec=importlib.util.spec_from_file_location("main_exp16c_grad",ROOT/"main-mner-robert.py");m=importlib.util.module_from_spec(spec);old=sys.argv;sys.argv=ARGS
 try:spec.loader.exec_module(m)
 finally:sys.argv=old
 m.args.labels=str((ROOT/"kais_extension/configs/labels_twitter2017.txt").resolve());return m
def remap(s):
 old="image_model._hf.encoder.vision_model.";new="image_model._hf.encoder.";return {new+k[len(old):] if k.startswith(old) else k:v for k,v in s.items()}
def groups(model):
 return {"text_gate":list(model.dist_t.polar_gate.parameters()),"text_mu":list(model.dist_t.mu_net.parameters()),"text_logvar":list(model.dist_t.logvar_net.parameters()),"visual_gate":list(model.dist_v.polar_gate.parameters()),"visual_mu":list(model.dist_v.mu_net.parameters()),"visual_logvar":list(model.dist_v.logvar_net.parameters())}
def grad(loss,params):
 gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True); vals=[g for g in gs if g is not None]
 return {"connected_parameters":len(vals),"total_parameters":len(params),"grad_l2":float(torch.sqrt(sum((g.float()**2).sum() for g in vals)).item()) if vals else 0.0,"grad_abs_max":float(max(g.abs().max().item() for g in vals)) if vals else 0.0,"finite":bool(all(torch.isfinite(g).all() for g in vals)) if vals else True}
def main():
 os.chdir(ROOT);RES.mkdir(parents=True,exist_ok=True);m=load_main();labels=m.get_labels(m.args.labels);cfg=m.AutoConfig.from_pretrained(m.args.model_name_or_path);tok=m.AutoTokenizer.from_pretrained(m.args.tokenizer_name);model=m.BertSpanNerBN(cfg,args=m.args,num_labels=len(labels));model.load_state_dict(remap(torch.load(CKPT,map_location="cpu",weights_only=True)),strict=True);model.to(m.args.device);model.train();model.set_polar_train_epoch(3);model.set_partial_ot_warmup(False)
 ex=m.load_examples(m.args.data_dir,"train",tok,max_seq_len=m.args.max_seq_len);from utils.datasets import SpanNerDataset,collate_fn
 ds=SpanNerDataset(ex,args=m.args,tokenizer=tok,labels=labels);sub=Subset(ds,list(range(8)));dl=DataLoader(sub,sampler=SequentialSampler(sub),batch_size=2,collate_fn=collate_fn); rows=[]; losses_rows=[]; t0=time.time();torch.cuda.reset_peak_memory_stats(m.args.device)
 for bi,bc in enumerate(dl):
  b={k:v.to(m.args.device) for k,v in bc.items()};torch.manual_seed(160300+bi);out,L=model(b)
  comps={"classification":L["c"],"supervised_contrastive":L["scl"],"mner_task":L["c"]+L["scl"],"ib_kl":L["loss_ib_tv"],"polarization":L["loss_polar"],"got":L["ot_total"],"total":L["loss"]}
  losses_rows.append({"batch":bi,**{k:float(v.detach()) for k,v in comps.items()}})
  for lname,loss in comps.items():
   for gname,ps in groups(model).items():rows.append({"batch":bi,"loss_component":lname,"parameter_group":gname,"requires_grad":all(p.requires_grad for p in ps),**grad(loss,ps)})
 # cancellation diagnostic: total should equal vector sum; compare combined norms per group and cosine IB/polar where both connect
 agg=[]
 for lname in sorted(set(r["loss_component"] for r in rows)):
  for gname in sorted(set(r["parameter_group"] for r in rows)):
   q=[r for r in rows if r["loss_component"]==lname and r["parameter_group"]==gname];agg.append({"batch":"mean","loss_component":lname,"parameter_group":gname,"requires_grad":q[0]["requires_grad"],"connected_parameters_mean":np.mean([r["connected_parameters"] for r in q]),"total_parameters":q[0]["total_parameters"],"grad_l2_mean":np.mean([r["grad_l2"] for r in q]),"grad_l2_std":np.std([r["grad_l2"] for r in q]),"grad_abs_max_mean":np.mean([r["grad_abs_max"] for r in q]),"finite":all(r["finite"] for r in q)})
 with (RES/"gradient_contribution.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (RES/"gradient_contribution_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 with (RES/"gradient_component_losses.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(losses_rows[0]));w.writeheader();w.writerows(losses_rows)
 meta={"checkpoint":str(CKPT),"train_batches":4,"batch_size":2,"optimizer_steps":0,"all_ppib_parameters_trainable":all(p.requires_grad for p in list(model.dist_t.parameters())+list(model.dist_v.parameters())),"ib_definition":"0.001 * mean(KL_text + KL_visual); sampled gated z not used by this term","polar_definition":"0.01 * (text Lpolar + visual Lpolar) at completed warmup","got_definition":"0.05 * Partial-OT loss on local span/patch path","seconds":time.time()-t0,"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_mb":torch.cuda.max_memory_allocated(m.args.device)/1024**2}
 (RES/"gradient_audit_metadata.json").write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
