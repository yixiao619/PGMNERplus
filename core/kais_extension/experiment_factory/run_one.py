#!/usr/bin/env python3
"""Execute one immutable Factory V2 job. DEV-only; never evaluates TEST."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from transformers import get_linear_schedule_with_warmup

ROOT=Path("/home/jyx/yixiao/MINER-My");FACTORY=ROOT/"kais_extension/experiment_factory"
# When invoked as a script, Python prepends FACTORY and would resolve its
# helper package as top-level ``models``, shadowing the original PGMNER package.
sys.path=[p for p in sys.path if Path(p or ".").resolve()!=FACTORY.resolve()];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
def cli():
 p=argparse.ArgumentParser();p.add_argument("--job",required=True);p.add_argument("--run-dir",required=True);p.add_argument("--smoke",action="store_true");return p.parse_args()
def main_module(out,seed):
 argv=["main-mner-robert.py","--data_dir","data/twitter2017","--output_dir",str(out),"--model_name_or_path","pretrain/roberta-large","--tokenizer_name","pretrain/roberta-large","--vit_name","pretrain/clip-vit-base-patch16","--use_hf_clip_vision","true","--auto_img_dims","true","--gpu_id","0","--batch_size","16","--max_seq_len","128","--max_span_len","4","--seed",str(seed),"--use_pp_ib","--beta_polar","0.01","--delta_minus","0.2","--delta_plus","0.8","--pp_ib_lambda_balance","1.0","--beta_polar_warmup_epochs","3","--use_got","--got_mode","partial_sinkhorn","--ot_alpha","0.05","--entropy_eps","0.1","--ot_sinkhorn_iter","30","--partial_mass_init","0.5","--dust_cost","1.0","--ot_warmup_epochs","0","--do_train","false","--do_eval","false","--do_predict","false"]
 spec=importlib.util.spec_from_file_location("factory_main",ROOT/"main-mner-robert.py");m=importlib.util.module_from_spec(spec);old=sys.argv;sys.argv=argv
 try:spec.loader.exec_module(m)
 finally:sys.argv=old
 m.args.labels=str(ROOT/"kais_extension/configs/labels_twitter2017.txt");m.args.output_dir=str(out);return m
def checkpoint():
 p=Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth");s=torch.load(p,map_location="cpu",weights_only=True);old="image_model._hf.encoder.vision_model.";new="image_model._hf.encoder.";return {new+k[len(old):] if k.startswith(old) else k:v for k,v in s.items()}
def write_csv(path,rows):
 if not rows:return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 tmp=path.with_suffix(".tmp");
 with tmp.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 tmp.replace(path)
def grad_norm(loss,ps):
 ps=[p for p in ps if p.requires_grad];gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True) if ps else [];gs=[g for g in gs if g is not None];return float(torch.sqrt(sum((g.float()**2).sum() for g in gs))) if gs else 0.0
def configure(job,out,seed):
 m=main_module(out,seed);m.set_seed(m.args);labels=m.get_labels(m.args.labels);cfg=m.AutoConfig.from_pretrained(m.args.model_name_or_path);tok=m.AutoTokenizer.from_pretrained(m.args.tokenizer_name);route=job["route"]
 if route in ("P2","P3"):
  m.args.exp20_kind="entity";m.args.exp20_gate_mode=job.get("gate_mode","learned");m.args.exp20_alpha=float(job.get("alpha",.5));m.args.exp20_bottleneck=128
  if route=="P3":
   m.args.factory_gate_temperature=float(job["gate_temperature"]);m.args.factory_gate_variance_target=float(job["gate_variance_target"]);m.args.factory_gate_reg_weight=float(job["gate_reg_weight"]);from kais_extension.experiment_factory.models.ppib_models import StablePCPPModel as Cls
  else:from kais_extension.experiment20_overnight_ppib.models.exp20_model import BertSpanNerBNExp20 as Cls
 else:from models.bn_bert_ner_OT_robert import BertSpanNerBN as Cls
 torch.manual_seed(2000);model=Cls(cfg,args=m.args,num_labels=len(labels));missing,unexpected=model.load_state_dict(checkpoint(),strict=False)
 if route not in ("P2","P3") and (missing or unexpected):raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
 if route in ("P2","P3") and (unexpected or any(not k.startswith("exp20_adapter") for k in missing)):raise RuntimeError(f"adapter checkpoint mismatch missing={missing} unexpected={unexpected}")
 if route=="G1" and job.get("ot_backend")=="full":
  from models.partial_ot import NodeSinkhorn;old=model.semantic_ot_aligner._partial_ot;model.semantic_ot_aligner._partial_ot=NodeSinkhorn(eps=old.eps,n_iters=old.n_iters)
 elif route=="G2":
  from kais_extension.experiment_factory.models.ot_backends import AdaptiveMassSinkhorn,FixedActualMassSinkhorn;old=model.semantic_ot_aligner._partial_ot
  model.semantic_ot_aligner._partial_ot=(AdaptiveMassSinkhorn(eps=old.eps,dust_cost=old.dust_cost,n_iters=old.n_iters,mass_min=job.get("mass_min",.3),mass_max=job.get("mass_max",.9)) if job["variant"]=="adaptive_mass" else FixedActualMassSinkhorn(eps=old.eps,dust_cost=old.dust_cost,n_iters=old.n_iters,actual_mass=job["fixed_mass"]))
 elif route=="G3":
  from kais_extension.experiment_factory.models.ot_backends import ReliabilityWeightedSinkhorn;old=model.semantic_ot_aligner._partial_ot;model.semantic_ot_aligner._partial_ot=ReliabilityWeightedSinkhorn(eps=old.eps,dust_cost=old.dust_cost,n_iters=old.n_iters,dim=int(m.args.hidden_dim),strength=float(job["reliability_strength"]));model.semantic_ot_aligner._partial_ot.s_logit.data.copy_(old.s_logit.data)
 model.to(m.args.device);model.set_partial_ot_warmup(False);return m,model,tok,labels,missing
def route_params(model,job):
 route=job["route"]
 if route in ("P2","P3"):return list(model.exp20_adapter.parameters())+list(model.dist_t.parameters())+list(model.dist_v.parameters())
 if route in ("G2","G3"):return list(model.semantic_ot_aligner._partial_ot.parameters())+list(model.semantic_ot_aligner.text_proj.parameters())+list(model.semantic_ot_aligner.visual_proj.parameters())
 if route in ("G1","G4"):return list(model.semantic_ot_aligner.parameters())
 if route=="P1":return list(model.dist_t.parameters())+list(model.dist_v.parameters())+list(model.bert.parameters())+list(model.image_model.parameters())
 return list(model.parameters())
def mechanism(model,job):
 g=[]
 for branch in (model.dist_t,model.dist_v):
  if getattr(branch,"last_probe",None) and "gate" in branch.last_probe:g.extend(branch.last_probe["gate"].detach().flatten().cpu().tolist())
 x=np.asarray(g or [.5]);d={"gate_mean":float(x.mean()),"gate_std":float(x.std()),"gate_polarization":float(np.mean((x<=.1)|(x>=.9))),"mass_mean":"","mass_std":"","transported_mass":"","unmatched_mass":"","matching_entropy":"","transport_concentration":"","reliability_std":"","residual_ratio":""}
 backend=getattr(model.semantic_ot_aligner,"_partial_ot",None);s=getattr(backend,"last_factory_stats",{}) if backend else {};d.update({k:s.get(k,d[k]) for k in ["mass_mean","mass_std","reliability_std"]})
 if hasattr(model,"last_exp20_summary"):d["residual_ratio"]=model.last_exp20_summary.get("residual_ratio","")
 if hasattr(model,"last_exp20_gate") and model.last_exp20_gate is not None:d["entity_gate_std"]=float(model.last_exp20_gate.detach().std())
 plan=getattr(model,"last_transport_plan",None)
 if plan is not None:
  t=plan.float().clamp_min(0);mass=t.sum((1,2));p=t/mass[:,None,None].clamp_min(1e-12);nz=(t>0).sum((1,2)).clamp_min(2);entropy=-(p*p.clamp_min(1e-12).log()).sum((1,2))/nz.float().log();d.update({"transported_mass":float(mass.mean()),"unmatched_mass":float((1-mass).clamp_min(0).mean()),"matching_entropy":float(entropy.mean()),"transport_concentration":float((t.amax((1,2))/mass.clamp_min(1e-12)).mean())})
 return d
def scan(model,loader,device):
 model.eval();pred=[];stats=[]
 with torch.no_grad():
  for bc in loader:
   b={k:v.to(device) for k,v in bc.items()};p=model(b)[0][0];mask=b["span_masks"].bool();pred.extend(p.argmax(-1)[mask].cpu().tolist());stats.append(mechanism(model,{}))
 out={"predictions":pred}
 for k in stats[0]:
  vals=[float(x[k]) for x in stats if x.get(k,"")!=""];out[k]=float(np.mean(vals)) if vals else ""
 return out
def smoke(job,run_dir):
 m,model,tok,labels,missing=configure(job,run_dir,42);from utils.datasets import SpanNerDataset,collate_fn
 ex=m.load_examples(m.args.data_dir,"train",tok,max_seq_len=128)[:8];ds=SpanNerDataset(ex,args=m.args,tokenizer=tok,labels=labels);batch=next(iter(DataLoader(ds,batch_size=2,collate_fn=collate_fn)));b={k:v.to(m.args.device) for k,v in batch.items()};model.train();_,L=model(b);task=L["c"]+L["scl"];gn=grad_norm(task,route_params(model,job));L["loss"].backward();finite=all(torch.isfinite(v).all() for v in L.values() if torch.is_tensor(v));supported=bool(finite and gn>0);result={"status":"complete" if supported else "unsupported","id":job["id"],"route":job["route"],"variant":job["variant"],"checkpoint_loaded":True,"finite":bool(finite),"task_gradient_norm":gn,"task_gradient_connected":gn>0,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"test_evaluated":False};(run_dir/"smoke.json").write_text(json.dumps(result,indent=2)+"\n");(run_dir/"status.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
def train(job,run_dir,snapshot):
 seed=int(job.get("seed",snapshot["seed"]));budget=snapshot["stage1"] if int(job["stage"])==1 else snapshot["stage2"];epochs=int(budget["max_epochs"]);patience=int(budget["patience"]);m,model,tok,labels,missing=configure(job,run_dir,seed);from utils.datasets import SpanNerDataset,collate_fn
 tr_ex=m.load_examples(m.args.data_dir,"train",tok,max_seq_len=128);dv_ex=m.load_examples(m.args.data_dir,"dev",tok,max_seq_len=128);tr=SpanNerDataset(tr_ex,args=m.args,tokenizer=tok,labels=labels);dv=SpanNerDataset(dv_ex,args=m.args,tokenizer=tok,labels=labels,dev=True);dlv=DataLoader(dv,sampler=SequentialSampler(dv),batch_size=16,collate_fn=collate_fn)
 baseline=scan(model,dlv,m.args.device)["predictions"];torch.save({"predictions":baseline},run_dir/"initial_dev_predictions.pt")
 # Full regularization/OT comparisons update all parameters; new-path routes update only scoped modules and optional head.
 if job["route"] in ("P2","P3","G2","G3"):
  for p in model.parameters():p.requires_grad=False
  for p in route_params(model,job):p.requires_grad=True
  if job.get("unfreeze_head"):
   for mod in (model.fusion_text,model.ln_text,model.span_classifier):
    for p in mod.parameters():p.requires_grad=True
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=float(snapshot["optimizer"]["module_lr"]),weight_decay=float(snapshot["optimizer"]["weight_decay"]))
 steps=math.ceil(len(tr)/16)*epochs;sched=get_linear_schedule_with_warmup(opt,int(steps*float(snapshot["optimizer"]["warmup_ratio"])),steps);rows=[];best=-1;bad=0;t0=time.time();torch.cuda.reset_peak_memory_stats()
 for ep in range(1,epochs+1):
  if job.get("kl_warmup_epochs"):m.args.ib_kl_weight=float(job["ib_kl_weight"])*min(1,ep/int(job["kl_warmup_epochs"]))
  else:m.args.ib_kl_weight=float(job.get("ib_kl_weight",.001))
  m.args.ot_alpha=float(job.get("ot_alpha",.05))*min(1,ep/int(job.get("ot_loss_warmup_epochs",1)))
  gen=torch.Generator().manual_seed(seed+ep);dlt=DataLoader(tr,sampler=RandomSampler(tr,generator=gen),batch_size=16,collate_fn=collate_fn);model.train();sums={};n=0;gn=[]
  for si,bc in enumerate(dlt):
   b={k:v.to(m.args.device) for k,v in bc.items()};opt.zero_grad(set_to_none=True);_,L=model(b);loss=L["loss"]
   if not torch.isfinite(loss):raise FloatingPointError("NaN/Inf loss")
   if si in (0,50,100):gn.append(grad_norm(L["c"]+L["scl"],route_params(model,job)))
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();sched.step();n+=1
   for k,v in L.items():
    if torch.is_tensor(v) and v.numel()==1:sums[k]=sums.get(k,0)+float(v.detach())
  met=m.evaluate(m.args,model,tok,labels,mode="dev",prefix=f"{job['id']}_e{ep}",only_test=False,epoch_num=ep,split_tag="dev")[0];sc=scan(model,dlv,m.args.device);current_predictions=sc.pop("predictions");changed=sum(a!=b for a,b in zip(current_predictions,baseline));row={"epoch":ep,"dev_precision":met["span_precision"],"dev_recall":met["span_recall"],"dev_f1":met["span_f1"],"task_gradient_norm":float(np.mean(gn)),"prediction_changes":changed,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"elapsed_seconds":time.time()-t0,**sc,**{f"train_{k}":v/n for k,v in sums.items()}};rows.append(row);write_csv(run_dir/"epoch_metrics.csv",rows);print(json.dumps(row),flush=True)
  if row["dev_f1"]>best:best=row["dev_f1"];bad=0;torch.save(model.state_dict(),run_dir/"best_model.pth");torch.save({"predictions":current_predictions},run_dir/"best_dev_predictions.pt")
  else:bad+=1
  if bad>=patience:break
 bestrow=max(rows,key=lambda x:x["dev_f1"]);collapse=(bestrow.get("gate_std",0)!="" and float(bestrow.get("gate_std",0))<1e-4) or (bestrow.get("gate_mean",.5)!="" and (float(bestrow.get("gate_mean",.5))<.05 or float(bestrow.get("gate_mean",.5))>.95));summary={"status":"complete","id":job["id"],"seed":seed,"route":job["route"],"variant":job["variant"],"stage":job["stage"],"best_dev_f1":bestrow["dev_f1"],"delta_dev_vs_baseline":bestrow["dev_f1"]-float(snapshot["baseline_dev_f1"]),"best_epoch":bestrow["epoch"],"epochs":len(rows),"prediction_changes":bestrow["prediction_changes"],"task_gradient_norm":bestrow["task_gradient_norm"],"gate_collapse":bool(collapse),"training_seconds":time.time()-t0,"peak_gpu_memory_mb":max(x["peak_gpu_memory_mb"] for x in rows),"mechanism":{k:bestrow.get(k,"") for k in ["gate_mean","gate_std","entity_gate_std","mass_mean","mass_std","transported_mass","unmatched_mass","matching_entropy","transport_concentration","reliability_std","residual_ratio"]},"test_evaluated":False,"job":job};(run_dir/"status.json").write_text(json.dumps(summary,indent=2)+"\n")
def main():
 a=cli();run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True);job=json.loads(Path(a.job).read_text());snapshot=json.loads((run.parent.parent/"submission.json").read_text());status=run/"status.json"
 if status.exists() and json.loads(status.read_text()).get("status")=="complete":return
 try:smoke(job,run) if a.smoke else train(job,run,snapshot)
 except Exception as e:status.write_text(json.dumps({"status":"failed","id":job.get("id"),"error":repr(e),"oom":"out of memory" in str(e).lower(),"traceback":traceback.format_exc(),"test_evaluated":False},indent=2)+"\n");raise
if __name__=="__main__":main()
