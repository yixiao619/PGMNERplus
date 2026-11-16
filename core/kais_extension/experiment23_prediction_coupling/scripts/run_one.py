#!/usr/bin/env python3
"""Run one DEV-only Experiment 23 short-training job."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from transformers import get_linear_schedule_with_warmup

ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment23_prediction_coupling"
sys.path=[p for p in sys.path if Path(p or ".").resolve()!=EXP.resolve()];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
def cli():
 p=argparse.ArgumentParser();p.add_argument("--job",required=True);p.add_argument("--run-dir",required=True);p.add_argument("--smoke",action="store_true");return p.parse_args()
def main_module(out,seed):
 argv=["main-mner-robert.py","--data_dir","data/twitter2017","--output_dir",str(out),"--model_name_or_path","pretrain/roberta-large","--tokenizer_name","pretrain/roberta-large","--vit_name","pretrain/clip-vit-base-patch16","--use_hf_clip_vision","true","--auto_img_dims","true","--gpu_id","0","--batch_size","16","--max_seq_len","128","--max_span_len","4","--seed",str(seed),"--use_pp_ib","--beta_polar","0.01","--delta_minus","0.2","--delta_plus","0.8","--use_got","--got_mode","partial_sinkhorn","--ot_alpha","0.05","--entropy_eps","0.1","--ot_sinkhorn_iter","30","--partial_mass_init","0.5","--dust_cost","1.0","--ot_warmup_epochs","0","--do_train","false","--do_eval","false","--do_predict","false","--eval_split_for_best","dev"]
 spec=importlib.util.spec_from_file_location("exp23_main",ROOT/"main-mner-robert.py");m=importlib.util.module_from_spec(spec);old=sys.argv;sys.argv=argv
 try:spec.loader.exec_module(m)
 finally:sys.argv=old
 m.args.labels=str(ROOT/"kais_extension/configs/labels_twitter2017.txt");m.args.output_dir=str(out);return m
def checkpoint():
 p=Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth");s=torch.load(p,map_location="cpu",weights_only=True);old="image_model._hf.encoder.vision_model.";new="image_model._hf.encoder.";return {new+k[len(old):] if k.startswith(old) else k:v for k,v in s.items()}
def setup(job,out):
 m=main_module(out,int(job["seed"]));m.set_seed(m.args);labels=m.get_labels(m.args.labels);cfg=m.AutoConfig.from_pretrained(m.args.model_name_or_path);tok=m.AutoTokenizer.from_pretrained(m.args.tokenizer_name);torch.manual_seed(2000)
 if job["route"]=="baseline":
  from models.bn_bert_ner_OT_robert import BertSpanNerBN as Cls
 else:
  m.args.e23_route=job["route"];from kais_extension.experiment23_prediction_coupling.models.e23_model import PredictionCoupledPGMNER as Cls
 model=Cls(cfg,args=m.args,num_labels=len(labels));missing,unexpected=model.load_state_dict(checkpoint(),strict=False)
 if job["route"]=="baseline" and (missing or unexpected):raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
 if job["route"]!="baseline" and (unexpected or any(not k.startswith("e23_coupler.") for k in missing)):raise RuntimeError(f"coupler mismatch missing={missing} unexpected={unexpected}")
 model.to(m.args.device);model.set_partial_ot_warmup(False);return m,model,tok,labels,missing
def grad_norm(loss,params):
 ps=[p for p in params if p.requires_grad];gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True) if ps else [];gs=[g for g in gs if g is not None];return float(torch.sqrt(sum((g.float()**2).sum() for g in gs))) if gs else 0.0
def scan(model,loader,device):
 model.eval();pred=[];gold=[];stats=[]
 with torch.no_grad():
  for bc in loader:
   b={k:v.to(device) for k,v in bc.items()};prob=model(b)[0][0];mask=b["span_masks"].bool();pred.extend(prob.argmax(-1)[mask].cpu().tolist());gold.extend(b["span_labels"][mask].cpu().tolist())
   if hasattr(model,"last_e23_summary") and model.last_e23_summary:stats.append(dict(model.last_e23_summary))
 out={"predictions":pred,"gold":gold}
 for k in (stats[0].keys() if stats else []):out[k]=float(np.mean([x[k] for x in stats]))
 return out
def transition(base,new,gold):
 keep=[i for i,(a,b,g) in enumerate(zip(base,new,gold)) if a!=0 or b!=0 or g!=0];changed=sum(base[i]!=new[i] for i in keep);cw=sum(base[i]==gold[i] and new[i]!=gold[i] for i in keep);wc=sum(base[i]!=gold[i] and new[i]==gold[i] for i in keep);return changed,cw,wc
def write_csv(path,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def smoke(job,out):
 m,model,tok,labels,missing=setup(job,out);from utils.datasets import SpanNerDataset,collate_fn
 ex=m.load_examples(m.args.data_dir,"train",tok,max_seq_len=128)[:8];ds=SpanNerDataset(ex,args=m.args,tokenizer=tok,labels=labels);bc=next(iter(DataLoader(ds,batch_size=2,collate_fn=collate_fn)));b={k:v.to(m.args.device) for k,v in bc.items()};model.train();torch.manual_seed(1919);p,L=model(b);finite=all(torch.isfinite(v).all() for v in L.values() if torch.is_tensor(v));task=L["c"]+L["scl"]
 if job["route"]!="baseline":
  gn=grad_norm(task,model.coupling_parameters());enabled=p[0].detach();model.e23_enabled=False;torch.manual_seed(1919);disabled=model(b)[0][0].detach();model.e23_enabled=True;model.e23_shuffle_plan=True;torch.manual_seed(1919);shuffled=model(b)[0][0].detach();model.e23_shuffle_plan=False;init_diff=float((enabled-disabled).abs().max());shuffle_diff=float((enabled-shuffled).abs().max());ratio=model.last_e23_summary.get("residual_ratio",0.0)
 else:gn=0.0;init_diff=0.0;shuffle_diff=0.0;ratio=0.0
 L["loss"].backward();ok=finite and (job["route"]=="baseline" or (gn>0 and init_diff>0 and shuffle_diff>0 and 0<ratio<.2));res={"id":job["id"],"route":job["route"],"status":"complete" if ok else "unsupported","finite":bool(finite),"checkpoint_loaded":True,"missing_keys":missing,"task_gradient_norm":gn,"enabled_vs_disabled_max_probability_diff":init_diff,"transport_shuffle_max_probability_diff":shuffle_diff,"residual_ratio":ratio,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"test_evaluated":False};(out/"smoke.json").write_text(json.dumps(res,indent=2)+"\n");print(json.dumps(res,indent=2))
 if not ok:raise RuntimeError(f"smoke unsupported: {res}")
def train(job,out):
 matrix=json.loads((EXP/"configs/experiment_matrix.json").read_text());epochs=int(matrix["stage1"]["max_epochs"]);patience=int(matrix["stage1"]["patience"]);m,model,tok,labels,missing=setup(job,out);from utils.datasets import SpanNerDataset,collate_fn
 tr=SpanNerDataset(m.load_examples(m.args.data_dir,"train",tok,max_seq_len=128),args=m.args,tokenizer=tok,labels=labels);dv=SpanNerDataset(m.load_examples(m.args.data_dir,"dev",tok,max_seq_len=128),args=m.args,tokenizer=tok,labels=labels,dev=True);dlv=DataLoader(dv,sampler=SequentialSampler(dv),batch_size=16,collate_fn=collate_fn);initial=scan(model,dlv,m.args.device);torch.save(initial,out/"initial_dev_predictions.pt")
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=float(matrix["optimizer"]["lr"]),weight_decay=float(matrix["optimizer"]["weight_decay"]));steps=math.ceil(len(tr)/16)*epochs;sched=get_linear_schedule_with_warmup(opt,int(steps*float(matrix["optimizer"]["warmup_ratio"])),steps);rows=[];best=-1;bad=0;t0=time.time();torch.cuda.reset_peak_memory_stats()
 for ep in range(1,epochs+1):
  gen=torch.Generator().manual_seed(int(job["seed"])+ep);dlt=DataLoader(tr,sampler=RandomSampler(tr,generator=gen),batch_size=16,collate_fn=collate_fn);model.train();sums={};n=0;gn=[]
  for si,bc in enumerate(dlt):
   b={k:v.to(m.args.device) for k,v in bc.items()};opt.zero_grad(set_to_none=True);_,L=model(b);loss=L["loss"]
   if not torch.isfinite(loss):raise FloatingPointError("NaN/Inf loss")
   if job["route"]!="baseline" and si in (0,50,100):gn.append(grad_norm(L["c"]+L["scl"],model.coupling_parameters()))
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();sched.step();n+=1
   for k,v in L.items():
    if torch.is_tensor(v) and v.numel()==1:sums[k]=sums.get(k,0.0)+float(v.detach())
  met=m.evaluate(m.args,model,tok,labels,mode="dev",prefix=f"{job['id']}_e{ep}",only_test=False,epoch_num=ep,split_tag="dev")[0];sc=scan(model,dlv,m.args.device);changed,cw,wc=transition(initial["predictions"],sc["predictions"],sc["gold"]);row={"epoch":ep,"dev_precision":met["span_precision"],"dev_recall":met["span_recall"],"dev_f1":met["span_f1"],"prediction_changes":changed,"correct_to_wrong":cw,"wrong_to_correct":wc,"task_gradient_norm":float(np.mean(gn)) if gn else 0.0,"peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1024**2,"elapsed_seconds":time.time()-t0,**{k:v for k,v in sc.items() if k not in ("predictions","gold")},**{f"train_{k}":v/n for k,v in sums.items()}};rows.append(row);write_csv(out/"epoch_metrics.csv",rows);print(json.dumps(row),flush=True)
  if row["dev_f1"]>best:best=row["dev_f1"];bad=0;torch.save(model.state_dict(),out/"best_model.pth");torch.save(sc,out/"best_dev_predictions.pt")
  else:bad+=1
  if bad>=patience:break
 model.load_state_dict(torch.load(out/"best_model.pth",map_location=m.args.device,weights_only=True));bestrow=max(rows,key=lambda x:x["dev_f1"]);disabled_f1="";disabled_changes=""
 if job["route"]!="baseline":
  enabled=scan(model,dlv,m.args.device);model.e23_enabled=False;dm=m.evaluate(m.args,model,tok,labels,mode="dev",prefix=f"{job['id']}_disabled",only_test=False,epoch_num=bestrow["epoch"],split_tag="dev_disabled")[0];disabled=scan(model,dlv,m.args.device);disabled_f1=dm["span_f1"];disabled_changes=sum(a!=b for a,b in zip(enabled["predictions"],disabled["predictions"]));model.e23_enabled=True
 added=sum(p.numel() for p in model.coupling_parameters()) if job["route"]!="baseline" else 0;summary={"status":"complete","id":job["id"],"seed":job["seed"],"route":job["route"],"best_dev_f1":bestrow["dev_f1"],"best_dev_precision":bestrow["dev_precision"],"best_dev_recall":bestrow["dev_recall"],"best_epoch":bestrow["epoch"],"epochs":len(rows),"prediction_changes":bestrow["prediction_changes"],"correct_to_wrong":bestrow["correct_to_wrong"],"wrong_to_correct":bestrow["wrong_to_correct"],"task_gradient_norm":bestrow["task_gradient_norm"],"residual_ratio":bestrow.get("residual_ratio",""),"gate_mean":bestrow.get("gate_mean",""),"gate_std":bestrow.get("gate_std",""),"span_gate_variance":bestrow.get("span_gate_variance",""),"row_mass_mean":bestrow.get("row_mass_mean",""),"concentration_mean":bestrow.get("concentration_mean",""),"disabled_dev_f1":disabled_f1,"enabled_vs_disabled_prediction_changes":disabled_changes,"added_parameters":added,"training_seconds":time.time()-t0,"peak_gpu_memory_mb":max(x["peak_gpu_memory_mb"] for x in rows),"test_evaluated":False};(out/"status.json").write_text(json.dumps(summary,indent=2)+"\n")
def main():
 a=cli();out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True);job=json.loads(Path(a.job).read_text());
 try:smoke(job,out) if a.smoke else train(job,out)
 except Exception as e:(out/"status.json").write_text(json.dumps({"status":"failed","id":job.get("id"),"route":job.get("route"),"seed":job.get("seed"),"error":repr(e),"oom":"out of memory" in str(e).lower(),"traceback":traceback.format_exc(),"test_evaluated":False},indent=2)+"\n");raise
if __name__=="__main__":main()
