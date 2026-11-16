#!/usr/bin/env python3
"""Five-epoch DEV-only PC-PP-IB screening from the historical checkpoint."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from transformers import get_linear_schedule_with_warmup
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'kais_extension/experiment19_pc_ppib';RES=EXP/'results';CKPT=Path('/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
BASE=['main-mner-robert.py','--data_dir','data/twitter2017','--output_dir',str(EXP/'cache'),'--model_name_or_path','pretrain/roberta-large','--tokenizer_name','pretrain/roberta-large','--vit_name','pretrain/clip-vit-base-patch16','--use_hf_clip_vision','true','--auto_img_dims','true','--gpu_id','0','--epoch','5','--batch_size','16','--max_seq_len','128','--max_span_len','4','--seed','42','--lr','3e-5','--bert_lr','1e-5','--warmup_ratio','0.1','--patience','2','--use_pp_ib','--beta_polar','0.01','--delta_minus','0.2','--delta_plus','0.8','--pp_ib_lambda_balance','1.0','--beta_polar_warmup_epochs','3','--use_got','--got_mode','partial_sinkhorn','--ot_alpha','0.05','--entropy_eps','0.1','--ot_sinkhorn_iter','30','--partial_mass_init','0.5','--dust_cost','1.0','--ot_warmup_epochs','3','--do_train','false','--do_eval','false','--do_predict','false']
def cli():p=argparse.ArgumentParser();p.add_argument('--gate-mode',choices=['learned','fixed_0.5'],required=True);p.add_argument('--epochs',type=int,default=5);return p.parse_args()
def mainmod(out):
 s=importlib.util.spec_from_file_location('main_exp19_train',ROOT/'main-mner-robert.py');m=importlib.util.module_from_spec(s);old=sys.argv;sys.argv=BASE
 try:s.loader.exec_module(m)
 finally:sys.argv=old
 m.args.labels=str((ROOT/'kais_extension/configs/labels_twitter2017.txt').resolve());m.args.output_dir=str(out);return m
def state(path=CKPT):
 d=torch.load(path,map_location='cpu',weights_only=True);o='image_model._hf.encoder.vision_model.';n='image_model._hf.encoder.';return {n+k[len(o):] if k.startswith(o) else k:v for k,v in d.items()}
def write_csv(path,rows):
 if not rows:return
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def gradnorm(loss,ps):
 gs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True);v=[g for g in gs if g is not None];return float(torch.sqrt(sum((g.float()**2).sum() for g in v))) if v else 0.0
def gate_summary(m,loader,device,epoch,mode):
 tg=[];vg=[];rr=[];m.eval();torch.manual_seed(1942+epoch)
 with torch.no_grad():
  for bc in loader:
   b={k:v.to(device) for k,v in bc.items()};m(b);tg.extend(m.dist_t.last_probe['gate'].reshape(-1).cpu().tolist());vg.extend(m.dist_v.last_probe['gate'].reshape(-1).cpu().tolist());rr.append(m.last_pc_summary['residual_ratio'])
 out=[]
 for name,x in [('text',np.array(tg)),('visual',np.array(vg))]:
  ent=-(x*np.log(np.clip(x,1e-12,1))+(1-x)*np.log(np.clip(1-x,1e-12,1)))/math.log(2)
  out.append({'variant':mode,'epoch':epoch,'modality':name,'n':len(x),'mean':x.mean(),'std':x.std(),'low_0_0.1':np.mean(x<=.1),'middle_0.1_0.9':np.mean((x>.1)&(x<.9)),'high_0.9_1':np.mean(x>=.9),'polarization_ratio':np.mean((x<=.1)|(x>=.9)),'normalized_entropy':ent.mean(),'residual_ratio_mean':np.mean(rr)})
 return out
def evaldev(mod,model,tok,labels,prefix,epoch):
 torch.manual_seed(4242+epoch);return mod.evaluate(mod.args,model,tok,labels,mode='dev',prefix=prefix,only_test=False,epoch_num=epoch,split_tag='dev')[0]
def main():
 a=cli();variant='learned_gate' if a.gate_mode=='learned' else 'fixed_gate_0.5';out=EXP/'checkpoints'/variant;out.mkdir(parents=True,exist_ok=True);mod=mainmod(out);mod.set_seed(mod.args)
 from kais_extension.experiment19_pc_ppib.models.pc_ppib_model import BertSpanNerBNPCPPIB
 labels=mod.get_labels(mod.args.labels);cfg=mod.AutoConfig.from_pretrained(mod.args.model_name_or_path);tok=mod.AutoTokenizer.from_pretrained(mod.args.tokenizer_name);mod.args.pc_gate_mode=a.gate_mode;mod.args.pc_alpha=.1;mod.args.pc_bottleneck=128
 torch.manual_seed(1900);model=BertSpanNerBNPCPPIB(cfg,args=mod.args,num_labels=len(labels));miss,unexp=model.load_state_dict(state(),strict=False);model.to(mod.args.device);model.set_partial_ot_warmup(False)
 for p in model.parameters():p.requires_grad=False
 for p in model.pc_adapter.parameters():p.requires_grad=True
 for p in list(model.dist_t.parameters())+list(model.dist_v.parameters()):p.requires_grad=True
 from utils.datasets import SpanNerDataset,collate_fn
 tr_ex=mod.load_examples(mod.args.data_dir,'train',tok,max_seq_len=mod.args.max_seq_len);dv_ex=mod.load_examples(mod.args.data_dir,'dev',tok,max_seq_len=mod.args.max_seq_len)
 tr=SpanNerDataset(tr_ex,args=mod.args,tokenizer=tok,labels=labels);dv=SpanNerDataset(dv_ex,args=mod.args,tokenizer=tok,labels=labels,dev=True);dv_loader=DataLoader(dv,sampler=SequentialSampler(dv),batch_size=16,collate_fn=collate_fn)
 groups=[{'params':list(model.pc_adapter.parameters()),'lr':1e-4,'weight_decay':1e-4},{'params':list(model.dist_t.parameters())+list(model.dist_v.parameters()),'lr':1e-5,'weight_decay':1e-4}];steps=math.ceil(len(tr)/16)*a.epochs;opt=torch.optim.AdamW(groups);sch=get_linear_schedule_with_warmup(opt,int(.1*steps),steps)
 epoch_rows=[];gate_rows=[];best=-1;best_ep=0;bad=0;t0=time.time();torch.cuda.reset_peak_memory_stats(mod.args.device)
 for ep in range(1,a.epochs+1):
  mod.set_seed(mod.args);gen=torch.Generator();gen.manual_seed(42+ep);dl=DataLoader(tr,sampler=RandomSampler(tr,generator=gen),batch_size=16,collate_fn=collate_fn);model.train();model.set_polar_train_epoch(3);sums={};n=0;gg=[];mg=[];lg=[];ag=[];rat=[]
  for si,bc in enumerate(dl):
   b={k:v.to(mod.args.device) for k,v in bc.items()};opt.zero_grad(set_to_none=True);_,L=model(b);loss=L['loss'];
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   if si>0 and si%25==0:
    task=L['c']+L['scl'];gg.append(gradnorm(task,list(model.dist_t.polar_gate.parameters())+list(model.dist_v.polar_gate.parameters())));mg.append(gradnorm(task,list(model.dist_t.mu_net.parameters())+list(model.dist_v.mu_net.parameters())));lg.append(gradnorm(task,list(model.dist_t.logvar_net.parameters())+list(model.dist_v.logvar_net.parameters())));ag.append(gradnorm(task,list(model.pc_adapter.parameters())))
   loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);opt.step();sch.step();n+=1;rat.append(model.last_pc_summary['residual_ratio'])
   for k,v in L.items():
    if torch.is_tensor(v) and v.numel()==1:sums[k]=sums.get(k,0)+float(v.detach())
  dev=evaldev(mod,model,tok,labels,f'{variant}_epoch{ep}',ep);gs=gate_summary(model,dv_loader,mod.args.device,ep,variant);gate_rows.extend(gs);row={'variant':variant,'epoch':ep,'dev_precision':dev['span_precision'],'dev_recall':dev['span_recall'],'dev_f1':dev['span_f1'],'task_gate_grad_l2':np.mean(gg) if gg else 0,'task_mu_grad_l2':np.mean(mg) if mg else 0,'task_logvar_grad_l2':np.mean(lg) if lg else 0,'task_adapter_grad_l2':np.mean(ag) if ag else 0,'residual_ratio_train_mean':np.mean(rat),'seconds_elapsed':time.time()-t0,**{f'train_{k}':v/n for k,v in sums.items()}};epoch_rows.append(row);print(json.dumps(row),flush=True)
  if float(dev['span_f1'])>best:
   best=float(dev['span_f1']);best_ep=ep;bad=0;torch.save(model.state_dict(),out/'best_model.pth')
  else:bad+=1
  if ep>=2 and float(dev['span_f1'])<.88462 and best<.89262: print('STOP: clear DEV degradation without recovery',flush=True);break
  if bad>=2:print('early stop patience=2',flush=True);break
 write_csv(RES/f'dev_epoch_metrics_{variant}.csv',epoch_rows);write_csv(RES/f'gate_statistics_{variant}.csv',gate_rows)
 result={'variant':variant,'gate_mode':a.gate_mode,'initial_checkpoint':str(CKPT),'baseline_dev_f1':.89462,'best_dev_f1':best,'best_epoch':best_ep,'delta_dev_vs_checkpoint':best-.89462,'epochs_completed':len(epoch_rows),'missing_keys':miss,'unexpected_keys':unexp,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'added_parameters':sum(p.numel() for p in model.pc_adapter.parameters()),'alpha':.1,'adapter_bottleneck':128,'optimizer':'AdamW adapter_lr=1e-4 ppib_lr=1e-5 wd=1e-4; frozen remainder','selection':'DEV only; TEST not evaluated','seconds':time.time()-t0,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated(mod.args.device)/1024**2,'gpu':torch.cuda.get_device_name(0)};(RES/f'{variant}_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
