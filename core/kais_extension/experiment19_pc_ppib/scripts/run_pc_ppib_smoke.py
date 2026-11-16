#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,os,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,SequentialSampler,Subset
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'kais_extension/experiment19_pc_ppib';RES=EXP/'results';CKPT=Path('/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth')
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
BASE=['main-mner-robert.py','--data_dir','data/twitter2017','--output_dir',str(EXP/'cache/smoke'),'--model_name_or_path','pretrain/roberta-large','--tokenizer_name','pretrain/roberta-large','--vit_name','pretrain/clip-vit-base-patch16','--use_hf_clip_vision','true','--auto_img_dims','true','--gpu_id','0','--batch_size','2','--max_seq_len','128','--max_span_len','4','--seed','42','--use_pp_ib','--beta_polar','0.01','--delta_minus','0.2','--delta_plus','0.8','--pp_ib_lambda_balance','1.0','--use_got','--got_mode','partial_sinkhorn','--ot_alpha','0.05','--entropy_eps','0.1','--ot_sinkhorn_iter','30','--partial_mass_init','0.5','--dust_cost','1.0','--do_train','false','--do_eval','false','--do_predict','false']
def mainmod():
 s=importlib.util.spec_from_file_location('main_exp19_smoke',ROOT/'main-mner-robert.py');m=importlib.util.module_from_spec(s);old=sys.argv;sys.argv=BASE
 try:s.loader.exec_module(m)
 finally:sys.argv=old
 m.args.labels=str((ROOT/'kais_extension/configs/labels_twitter2017.txt').resolve());return m
def state():
 d=torch.load(CKPT,map_location='cpu',weights_only=True);o='image_model._hf.encoder.vision_model.';n='image_model._hf.encoder.';return {n+k[len(o):] if k.startswith(o) else k:v for k,v in d.items()}
def gn(loss,params):
 g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);v=[x for x in g if x is not None];return {'connected':len(v),'l2':float(torch.sqrt(sum((x.float()**2).sum() for x in v))) if v else 0.0,'finite':all(bool(torch.isfinite(x).all()) for x in v)}
def main():
 os.chdir(ROOT);RES.mkdir(parents=True,exist_ok=True);m=mainmod();from kais_extension.experiment19_pc_ppib.models.pc_ppib_model import BertSpanNerBNPCPPIB
 labels=m.get_labels(m.args.labels);cfg=m.AutoConfig.from_pretrained(m.args.model_name_or_path);tok=m.AutoTokenizer.from_pretrained(m.args.tokenizer_name);m.args.pc_gate_mode='learned';m.args.pc_alpha=.1;m.args.pc_bottleneck=128
 torch.manual_seed(1900);orig=m.BertSpanNerBN(cfg,args=m.args,num_labels=len(labels));orig.load_state_dict(state(),strict=True);orig.to(m.args.device)
 torch.manual_seed(1900);pc=BertSpanNerBNPCPPIB(cfg,args=m.args,num_labels=len(labels));miss,unexp=pc.load_state_dict(state(),strict=False);pc.to(m.args.device);zero_init=bool(torch.count_nonzero(pc.pc_adapter.proj_out.weight.detach())==0 and torch.count_nonzero(pc.pc_adapter.proj_out.bias.detach())==0)
 ex=m.load_examples(m.args.data_dir,'train',tok,max_seq_len=m.args.max_seq_len);from utils.datasets import SpanNerDataset,collate_fn
 ds=SpanNerDataset(ex,args=m.args,tokenizer=tok,labels=labels);b=next(iter(DataLoader(Subset(ds,[0,1]),sampler=SequentialSampler(Subset(ds,[0,1])),batch_size=2,collate_fn=collate_fn)));b={k:v.to(m.args.device) for k,v in b.items()}
 torch.cuda.reset_peak_memory_stats(m.args.device);t0=time.time();orig.eval();pc.eval();torch.manual_seed(1919);po,lo=orig(b);torch.manual_seed(1919);pp,lp=pc(b);diff=float((po[0]-pp[0]).abs().max());same=bool(torch.equal(po[0].argmax(-1),pp[0].argmax(-1)))
 for p in pc.parameters():p.requires_grad=False
 for p in pc.pc_adapter.parameters():p.requires_grad=True
 for p in list(pc.dist_t.parameters())+list(pc.dist_v.parameters()):p.requires_grad=True
 pc.train();torch.manual_seed(1920);_,L=pc(b);task=L['c']+L['scl'];gate=list(pc.dist_t.polar_gate.parameters())+list(pc.dist_v.polar_gate.parameters());mus=list(pc.dist_t.mu_net.parameters())+list(pc.dist_v.mu_net.parameters());lvs=list(pc.dist_t.logvar_net.parameters())+list(pc.dist_v.logvar_net.parameters());ad=list(pc.pc_adapter.parameters())
 before={'adapter':gn(task,ad),'gate':gn(task,gate),'mu':gn(task,mus),'logvar':gn(task,lvs)}
 opt=torch.optim.AdamW([{'params':ad,'lr':1e-4},{'params':list(pc.dist_t.parameters())+list(pc.dist_v.parameters()),'lr':1e-5}]);opt.zero_grad();L['loss'].backward();allfinite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pc.parameters());opt.step();opt.zero_grad()
 torch.manual_seed(1921);_,L2=pc(b);task2=L2['c']+L2['scl'];after={'adapter':gn(task2,ad),'gate':gn(task2,gate),'mu':gn(task2,mus),'logvar':gn(task2,lvs)}
 finite_losses={k:bool(torch.isfinite(v)) for k,v in L2.items() if torch.is_tensor(v) and v.numel()==1};conn={'checkpoint':str(CKPT),'missing_keys':miss,'unexpected_keys':unexp,'missing_only_pc_adapter':all(x.startswith('pc_adapter.') for x in miss) and not unexp,'task_grad_before_step':before,'task_grad_after_one_step':after,'losses_after_step':{k:float(v.detach()) for k,v in L2.items() if torch.is_tensor(v) and v.numel()==1},'finite_losses':finite_losses,'optimizer_step_gradients_finite':allfinite,'residual_summary_after_step':pc.last_pc_summary,'seconds':time.time()-t0,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated(m.args.device)/1024**2,'gpu':torch.cuda.get_device_name(0)}
 equiv={'max_abs_probability_difference':diff,'argmax_predictions_identical':same,'zero_initialized_output_layer':zero_init,'note':'Equivalence was measured before the smoke optimizer step.','pass':diff<=1e-7 and same and zero_init}
 (RES/'gradient_connectivity.json').write_text(json.dumps(conn,indent=2)+'\n');(RES/'initialization_equivalence.json').write_text(json.dumps(equiv,indent=2)+'\n');print(json.dumps({'equivalence':equiv,'connectivity':conn},indent=2))
if __name__=='__main__':main()
