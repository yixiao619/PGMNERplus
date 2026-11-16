#!/usr/bin/env python3
import importlib.util,json,os,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,SequentialSampler
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'kais_extension/experiment19_pc_ppib';RES=EXP/'results';CK=EXP/'checkpoints/learned_gate/best_model.pth'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
ARGV=['main-mner-robert.py','--data_dir','data/twitter2017','--output_dir',str(EXP/'cache/residual'),'--model_name_or_path','pretrain/roberta-large','--tokenizer_name','pretrain/roberta-large','--vit_name','pretrain/clip-vit-base-patch16','--use_hf_clip_vision','true','--auto_img_dims','true','--gpu_id','0','--batch_size','16','--max_seq_len','128','--max_span_len','4','--seed','42','--use_pp_ib','--beta_polar','0.01','--delta_minus','0.2','--delta_plus','0.8','--pp_ib_lambda_balance','1.0','--use_got','--got_mode','partial_sinkhorn','--ot_alpha','0.05','--entropy_eps','0.1','--ot_sinkhorn_iter','30','--partial_mass_init','0.5','--dust_cost','1.0','--do_train','false','--do_eval','false','--do_predict','false']
def main():
 os.chdir(ROOT);s=importlib.util.spec_from_file_location('m19r',ROOT/'main-mner-robert.py');m=importlib.util.module_from_spec(s);old=sys.argv;sys.argv=ARGV
 try:s.loader.exec_module(m)
 finally:sys.argv=old
 m.args.labels=str((ROOT/'kais_extension/configs/labels_twitter2017.txt').resolve());m.args.pc_gate_mode='learned';m.args.pc_alpha=.1;m.args.pc_bottleneck=128
 from kais_extension.experiment19_pc_ppib.models.pc_ppib_model import BertSpanNerBNPCPPIB
 labs=m.get_labels(m.args.labels);cfg=m.AutoConfig.from_pretrained(m.args.model_name_or_path);tok=m.AutoTokenizer.from_pretrained(m.args.tokenizer_name);model=BertSpanNerBNPCPPIB(cfg,args=m.args,num_labels=len(labs));model.load_state_dict(torch.load(CK,map_location='cpu',weights_only=True),strict=True);model.to(m.args.device).eval()
 ex=m.load_examples(m.args.data_dir,'dev',tok,max_seq_len=m.args.max_seq_len);from utils.datasets import SpanNerDataset,collate_fn
 ds=SpanNerDataset(ex,args=m.args,tokenizer=tok,labels=labs,dev=True);dl=DataLoader(ds,sampler=SequentialSampler(ds),batch_size=16,collate_fn=collate_fn);changed=0;span_changed=0;maxd=0.;means=[];rat=[]
 with torch.no_grad():
  for bi,bc in enumerate(dl):
   b={k:v.to(m.args.device) for k,v in bc.items()};model.pc_adapter.alpha.fill_(.1);torch.manual_seed(9000+bi);a=model(b)[0][0];rat.append(model.last_pc_summary['residual_ratio']);model.pc_adapter.alpha.zero_();torch.manual_seed(9000+bi);z=model(b)[0][0];d=(a-z).abs();maxd=max(maxd,float(d.max()));means.append(float(d.mean()));pa=a.argmax(-1);pz=z.argmax(-1);mask=b['span_masks']>0;span_changed+=int(((pa!=pz)&mask).sum());changed+=int((((pa!=pz)&mask).any(1)).sum())
 out={'split':'Twitter-2017 DEV','n_samples':len(ds),'checkpoint':str(CK),'alpha_on':.1,'alpha_off':0,'max_abs_probability_change':maxd,'mean_abs_probability_change':sum(means)/len(means),'changed_samples':changed,'changed_valid_spans':span_changed,'mean_residual_ratio':sum(rat)/len(rat),'same_matched_vib_noise':True};(RES/'residual_effect.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
