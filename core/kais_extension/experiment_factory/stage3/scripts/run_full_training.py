#!/usr/bin/env python3
"""Run one formal from-scratch PGMNER job. DEV-only selection; no TEST."""
from __future__ import annotations
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My")
def cli():
 p=argparse.ArgumentParser();p.add_argument("--job",required=True);p.add_argument("--run-dir",required=True);return p.parse_args()
def command(job,out):
 # Disable post-paper PP-IB repair defaults. The May-2026 run had no balance,
 # polar warmup, or custom gate initialization.
 return [sys.executable,"-u",str(ROOT/"main-mner-robert.py"),"--data_dir","data/twitter2017","--model_name_or_path","pretrain/roberta-large","--tokenizer_name","pretrain/roberta-large","--vit_name","pretrain/clip-vit-base-patch16","--use_hf_clip_vision","true","--auto_img_dims","true","--gpu_id","0","--epoch","40","--batch_size","16","--max_seq_len","128","--max_span_len","4","--lr","3e-5","--bert_lr","1e-5","--weight_decay","0.01","--warmup_ratio","0.1","--patience","0","--seed",str(job["seed"]),"--logging_steps","100","--do_train","true","--do_eval","true","--do_predict","false","--eval_split_for_best","dev","--use_pp_ib","--ib_kl_weight",str(job["ib_kl_weight"]),"--beta_polar","0.01","--delta_minus","0.2","--delta_plus","0.8","--pp_ib_lambda_balance","0","--beta_polar_warmup_epochs","0","--pp_ib_gate_init_std","0","--pp_ib_probe","--use_got","--got_mode","partial_sinkhorn","--ot_alpha","0.05","--entropy_eps","0.1","--ot_sinkhorn_iter","30","--partial_mass_init","0.5","--dust_cost","1.0","--ot_warmup_epochs","3","--use_ema","--ema_decay","0.999","--output_dir",str(out)]
def main():
 a=cli();job=json.loads(Path(a.job).read_text());out=Path(a.run_dir);out.mkdir(parents=True,exist_ok=True);status=out/"status.json"
 if status.exists() and json.loads(status.read_text()).get("status")=="complete":return
 cmd=command(job,out);(out/"command.json").write_text(json.dumps(cmd,indent=2)+"\n");t=time.time();base={"id":job["id"],"group":"A","seed":job["seed"],"variant":job["variant"],"ib_kl_weight":job["ib_kl_weight"],"selection_split":"dev","test_evaluated":False,"lossless_resume_supported":False};status.write_text(json.dumps({**base,"status":"running","started_at":t},indent=2)+"\n")
 try:
  rc=subprocess.run(cmd,cwd=ROOT).returncode
  if rc:raise RuntimeError(f"training exited {rc}")
  hist=[];hp=out/"per_class_history.jsonl"
  if hp.exists():
   for line in hp.read_text().splitlines():
    row=json.loads(line)
    if row.get("split")=="dev":hist.append(row)
  if not hist:raise RuntimeError("no DEV epoch history produced")
  best=max(hist,key=lambda x:float(x["overall"]["f1"]));probe={};pp=out/"probe/history.json"
  if pp.exists():
   ph=json.loads(pp.read_text());probe=ph[int(best["epoch"])-1] if len(ph)>=int(best["epoch"]) else (ph[-1] if ph else {})
  final={**base,"status":"complete","finished_at":time.time(),"training_seconds":time.time()-t,"epochs":len(hist),"best_epoch":best["epoch"],"best_dev_precision":best["overall"]["p"],"best_dev_recall":best["overall"]["r"],"best_dev_f1":best["overall"]["f1"],"gate_statistics":probe};status.write_text(json.dumps(final,indent=2)+"\n")
 except BaseException as e:
  status.write_text(json.dumps({**base,"status":"failed","finished_at":time.time(),"training_seconds":time.time()-t,"error":repr(e),"oom":"out of memory" in str(e).lower(),"traceback":traceback.format_exc()},indent=2)+"\n");raise
if __name__=="__main__":main()
