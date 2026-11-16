import os,sys,json,time,subprocess,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).parent;SOURCE=Path('/data/hfy/VG-Think/E2E-GMNER');OUT=ROOT/'results';OUT.mkdir(exist_ok=True);PY=sys.executable
BASE=SOURCE/'model/Qwen2.5-VL-7B-Instruct';ADAPTER=SOURCE/'outputs/exp_cot_paper/best'
def status(stage,**kw):
 d=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z'),**kw);(OUT/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
def env(gpu):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT/'src')+':/data/cyc/LLARA/peft/src',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');return e
def command(name,smoke=False):
 c=[PY,'-u',str(ROOT/'train.py'),'--model_name',str(BASE),'--init_adapter',str(ADAPTER),'--train_jsonl',str(ROOT/f'data/{name}_train.jsonl'),'--image_root',str(SOURCE/'data'),'--output_dir',str(OUT/('smoke' if smoke else name)),'--device','cuda:0','--epochs','1' if smoke else '2','--batch_size','1','--grad_accum_steps','2' if smoke else '16','--lr','5e-6','--weight_decay','1e-3','--warmup_ratio','.05','--max_length','2048','--bf16','--grad_ckpt','--attn_impl','sdpa','--dev_batch_size','1','--eval_max_new_tokens','512','--num_workers','2','--seed','42']
 if smoke:c+=['--max_train_samples','4']
 else:c+=['--dev_jsonl',str(ROOT/('data/correction_dev.jsonl' if name=='correction' else 'data/dev.jsonl'))]
 return c

def main():
 status('smoke_training');(OUT/'source_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if 'results' not in p.parts},indent=2))
 with (OUT/'smoke.log').open('a') as f:subprocess.run(command('control',True),cwd=ROOT,env=env(2),stdout=f,stderr=subprocess.STDOUT,check=True)
 status('parallel_training');jobs=[]
 for gpu,name in enumerate(['control','answer','correction']):
  cmd=command(name);(OUT/f'{name}_command.json').write_text(json.dumps(cmd,indent=2));f=(OUT/f'{name}.log').open('a');p=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=f,stderr=subprocess.STDOUT);jobs.append((name,p,f))
 failures=[]
 for name,p,f in jobs:
  code=p.wait();f.close()
  if code:failures.append(dict(name=name,code=code))
 if failures:raise RuntimeError(f'Training failures: {failures}')
 status('dev_selection')
 sys.path.insert(0,str(ROOT/'src'));from e2egmner.evaluation.parsing import parse_triples
 from e2egmner.evaluation.metrics import count_correct_gmner,count_correct_mner
 def metric(path):
  rows=[json.loads(x) for x in path.open()];counts=[0,0,0,0]
  for r in rows:
   p=parse_triples(r['response_answer_scaled'],strict=False,where='pred');g=parse_triples(r['labels_answer_scaled'],strict=True,where='gold');counts[0]+=count_correct_gmner(p,g,.5);counts[1]+=len(p);counts[2]+=len(g);counts[3]+=count_correct_mner(p,g)
  return dict(gmner_f1=2*counts[0]/max(counts[1]+counts[2],1),mner_f1=2*counts[3]/max(counts[1]+counts[2],1),counts=counts,posts=len(rows))
 trials=[dict(name='original_teacher',epoch=3,**metric(SOURCE/'outputs/exp_cot_paper/dev_pred_epoch3.jsonl'))]
 for name in ['control','answer','correction']:
  for ep in [1,2]:trials.append(dict(name=name,epoch=ep,**metric(OUT/name/f'dev_pred_epoch{ep}.jsonl')))
 winner=max(trials,key=lambda x:x['gmner_f1']);(OUT/'locked.json').write_text(json.dumps(dict(winner=winner,trials=trials,rule='DEV joint F1 only; original teacher wins ties; test unseen by trainer.'),indent=2))
 if winner['name']=='original_teacher':
  (OUT/'final_results.json').write_text(json.dumps(dict(winner=winner,teacher_test_f1=.6465234801601747,note='No DEV improvement; original teacher retained, not an adaptation gain.'),indent=2));status('complete');return
 name=winner['name'];status('test_evaluation',winner=winner)
 cmd=[PY,'-u',str(ROOT/'eval.py'),'--checkpoint',str(OUT/name/'best'),'--model_name',str(BASE),'--test_jsonl',str(ROOT/('data/correction_test.jsonl' if name=='correction' else 'data/test835.jsonl')),'--image_root',str(SOURCE/'data'),'--output_dir',str(OUT/'selected_test'),'--tag','selected','--bf16','--attn_impl','sdpa','--test_batch_size','1','--num_workers','2','--max_length','2048','--eval_max_new_tokens','512']
 with (OUT/'test.log').open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env(2),stdout=f,stderr=subprocess.STDOUT,check=True)
 paths=list((OUT/'selected_test').glob('test_pred*selected*.jsonl'));assert len(paths)==1,paths
 result=metric(paths[0]);assert result['posts']==835
 (OUT/'final_results.json').write_text(json.dumps(dict(winner=winner,selected_test=result,teacher_test_f1=.6465234801601747,delta_vs_teacher_pp=100*(result['gmner_f1']-.6465234801601747),scope='TRAIN-only LoRA adaptation; full DEV model/epoch selection; same previously examined835 TEST. Control/answer variants are teacher adaptation, correction additionally uses PGMNER candidates. No PPIB/GOTA ablation claim.'),indent=2));status('complete')
try:main()
except Exception:status('failed',error=traceback.format_exc());raise
