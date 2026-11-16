import os,json,time,sys,subprocess,hashlib,argparse,re,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/jyx/PGMNER_day_20261104')
ROUTES=['both','visual_only','text_only']
ARMS=['full','control']
def dump(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False));q.replace(p)
def env_for(config):return {**os.environ,'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PGMNER_EXPERIMENT':json.dumps(config)}
def worker(gpu,smoke):
 for name,h in json.loads((BASE/'source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 route=ROUTES[gpu]
 for arm in (['full'] if smoke else ARMS):
  name=route+'_'+arm;out=BASE/('smoke' if smoke else 'runs')/name;out.mkdir(parents=True,exist_ok=False)
  config={'structure':0,'normalize_fusion':False,'context':True,'contrastive':0 if gpu==2 and arm=='control' else .03,'text_keep':.85,'image_keep':.6,'noise_scale':.25,'gate_cost':.15,'text_mix':.2,'interaction':False,'clip_anchor':True,'ib_side':['both','visual','text'][gpu]}
  cmd=json.loads(Path('/home/jyx/PGMNER_positive_seed42_20261103/selected/sharp_full_s42/command.json').read_text());cmd[0]=sys.executable
  options={'--output_dir':str(out),'--gpu_id':str(gpu),'--entropy_eps':'.07','--beta_polar':'.001','--ib_kl_weight':'.0001','--vit_name':'pretrain/clip-vit-base-patch16'}
  for k,v in options.items():cmd[cmd.index(k)+1]=v
  cmd+=['--repair_gate_balance','bernoulli','--repair_gate_target','.6','--repair_gate_balance_weight','2','--repair_ib','off' if gpu==0 and arm=='control' else 'polarized','--repair_transport','none' if gpu==1 and arm=='control' else 'partial']
  if smoke:cmd+=['--repair_smoke_steps','3']
  dump(out/'command.json',cmd);dump(out/'experiment.json',config)
  status={'state':'running','start':time.time(),'gpu':gpu};dump(out/'status.json',status)
  with (out/'train.log').open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env_for(config),stdout=f,stderr=subprocess.STDOUT).returncode
  status.update(state='complete' if code==0 else 'failed',returncode=code,end=time.time());dump(out/'status.json',status)
  # Preserve failures and continue independent arms.
 if smoke:return
 for arm in ARMS:
  name=route+'_'+arm;out=BASE/'runs'/name
  if json.loads((out/'status.json').read_text())['state']!='complete':continue
  dest=BASE/'audits'/name;dest.parent.mkdir(exist_ok=True)
  config=json.loads((out/'experiment.json').read_text());status={'state':'running','start':time.time(),'gpu':gpu};dump(dest.parent/(name+'_status.json'),status)
  cmd=[sys.executable,str(ROOT/'scripts/audit_visual.py'),'--run',str(out),'--output',str(dest),'--gpu',str(gpu)]
  with (dest.parent/(name+'.log')).open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env_for(config),stdout=f,stderr=subprocess.STDOUT).returncode
  status.update(state='complete' if code==0 else 'failed',returncode=code,end=time.time());dump(dest.parent/(name+'_status.json'),status)
 # Per-GPU summary avoids concurrent writes.
 rows=[]
 for arm in ARMS:
  name=route+'_'+arm;p=BASE/'runs'/name/'eval_results.txt'
  scores=list(map(float,re.findall(r'^span_f1 = ([0-9.]+)',p.read_text(),re.M))) if p.exists() else []
  row={'name':name,'dev_curve':scores,'best_dev':max(scores) if scores else None}
  p=BASE/'audits'/name/'results.json'
  if p.exists():row['audit']=json.loads(p.read_text())
  rows.append(row)
 dump(BASE/(route+'_summary.json'),rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.worker is not None:return worker(a.worker,a.smoke)
 BASE.mkdir(exist_ok=True)
 if not (BASE/'source_hashes.json').exists():
  files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts and 'pretrain' not in p.parts]
  dump(BASE/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
  dump(BASE/'protocol.json',{'routes':ROUTES,'arms':ARMS,'seed':42,'epochs':8,'selection':'DEV best; no TEST','interpretation':'Exploratory matched controls; retain all results. Capacity comparison interaction_base vs interaction_large. No guarantee of improvement.'})
 children=[]
 for gpu in range(3):
  cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(gpu)]+(['--smoke'] if a.smoke else [])
  with (BASE/f'{"smoke" if a.smoke else "run"}_gpu{gpu}.log').open('w') as f:
   p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children.append({'gpu':gpu,'pid':p.pid,'route':ROUTES[gpu]})
 dump(BASE/('smoke_launch.json' if a.smoke else 'launch.json'),children);print(json.dumps(children))
if __name__=='__main__':main()
