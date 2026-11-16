"""Immutable seed-42 exploratory six-arm study; no test-set selection."""
import os,json,time,sys,subprocess,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/jyx/PGMNER_visual_rebuild_20261103')
ARMS=[('full',{},'polarized','partial'),('no_ppib',{},'off','partial'),('no_gota',{},'polarized','none'),('no_structure',{'structure':0},'polarized','partial'),('no_contrastive',{'contrastive':0},'polarized','partial'),('weak_fusion',{'normalize_fusion':False},'polarized','partial')]
def dump(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False));q.replace(p)
def run(gpu,smoke):
 hashes=json.loads((BASE/'source_hashes.json').read_text())
 for name,h in hashes.items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 choices=[ARMS[gpu]] if smoke else [ARMS[gpu],ARMS[gpu+3]]
 for name,config,ib,transport in choices:
  out=BASE/('smoke' if smoke else 'runs')/name;out.mkdir(parents=True,exist_ok=False)
  cmd=json.loads(Path('/home/jyx/PGMNER_positive_seed42_20261103/selected/sharp_full_s42/command.json').read_text())
  cmd[0]=sys.executable
  for k,v in {'--output_dir':str(out),'--gpu_id':str(gpu),'--entropy_eps':'0.07','--repair_probe_every':'100'}.items():cmd[cmd.index(k)+1]=v
  cmd+=['--repair_gate_balance','bernoulli','--repair_gate_target','0.6','--repair_gate_balance_weight','2','--repair_polarization_weight','1','--repair_ib',ib,'--repair_transport',transport]
  if smoke:cmd+=['--repair_smoke_steps','3']
  env={**os.environ,'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PGMNER_EXPERIMENT':json.dumps(config)}
  dump(out/'command.json',cmd);dump(out/'experiment.json',config)
  status={'state':'running','start':time.time(),'gpu':gpu};dump(out/'status.json',status)
  with (out/'train.log').open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  status.update(state='complete' if code==0 else 'failed',end=time.time(),returncode=code);dump(out/'status.json',status)
  if code:raise SystemExit(code)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.worker is not None:return run(a.worker,a.smoke)
 BASE.mkdir(exist_ok=True)
 if not (BASE/'source_hashes.json').exists():
  files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts and 'pretrain' not in p.parts]
  dump(BASE/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
  dump(BASE/'protocol.json',{'seed':42,'epochs':8,'selection':'best DEV only','test_used':False,'arms':ARMS,'common_changes':['CLIP preprocessing','freeze vision','missing image zero-mask','entropy .07'],'notes':'Exploratory new implementation; approximate detached structural refinement; all outcomes retained. No manuscript edits.'})
 children=[]
 for gpu in range(3):
  cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(gpu)]+(['--smoke'] if a.smoke else [])
  with (BASE/f'{"smoke" if a.smoke else "run"}_gpu{gpu}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children.append({'gpu':gpu,'pid':proc.pid})
 dump(BASE/('smoke_launch.json' if a.smoke else 'launch.json'),children);print(json.dumps(children))
if __name__=='__main__':main()
