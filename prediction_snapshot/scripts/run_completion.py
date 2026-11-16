"""Complete matched six-arm study without changing model implementation."""
import os,sys,json,time,hashlib,subprocess,argparse,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data/jyx/PGMNER_day_20261104')
BASE=Path('/data/jyx/PGMNER_factorial_20261104')
JOBS=[('no_ib_no_alignment','off','none'),('visual_uniform','polarized','uniform_partial'),('visual_vib','vib','partial')]
REUSE={'visual_full':'visual_only_full','visual_no_alignment':'visual_only_control','no_ib_alignment':'both_control'}
def dump(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False));q.replace(p)
def run(gpu,smoke):
 for n,h in json.loads((BASE/'source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 name,ib,transport=JOBS[gpu];out=BASE/('smoke' if smoke else 'runs')/name;out.mkdir(parents=True,exist_ok=False)
 cmd=json.loads((OLD/'runs/visual_only_full/command.json').read_text());cmd[0]=sys.executable
 for k,v in {'--output_dir':str(out),'--gpu_id':str(gpu),'--repair_ib':ib,'--repair_transport':transport}.items():cmd[cmd.index(k)+1]=v
 if smoke:cmd+=['--repair_smoke_steps','3']
 config=json.loads((OLD/'runs/visual_only_full/experiment.json').read_text())
 env={**os.environ,'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PGMNER_EXPERIMENT':json.dumps(config)}
 dump(out/'command.json',cmd);dump(out/'experiment.json',config)
 s={'state':'running','start':time.time(),'gpu':gpu};dump(out/'status.json',s)
 with (out/'train.log').open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 s.update(state='complete' if code==0 else 'failed',returncode=code,end=time.time());dump(out/'status.json',s)
 if code or smoke:return
 dest=BASE/'audits'/name;dest.parent.mkdir(exist_ok=True)
 s={'state':'running','start':time.time(),'gpu':gpu};dump(dest.parent/(name+'_status.json'),s)
 with (dest.parent/(name+'.log')).open('w') as f:
  code=subprocess.run([sys.executable,str(ROOT/'scripts/audit_visual.py'),'--run',str(out),'--output',str(dest),'--gpu',str(gpu)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 s.update(state='complete' if code==0 else 'failed',returncode=code,end=time.time());dump(dest.parent/(name+'_status.json'),s)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.worker is not None:return run(a.worker,a.smoke)
 BASE.mkdir(exist_ok=True)
 if not (BASE/'source_hashes.json').exists():
  h=json.loads((OLD/'source_hashes.json').read_text())
  for n,v in h.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v,n
  n='scripts/run_completion.py';h[n]=hashlib.sha256((ROOT/n).read_bytes()).hexdigest();dump(BASE/'source_hashes.json',h)
  dump(BASE/'protocol.json',{'seed':42,'epochs':8,'new_jobs':JOBS,'reuse':{k:str(OLD/'runs'/v) for k,v in REUSE.items()},'test_used':False,'note':'Same model source verified SHA256. no_ib_alignment reused: ib_side ignored when IB off. Uniform plan intervention audit is identity by construction; use retrained comparison. No manuscript changes.'})
  snap=BASE/'source';snap.mkdir()
  for n in h:
   dst=snap/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
 children=[]
 for gpu in range(3):
  cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',str(gpu)]+(['--smoke'] if a.smoke else [])
  with (BASE/f'{"smoke" if a.smoke else "run"}_gpu{gpu}.log').open('w') as f:
   proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children.append({'gpu':gpu,'pid':proc.pid,'job':JOBS[gpu][0]})
 dump(BASE/('smoke_launch.json' if a.smoke else 'launch.json'),children);print(json.dumps(children))
if __name__=='__main__':main()
