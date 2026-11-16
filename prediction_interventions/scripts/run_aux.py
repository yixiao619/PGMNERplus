import os,sys,json,time,hashlib,subprocess,shutil,re,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data/jyx/PGMNER_targeted_fixes_20261105');OLD=Path('/data/jyx/PGMNER_separate_clip_20261105/runs/aux_full')
ARMS=[('mu_gate','polarized','partial',.05,'visual'),('centered_cost','polarized','partial',.05,'visual'),('both_fixes','polarized','partial',.05,'visual')]
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def one(gpu,arm,smoke):
 for n,h in json.loads((BASE/'source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 name,ib,transport,weight,mode=arm;out=BASE/('smoke' if smoke else 'runs')/name;out.mkdir(parents=True,exist_ok=False)
 cmd=json.loads((OLD/'command.json').read_text());cmd[0]=sys.executable
 for k,v in {'--output_dir':str(out),'--gpu_id':str(gpu),'--repair_ib':ib,'--repair_transport':transport}.items():cmd[cmd.index(k)+1]=v
 if smoke:cmd+=['--repair_smoke_steps','3']
 cfg=json.loads((OLD/'experiment.json').read_text());cfg.update(local_aux_weight=weight,local_aux_mode=mode,fix_mu_gate=name in ('mu_gate','both_fixes'),fix_centered_cost=name in ('centered_cost','both_fixes'))
 dump(out/'command.json',cmd);dump(out/'experiment.json',cfg)
 env={**os.environ,'PGMNER_EXPERIMENT':json.dumps(cfg),'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
 s={'state':'running','start':time.time(),'gpu':gpu};dump(out/'status.json',s)
 with (out/'train.log').open('w') as f:c=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 s.update(state='complete' if c==0 else 'failed',end=time.time(),returncode=c);dump(out/'status.json',s)
 if c or smoke:return
 for script,dest in [('audit_visual.py','audit')]+([('entity_signal_probe.py','entity_probe')] if name=='both_fixes' else []):
  st={'state':'running','start':time.time()};dump(out/(dest+'_status.json'),st)
  with (out/(dest+'.log')).open('w') as f:c=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--run',str(out),'--output',str(out/dest),'--gpu',str(gpu)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
  st.update(state='complete' if c==0 else 'failed',end=time.time(),returncode=c);dump(out/(dest+'_status.json'),st)
def worker(smoke):
 def lane(gpu):
  for i,a in enumerate(ARMS):
   if i%3==gpu and (not smoke or i<3):one(gpu,a,smoke)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:list(p.map(lane,range(3)))
 rows=[]
 for p in (BASE/('smoke' if smoke else 'runs')).glob('*/status.json'):
  row={'name':p.parent.name,'status':json.loads(p.read_text())};e=p.parent/'eval_results.txt'
  if e.exists():
   scores=list(map(float,re.findall(r'^span_f1 = ([0-9.]+)',e.read_text(),re.M)));row['dev_curve']=scores;row['best_dev']=max(scores) if scores else None
  rows.append(row)
 dump(BASE/('smoke_summary.json' if smoke else 'summary.json'),rows)
if __name__=='__main__':
 smoke='--smoke' in sys.argv
 if '--worker' in sys.argv:worker(smoke)
 else:
  BASE.mkdir(exist_ok=True)
  if not (BASE/'source_hashes.json').exists():
   files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts and 'pretrain' not in p.parts]
   dump(BASE/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
   snap=BASE/'source';snap.mkdir()
   for p in files:
    dst=snap/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
   dump(BASE/'protocol.json',{'seed':42,'epochs':8,'arms':ARMS,'test_used':False,'selection':'best DEV','auxiliary':'training only, balanced entity/nonentity CE, detached text context with30% sample dropout, normalized local values, weight.05; matching main inference path unchanged','controls':'all arms consume identical extra dropout RNG; no_aux rerun rather than reusing original round1','bound':'three prespecified repairs; baseline separate-clip aux_full; no result-dependent configuration selection'})
  if not smoke:
   for name in ['mu_gate','centered_cost','both_fixes']:assert json.loads((BASE/'smoke'/name/'status.json').read_text())['state']=='complete'
  cmd=[sys.executable,str(Path(__file__).resolve()),'--worker']+(['--smoke'] if smoke else [])
  with (BASE/('smoke_worker.log' if smoke else 'worker.log')).open('w') as f:p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump(BASE/('smoke_launch.json' if smoke else 'launch.json'),{'pid':p.pid,'command':cmd});print('PID',p.pid)
