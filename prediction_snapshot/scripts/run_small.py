import os,sys,json,time,subprocess,concurrent.futures,traceback,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data/jyx/PGMNER_small_function_20261108');RUN=Path('/data/jyx/PGMNER_stage_archive_20261108/experiment/runs/selected_reduced')
def dump(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def job(name,kind,gpu,repeat=0,limit=0):
 env={**os.environ,'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
 cmd=[sys.executable,str(ROOT/'scripts/small_function_audit.py'),'--run',str(RUN),'--output',str(BASE/name),'--gpu',str(gpu),'--kind',kind,'--repeat',str(repeat),'--limit',str(limit)]
 s={'state':'running','start':time.time(),'command':cmd};dump(BASE/(name+'_status.json'),s)
 with (BASE/(name+'.log')).open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 s.update(state='complete' if code==0 else 'failed',end=time.time(),returncode=code);dump(BASE/(name+'_status.json'),s)
 if code:raise RuntimeError(name+' failed')
def worker():
 s={'state':'checking','start':time.time()};dump(BASE/'pipeline_status.json',s)
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:
   fs=[p.submit(job,'smoke_delete','deletion',0,0,8),p.submit(job,'smoke_corrupt','corruption',1,0,8)]
   for f in fs:f.result()
  s.update(state='auditing');dump(BASE/'pipeline_status.json',s)
  def lane(g):
   if g==0:job('deletion','deletion',0)
   elif g==1:job('corruption0','corruption',1,0)
   else:
    job('corruption1','corruption',2,1);job('corruption2','corruption',2,2)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:list(p.map(lane,range(3)))
  with (BASE/'summary.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts/summarize_small.py'),'--base',str(BASE)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  s.update(state='complete',end=time.time());dump(BASE/'pipeline_status.json',s)
 except Exception:
  s.update(state='failed',end=time.time(),error=traceback.format_exc());dump(BASE/'pipeline_status.json',s);raise
if __name__=='__main__':
 if '--worker' in sys.argv:worker()
 else:
  BASE.mkdir(exist_ok=False)
  old=json.loads(Path('/data/jyx/PGMNER_stage_archive_20261108/experiment/source_hashes.json').read_text())
  for n,h in old.items():
   if n.startswith(('models/','utils/')) or n=='main-mner-robert.py':assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
  hashes={}
  for p in ROOT.rglob('*.py'):
   if '__pycache__' in p.parts:continue
   n=p.relative_to(ROOT);q=BASE/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);hashes[str(n)]=hashlib.sha256(p.read_bytes()).hexdigest()
  dump(BASE/'source_hashes.json',hashes)
  with (BASE/'worker.log').open('w') as f:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump(BASE/'launch.json',{'pid':p.pid,'checkpoint':str(RUN)});print('PID',p.pid)
