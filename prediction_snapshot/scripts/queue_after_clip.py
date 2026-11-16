import os,sys,time,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data/jyx/PGMNER_targeted_fixes_20261105');PRIOR=Path('/data/jyx/PGMNER_separate_clip_20261105')
ARMS=['mu_gate','centered_cost','both_fixes']
def dump(value):
 p=BASE/'queue_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def wait_status(paths,deadline):
 while True:
  states=[]
  for p in paths:
   if not p.exists():states.append('pending');continue
   d=json.loads(p.read_text());state=d['state'];states.append(state)
   if state=='failed':raise RuntimeError('Dependency failed: '+str(p))
  if all(s=='complete' for s in states):return
  if time.time()>deadline:raise TimeoutError('Waiting deadline exceeded; no score-based fallback')
  time.sleep(30)
def worker():
 s={'state':'waiting_for_clip','start':time.time(),'baseline':str(PRIOR/'runs/aux_full'),'deadline_hours':18};dump(s)
 deadline=time.time()+18*3600
 try:
  paths=[]
  for name in ['aux_full','no_aux','text_only_aux']:
   paths.extend([PRIOR/'runs'/name/'status.json',PRIOR/'runs'/name/'audit_status.json'])
  paths.append(PRIOR/'runs/aux_full/entity_probe_status.json');wait_status(paths,deadline)
  for name,h in json.loads((BASE/'queued_source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
  s.update(state='smoke',dependency_complete=time.time());dump(s)
  subprocess.run([sys.executable,str(ROOT/'scripts/run_aux.py'),'--smoke'],cwd=ROOT,check=True)
  wait_status([BASE/'smoke'/name/'status.json' for name in ARMS],time.time()+1800)
  s.update(state='training',training_start=time.time());dump(s)
  subprocess.run([sys.executable,str(ROOT/'scripts/run_aux.py')],cwd=ROOT,check=True)
  paths=[]
  for name in ARMS:paths.extend([BASE/'runs'/name/'status.json',BASE/'runs'/name/'audit_status.json'])
  paths.append(BASE/'runs/both_fixes/entity_probe_status.json');wait_status(paths,deadline)
  s.update(state='complete',end=time.time());dump(s)
 except Exception as e:
  s.update(state='failed',end=time.time(),error=repr(e));dump(s);raise
if __name__=='__main__':
 if '--worker' in sys.argv:worker()
 else:
  BASE.mkdir(exist_ok=False)
  files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts and 'pretrain' not in p.parts]
  (BASE/'queued_source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
  with (BASE/'queue.log').open('w') as f:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (BASE/'queue_launch.json').write_text(json.dumps({'pid':p.pid}));print('queue PID',p.pid)
