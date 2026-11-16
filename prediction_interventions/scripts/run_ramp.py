import os,sys,json,time,re,hashlib,subprocess,shutil,concurrent.futures,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data/jyx/PGMNER_ppib_ramp_20261107');OLD=Path('/data/jyx/PGMNER_functional_2x2_20261106/runs/ppib_partial')
ARMS=[('local02','polarized','partial'),('local03','polarized','partial'),('ramp05','polarized','partial')]
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def train(name,ib,transport,gpu,steps=0):
 out=BASE/('checks' if steps else 'runs')/name;out.mkdir(parents=True,exist_ok=False)
 cmd=json.loads((OLD/'command.json').read_text());cmd[0]=sys.executable
 for k,v in {'--output_dir':str(out),'--gpu_id':str(gpu),'--repair_ib':ib,'--repair_transport':transport}.items():cmd[cmd.index(k)+1]=v
 cmd[cmd.index('--beta_polar')+1]='.001'
 if steps:cmd+=['--repair_smoke_steps',str(steps)]
 cfg=json.loads((OLD/'experiment.json').read_text());cfg.update(local_aux_weight=0.,local_aux_mode='visual',fix_mu_gate=True,fix_centered_cost=False,image_keep=.6,polar_form='quadratic',local_coefficient=.2 if 'local02' in name else (.3 if 'local03' in name else .1),local_schedule='ramp' if 'ramp05' in name else 'constant')
 dump(out/'command.json',cmd);dump(out/'experiment.json',cfg)
 env={**os.environ,'PGMNER_EXPERIMENT':json.dumps(cfg),'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
 s={'state':'running','start':time.time(),'gpu':gpu};dump(out/'status.json',s)
 with (out/'train.log').open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 s.update(state='complete' if code==0 else 'failed',returncode=code,end=time.time());dump(out/'status.json',s)
 if code:raise RuntimeError('Training failed: '+str(out))
 if steps:return json.loads((out/'determinism.json').read_text())
 st={'state':'running','start':time.time()};dump(out/'audit_status.json',st)
 with (out/'audit.log').open('w') as f:code=subprocess.run([sys.executable,str(ROOT/'scripts/functional_audit.py'),'--run',str(out),'--output',str(out/'audit'),'--gpu',str(gpu)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 st.update(state='complete' if code==0 else 'failed',returncode=code,end=time.time());dump(out/'audit_status.json',st)
 if code:raise RuntimeError('Audit failed: '+str(out))
 with (out/'path_probe.log').open('w') as f:code=subprocess.run([sys.executable,str(ROOT/'scripts/path_probe.py'),'--run',str(out),'--output',str(out/'path_probe'),'--gpu',str(gpu),'--limit','64'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError('Path probe failed: '+str(out))
def worker():
 s={'state':'checking_variants','start':time.time()};dump(BASE/'pipeline_status.json',s)
 try:
  for n,h in json.loads((BASE/'source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
  s.update(state='checking_variants');dump(BASE/'pipeline_status.json',s)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
   list(p.map(lambda item:train('smoke_'+item[1][0],item[1][1],item[1][2],item[0],3),enumerate(ARMS)))
  s.update(state='training');dump(BASE/'pipeline_status.json',s)
  def lane(g):
   for i,arm in enumerate(ARMS):
    if i%3==g:train(*arm,g)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:list(p.map(lane,range(3)))
  rows=[]
  for name,_,_ in ARMS:
   r=BASE/'runs'/name;sc=list(map(float,re.findall(r'^span_f1 = ([0-9.]+)',(r/'eval_results.txt').read_text(),re.M)))
   rows.append({'name':name,'dev_curve':sc,'best_dev':max(sc),'audit':json.loads((r/'audit/results.json').read_text())})
  dump(BASE/'summary.json',rows);s.update(state='complete',end=time.time());dump(BASE/'pipeline_status.json',s)
 except Exception:
  s.update(state='failed',end=time.time(),error=traceback.format_exc());dump(BASE/'pipeline_status.json',s);raise
if __name__=='__main__':
 if '--worker' in sys.argv:worker()
 else:
  BASE.mkdir(exist_ok=False)
  files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts and 'pretrain' not in p.parts]
  dump(BASE/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
  for p in files:
   dst=BASE/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  with (BASE/'worker.log').open('w') as f:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump(BASE/'launch.json',{'pid':p.pid});print('PID',p.pid)
