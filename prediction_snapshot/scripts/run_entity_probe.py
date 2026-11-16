import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data/jyx/PGMNER_entity_signal_20261105')
if len(sys.argv)==1:
 BASE.mkdir(exist_ok=False)
 with (BASE/'worker.log').open('w') as f:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('PID',p.pid)
else:
 cmd=[sys.executable,str(ROOT/'scripts/entity_signal_probe.py'),'--run','/data/jyx/PGMNER_adaptive_split_20261104/round1/full','--output',str(BASE/'results'),'--gpu','0']
 status={'state':'running','start':time.time(),'command':cmd};p=BASE/'status.json';p.write_text(json.dumps(status))
 env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'}
 with (BASE/'run.log').open('w') as f:c=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 status.update(state='complete' if c==0 else 'failed',returncode=c,end=time.time());p.write_text(json.dumps(status))
