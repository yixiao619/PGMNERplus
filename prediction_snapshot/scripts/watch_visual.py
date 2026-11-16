import os,json,time,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/jyx/PGMNER_visual_rebuild_20261103')
gpu=int(sys.argv[1]); names=[['full','no_structure'],['no_ppib','no_contrastive'],['no_gota','weak_fusion']][gpu]
# Wait until both assigned training jobs exit; then use the same GPU for audits.
while True:
 statuses=[BASE/'runs'/n/'status.json' for n in names]
 if all(p.exists() and json.loads(p.read_text())['state'] in ('complete','failed') for p in statuses): break
 # An early training failure means the sequential worker never starts the second.
 if statuses[0].exists() and json.loads(statuses[0].read_text())['state']=='failed': break
 time.sleep(30)
for name in names:
 run=BASE/'runs'/name
 if not (run/'status.json').exists() or json.loads((run/'status.json').read_text())['state']!='complete':continue
 dest=BASE/'audits'/name;dest.parent.mkdir(exist_ok=True)
 env={**os.environ,'PGMNER_EXPERIMENT':(run/'experiment.json').read_text(),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
 status={'state':'running','start':time.time(),'gpu':gpu};p=dest.parent/f'{name}_status.json';p.write_text(json.dumps(status))
 with (dest.parent/f'{name}.log').open('w') as f:
  code=subprocess.run([sys.executable,str(ROOT/'scripts/audit_visual.py'),'--run',str(run),'--output',str(dest),'--gpu',str(gpu)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 status.update(state='complete' if code==0 else 'failed',end=time.time(),returncode=code);p.write_text(json.dumps(status))
