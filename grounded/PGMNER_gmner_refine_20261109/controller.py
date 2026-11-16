import os,sys,subprocess,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);PY=sys.executable

def status(stage,**kw):
 d=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z'),**kw);(OUT/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
def gpu(split):
 status('scoring_'+split);jobs=[]
 for shard in range(3):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(shard),PYTHONPATH='/data/cyc/LLARA/peft/src',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  f=(OUT/f'{split}_worker{shard}.log').open('a');p=subprocess.Popen([PY,'-u',str(ROOT/'worker.py'),split,str(shard)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);jobs.append((p,f))
 codes=[]
 for p,f in jobs:codes.append(p.wait());f.close()
 if any(codes):raise RuntimeError(f'{split} worker exits {codes}')
def cpu(stage):
 status(stage)
 with (OUT/f'{stage}.log').open('a') as f:subprocess.run([PY,'-u',str(ROOT/'train.py'),stage],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 status('starting');(OUT/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},indent=2))
 gpu('dev');cpu('fit');gpu('test');cpu('evaluate');status('complete')
except Exception:status('failed',error=traceback.format_exc());raise
