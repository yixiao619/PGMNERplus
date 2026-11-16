import subprocess,sys,os,json,time,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);PY=sys.executable
def status(stage,**kw):
    d=dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S %z'),**kw)
    (OUT/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
def cpu(stage):
    status(stage)
    with (OUT/f'{stage}.log').open('a') as f:subprocess.run([PY,'-u',str(ROOT/'experiment.py'),stage],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
def gpu(split):
    status('review_'+split);jobs=[]
    for shard in range(3):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='/data/cyc/LLARA/peft/src')
        f=(OUT/f'review_{split}_shard{shard}.log').open('a')
        p=subprocess.Popen([PY,'-u',str(ROOT/'review_worker.py'),'--split',split,'--shard',str(shard)],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT);jobs.append((p,f))
    codes=[]
    for p,f in jobs:codes.append(p.wait());f.close()
    if any(codes):raise RuntimeError(f'Worker exit codes: {codes}')
try:
    status('starting')
    (OUT/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},indent=2))
    cpu('cached');gpu('dev');cpu('select')
    if json.loads((OUT/'locked_final.json').read_text())['winner']=='review':gpu('test')
    cpu('final');status('complete')
except Exception:status('failed',error=traceback.format_exc());raise
