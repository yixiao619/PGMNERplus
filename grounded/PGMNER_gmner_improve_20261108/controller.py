"""Resumable fixed experimental schedule, no TEST-dependent follow-up search."""
import subprocess, sys, os, json, time, traceback, hashlib
from pathlib import Path
ROOT=Path(__file__).parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
PY=sys.executable

def status(stage, **extra):
    value=dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S %z'),pid=os.getpid(),**extra)
    (OUT/'status.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)

def cpu(stage):
    status(stage)
    with (OUT/f'{stage}.log').open('a') as log:
        subprocess.run([PY,'-u',str(ROOT/'experiment.py'),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def gpu(split):
    status('context_'+split)
    handles=[]
    for shard, gpu_id in enumerate([1,2]):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu_id),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH='/data/cyc/LLARA/peft/src'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
        log=(OUT/f'context_{split}_shard{shard}.log').open('a')
        proc=subprocess.Popen([PY,'-u',str(ROOT/'context_worker.py'),'--split',split,'--shard',str(shard)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        handles.append((proc,log))
    failures=[]
    for proc,log in handles:
        code=proc.wait();log.close()
        if code:failures.append(dict(pid=proc.pid,code=code))
    if failures:raise RuntimeError(f'Worker failure: {failures}')

try:
    status('starting')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    (OUT/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
    cpu('baseline');cpu('pilot');gpu('dev');cpu('select')
    lock=json.loads((OUT/'locked_policy.json').read_text())
    if lock['winner']['source']=='context':gpu('test')
    cpu('final');status('complete')
except Exception:
    status('failed',error=traceback.format_exc());raise
