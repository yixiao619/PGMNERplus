"""Bounded two-round development exploration; fixed decision rules and all outcomes retained."""
import os,sys,json,time,re,hashlib,subprocess,shutil,concurrent.futures,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data/jyx/PGMNER_adaptive_split_20261104');OLD=Path('/data/jyx/PGMNER_day_20261104/runs/visual_only_full')
ARMS=[('full','polarized','partial'),('no_gate','off','partial'),('uniform','polarized','uniform_partial'),('no_local','polarized','none')]
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False));q.replace(p)
def verify():
 for n,h in json.loads((BASE/'source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
def one(stage,arm,gpu,config,smoke=False):
 verify();name,ib,transport=arm;out=BASE/stage/name;out.mkdir(parents=True,exist_ok=False)
 cmd=json.loads((OLD/'command.json').read_text());cmd[0]=sys.executable
 for k,v in {'--output_dir':str(out),'--gpu_id':str(gpu),'--repair_ib':ib,'--repair_transport':transport}.items():cmd[cmd.index(k)+1]=v
 if smoke:cmd+=['--repair_smoke_steps','3']
 dump(out/'command.json',cmd);dump(out/'experiment.json',config)
 env={**os.environ,'PGMNER_EXPERIMENT':json.dumps(config),'PYTHONHASHSEED':'42','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
 s={'state':'running','gpu':gpu,'start':time.time()};dump(out/'status.json',s)
 with (out/'train.log').open('w') as f:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 s.update(state='complete' if code==0 else 'failed',end=time.time(),returncode=code);dump(out/'status.json',s)
 if code or smoke:return {'arm':name,'train_code':code}
 audit=out/'audit';a={'state':'running','start':time.time()};dump(out/'audit_status.json',a)
 with (out/'audit.log').open('w') as f:acode=subprocess.run([sys.executable,str(ROOT/'scripts/audit_visual.py'),'--run',str(out),'--output',str(audit),'--gpu',str(gpu)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 a.update(state='complete' if acode==0 else 'failed',end=time.time(),returncode=acode);dump(out/'audit_status.json',a)
 scores=list(map(float,re.findall(r'^span_f1 = ([0-9.]+)',(out/'eval_results.txt').read_text(),re.M)))
 row={'arm':name,'train_code':code,'audit_code':acode,'curve':scores,'best_dev':max(scores)}
 if (audit/'results.json').exists():row['audit']=json.loads((audit/'results.json').read_text())
 dump(out/'summary.json',row);return row
def round_run(stage,config,smoke=False):
 def worker(gpu):
  choices=[ARMS[gpu]] if smoke else [a for i,a in enumerate(ARMS) if i%3==gpu]
  return [one(stage,a,gpu,config,smoke) for a in choices]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(worker,g) for g in range(3)];rows=[]
  for f in futures:rows.extend(f.result())
 dump(BASE/(stage+'_summary.json'),rows);return {r['arm']:r for r in rows}
def execute(smoke):
 verify();diag=json.loads((BASE/'vision_comparison.json').read_text())
 gain=2. if diag['clip_pretrained']['mean_residual_energy']<.01 else 1.
 config={'clip_anchor':True,'contrastive':.03,'image_keep':.6,'noise_scale':.25,'residual_gain':gain,'local_coefficient':.1}
 dump(BASE/'initial_config.json',config)
 if smoke:return round_run('smoke',config,True)
 for arm in ARMS[:3]:assert json.loads((BASE/'smoke'/arm[0]/'status.json').read_text())['state']=='complete'
 state={'state':'running','start':time.time(),'max_rounds':2};dump(BASE/'pipeline_status.json',state)
 try:
  r=round_run('round1',config)
  if any(v.get('train_code') or v.get('audit_code') for v in r.values()):raise RuntimeError('Round1 incomplete; preserve failure, no score-driven retry')
  f=r['full']['best_dev'];audit=r['full']['audit'];correct=audit['paired']['E2_correct_image']['delta_f1']
  good=all(f>r[k]['best_dev'] for k in ['no_gate','uniform','no_local']) and correct>0
  decision={'round1_full':f,'correct_image_delta':correct,'passed_exploratory_direction_check':good,'warning':'One seed and DEV-selected checkpoints; not confirmatory evidence.'}
  if good:decision.update(action='stop_after_round1',reason='All prespecified directions positive; no additional tuning')
  else:
   cfg=dict(config)
   if f<=r['no_local']['best_dev']:
    cfg['local_coefficient']=.05;reason='Local branch underperforms common-only control: halve only local coefficient'
   elif f<=r['no_gate']['best_dev']:
    cfg['image_keep']=.8;reason='Gate underperforms no-gate control: change only retention target .6 to .8'
   else:
    cfg['residual_gain']=min(4.,gain*2);reason='Assignment/content gains incomplete: double only residual gain, capped at4'
   decision.update(action='run_round2',reason=reason,next_config=cfg)
  dump(BASE/'decision.json',decision)
  if not good:
   rr=round_run('round2',cfg)
   if any(v.get('train_code') or v.get('audit_code') for v in rr.values()):raise RuntimeError('Round2 incomplete; see retained logs')
  state.update(state='complete',end=time.time());dump(BASE/'pipeline_status.json',state)
 except Exception:
  state.update(state='failed',end=time.time(),error=traceback.format_exc());dump(BASE/'pipeline_status.json',state);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
 if a.worker:execute(a.smoke)
 else:
  if not (BASE/'source_hashes.json').exists():
   files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'data' not in p.parts and 'pretrain' not in p.parts]
   dump(BASE/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
   snap=BASE/'source';snap.mkdir()
   for p in files:
    dst=snap/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  cmd=[sys.executable,str(Path(__file__).resolve()),'--worker']+(['--smoke'] if a.smoke else [])
  with (BASE/('smoke_pipeline.log' if a.smoke else 'pipeline.log')).open('w') as f:proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  dump(BASE/('smoke_launch.json' if a.smoke else 'launch.json'),{'pid':proc.pid,'command':cmd});print('pipeline PID',proc.pid)
