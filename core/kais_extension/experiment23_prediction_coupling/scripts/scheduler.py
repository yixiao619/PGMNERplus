#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,pwd,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment23_prediction_coupling"
def state():
 out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True);g={}
 for r in csv.reader(out.splitlines()):i,u,b,t,f,z=[x.strip() for x in r];g[u]={"index":int(i),"uuid":u,"pci_bus_id":b,"total_mib":int(t),"free_mib":int(f),"utilization":int(z),"processes":[]}
 try:
  out=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True)
  for r in csv.reader(out.splitlines()):
   u,pid,name,mem=[x.strip() for x in r]
   if u in g:
    try:user=pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:user="unknown"
    g[u]["processes"].append({"pid":int(pid),"user":user,"name":name,"used_mib":int(mem)})
 except subprocess.CalledProcessError:pass
 return g
def log(p,msg):
 s=f"[{time.strftime('%F %T')}] {msg}";print(s,flush=True)
 with p.open("a") as f:f.write(s+"\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);pol=json.loads((EXP/"configs/gpu_policy.json").read_text());jobs=[json.loads(x.read_text()) for x in sorted((sub/"jobs").glob("*.json"))];pending=[];attempt={}
 for j in jobs:
  sp=sub/"runs"/j["id"]/"status.json"
  if not (sp.exists() and json.loads(sp.read_text()).get("status")=="complete"):pending.append(j)
 active={};lp=sub/"logs/scheduler.log";log(lp,f"start jobs={len(pending)} concurrency={pol['global_max_concurrency']}")
 while pending or active:
  for pid,(pr,j,u,fh,start) in list(active.items()):
   rc=pr.poll()
   if rc is not None:
    fh.close();sp=sub/"runs"/j["id"]/"status.json";x=json.loads(sp.read_text()) if sp.exists() else {"status":"failed"};log(lp,f"finish {j['id']} rc={rc} status={x.get('status')}")
    if x.get("status")!="complete" and attempt[j["id"]]<int(pol["max_attempts"]):pending.append(j);log(lp,f"requeue {j['id']} from scratch")
    del active[pid]
  if (sub/"stop.request").exists():
   if not active:break
  elif pending and len(active)<int(pol["global_max_concurrency"]):
   gs=state();used={x[2] for x in active.values()};pick=None
   for u in pol["allowed_gpu_uuids"]:
    if u in used or u not in gs:continue
    s=gs[u];reserve=max(int(s["total_mib"]*float(pol["reserve_memory_fraction"])),int(pol["reserve_memory_mib_min"]));
    if s["free_mib"]>=int(pol["expected_peak_mib"])+reserve:pick=(u,s,reserve);break
   if pick:
    j=pending.pop(0);u,s,reserve=pick;attempt[j["id"]]=attempt.get(j["id"],0)+1;run=sub/"runs"/j["id"];run.mkdir(parents=True,exist_ok=True);(run/"assignment.json").write_text(json.dumps({"gpu":s,"reserve_mib":reserve,"attempt":attempt[j["id"]]},indent=2)+"\n");fh=(sub/"logs"/f"{j['id']}.log").open("a");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=u;cmd=[sys.executable,"-u",str(EXP/"scripts/run_one.py"),"--job",str(sub/"jobs"/f"{j['id']}.json"),"--run-dir",str(run)];pr=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,env=env,start_new_session=True);active[pr.pid]=(pr,j,u,fh,time.time());log(lp,f"launch {j['id']} gpu={u} bus={s['pci_bus_id']} free={s['free_mib']} util={s['utilization']} expected={pol['expected_peak_mib']} reserve={reserve} existing={s['processes']}")
  if pending or active:time.sleep(int(pol["monitor_seconds"]) if active else int(pol["poll_seconds"]))
 log(lp,"scheduler finished")
if __name__=="__main__":main()
