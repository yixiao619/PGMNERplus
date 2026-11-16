#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,pwd,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory"
def args():p=argparse.ArgumentParser();p.add_argument("--submission",required=True);return p.parse_args()
def gpu_state():
 out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True);g={}
 for row in csv.reader(out.splitlines()):
  i,u,b,total,used,free,util=[x.strip() for x in row];g[u]={"index":int(i),"uuid":u,"pci_bus_id":b,"total_mib":int(total),"used_mib":int(used),"free_mib":int(free),"utilization":int(util),"compute_processes":[]}
 try:
  p=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True)
  for row in csv.reader(p.splitlines()):
   u,pid,name,mem=[x.strip() for x in row]
   if u in g:
    try:owner=pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (FileNotFoundError,KeyError,PermissionError):owner="unknown"
    g[u]["compute_processes"].append({"pid":int(pid),"user":owner,"name":name,"used_mib":int(mem)})
 except subprocess.CalledProcessError:pass
 return g
def log(path,msg):
 line=f"[{time.strftime('%F %T')}] {msg}";print(line,flush=True)
 with path.open("a") as f:f.write(line+"\n")
def main():
 a=args();sub=Path(a.submission).resolve();policy=json.loads((sub/"gpu_policy.json").read_text());snapshot=json.loads((sub/"submission.json").read_text());mainlog=sub/"logs/scheduler.log";jobs=[json.loads(p.read_text()) for p in sorted((sub/"jobs").glob("*.json"))];active={};pending=[]
 for j in jobs:
  status=sub/"runs"/j["id"]/"status.json"
  if status.exists() and json.loads(status.read_text()).get("status") in {"complete","failed","unsupported"}:continue
  pending.append(j)
 (sub/"scheduler.pid").write_text(str(os.getpid())+"\n");log(mainlog,f"start shared policy jobs={len(pending)} concurrency={policy['global_max_concurrency']}")
 while pending or active:
  for pid,(proc,j,u,fh,started,timeout_s) in list(active.items()):
   rc=proc.poll()
   if rc is None and time.time()-started>timeout_s:
    os.killpg(proc.pid,signal.SIGTERM);log(mainlog,f"timeout {j['id']} after {timeout_s}s; terminating framework process group")
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=proc.returncode
   if rc is not None:
    fh.close();log(mainlog,f"finish {j['id']} gpu={u} rc={rc}");del active[pid]
  if (sub/"stop.request").exists():
   if not active:break
   log(mainlog,"stop requested; no new jobs, waiting for active jobs")
  elif len(active)<int(policy["global_max_concurrency"]):
   states=gpu_state();used_gpu={v[2] for v in active.values()};chosen=None
   for u in policy["allowed_gpu_uuids"]:
    if u in used_gpu or u not in states:continue
    reserve=max(int(states[u]["total_mib"]*float(policy["reserve_memory_fraction"])),int(policy["reserve_memory_mib_min"]));needed=int(pending[0]["expected_peak_mib"]) if pending else 0
    if states[u]["free_mib"]>=reserve+needed:chosen=(u,states[u],reserve);break
   if chosen and pending:
    j=pending.pop(0);u,state,reserve=chosen;run=sub/"runs"/j["id"];run.mkdir(parents=True,exist_ok=True);(run/"assignment.json").write_text(json.dumps({"assigned_at":time.time(),"gpu":state,"reserve_mib":reserve,"expected_peak_mib":j["expected_peak_mib"]},indent=2)+"\n");lf=(sub/"logs"/f"{j['id']}.log").open("a");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=u;cmd=[sys.executable,"-u",str(F/"run_one.py"),"--job",str(sub/"jobs"/f"{j['id']}.json"),"--run-dir",str(run)]
    if int(snapshot["requested_stage"])==0:cmd.append("--smoke")
    proc=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,env=env,start_new_session=True);timeout_s=int(j.get("timeout_seconds",snapshot.get("stage1",{}).get("timeout_seconds",10800)));active[proc.pid]=(proc,j,u,lf,time.time(),timeout_s);log(mainlog,f"launch {j['id']} gpu={u} free={state['free_mib']} util={state['utilization']} existing_processes={len(state['compute_processes'])} expected={j['expected_peak_mib']} reserve={reserve} timeout={timeout_s}s")
  if pending or active:time.sleep(5 if active else int(policy["poll_seconds"]))
 if not snapshot.get("custom_summary",False):subprocess.run([sys.executable,str(F/"summarize.py"),"--submission",str(sub)],check=False)
 log(mainlog,"scheduler finished")
if __name__=="__main__":main()
