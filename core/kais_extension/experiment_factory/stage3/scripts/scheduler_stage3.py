#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,pwd,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory";S=F/"stage3"
def gpu_state():
 out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True);g={}
 for r in csv.reader(out.splitlines()):
  i,u,b,t,used,free,util=[x.strip() for x in r];g[u]={"index":int(i),"uuid":u,"pci_bus_id":b,"total_mib":int(t),"used_mib":int(used),"free_mib":int(free),"utilization":int(util),"compute_processes":[]}
 try:
  out=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True)
  for r in csv.reader(out.splitlines()):
   u,pid,name,mem=[x.strip() for x in r]
   if u in g:
    try:user=pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:user="unknown"
    g[u]["compute_processes"].append({"pid":int(pid),"user":user,"name":name,"used_mib":int(mem)})
 except subprocess.CalledProcessError:pass
 return g
def log(path,msg):
 line=f"[{time.strftime('%F %T')}] {msg}";print(line,flush=True)
 with path.open("a") as f:f.write(line+"\n")
def descendants(root):
 tree={}
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:ppid=int((p/"stat").read_text().split()[3])
  except Exception:continue
  tree.setdefault(ppid,[]).append(int(p.name))
 out={root};stack=[root]
 while stack:
  x=stack.pop()
  for c in tree.get(x,[]):
   if c not in out:out.add(c);stack.append(c)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);policy=json.loads((sub/"gpu_policy.json").read_text());jobs=[json.loads(x.read_text()) for x in sorted((sub/"jobs").glob("*.json"))];jobs.sort(key=lambda x:(0 if x["group"]=="B" else 1,x["id"]));pending=[]
 for j in jobs:
  st=sub/"runs"/j["id"]/"status.json"
  if st.exists() and json.loads(st.read_text()).get("status")=="complete":continue
  pending.append(j)
 active={};peaks={};attempts={};mainlog=sub/"logs/scheduler.log";(sub/"scheduler.pid").write_text(str(os.getpid())+"\n");log(mainlog,f"start jobs={len(pending)} shared concurrency={policy['global_max_concurrency']}")
 while pending or active:
  states=gpu_state()
  for pid,v in list(active.items()):
   proc,j,u,fh,started,initial_free,known=v;rc=proc.poll();state=states.get(u,{})
   if rc is None:
    family=descendants(pid);own_mem=sum(x["used_mib"] for x in state.get("compute_processes",[]) if x["pid"] in family);peaks[pid]=max(peaks.get(pid,0),own_mem)
    with (sub/"runs"/j["id"]/"resource_monitor.jsonl").open("a") as f:f.write(json.dumps({"time":time.time(),"gpu_free_mib":state.get("free_mib"),"gpu_utilization":state.get("utilization"),"job_memory_mib":own_mem,"compute_processes":state.get("compute_processes",[])})+"\n")
    current={x["pid"] for x in state.get("compute_processes",[])};foreign_new=current-known
    if foreign_new and initial_free-state.get("free_mib",initial_free)>=int(policy["foreign_process_drop_event_mib"]):
     with (sub/"runs"/j["id"]/"memory_events.jsonl").open("a") as f:f.write(json.dumps({"time":time.time(),"event":"foreign_process_and_free_memory_drop","initial_free_mib":initial_free,"current_free_mib":state.get("free_mib"),"new_pids":sorted(foreign_new),"gpu":u})+"\n")
     known.update(foreign_new)
    if time.time()-started>int(j["timeout_seconds"]):os.killpg(pid,signal.SIGTERM);rc=proc.wait();log(mainlog,f"timeout {j['id']}")
   if rc is not None:
    fh.close();st=sub/"runs"/j["id"]/"status.json";sv=json.loads(st.read_text()) if st.exists() else {"status":"failed","error":"missing status"};sv["scheduler_peak_gpu_memory_mib"]=peaks.get(pid,0);st.write_text(json.dumps(sv,indent=2)+"\n");log(mainlog,f"finish {j['id']} rc={rc} status={sv.get('status')} peak_mib={peaks.get(pid,0)}")
    if sv.get("status")!="complete" and attempts[j["id"]]<int(j.get("max_attempts",1)):
     pending.append(j);log(mainlog,f"requeue {j['id']} attempt={attempts[j['id']]+1}; restart is from beginning, not lossless resume")
    del active[pid];peaks.pop(pid,None)
  if (sub/"stop.request").exists():
   if not active:break
  elif pending and len(active)<int(policy["global_max_concurrency"]):
   used={x[2] for x in active.values()};chosen=None
   for u in policy["allowed_gpu_uuids"]:
    if u in used or u not in states:continue
    st=states[u];reserve=max(int(st["total_mib"]*float(policy["reserve_memory_fraction"])),int(policy["reserve_memory_mib_min"]));need=int(pending[0]["expected_peak_mib"])
    if st["free_mib"]>=need+reserve:chosen=(u,st,reserve);break
   if chosen:
    j=pending.pop(0);u,st,reserve=chosen;attempts[j["id"]]=attempts.get(j["id"],0)+1;run=sub/"runs"/j["id"];run.mkdir(parents=True,exist_ok=True);(run/"assignment.json").write_text(json.dumps({"assigned_at":time.time(),"attempt":attempts[j["id"]],"gpu":st,"reserve_mib":reserve,"expected_peak_mib":j["expected_peak_mib"],"lossless_resume_supported":False},indent=2)+"\n");lf=(sub/"logs"/f"{j['id']}.log").open("a");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=u
    runner=F/"run_one.py" if j["runner"]=="factory" else S/"scripts/run_full_training.py";cmd=[sys.executable,"-u",str(runner),"--job",str(sub/"jobs"/f"{j['id']}.json"),"--run-dir",str(run)];proc=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,env=env,start_new_session=True);known={x["pid"] for x in st["compute_processes"]};peaks[proc.pid]=0;active[proc.pid]=(proc,j,u,lf,time.time(),st["free_mib"],known);log(mainlog,f"launch {j['id']} attempt={attempts[j['id']]} gpu={u} bus={st['pci_bus_id']} free={st['free_mib']} util={st['utilization']} existing={st['compute_processes']} expected={j['expected_peak_mib']} reserve={reserve}")
  if pending or active:time.sleep(int(policy["monitor_seconds"]) if active else int(policy["poll_seconds"]))
 log(mainlog,"scheduler finished")
if __name__=="__main__":main()
