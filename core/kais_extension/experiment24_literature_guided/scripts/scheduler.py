#!/usr/bin/env python3
from __future__ import annotations
import csv,json,os,pwd,subprocess,sys,time
from pathlib import Path

ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment24_literature_guided"

def gpu_state():
    out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    g={}
    for r in csv.reader(out.splitlines()):
        i,u,b,t,f,z=[x.strip() for x in r];g[u]={"index":int(i),"uuid":u,"pci_bus_id":b,"total_mib":int(t),"free_mib":int(f),"utilization":int(z),"processes":[]}
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

def log(path,msg):
    s=f"[{time.strftime('%F %T')}] {msg}";print(s,flush=True)
    with path.open("a") as f:f.write(s+"\n")

def main():
    matrix=json.loads((EXP/"configs/experiment_matrix.json").read_text());policy=json.loads((EXP/"configs/gpu_policy.json").read_text())
    skip_ids={x for x in os.environ.get("E24_SKIP_IDS","").split(",") if x}
    pending=[]
    for j in matrix["jobs"]:
        if j["id"] in skip_ids:continue
        sp=EXP/"runs"/j["id"]/"status.json"
        if not(sp.exists() and json.loads(sp.read_text()).get("status")=="complete"):pending.append(j)
    active={};lp=EXP/"logs/scheduler.log";log(lp,f"start pending={len(pending)} concurrency={policy['global_max_concurrency']} skip={sorted(skip_ids)}")
    while pending or active:
        for pid,(proc,j,u,fh) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                fh.close();sp=EXP/"runs"/j["id"]/"status.json";st=json.loads(sp.read_text()).get("status") if sp.exists() else "missing"
                log(lp,f"finish id={j['id']} rc={rc} status={st}");del active[pid]
        if (EXP/"stop.request").exists():
            if not active:break
        elif pending and len(active)<int(policy["global_max_concurrency"]):
            gs=gpu_state();used={x[2] for x in active.values()};pick=None;j=pending[0]
            for u in policy["allowed_gpu_uuids"]:
                if u in used or u not in gs:continue
                s=gs[u];reserve=max(int(s["total_mib"]*float(policy["reserve_memory_fraction"])),int(policy["reserve_memory_mib_min"]))
                if s["free_mib"]>=int(j["expected_peak_mib"])+reserve:pick=(u,s,reserve);break
            if pick:
                pending.pop(0);u,s,reserve=pick;run=EXP/"runs"/j["id"];run.mkdir(parents=True,exist_ok=True)
                (run/"assignment.json").write_text(json.dumps({"job":j,"gpu":s,"reserve_mib":reserve},indent=2)+"\n")
                fh=(EXP/"logs"/f"{j['id']}.log").open("a");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=u
                if j["route"]=="B":cmd=[sys.executable,"-u",str(EXP/"scripts/run_grounding_route.py"),"--run-dir",str(run)]
                else:cmd=[sys.executable,"-u",str(EXP/"scripts/run_twitter_route.py"),"--route",j["route"],"--run-dir",str(run),"--seed",str(matrix["seed"])]
                proc=subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active[proc.pid]=(proc,j,u,fh);log(lp,f"launch id={j['id']} pid={proc.pid} gpu={u} bus={s['pci_bus_id']} free={s['free_mib']} util={s['utilization']} existing={s['processes']}")
        if pending or active:time.sleep(int(policy["monitor_seconds"]) if active else int(policy["poll_seconds"]))
    subprocess.run([sys.executable,str(EXP/"scripts/summarize.py")],check=False)
    log(lp,"scheduler finished")

if __name__=="__main__":main()
