#!/usr/bin/env python3
from __future__ import annotations
import csv,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");EXP=ROOT/"kais_extension/experiment24_literature_guided"
policy=json.loads((EXP/"configs/gpu_policy.json").read_text());log=EXP/"logs/posthoc_launcher.log"
def note(s):
    x=f"[{time.strftime('%F %T')}] {s}";print(x,flush=True)
    with log.open("a") as f:f.write(x+"\n")
while True:
    out=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.total,memory.free","--format=csv,noheader,nounits"],text=True)
    rows={r[0].strip():(int(r[1]),int(r[2])) for r in csv.reader(out.splitlines())};pick=None
    for u in policy["allowed_gpu_uuids"]:
        if u not in rows:continue
        total,free=rows[u];reserve=max(int(total*float(policy["reserve_memory_fraction"])),int(policy["reserve_memory_mib_min"]))
        if free>=20480+reserve:pick=(u,free,reserve);break
    if pick:break
    note("waiting for posthoc GPU capacity");time.sleep(int(policy["poll_seconds"]))
u,free,reserve=pick;note(f"launch posthoc gpu={u} free={free} reserve={reserve}")
env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=u
with (EXP/"logs/posthoc_mechanism.log").open("a") as fh:
    rc=subprocess.call([sys.executable,"-u",str(EXP/"scripts/posthoc_mechanism.py")],stdout=fh,stderr=subprocess.STDOUT,env=env)
note(f"posthoc exit={rc}");sys.exit(rc)
