#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");F=ROOT/"kais_extension/experiment_factory";S=F/"stage2"
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);state=S/"runner_status.json";(S/"runner.pid").write_text(str(os.getpid())+"\n")
 state.write_text(json.dumps({"status":"running","submission":str(sub),"started_at":time.time()},indent=2)+"\n")
 try:
  rc=subprocess.run([sys.executable,"-u",str(F/"scheduler.py"),"--submission",str(sub)]).returncode
  subprocess.run([sys.executable,str(S/"summarize_stage2.py"),"--submission",str(sub)],check=False)
  state.write_text(json.dumps({"status":"complete" if rc==0 else "failed","submission":str(sub),"scheduler_returncode":rc,"finished_at":time.time()},indent=2)+"\n")
  raise SystemExit(rc)
 except BaseException as e:
  if not isinstance(e,SystemExit):state.write_text(json.dumps({"status":"failed","submission":str(sub),"error":repr(e),"traceback":traceback.format_exc()},indent=2)+"\n")
  raise
if __name__=="__main__":main()
