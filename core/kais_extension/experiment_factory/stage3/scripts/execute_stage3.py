#!/usr/bin/env python3
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/jyx/yixiao/MINER-My");S=ROOT/"kais_extension/experiment_factory/stage3"
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission",required=True);a=p.parse_args();sub=Path(a.submission);(S/"runner.pid").write_text(str(os.getpid())+"\n");state=S/"runner_status.json";state.write_text(json.dumps({"status":"running","submission":str(sub),"started_at":time.time()},indent=2)+"\n")
 try:
  rc=subprocess.run([sys.executable,"-u",str(S/"scripts/scheduler_stage3.py"),"--submission",str(sub)]).returncode;subprocess.run([sys.executable,str(S/"scripts/summarize_stage3.py"),"--submission",str(sub)],check=False);state.write_text(json.dumps({"status":"complete" if rc==0 else "failed","submission":str(sub),"returncode":rc,"finished_at":time.time()},indent=2)+"\n");raise SystemExit(rc)
 except BaseException as e:
  if not isinstance(e,SystemExit):state.write_text(json.dumps({"status":"failed","error":repr(e),"traceback":traceback.format_exc()},indent=2)+"\n")
  raise
if __name__=="__main__":main()
