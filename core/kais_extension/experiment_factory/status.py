#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from pathlib import Path
F=Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory")
def main():
 p=argparse.ArgumentParser();p.add_argument("--submission");p.add_argument("--request-stop",action="store_true");a=p.parse_args();sub=Path(a.submission or (F/"runs/latest_submission.txt").read_text().strip())
 if a.request_stop:(sub/"stop.request").write_text("stop scheduling new jobs; let active jobs finish\n");print(f"Stop requested: {sub}");return
 pid=(sub/"scheduler.pid").read_text().strip() if (sub/"scheduler.pid").exists() else "";print(f"submission={sub}\nscheduler_pid={pid}\nrunning={bool(pid and Path('/proc',pid).exists())}")
 for pth in sorted((sub/"jobs").glob("*.json")):
  j=json.loads(pth.read_text());s=sub/"runs"/j["id"]/"status.json";state=json.loads(s.read_text()).get("status") if s.exists() else "pending";print(f"{j['id']}: {state}")
if __name__=="__main__":main()
