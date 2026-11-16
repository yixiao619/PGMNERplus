#!/usr/bin/env python3
from __future__ import annotations
import json,os
from pathlib import Path
S=Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory/stage2")
sub=Path((S/"latest_submission.txt").read_text().strip());pid=(S/"runner.pid").read_text().strip() if (S/"runner.pid").exists() else ""
print(f"submission={sub}\nrunner_pid={pid}\nrunning={bool(pid and Path('/proc',pid).exists())}")
for j in sorted((sub/"jobs").glob("*.json")):
 d=json.loads(j.read_text());p=sub/"runs"/d["id"]/"status.json";print(f"{d['id']}: {json.loads(p.read_text()).get('status') if p.exists() else 'pending'}")
