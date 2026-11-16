#!/usr/bin/env python3
"""Create an immutable real-batch Stage-0 gradient-audit submission."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
F=Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment_factory")
def main():
 p=argparse.ArgumentParser();p.add_argument("routes",nargs="*");p.add_argument("--dry-run",action="store_true");a=p.parse_args();cmd=[sys.executable,str(F/"submit.py"),"--stage","0"]
 if a.routes:cmd.extend(["--routes",*a.routes])
 if a.dry_run:cmd.append("--dry-run")
 raise SystemExit(subprocess.run(cmd).returncode)
if __name__=="__main__":main()
