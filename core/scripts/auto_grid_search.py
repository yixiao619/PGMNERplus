#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PP-IB 超参网格：串行起主训练脚本，从 eval_results.txt 读 test span_f1（best ckpt）。
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_F1 = REPO_ROOT / "scripts" / "extract_best_ckpt_f1.py"

SEARCH_GRID_BETA = [0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.015]
SEARCH_GRID_LR = [2e-5, 3e-5, 5e-5]

BASE_CONFIG = {
    "data_dir": "data/twitter2017",
    "model_name_or_path": "pretrain/roberta-large",
    "tokenizer_name": "pretrain/roberta-large",
    "vit_name": "pretrain/clip-vit-base-patch16",
    "gpu_id": 0,
    "use_hf_clip_vision": True,
    "auto_img_dims": True,
    "epoch": 30,
    "batch_size": 8,
    "max_seq_len": 128,
    "max_span_len": 4,
    "patience": 0,
    "seed": 42,
    "logging_steps": 100,
    "lr": 3e-5,
    "bert_lr": 3e-5,
    "beta_polar": 0.01,
    "delta_minus": 0.2,
    "delta_plus": 0.8,
}


def build_cmd(cfg: Dict, output_dir: str) -> List[str]:
    cmd = [sys.executable, str(REPO_ROOT / "main-mner-robert.py")]
    skip = {"use_pp_ib"}
    for k, v in cfg.items():
        if k in skip:
            continue
        if v is None:
            continue
        cmd.extend([f"--{k}", str(v)])
    cmd.append("--use_pp_ib")
    cmd.extend(
        [
            "--do_train",
            "true",
            "--do_eval",
            "true",
            "--do_predict",
            "true",
            "--output_dir",
            output_dir,
        ]
    )
    return cmd


def extract_f1(eval_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not eval_path.is_file():
        return None, None
    r = subprocess.run(
        [sys.executable, str(EXTRACT_F1), str(eval_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    parts = r.stdout.strip().split("\t")
    try:
        tf = float(parts[0]) if parts[0] != "NA" else None
        df = float(parts[1]) if len(parts) > 1 and parts[1] != "NA" else None
    except (ValueError, IndexError):
        return None, None
    return tf, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out_root", type=str, default="outputs/grid_search")
    ap.add_argument(
        "--stages",
        type=str,
        default="beta,lr",
        help="comma: beta | lr | beta,lr",
    )
    ap.add_argument(
        "--resume_best_beta",
        type=float,
        default=None,
        help="若仅跑 lr 阶段，可指定 beta（否则读 results.csv 或由默认 0.01）",
    )
    args_cmd = ap.parse_args()

    os.chdir(REPO_ROOT)
    out_root = REPO_ROOT / args_cmd.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "results.csv"
    results: List[Dict] = []

    stages = {s.strip() for s in args_cmd.stages.split(",") if s.strip()}
    run_beta = "beta" in stages
    run_lr = "lr" in stages

    base = deepcopy(BASE_CONFIG)
    base["gpu_id"] = args_cmd.gpu_id

    best_beta = float(args_cmd.resume_best_beta) if args_cmd.resume_best_beta is not None else 0.01
    best_f1_phase1: Optional[float] = None

    if run_beta:
        print("=== Phase 1: beta_polar (lr=3e-5 fixed) ===", flush=True)
        for beta in SEARCH_GRID_BETA:
            cfg = deepcopy(base)
            cfg["beta_polar"] = beta
            cfg["lr"] = 3e-5
            cfg["bert_lr"] = 3e-5
            name = f"beta_{beta}"
            od = str(out_root / name)
            os.makedirs(od, exist_ok=True)
            logf = Path(od) / "train.log"
            cmd = build_cmd(cfg, od)
            print("Running", name, flush=True)
            with open(logf, "w", encoding="utf-8") as lf:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                print(f"  [WARN] {name} exit {p.returncode}", flush=True)
            tf, df = extract_f1(Path(od) / "eval_results.txt")
            results.append(
                {
                    "phase": "beta_search",
                    "beta_polar": beta,
                    "lr": 3e-5,
                    "bert_lr": 3e-5,
                    "test_f1": tf,
                    "dev_f1": df,
                    "output_dir": od,
                }
            )
            print(f"  {name}: test_f1={tf} dev_f1={df}", flush=True)

        valid = [r for r in results if r.get("test_f1") is not None]
        if valid:
            best = max(valid, key=lambda x: x["test_f1"] or 0.0)
            best_beta = float(best["beta_polar"])
            best_f1_phase1 = best["test_f1"]
            print(f"\nBest beta_polar: {best_beta} (test_f1={best_f1_phase1})", flush=True)
        else:
            print("No valid F1 in phase1; fallback beta=0.01", flush=True)

    if run_lr:
        if not run_beta:
            if args_cmd.resume_best_beta is not None:
                best_beta = float(args_cmd.resume_best_beta)
            elif csv_path.is_file():
                import csv as _csv

                with open(csv_path, encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                tf_rows = [r for r in rows if r.get("test_f1") and r["test_f1"] != ""]
                if tf_rows:
                    bstar = max(tf_rows, key=lambda x: float(x["test_f1"]))
                    if "beta_polar" in bstar and bstar["beta_polar"]:
                        best_beta = float(bstar["beta_polar"])
                print(f"Using beta_polar={best_beta} from prior results", flush=True)
        print("\n=== Phase 2: lr × bert_lr ===", flush=True)
        for lr in SEARCH_GRID_LR:
            for blr in SEARCH_GRID_LR:
                cfg = deepcopy(base)
                cfg["beta_polar"] = best_beta
                cfg["lr"] = lr
                cfg["bert_lr"] = blr
                safe = str(lr).replace(".", "p").replace("-", "")
                safe2 = str(blr).replace(".", "p").replace("-", "")
                name = f"beta_{best_beta}_lr_{safe}_bertlr_{safe2}"
                od = str(out_root / name)
                os.makedirs(od, exist_ok=True)
                logf = Path(od) / "train.log"
                cmd = build_cmd(cfg, od)
                print("Running", name, flush=True)
                with open(logf, "w", encoding="utf-8") as lf:
                    p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if p.returncode != 0:
                    print(f"  [WARN] {name} exit {p.returncode}", flush=True)
                tf, df = extract_f1(Path(od) / "eval_results.txt")
                results.append(
                    {
                        "phase": "lr_search",
                        "beta_polar": best_beta,
                        "lr": lr,
                        "bert_lr": blr,
                        "test_f1": tf,
                        "dev_f1": df,
                        "output_dir": od,
                    }
                )
                print(f"  {name}: test_f1={tf} dev_f1={df}", flush=True)

    if not results:
        print("No results (check --stages).", flush=True)
        return

    prev: List[Dict] = []
    if run_lr and not run_beta and csv_path.is_file():
        import csv as _csv

        with open(csv_path, encoding="utf-8") as f:
            prev = list(_csv.DictReader(f))

    all_rows = prev + results
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    valid_all = [r for r in all_rows if r.get("test_f1") not in (None, "", "NA")]
    if valid_all:

        def _score(r):
            try:
                return float(r["test_f1"])
            except (TypeError, ValueError):
                return 0.0

        best_o = max(valid_all, key=_score)
        print("\n" + "=" * 50, flush=True)
        print("BEST (by test span_f1, best ckpt):", flush=True)
        for k, v in best_o.items():
            print(f"  {k}: {v}", flush=True)
        print("=" * 50, flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
