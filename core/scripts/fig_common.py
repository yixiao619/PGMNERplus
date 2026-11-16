#!/usr/bin/env python3
"""Shared helpers for paper figure scripts (Fig 3–5)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
VIZ_DIR = ROOT / "viz"
FIG_DIR = ROOT / "figures"
JSONL = RESULTS_DIR / "hparam_sweep.jsonl"
CONFIG = ROOT / "scripts" / "fig345_config.json"

# Paper Fig.3 fixed hyperparameters (mapped to CLI flags)
FIG3_FIXED: Dict[str, Any] = {
    "partial_mass_init": 0.6,   # s_init (beta sweep)
    "beta_polar": 0.01,         # fixed for s_init sweep
    "delta_minus": 0.2,         # rho
    "delta_plus": 0.8,
    "pp_ib_lambda_balance": 1.0,
    "entropy_eps": 0.1,         # epsilon
    "ot_alpha": 0.2,            # lambda (OT loss scale)
    "dust_cost": 0.5,           # tau
    "ot_sinkhorn_iter": 30,
    "ot_warmup_epochs": 3,
    "use_pp_ib": True,
    "use_got": True,
    "got_mode": "partial_sinkhorn",
    "use_ema": True,
    "ema_decay": 0.999,
}

BETA_VALUES = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
SINIT_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [42, 1337, 2024]

DEFAULT_CKPT = ROOT / "outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth"
DEFAULT_RUN_DIR = ROOT / "outputs/pp_ib_partial_ot_grid_dc10_m05_seed42"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    return {
        "pgmner_run_dir": str(DEFAULT_RUN_DIR),
        "pgmner_ckpt": str(DEFAULT_CKPT),
        "hvpnet_preds": "",
        "smner_preds": "",
        "case_a_idx": 42,
        "case_b_idx": 17,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_keys(path: Path = JSONL) -> Set[Tuple[str, str, int]]:
    """Only treat successful runs (rc=0, sensible F1) as done."""
    done: Set[Tuple[str, str, int]] = set()
    for r in read_jsonl(path):
        if r.get("rc", 0) not in (0, None):
            continue
        f1 = r.get("best_f1")
        if f1 is None or float(f1) < 0.5:
            continue
        done.add((r["hparam_name"], str(r["value"]), int(r["seed"])))
    return done


def best_test_f1_from_run(out_dir: Path) -> Optional[float]:
    hist = out_dir / "per_class_history.jsonl"
    if hist.exists():
        best = None
        for line in hist.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "test":
                continue
            f1 = float(r["overall"]["f1"])
            best = f1 if best is None else max(best, f1)
        if best is not None:
            return best
    er = out_dir / "eval_results.txt"
    if er.exists():
        text = er.read_text(errors="ignore")
        if "test best ckpt" in text:
            m = re.search(r"span_f1\s*=\s*([\d.]+)", text.split("test best ckpt")[-1])
            if m:
                return float(m.group(1))
    tl = out_dir / "train.log"
    if tl.exists():
        text = tl.read_text(errors="ignore")
        if "test best ckpt" in text:
            m = re.search(
                r"test best ckpt.*?span_f1\s*=\s*([\d.]+)", text, re.S
            )
            if m:
                return float(m.group(1))
    return None


def build_train_cmd(
    out_dir: Path,
    seed: int,
    gpu: int,
    extra: Dict[str, Any],
) -> List[str]:
    args = {
        "data_dir": "data/twitter2017",
        "model_name_or_path": "pretrain/roberta-large",
        "tokenizer_name": "pretrain/roberta-large",
        "vit_name": "pretrain/clip-vit-base-patch16",
        "use_hf_clip_vision": "true",
        "auto_img_dims": "true",
        "gpu_id": 0,
        "batch_size": 16,
        "max_seq_len": 128,
        "max_span_len": 4,
        "lr": "3e-5",
        "bert_lr": "1e-5",
        "warmup_ratio": 0.1,
        "logging_steps": 100,
        "do_train": "true",
        "do_eval": "true",
        "do_predict": "true",
        "epoch": 30,
        "patience": 5,
        "seed": seed,
        "output_dir": str(out_dir),
    }
    args.update(FIG3_FIXED)
    args.update(extra)
    cmd = [sys.executable, str(ROOT / "main-mner-robert.py")]
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    return cmd, env


def build_eval_argv(gpu_id: int, ckpt_run: Path) -> List[str]:
    """Argv for importing main-mner-robert.py in eval/collect mode."""
    return [
        "main-mner-robert.py",
        "--data_dir",
        "data/twitter2017",
        "--output_dir",
        str(ckpt_run),
        "--model_name_or_path",
        "pretrain/roberta-large",
        "--tokenizer_name",
        "pretrain/roberta-large",
        "--vit_name",
        "pretrain/clip-vit-base-patch16",
        "--use_hf_clip_vision",
        "true",
        "--auto_img_dims",
        "true",
        "--gpu_id",
        str(gpu_id),
        "--batch_size",
        "8",
        "--max_seq_len",
        "128",
        "--max_span_len",
        "4",
        "--do_train",
        "false",
        "--do_eval",
        "true",
        "--do_predict",
        "false",
        "--seed",
        "42",
        "--use_pp_ib",
        "--beta_polar",
        "0.01",
        "--delta_minus",
        "0.2",
        "--delta_plus",
        "0.8",
        "--use_got",
        "--got_mode",
        "partial_sinkhorn",
        "--ot_alpha",
        "0.05",
        "--entropy_eps",
        "0.1",
        "--partial_mass_init",
        "0.5",
        "--dust_cost",
        "0.8",
        "--use_ema",
        "--ema_decay",
        "0.999",
    ]


def load_main_module(gpu_id: int, ckpt_run: Path):
    """Import main-mner-robert with stable argv (avoid --gpu -> --gpu_id abbreviation)."""
    import importlib.util

    import torch

    old_argv = sys.argv
    sys.argv = build_eval_argv(gpu_id, ckpt_run)
    spec = importlib.util.spec_from_file_location(
        "main_mner_robert", ROOT / "main-mner-robert.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = old_argv
    args = mod.args
    if torch.cuda.is_available() and args.gpu_id >= 0:
        torch.cuda.set_device(int(args.gpu_id))
    return mod, args


def to_infer_batch(batch: Dict[str, Any], device) -> Dict[str, Any]:
    """Move batch to device and skip loss branches that need span labels."""
    out = {k: v.to(device) for k, v in batch.items()}
    out["span_labels"] = None
    return out


def run_training(out_dir: Path, seed: int, gpu: int, extra: Dict[str, Any]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best_model.pth").exists():
        f1 = best_test_f1_from_run(out_dir)
        if f1 is not None and f1 >= 0.5:
            return 0
    cmd, env = build_train_cmd(out_dir, seed, gpu, extra)
    log_path = out_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode
