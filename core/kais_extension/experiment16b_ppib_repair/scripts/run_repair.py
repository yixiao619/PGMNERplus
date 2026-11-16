#!/usr/bin/env python3
"""Run one pre-registered PP-IB repair using the unmodified PGMNER pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment16b_ppib_repair"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main_args(output_dir: Path, warmup: int):
    return [
        "--data_dir", "data/twitter2017", "--output_dir", str(output_dir),
        "--model_name_or_path", "pretrain/roberta-large", "--tokenizer_name", "pretrain/roberta-large",
        "--vit_name", "pretrain/clip-vit-base-patch16", "--use_hf_clip_vision", "true", "--auto_img_dims", "true",
        "--gpu_id", "0", "--epoch", "40", "--batch_size", "16", "--max_seq_len", "128", "--max_span_len", "4",
        "--lr", "3e-5", "--bert_lr", "1e-5", "--warmup_ratio", "0.1", "--patience", "0", "--seed", "42",
        "--logging_steps", "100", "--do_train", "true", "--do_eval", "true", "--do_predict", "true",
        "--use_pp_ib", "--beta_polar", "0.01", "--delta_minus", "0.2", "--delta_plus", "0.8",
        "--pp_ib_lambda_balance", "1.0", "--pp_ib_gate_init_std", "0.05", "--pp_ib_gate_init_bias", "0.0",
        "--beta_polar_warmup_epochs", str(warmup), "--pp_ib_probe",
        "--use_got", "--got_mode", "partial_sinkhorn", "--ot_alpha", "0.05", "--entropy_eps", "0.1",
        "--ot_sinkhorn_iter", "30", "--partial_mass_init", "0.5", "--dust_cost", "1.0", "--ot_warmup_epochs", "3",
        "--use_ema", "--ema_decay", "0.999",
    ]


def load_main(argv):
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert_exp16b_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old = sys.argv[:]
    sys.argv = [str(path), *argv]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old
    return module


def parse_results(path):
    result = {}
    if not path.exists():
        return result
    for block in re.split(r"\*{5} Eval results ", path.read_text(errors="replace")):
        if not block.strip():
            continue
        head = block.splitlines()[0].lower()
        split = "test" if "test" in head else "dev" if "dev" in head else None
        if not split or "best ckpt" not in head:
            continue
        for key, val in re.findall(r"(span_precision|span_recall|span_f1)\s*=\s*([0-9.]+)", block):
            result[f"{split}_{key}"] = float(val)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("repair_a", "repair_ab"), required=True)
    cli = ap.parse_args()
    warmup = 3 if cli.variant == "repair_a" else 10
    out = EXP / "checkpoints" / cli.variant
    out.mkdir(parents=True, exist_ok=True)
    argv = main_args(out, warmup)
    cfg = {
        "variant": cli.variant, "selection": "best DEV span F1", "test_used_for_selection": False,
        "repair": {"gate_weight_std": 0.05, "gate_bias": 0.0, "polar_warmup_epochs": warmup},
        "argv": argv, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (EXP / "configs" / f"{cli.variant}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    from kais_extension.experiment16b_ppib_repair.scripts.dual_gate_probe import DualGateProbe
    import pp_ib_probe
    pp_ib_probe.PPIBProbe = DualGateProbe
    mod = load_main(argv)
    start = time.time()
    mod.main()
    result = {
        "variant": cli.variant, "training_time_sec": time.time() - start,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_gpu": 0,
        "selection": "best DEV span F1", "test_used_for_selection": False,
    }
    result.update(parse_results(out / "eval_results.txt"))
    (EXP / "results" / f"{cli.variant}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
