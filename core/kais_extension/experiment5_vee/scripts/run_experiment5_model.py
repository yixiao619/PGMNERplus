#!/usr/bin/env python3
"""Run Experiment 5 baseline or one VEE variant with the original pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment5_vee"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_ARGS = [
    "--data_dir",
    "data/twitter2017",
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
    "0",
    "--epoch",
    "40",
    "--batch_size",
    "16",
    "--max_seq_len",
    "128",
    "--max_span_len",
    "4",
    "--lr",
    "3e-5",
    "--bert_lr",
    "1e-5",
    "--warmup_ratio",
    "0.1",
    "--patience",
    "0",
    "--seed",
    "42",
    "--logging_steps",
    "100",
    "--do_train",
    "true",
    "--do_eval",
    "true",
    "--do_predict",
    "true",
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
    "--ot_sinkhorn_iter",
    "30",
    "--partial_mass_init",
    "0.5",
    "--dust_cost",
    "1.0",
    "--ot_warmup_epochs",
    "3",
    "--use_ema",
    "--ema_decay",
    "0.999",
]


def parse_eval_results(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    blocks = re.split(r"\*{5} Eval results ", text)
    for block in blocks:
        if not block.strip():
            continue
        head = block.splitlines()[0].lower()
        target = "test" if "test" in head else "dev" if "dev" in head else "eval"
        for key, val in re.findall(r"(span_precision|span_recall|span_f1)\s*=\s*([0-9.]+)", block):
            out[f"{target}_{key}"] = float(val)
    return out


def best_epoch_from_log(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    matches = re.findall(r"epoch_num:\s*([0-9]+)", text)
    if matches:
        return int(matches[-1])
    return None


def load_main(main_args: List[str]):
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert_exp5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_argv = sys.argv[:]
    sys.argv = [str(path), *main_args]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--vee_type", default="none", choices=("none", "mean", "attention", "gated_attention"))
    ap.add_argument("--vee_initial_gate", type=float, default=0.1)
    ap.add_argument("--vee_residual_scale", type=float, default=1.0)
    ap.add_argument("--vee_attention_temperature", type=float, default=1.0)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--no_train", action="store_true")
    args_cli = ap.parse_args()

    out_dir = Path(args_cli.output_dir) if args_cli.output_dir else EXP / "checkpoints" / args_cli.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    main_args = [*BASE_ARGS, "--output_dir", str(out_dir)]
    if args_cli.no_train:
        main_args = [
            ("false" if prev == "--do_train" else item)
            for prev, item in zip([""] + main_args[:-1], main_args)
        ]

    config = {
        "run_name": args_cli.run_name,
        "vee_type": args_cli.vee_type,
        "vee_initial_gate": args_cli.vee_initial_gate,
        "vee_residual_scale": args_cli.vee_residual_scale,
        "vee_attention_temperature": args_cli.vee_attention_temperature,
        "main_args": main_args,
        "output_dir": str(out_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (EXP / "configs" / f"{args_cli.run_name}.json").write_text(json.dumps(config, indent=2) + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this execution path")

    main_mod = load_main(main_args)
    main_mod.args.vee_type = args_cli.vee_type
    main_mod.args.vee_initial_gate = args_cli.vee_initial_gate
    main_mod.args.vee_residual_scale = args_cli.vee_residual_scale
    main_mod.args.vee_attention_temperature = args_cli.vee_attention_temperature
    main_mod.args.disable_vee_visual = False

    if args_cli.vee_type != "none":
        from kais_extension.experiment5_vee.models.pgmner_vee import BertSpanNerBNVEE

        main_mod.BertSpanNerBN = BertSpanNerBNVEE

    t0 = time.time()
    main_mod.main()
    elapsed = time.time() - t0

    result = {
        "run_name": args_cli.run_name,
        "model": "PGMNER" if args_cli.vee_type == "none" else f"PGMNER+{args_cli.vee_type}",
        "vee_type": args_cli.vee_type,
        "training_time_sec": elapsed,
        "best_epoch": best_epoch_from_log(out_dir / "train.log"),
    }
    result.update(parse_eval_results(out_dir / "eval_results.txt"))
    (EXP / "results" / f"{args_cli.run_name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
