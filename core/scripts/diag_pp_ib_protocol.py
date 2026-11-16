#!/usr/bin/env python3
"""30-min PP-IB beta_polar diagnostic protocol (Steps 0-3)."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pp_ib_diag_utils import (  # noqa: E402
    _verdict,
    collect_pp_ib_stats,
    make_dev_loader,
    plot_diag,
    save_history_json,
)


def log_progress(out_dir: Path, msg: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with (out_dir / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_argv(
    gpu_id: int,
    out_dir: Path,
    epochs: int,
    seed: int = 42,
    beta_polar: float = 0.01,
    lambda_balance: float = 1.0,
    delta_minus: float = 0.2,
    gate_init_std: float = 0.01,
    gate_on_hidden: bool = False,
) -> list:
    argv = [
        "main-mner-robert.py",
        "--data_dir",
        "data/twitter2017",
        "--output_dir",
        str(out_dir),
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
        "--logging_steps",
        "999999",
        "--do_train",
        "true",
        "--do_eval",
        "false",
        "--do_predict",
        "false",
        "--patience",
        "0",
        "--seed",
        str(seed),
        "--epoch",
        str(epochs),
        "--use_pp_ib",
        "--beta_polar",
        str(beta_polar),
        "--delta_minus",
        str(delta_minus),
        "--delta_plus",
        "0.8",
        "--pp_ib_lambda_balance",
        str(lambda_balance),
        "--beta_polar_warmup_epochs",
        "0",
        "--pp_ib_gate_init_std",
        str(gate_init_std),
        "--use_got",
        "--got_mode",
        "partial_sinkhorn",
        "--ot_alpha",
        "0.2",
        "--entropy_eps",
        "0.1",
        "--partial_mass_init",
        "0.6",
        "--dust_cost",
        "0.5",
        "--ot_sinkhorn_iter",
        "30",
        "--ot_warmup_epochs",
        "3",
    ]
    if gate_on_hidden:
        argv.append("--pp_ib_gate_on_hidden")
    return argv


def load_main(gpu_id: int, out_dir: Path, **kw):
    import importlib.util

    old = sys.argv
    sys.argv = build_argv(gpu_id, out_dir, **kw)
    spec = importlib.util.spec_from_file_location(
        "main_mner_robert_diag", ROOT / "main-mner-robert.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = old
    return mod, mod.args


def train_with_diag(
    main_mod,
    args,
    model,
    tokenizer,
    labels,
    n_epochs: int,
    dev_loader,
    out_dir: Path,
    step_tag: str,
) -> list:
    from engine_utils import prepare_optimizer_scheduler
    from utils.datasets import SpanNerDataset, collate_fn, load_examples

    train_examples = load_examples(
        args.data_dir, mode="train", tokenizer=tokenizer, visual_suffix_map=args.visual_suffix_map
    )
    train_dataset = SpanNerDataset(
        train_examples, args=args, tokenizer=tokenizer, labels=labels
    )
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = steps_per_epoch * n_epochs
    optimizer, scheduler = prepare_optimizer_scheduler(args, model, total_steps)

    history = []
    model.train()
    global_step = 0

    for epoch in trange(1, n_epochs + 1, desc=f"{step_tag} epochs"):
        if hasattr(model, "set_polar_train_epoch"):
            model.set_polar_train_epoch(epoch)
        ot_we = int(getattr(args, "ot_warmup_epochs", 0) or 0)
        rel_ep = epoch - 1
        in_ot_wu = ot_we > 0 and rel_ep >= 0 and rel_ep < ot_we
        if hasattr(model, "set_partial_ot_warmup"):
            model.set_partial_ot_warmup(in_ot_wu)

        sampler = RandomSampler(train_dataset)
        loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )
        pbar = tqdm(loader, desc=f"{step_tag} ep{epoch}", leave=False)
        for batch in pbar:
            model.train()
            batch_d = {k: v.to(args.device) for k, v in batch.items()}
            _, loss_dic = model(batch_d)
            loss = loss_dic["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

        stats = collect_pp_ib_stats(model, dev_loader, args.device)
        verdict = _verdict(stats)
        row = {
            "epoch": epoch,
            "step_tag": step_tag,
            "verdict": verdict,
            **{k: v for k, v in stats.items() if k != "gates"},
            "gates": stats["gates"],
        }
        history.append(row)
        log_progress(
            out_dir,
            f"{step_tag} epoch {epoch}/{n_epochs}: "
            f"gate_mean={row['gate_mean']:.4f} gate_std={row['gate_std']:.4f} "
            f"bimod={row['bimodality']:.3f} frac_mid={row['frac_mid']:.3f} "
            f"mu_norm={row['mu_norm_mean']:.3f} verdict={verdict}",
        )

    return history


def run_phase(
    gpu_id: int,
    root_out: Path,
    phase: str,
    n_epochs: int,
    beta_polar: float = 0.01,
    gate_init_std: float = 0.01,
    gate_on_hidden: bool = False,
    lambda_balance: float = 1.0,
) -> tuple:
    out_dir = root_out / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    log_progress(root_out, f"=== {phase} START ({n_epochs} epochs, beta={beta_polar}, gate_std={gate_init_std}, gate_on_h={gate_on_hidden}) ===")

    main_mod, args = load_main(
        gpu_id,
        out_dir,
        epochs=n_epochs,
        beta_polar=beta_polar,
        gate_init_std=gate_init_std,
        gate_on_hidden=gate_on_hidden,
        lambda_balance=lambda_balance,
    )
    from utils.datasets import get_labels, load_examples

    labels = get_labels(args.labels)
    tokenizer = main_mod.AutoTokenizer.from_pretrained(args.tokenizer_name)
    dev_examples = load_examples(args.data_dir, mode="dev", tokenizer=tokenizer)
    from utils.datasets import collate_fn

    dev_loader = make_dev_loader(dev_examples, args, tokenizer, labels, collate_fn)

    config = main_mod.AutoConfig.from_pretrained(args.model_name_or_path)
    model = main_mod.BertSpanNerBN(config, args=args, num_labels=len(labels))
    model.to(args.device)

    history = train_with_diag(
        main_mod, args, model, tokenizer, labels, n_epochs, dev_loader, root_out, phase
    )
    verdict = _verdict(history[-1])
    png = root_out / f"{phase}.png"
    if phase == "step0_baseline":
        png = root_out / "diag_baseline.png"
    elif phase == "step3_validate":
        png = root_out / "diag.png"
    plot_diag(history, png.with_suffix(".pdf"), title=phase, verdict=verdict)
    save_history_json(history, out_dir / "history.json")
    summary = {
        "phase": phase,
        "verdict": verdict,
        "last": {k: v for k, v in history[-1].items() if k != "gates"},
        "plot": str(png),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress(root_out, f"=== {phase} DONE verdict={verdict} plot={png} ===")
    return verdict, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="results/diag_pp_ib")
    parser.add_argument("--step", type=str, default="all", choices=["all", "0", "1", "2", "3"])
    args_cli = parser.parse_args()

    root_out = ROOT / args_cli.out_dir
    gpu = int(args_cli.gpu)
    log_progress(root_out, f"Diagnostic protocol on GPU {gpu}")

    if args_cli.step in ("all", "0"):
        verdict0, _ = run_phase(
            gpu, root_out, "step0_baseline", n_epochs=2,
            beta_polar=0.01, gate_init_std=0.01, gate_on_hidden=False,
            lambda_balance=1.0,
        )
    else:
        p = root_out / "step0_baseline" / "summary.json"
        verdict0 = json.loads(p.read_text())["verdict"] if p.exists() else "OK"

    if args_cli.step in ("all", "1"):
        if verdict0 in ("VIB-collapse", "Logit-stuck"):
            run_phase(
                gpu, root_out, "step1_gate_fix", n_epochs=2,
                beta_polar=0.01, gate_init_std=0.5, gate_on_hidden=True,
                lambda_balance=1.0,
            )
            fix_verdict_path = root_out / "step1_gate_fix" / "summary.json"
            verdict_fix = json.loads(fix_verdict_path.read_text())["verdict"]
        else:
            log_progress(root_out, f"Step1 skipped (baseline verdict={verdict0})")
            verdict_fix = verdict0
    else:
        verdict_fix = verdict0

    if args_cli.step in ("all", "2"):
        if verdict_fix in ("beta too weak", "balance dominates") or verdict0 == "beta too weak":
            run_phase(
                gpu, root_out, "step2_beta_up", n_epochs=2,
                beta_polar=0.1, gate_init_std=0.5, gate_on_hidden=True,
                lambda_balance=0.0,
            )
            validate_cfg = dict(beta_polar=0.1, gate_init_std=0.5, gate_on_hidden=True, lambda_balance=0.0)
        elif verdict0 in ("VIB-collapse", "Logit-stuck"):
            validate_cfg = dict(beta_polar=0.01, gate_init_std=0.5, gate_on_hidden=True, lambda_balance=1.0)
        else:
            log_progress(root_out, f"Step2 skipped (verdict={verdict_fix})")
            validate_cfg = dict(beta_polar=0.01, gate_init_std=0.01, gate_on_hidden=False, lambda_balance=1.0)
    else:
        validate_cfg = dict(beta_polar=0.01, gate_init_std=0.5, gate_on_hidden=True, lambda_balance=1.0)

    if args_cli.step in ("all", "3"):
        run_phase(
            gpu, root_out, "step3_validate", n_epochs=5, **validate_cfg
        )

    log_progress(root_out, "=== PROTOCOL COMPLETE ===")


if __name__ == "__main__":
    main()
