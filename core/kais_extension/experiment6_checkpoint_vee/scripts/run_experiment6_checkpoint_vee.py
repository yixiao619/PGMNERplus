#!/usr/bin/env python3
"""Experiment 6: initialize Mean-VEE from the ICDM PGMNER checkpoint.

This is KAIS extension code only. It imports the original data/model/evaluation
pipeline but does not modify original model files or historical artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, RobertaTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment6_checkpoint_vee"
EXP5 = ROOT / "kais_extension" / "experiment5_vee"
ICDM_CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
REF_F1 = 0.90647
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_ARGS = [
    "--data_dir", "data/twitter2017",
    "--model_name_or_path", "pretrain/roberta-large",
    "--tokenizer_name", "pretrain/roberta-large",
    "--vit_name", "pretrain/clip-vit-base-patch16",
    "--use_hf_clip_vision", "true",
    "--auto_img_dims", "true",
    "--gpu_id", "0",
    "--epoch", "1",
    "--batch_size", "16",
    "--max_seq_len", "128",
    "--max_span_len", "4",
    "--lr", "3e-5",
    "--bert_lr", "1e-5",
    "--warmup_ratio", "0.1",
    "--patience", "0",
    "--seed", "42",
    "--logging_steps", "100",
    "--do_train", "false",
    "--do_eval", "false",
    "--do_predict", "false",
    "--use_pp_ib",
    "--beta_polar", "0.01",
    "--delta_minus", "0.2",
    "--delta_plus", "0.8",
    "--use_got",
    "--got_mode", "partial_sinkhorn",
    "--ot_alpha", "0.05",
    "--entropy_eps", "0.1",
    "--ot_sinkhorn_iter", "30",
    "--partial_mass_init", "0.5",
    "--dust_cost", "1.0",
    "--ot_warmup_epochs", "3",
]


def load_main_module(extra_args: List[str]):
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert_exp6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv[:]
    sys.argv = [str(path), *BASE_ARGS, *extra_args]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def ensure_dirs() -> None:
    for sub in ["scripts", "configs", "logs", "results", "checkpoints", "figures"]:
        (EXP / sub).mkdir(parents=True, exist_ok=True)


def load_state(path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def build_model(mod, labels: List[str], output_dir: Path, vee: bool):
    from kais_extension.experiment5_vee.models.pgmner_vee import BertSpanNerBNVEE

    args = mod.args
    args.output_dir = str(output_dir)
    args.vee_type = "mean" if vee else "none"
    args.vee_initial_gate = 0.1
    args.vee_residual_scale = 1.0
    args.vee_attention_temperature = 1.0
    args.disable_vee_visual = False
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None,
    )
    model = BertSpanNerBNVEE(config, args=args, num_labels=len(labels))
    model.to(args.device)
    return model


def prepare_common(mod):
    labels = mod.get_labels(mod.args.labels)
    tokenizer = RobertaTokenizer.from_pretrained(
        mod.args.tokenizer_name if mod.args.tokenizer_name else mod.args.model_name_or_path
    )
    return tokenizer, labels


def reset_eval_files(out_dir: Path) -> None:
    for name in ["eval_results.txt", "per_class_history.jsonl"]:
        p = out_dir / name
        if p.exists():
            p.unlink()


def evaluate_split(mod, model, tokenizer, labels, mode: str, prefix: str, epoch: Optional[int] = None) -> Dict[str, float]:
    return mod.evaluate(
        mod.args,
        model,
        tokenizer,
        labels,
        mode=mode,
        prefix=prefix,
        only_test=False,
        epoch_num=epoch,
        split_tag=mode,
    )[0]


def current_alpha(model) -> Optional[float]:
    vee = getattr(model, "vee", None)
    if vee is None or not hasattr(vee, "alpha_logit"):
        return None
    return float(torch.sigmoid(vee.alpha_logit.detach()).cpu())


def summarize_vee_on_split(mod, model, tokenizer, labels, mode: str = "dev") -> Dict[str, float]:
    from utils.datasets import SpanNerDataset, collate_fn, load_examples

    args = mod.args
    examples = load_examples(
        args.data_dir,
        mode=mode,
        tokenizer=tokenizer,
        visual_suffix_map=args.visual_suffix_map,
        aux_knowledge_map=mod.resolve_aux_knowledge_map(args, mode),
        vllm_desc_map=mod.resolve_vllm_desc_map(args, mode),
        max_seq_len=args.max_seq_len,
        max_aux_words=int(getattr(args, "max_aux_words", 12)),
        aux_dropout=0.0,
        desc_max_words=int(getattr(args, "desc_max_words", 60)),
        vllm_desc_dropout=0.0,
        desc_sep_token=str(getattr(args, "desc_sep_token", "</s>")),
    )
    ds = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=8, collate_fn=collate_fn)
    vals: Dict[str, List[float]] = {}
    ratios: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            tensors = {k: v.to(args.device) for k, v in batch.items()}
            _ = model(tensors)
            summary = getattr(model, "last_vee_summary", {}) or {}
            for k, v in summary.items():
                vals.setdefault(k, []).append(float(v))
            # last_vee_summary contains delta norm but not hidden norm; use a conservative proxy
            if "delta_norm_mean" in summary and "evidence_norm_mean" in summary:
                denom = max(float(summary.get("evidence_norm_mean", 0.0)), 1e-12)
                ratios.append(float(summary["delta_norm_mean"]) / denom)
    out = {f"{k}_mean": sum(v) / len(v) for k, v in vals.items() if v}
    if ratios:
        out["delta_to_evidence_ratio_mean"] = sum(ratios) / len(ratios)
    a = current_alpha(model)
    if a is not None:
        out["alpha"] = a
    return out


def set_trainable(model, strategy: str, phase: int = 1) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False
    if strategy == "A" or (strategy == "B" and phase == 1):
        prefixes = ("vee.",)
    elif strategy == "B" and phase == 2:
        prefixes = ("vee.", "fusion_text.", "ln_text.", "span_classifier.")
    elif strategy == "C":
        for _, p in model.named_parameters():
            p.requires_grad = True
        return
    else:
        raise ValueError(f"unknown strategy={strategy} phase={phase}")
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = True


def lr_for_name(strategy: str, name: str) -> float:
    if name.startswith("vee."):
        return 1e-4
    if strategy == "B":
        return 1e-5
    if strategy == "C":
        if name.startswith("bert.") or name.startswith("image_model."):
            return 1e-6
        return 1e-5
    return 1e-4


def make_optimizer(args, model, strategy: str, total_steps: int):
    no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm", "ln_text")
    groups: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = lr_for_name(strategy, name)
        wd = 0.0 if any(k in name for k in no_decay_keys) else float(args.weight_decay)
        groups.setdefault((lr, wd), []).append(p)
    opt_groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1]))
    ]
    optimizer = torch.optim.AdamW(opt_groups, eps=float(args.adam_epsilon))
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler, opt_groups


def train_phase(
    mod,
    model,
    tokenizer,
    labels,
    out_dir: Path,
    strategy: str,
    phase: int,
    max_epochs: int,
    patience: int,
    start_best: float,
    start_epoch_offset: int = 0,
) -> Tuple[float, int, int, int]:
    from utils.datasets import SpanNerDataset, collate_fn, load_examples

    args = mod.args
    train_examples = load_examples(
        args.data_dir,
        mode="train",
        tokenizer=tokenizer,
        visual_suffix_map=args.visual_suffix_map,
        aux_knowledge_map=mod.resolve_aux_knowledge_map(args, "train"),
        vllm_desc_map=mod.resolve_vllm_desc_map(args, "train"),
        max_seq_len=args.max_seq_len,
        max_aux_words=int(getattr(args, "max_aux_words", 12)),
        aux_dropout=float(getattr(args, "aux_dropout", 0.0)),
        desc_max_words=int(getattr(args, "desc_max_words", 60)),
        vllm_desc_dropout=float(getattr(args, "vllm_desc_dropout", 0.0)),
        desc_sep_token=str(getattr(args, "desc_sep_token", "</s>")),
    )
    ds = SpanNerDataset(train_examples, args=args, tokenizer=tokenizer, labels=labels)
    total_steps = math.ceil(len(ds) / int(args.batch_size)) * max_epochs
    optimizer, scheduler, opt_groups = make_optimizer(args, model, strategy, total_steps)
    print(f"optimizer groups strategy={strategy} phase={phase}: " + json.dumps([
        {"lr": g["lr"], "weight_decay": g["weight_decay"], "n_params": sum(p.numel() for p in g["params"])} for g in opt_groups
    ], indent=2))
    best = start_best
    best_epoch = 0
    bad = 0
    global_step = 0
    for ep in range(1, max_epochs + 1):
        epoch_num = start_epoch_offset + ep
        if hasattr(model, "set_polar_train_epoch"):
            model.set_polar_train_epoch(epoch_num)
        if hasattr(model, "set_partial_ot_warmup"):
            # Fine-tuning starts from a trained checkpoint; do not force s=1 warmup.
            model.set_partial_ot_warmup(False)
        dl = DataLoader(ds, sampler=RandomSampler(ds), batch_size=args.batch_size, collate_fn=collate_fn)
        model.train()
        losses: List[float] = []
        for batch in tqdm(dl, desc=f"strategy {strategy} phase {phase} epoch {ep}/{max_epochs}"):
            tensors = {k: v.to(args.device) for k, v in batch.items()}
            _, loss_dict = model(tensors)
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss strategy={strategy} phase={phase} epoch={ep}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            model.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            global_step += 1
        dev = evaluate_split(mod, model, tokenizer, labels, "dev", f"strategy_{strategy}_phase_{phase}_epoch_{ep}", epoch_num)
        dev_f1 = float(dev["span_f1"])
        print(json.dumps({"strategy": strategy, "phase": phase, "epoch": ep, "global_epoch": epoch_num, "mean_loss": sum(losses)/max(len(losses),1), "dev_f1": dev_f1, "alpha": current_alpha(model)}, sort_keys=True))
        if dev_f1 >= best:
            best = dev_f1
            best_epoch = epoch_num
            bad = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"saved best strategy={strategy} epoch={best_epoch} dev_f1={best}")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop strategy={strategy} phase={phase} bad={bad} patience={patience}")
                break
    return best, best_epoch, bad, global_step


def train_strategy(mod, tokenizer, labels, strategy: str, checkpoint: Path) -> Dict[str, object]:
    out_dir = EXP / "checkpoints" / f"strategy_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_eval_files(out_dir)
    model = build_model(mod, labels, out_dir, vee=True)
    missing, unexpected = model.load_state_dict(load_state(checkpoint, mod.args.device), strict=False)
    print(f"strategy {strategy}: loaded ICDM checkpoint missing={len(missing)} unexpected={len(unexpected)}")
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t0 = time.time()
    best = -1.0
    best_epoch = 0
    if strategy == "A":
        set_trainable(model, "A", 1)
        best, best_epoch, _, _ = train_phase(mod, model, tokenizer, labels, out_dir, "A", 1, 15, 4, best)
    elif strategy == "B":
        set_trainable(model, "B", 1)
        best, best_epoch, _, _ = train_phase(mod, model, tokenizer, labels, out_dir, "B", 1, 5, 4, best)
        set_trainable(model, "B", 2)
        best, best_epoch2, _, _ = train_phase(mod, model, tokenizer, labels, out_dir, "B", 2, 10, 4, best, 5)
        best_epoch = best_epoch2 or best_epoch
    elif strategy == "C":
        set_trainable(model, "C", 1)
        best, best_epoch, _, _ = train_phase(mod, model, tokenizer, labels, out_dir, "C", 1, 10, 3, best)
    else:
        raise ValueError(strategy)
    elapsed = time.time() - t0
    # Load best dev checkpoint for final test.
    model.load_state_dict(load_state(out_dir / "best_model.pth", mod.args.device), strict=False)
    dev = evaluate_split(mod, model, tokenizer, labels, "dev", f"strategy_{strategy}_best", best_epoch)
    test = evaluate_split(mod, model, tokenizer, labels, "test", f"strategy_{strategy}_best", best_epoch)
    vee_summary = summarize_vee_on_split(mod, model, tokenizer, labels, "test")
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "strategy": f"Strategy {strategy}",
        "initial_checkpoint": str(checkpoint),
        "best_dev_f1": best,
        "best_epoch": best_epoch,
        "test_precision": test["span_precision"],
        "test_recall": test["span_recall"],
        "test_f1": test["span_f1"],
        "delta_vs_0.90647": round(float(test["span_f1"]) - REF_F1, 5),
        "learned_alpha": vee_summary.get("alpha", current_alpha(model)),
        "visual_residual_ratio": vee_summary.get("delta_to_evidence_ratio_mean", ""),
        "trainable_parameters": trainable_after,
        "trainable_parameters_before_freeze": trainable_before,
        "training_time": elapsed,
        "vee_summary": vee_summary,
    }
    (EXP / "results" / f"strategy_{strategy}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_main_csv(rows: List[Dict[str, object]]) -> None:
    out = EXP / "results" / "checkpoint_vee_main.csv"
    fields = [
        "strategy", "initial_checkpoint", "best_dev_f1", "best_epoch", "test_precision", "test_recall", "test_f1",
        "delta_vs_0.90647", "learned_alpha", "visual_residual_ratio", "trainable_parameters", "training_time",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print(out)


def write_report(rows: List[Dict[str, object]]) -> None:
    best_dev = max([r for r in rows if r["strategy"] != "Original ICDM checkpoint"], key=lambda r: float(r.get("best_dev_f1", -1)), default=None)
    best_test = max([r for r in rows if r["strategy"] != "Original ICDM checkpoint"], key=lambda r: float(r.get("test_f1", -1)), default=None)
    lines = [
        "# Experiment 6: Conference-Checkpoint VEE Enhancement",
        "",
        "## 1. Motivation",
        "Test whether lightweight Mean-VEE improves the recovered ICDM PGMNER checkpoint.",
        "",
        "## 2. Relation to Experiment 5",
        "Experiment 5 selected Mean-VEE under a controlled rerun baseline; Experiment 6 initializes from the historical ICDM checkpoint.",
        "",
        "## 3. Original ICDM Checkpoint Reproduction",
    ]
    orig = rows[0]
    lines += [f"Original test P/R/F1: {orig.get('test_precision')} / {orig.get('test_recall')} / {orig.get('test_f1')}.", ""]
    lines += ["## 4. Mean-VEE Architecture", "Mean pooled post-GOT visual representation projected and added residually before the span classifier with a learned scalar alpha initialized near 0.1.", ""]
    for s in ["A", "B", "C"]:
        rr = next((r for r in rows if r["strategy"] == f"Strategy {s}"), None)
        lines += [f"## {4 + ord(s)-64}. Strategy {s}"]
        if rr:
            lines += [f"Best dev F1={rr.get('best_dev_f1')}; test F1={rr.get('test_f1')}; delta={rr.get('delta_vs_0.90647')}; alpha={rr.get('learned_alpha')}.", ""]
        else:
            lines += ["Not completed.", ""]
    lines += ["## 9. Main Results", "See `results/checkpoint_vee_main.csv`.", ""]
    lines += ["## 10. Visual Dependency", "Pending detailed intervention analysis after strategy training completes.", ""]
    lines += ["## 11. Entity-Type Analysis", "Pending detailed entity-type comparison for BEST_DEV_STRATEGY.", ""]
    lines += ["## 12. Implications for KAIS", "To be interpreted using DEV-first strategy selection and final test evaluation only.", ""]
    lines += ["## 13. Limitations", "Single seed and Twitter-2017 only; no test-based hyperparameter search.", ""]
    if best_dev:
        lines += ["## 14. Recommended Next Experiment", f"BEST_DEV_STRATEGY={best_dev['strategy']}; BEST_TEST_STRATEGY={best_test['strategy'] if best_test else 'N/A'}. Do not start Experiment 7 automatically.", ""]
    (EXP / "EXPERIMENT_6_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ICDM_CKPT))
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--strategies", default="A,B,C")
    args_cli = parser.parse_args()
    ensure_dirs()
    mod = load_main_module(["--output_dir", str(EXP / "checkpoints" / "original_icdm")])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this execution path")
    tokenizer, labels = prepare_common(mod)
    ckpt = Path(args_cli.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    orig_dir = EXP / "checkpoints" / "original_icdm"
    orig_dir.mkdir(parents=True, exist_ok=True)
    reset_eval_files(orig_dir)
    orig = build_model(mod, labels, orig_dir, vee=False)
    missing, unexpected = orig.load_state_dict(load_state(ckpt, mod.args.device), strict=False)
    print(f"original checkpoint load missing={len(missing)} unexpected={len(unexpected)}")
    test = evaluate_split(mod, orig, tokenizer, labels, "test", "original_icdm", None)
    original_row = {
        "strategy": "Original ICDM checkpoint",
        "initial_checkpoint": str(ckpt),
        "best_dev_f1": "",
        "best_epoch": "",
        "test_precision": test["span_precision"],
        "test_recall": test["span_recall"],
        "test_f1": test["span_f1"],
        "delta_vs_0.90647": round(float(test["span_f1"]) - REF_F1, 5),
        "learned_alpha": "",
        "visual_residual_ratio": "",
        "trainable_parameters": 0,
        "training_time": 0.0,
    }
    (EXP / "results" / "original_icdm_reproduction.json").write_text(json.dumps(original_row, indent=2, sort_keys=True) + "\n")
    if abs(float(test["span_f1"]) - REF_F1) > 0.001:
        write_main_csv([original_row])
        write_report([original_row])
        raise SystemExit(f"STOP_AND_INVESTIGATE: original F1={test['span_f1']} differs from {REF_F1}")
    if args_cli.verify_only:
        write_main_csv([original_row])
        write_report([original_row])
        return 0

    rows: List[Dict[str, object]] = [original_row]
    for strat in [s.strip().upper() for s in args_cli.strategies.split(",") if s.strip()]:
        rows.append(train_strategy(mod, tokenizer, labels, strat, ckpt))
        write_main_csv(rows)
        write_report(rows)
    write_main_csv(rows)
    write_report(rows)
    print("FINAL_ROWS")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
