#!/usr/bin/env python3
"""Experiment 7B: uncertainty-gated entity-aware VEE screening."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, RobertaTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment7_lightweight"
CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
REF_F1 = 0.90647
LABELS = ["O", "LOC", "MISC", "ORG", "PER"]
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


def load_main(extra: List[str]):
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert_exp7b", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv[:]
    sys.argv = [str(path), *BASE_ARGS, *extra]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def load_state(path: Path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def margin_thresholds_from_dev() -> Dict[str, float]:
    p = EXP / "results" / "cached_dev_predictions.pt"
    if not p.exists():
        raise FileNotFoundError("Run 7A first; cached_dev_predictions.pt missing")
    data = torch.load(p, map_location="cpu", weights_only=False)
    probs = data["probs"]
    masks = data["masks"].bool()
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1])[masks]
    # Active lowest-confidence fraction = lowest margins.
    return {
        "U50": float(torch.quantile(margin, 0.50)),
        "U30": float(torch.quantile(margin, 0.30)),
        "U15": float(torch.quantile(margin, 0.15)),
    }


def build_model(mod, labels: List[str], out_dir: Path, margin_threshold: float):
    from kais_extension.experiment7_lightweight.models.ug_vee import BertSpanNerBNUgVEE
    args = mod.args
    args.output_dir = str(out_dir)
    args.ug_margin_threshold = float(margin_threshold)
    args.ug_initial_alpha = 0.1
    args.ug_temperature = 1.0
    args.disable_new_visual_path = False
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None,
    )
    model = BertSpanNerBNUgVEE(config, args=args, num_labels=len(labels))
    model.to(args.device)
    missing, unexpected = model.load_state_dict(load_state(CKPT, args.device), strict=False)
    print(f"loaded ICDM into UG-VEE missing={len(missing)} unexpected={len(unexpected)}")
    return model


def set_trainable(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("ug_vee."):
            p.requires_grad = True


def make_optimizer(model, total_steps: int, warmup_ratio: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": 1e-4, "weight_decay": 0.01})
    if no_decay:
        groups.append({"params": no_decay, "lr": 1e-4, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, eps=1e-8)
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * warmup_ratio), total_steps)
    return opt, sched


def prepare_data(mod, tokenizer, labels, split: str, train: bool):
    from utils.datasets import SpanNerDataset, collate_fn, load_examples
    args = mod.args
    examples = load_examples(
        args.data_dir,
        mode=split,
        tokenizer=tokenizer,
        visual_suffix_map=args.visual_suffix_map,
        aux_knowledge_map=mod.resolve_aux_knowledge_map(args, split),
        vllm_desc_map=mod.resolve_vllm_desc_map(args, split),
        max_seq_len=args.max_seq_len,
        max_aux_words=int(getattr(args, "max_aux_words", 12)),
        aux_dropout=0.0,
        desc_max_words=int(getattr(args, "desc_max_words", 60)),
        vllm_desc_dropout=0.0,
        desc_sep_token=str(getattr(args, "desc_sep_token", "</s>")),
    )
    ds = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=not train)
    return ds, collate_fn


def evaluate(mod, model, tokenizer, labels, mode: str, prefix: str, epoch: Optional[int] = None):
    return mod.evaluate(mod.args, model, tokenizer, labels, mode=mode, prefix=prefix, only_test=False, epoch_num=epoch, split_tag=mode)[0]


def mean_summary(model) -> Dict[str, float]:
    return dict(getattr(model, "last_ug_summary", {}) or {})


def train_variant(mod, tokenizer, labels, name: str, margin_threshold: float) -> Dict[str, object]:
    out_dir = EXP / "checkpoints" / f"7b_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in ["eval_results.txt", "per_class_history.jsonl"]:
        p = out_dir / old
        if p.exists():
            p.unlink()
    model = build_model(mod, labels, out_dir, margin_threshold)
    set_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ds, collate_fn = prepare_data(mod, tokenizer, labels, "train", True)
    total_steps = math.ceil(len(train_ds) / int(mod.args.batch_size)) * 10
    opt, sched = make_optimizer(model, total_steps, float(mod.args.warmup_ratio))
    best = -1.0
    best_epoch = 0
    bad = 0
    t0 = time.time()
    for epoch in range(1, 11):
        if hasattr(model, "set_partial_ot_warmup"):
            model.set_partial_ot_warmup(False)
        dl = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=mod.args.batch_size, collate_fn=collate_fn)
        model.train()
        losses = []
        summaries = []
        for batch in tqdm(dl, desc=f"7B {name} epoch {epoch}/10"):
            tensors = {k: v.to(mod.args.device) for k, v in batch.items()}
            _, loss_dict = model(tensors)
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {name} epoch={epoch}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(mod.args.max_grad_norm))
            opt.step(); sched.step(); model.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            summaries.append(mean_summary(model))
        dev = evaluate(mod, model, tokenizer, labels, "dev", f"7b_{name}_epoch_{epoch}", epoch)
        dev_f1 = float(dev["span_f1"])
        avg_summary = {}
        for k in ["alpha", "active_fraction", "gate_mean", "residual_ratio", "attention_entropy", "attention_max"]:
            vals = [s.get(k) for s in summaries if k in s]
            if vals:
                avg_summary[k] = sum(vals) / len(vals)
        print(json.dumps({"variant": name, "epoch": epoch, "dev_f1": dev_f1, "loss": sum(losses)/max(len(losses),1), **avg_summary}, sort_keys=True))
        if dev_f1 >= best:
            best = dev_f1
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
        else:
            bad += 1
            if bad >= 3:
                print(f"early stop {name} at epoch={epoch}")
                break
    elapsed = time.time() - t0
    model.load_state_dict(load_state(out_dir / "best_model.pth", mod.args.device), strict=False)
    dev = evaluate(mod, model, tokenizer, labels, "dev", f"7b_{name}_best", best_epoch)
    test = evaluate(mod, model, tokenizer, labels, "test", f"7b_{name}_best", best_epoch)
    # One test pass for final diagnostic summary.
    _ = summarize_on_split(mod, model, tokenizer, labels, "test")
    diag = mean_summary(model)
    result = {
        "variant": name,
        "margin_threshold": margin_threshold,
        "best_dev_f1": best,
        "best_epoch": best_epoch,
        "test_precision": test["span_precision"],
        "test_recall": test["span_recall"],
        "test_f1": test["span_f1"],
        "delta_vs_0.90647": round(float(test["span_f1"]) - REF_F1, 5),
        "trainable_parameters": trainable,
        "training_time": elapsed,
        **diag,
    }
    (EXP / "results" / f"7b_{name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarize_on_split(mod, model, tokenizer, labels, split: str):
    ds, collate_fn = prepare_data(mod, tokenizer, labels, split, False)
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=8, collate_fn=collate_fn)
    model.eval()
    vals: Dict[str, List[float]] = {}
    with torch.no_grad():
        for batch in dl:
            tensors = {k: v.to(mod.args.device) for k, v in batch.items()}
            model(tensors)
            for k, v in mean_summary(model).items():
                vals.setdefault(k, []).append(float(v))
    out = {k: sum(v) / len(v) for k, v in vals.items() if v}
    model.last_ug_summary = out
    return out


def write_csv(rows: List[Dict[str, object]]):
    out = EXP / "results" / "7b_uncertainty_vee.csv"
    fields = ["variant", "margin_threshold", "best_dev_f1", "best_epoch", "test_precision", "test_recall", "test_f1", "delta_vs_0.90647", "alpha", "active_fraction", "gate_mean", "residual_ratio", "attention_entropy", "attention_max", "trainable_parameters", "training_time"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(out)


def main() -> int:
    for sub in ["scripts", "configs", "logs", "results", "checkpoints", "figures"]:
        (EXP / sub).mkdir(parents=True, exist_ok=True)
    thresholds = margin_thresholds_from_dev()
    (EXP / "results" / "7b_margin_thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    mod = load_main(["--output_dir", str(EXP / "checkpoints" / "7b_tmp")])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    labels = mod.get_labels(mod.args.labels)
    tokenizer = RobertaTokenizer.from_pretrained(mod.args.tokenizer_name if mod.args.tokenizer_name else mod.args.model_name_or_path)
    rows = []
    for name in ["U50", "U30", "U15"]:
        rows.append(train_variant(mod, tokenizer, labels, name, thresholds[name]))
        write_csv(rows)
    best_dev = max(rows, key=lambda r: float(r["best_dev_f1"]))
    best_test = max(rows, key=lambda r: float(r["test_f1"]))
    (EXP / "results" / "best_7b.json").write_text(json.dumps({"BEST_DEV_7B": best_dev, "BEST_TEST_7B": best_test}, indent=2, sort_keys=True) + "\n")
    print("BEST_7B")
    print(json.dumps(best_dev, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
