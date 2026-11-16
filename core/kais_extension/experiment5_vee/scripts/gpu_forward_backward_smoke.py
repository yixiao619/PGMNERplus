#!/usr/bin/env python3
"""Minimal original PGMNER GPU forward/backward smoke test.

This script does not train: it runs one small real batch, calls backward once,
and exits without optimizer.step().
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "kais_extension" / "experiment5_vee" / "results" / "pgmner_gpu_smoke.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PGMNER_ARGS = [
    "--data_dir",
    "data/twitter2017",
    "--output_dir",
    "kais_extension/experiment5_vee/cache/gpu_smoke",
    "--model_name_or_path",
    "pretrain/roberta-large",
    "--tokenizer_name",
    "pretrain/roberta-large",
    "--vit_name",
    "pretrain/clip-vit-base-patch16",
    "--gpu_id",
    "0",
    "--epoch",
    "1",
    "--batch_size",
    "1",
    "--max_seq_len",
    "128",
    "--max_span_len",
    "4",
    "--seed",
    "42",
    "--txt_global_dim",
    "1024",
    "--txt_local_dim",
    "1024",
    "--img_global_dim",
    "768",
    "--img_local_dim",
    "768",
    "--hidden_dim",
    "1024",
    "--bert_lr",
    "1e-5",
    "--lr",
    "3e-5",
    "--warmup_ratio",
    "0.1",
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
    "--do_train",
    "false",
    "--do_eval",
    "false",
    "--do_predict",
    "false",
]


def load_main_module():
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_argv = sys.argv[:]
    sys.argv = [str(path), *PGMNER_ARGS]
    spec.loader.exec_module(module)
    sys.argv = old_argv
    return module


def main() -> int:
    os.chdir(ROOT)
    main_mod = load_main_module()

    args = main_mod.args

    main_mod.set_seed(args)
    labels = main_mod.get_labels(args.labels)
    config = main_mod.AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = main_mod.AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = main_mod.BertSpanNerBN(config, args=args, num_labels=len(labels))
    model.to(args.device)
    model.train()

    examples = main_mod.load_examples(args.data_dir, mode="train", tokenizer=tokenizer)
    dataset = main_mod.SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels)
    subset = Subset(dataset, [0])
    loader = DataLoader(
        subset,
        sampler=SequentialSampler(subset),
        batch_size=1,
        collate_fn=main_mod.collate_fn,
    )
    batch = next(iter(loader))
    ori_fea_tensors = {k: v.to(args.device) for k, v in batch.items()}

    torch.cuda.reset_peak_memory_stats(args.device)
    outputs = model(ori_fea_tensors)
    loss = outputs[1]["loss"]
    finite_loss = bool(torch.isfinite(loss).item())
    loss.backward()

    grad_checks = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach()
            grad_checks[name] = {
                "finite": bool(torch.isfinite(g).all().item()),
                "abs_mean": float(g.abs().mean().item()),
                "abs_max": float(g.abs().max().item()),
            }
        if len(grad_checks) >= 8:
            break

    result = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(args.device),
        "forward_success": True,
        "backward_success": True,
        "loss": float(loss.detach().cpu().item()),
        "loss_finite": finite_loss,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(args.device) / 1024 / 1024),
        "grad_checks": grad_checks,
        "all_checked_grads_finite": all(v["finite"] for v in grad_checks.values()),
        "all_checked_grads_nonzero": all(v["abs_max"] > 0 for v in grad_checks.values()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if finite_loss and result["all_checked_grads_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
