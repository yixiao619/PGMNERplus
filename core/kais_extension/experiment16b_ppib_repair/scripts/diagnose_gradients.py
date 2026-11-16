#!/usr/bin/env python3
"""One-real-batch PP-IB gradient-path audit; never saves or updates a checkpoint."""

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
EXP = ROOT / "kais_extension" / "experiment16b_ppib_repair"
CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_ARGS = [
    "--data_dir", "data/twitter2017",
    "--output_dir", str(EXP / "cache" / "gradient_diagnostic"),
    "--model_name_or_path", "pretrain/roberta-large",
    "--tokenizer_name", "pretrain/roberta-large",
    "--vit_name", "pretrain/clip-vit-base-patch16",
    "--use_hf_clip_vision", "true", "--auto_img_dims", "true",
    "--gpu_id", "0", "--epoch", "1", "--batch_size", "2",
    "--max_seq_len", "128", "--max_span_len", "4", "--seed", "42",
    "--txt_global_dim", "1024", "--txt_local_dim", "1024",
    "--img_global_dim", "768", "--img_local_dim", "768", "--hidden_dim", "1024",
    "--bert_lr", "1e-5", "--lr", "3e-5", "--warmup_ratio", "0.1",
    "--use_pp_ib", "--beta_polar", "0.01", "--delta_minus", "0.2",
    "--delta_plus", "0.8", "--pp_ib_lambda_balance", "1.0",
    "--beta_polar_warmup_epochs", "3",
    "--use_got", "--got_mode", "partial_sinkhorn", "--ot_alpha", "0.05",
    "--entropy_eps", "0.1", "--ot_sinkhorn_iter", "30",
    "--partial_mass_init", "0.5", "--dust_cost", "1.0", "--ot_warmup_epochs", "3",
    "--do_train", "false", "--do_eval", "false", "--do_predict", "false",
]


def load_main():
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert_exp16b_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_argv = sys.argv[:]
    sys.argv = [str(path), *BASE_ARGS]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module


def gate_params(model):
    return {
        "text_weight": model.dist_t.polar_gate[0].weight,
        "text_bias": model.dist_t.polar_gate[0].bias,
        "visual_weight": model.dist_v.polar_gate[0].weight,
        "visual_bias": model.dist_v.polar_gate[0].bias,
    }


def grad_record(loss, params, retain_graph=True):
    grads = torch.autograd.grad(
        loss, list(params.values()), retain_graph=retain_graph, allow_unused=True
    )
    out = {}
    for (name, _), grad in zip(params.items(), grads):
        out[name] = {
            "exists": grad is not None,
            "finite": bool(torch.isfinite(grad).all().item()) if grad is not None else None,
            "l2": float(grad.norm().item()) if grad is not None else 0.0,
            "abs_max": float(grad.abs().max().item()) if grad is not None else 0.0,
        }
    return out


def gate_stats(model):
    out = {}
    for name, branch in (("text", model.dist_t), ("visual", model.dist_v)):
        g = branch.last_probe["gate"].detach().float().reshape(-1)
        entropy = -(g * torch.log(g.clamp_min(1e-12)) + (1-g) * torch.log((1-g).clamp_min(1e-12)))
        out[name] = {
            "mean": float(g.mean().item()),
            "std": float(g.std(unbiased=False).item()),
            "min": float(g.min().item()),
            "max": float(g.max().item()),
            "extreme_0.1": float(((g <= .1) | (g >= .9)).float().mean().item()),
            "normalized_entropy": float((entropy / math.log(2)).mean().item()),
        }
    return out


def run_case(model, batch, name, repair=False):
    if repair:
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(1602)
        torch.nn.init.normal_(model.dist_t.polar_gate[0].weight, mean=0.0, std=0.05)
        torch.nn.init.normal_(model.dist_v.polar_gate[0].weight, mean=0.0, std=0.05)
        torch.nn.init.constant_(model.dist_t.polar_gate[0].bias, 0.0)
        torch.nn.init.constant_(model.dist_v.polar_gate[0].bias, 0.0)
        torch.random.set_rng_state(generator_state)

    model.train()
    model.set_polar_train_epoch(3)
    torch.manual_seed(42)
    _, losses = model(batch)
    params = gate_params(model)
    task = losses["c"]
    nonpolar_total = losses["loss"] - losses["loss_polar"]
    polar = losses["loss_polar"]
    result = {
        "case": name,
        "losses": {k: float(v.detach().item()) for k, v in losses.items() if torch.is_tensor(v) and v.numel() == 1},
        "gate_stats_before": gate_stats(model),
        "task_classification_grad": grad_record(task, params),
        "all_nonpolar_loss_grad": grad_record(nonpolar_total, params),
        "polarization_loss_grad": grad_record(polar, params),
        "total_loss_grad": grad_record(losses["loss"], params),
    }

    before = {k: p.detach().clone() for k, p in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=3e-5, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()
    result["one_step_parameter_change"] = {
        k: {
            "l2": float((p.detach() - before[k]).norm().item()),
            "abs_max": float((p.detach() - before[k]).abs().max().item()),
        }
        for k, p in params.items()
    }
    return result


def main():
    os.chdir(ROOT)
    main_mod = load_main()
    args = main_mod.args
    main_mod.set_seed(args)
    labels = main_mod.get_labels(args.labels)
    config = main_mod.AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = main_mod.AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = main_mod.BertSpanNerBN(config, args=args, num_labels=len(labels))
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    old = "image_model._hf.encoder.vision_model."
    new = "image_model._hf.encoder."
    state = {new + k[len(old):] if k.startswith(old) else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(args.device)

    examples = main_mod.load_examples(args.data_dir, mode="train", tokenizer=tokenizer)
    dataset = main_mod.SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels)
    subset = Subset(dataset, [0, 1])
    batch = next(iter(DataLoader(subset, sampler=SequentialSampler(subset), batch_size=2, collate_fn=main_mod.collate_fn)))
    batch = {k: v.to(args.device) for k, v in batch.items()}

    torch.cuda.reset_peak_memory_stats(args.device)
    original = run_case(model, batch, "paper_full_checkpoint", repair=False)
    model.load_state_dict(state, strict=True)
    repaired = run_case(model, batch, "repair_A_reinitialized_gate", repair=True)
    output = {
        "checkpoint": str(CKPT),
        "batch_size": 2,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(args.device) / 1024**2),
        "static_path_finding": "gamma*z is not on the active classification/logit path; mi(v,t) is computed but not added unless optional ib_consistency_weight>0 (paper-full uses 0)",
        "repair_A_definition": {"gate_weight_std": 0.05, "gate_bias": 0.0, "seed": 1602},
        "cases": [original, repaired],
    }
    out = EXP / "results" / "gradient_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
