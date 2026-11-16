#!/usr/bin/env python3
"""Collect case-study data for Fig.5."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fig_common import VIZ_DIR, ensure_dirs, load_config, load_main_module, to_infer_batch
from collect_fig4_viz import (
    forward_ot_batch,
    load_model,
    partial_plan_aug,
    test_loader,
)
from utils.datasets import collate_fn, get_labels, load_examples


def parse_predictions(path: Path) -> List[List[str]]:
    """Parse test_predictions.txt into list of samples (token pred per line)."""
    samples: List[List[str]] = []
    cur: List[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            if cur:
                samples.append(cur)
                cur = []
            continue
        parts = line.split()
        if len(parts) >= 3:
            cur.append(parts[2])
        elif len(parts) == 2:
            cur.append(parts[1])
    if cur:
        samples.append(cur)
    return samples


def tokens_gold_from_example(ex) -> tuple:
    words = list(getattr(ex, "words", []) or [])
    labels = list(getattr(ex, "label", []) or [])
    gold = []
    for lab in labels:
        if lab == "O":
            gold.append("O")
        elif "-" in lab:
            gold.append(lab.split("-", 1)[1])
        else:
            gold.append(lab)
    return words, gold


def dustbin_mass_per_token(plan: np.ndarray) -> List[float]:
    """Sum transport into/out of dust-bin row/col per text token."""
    n, m = plan.shape[0] - 1, plan.shape[1] - 1
    row_dust = plan[:n, m]
    col_dust = plan[n, :m]
    return (row_dust + col_dust).tolist()


def find_case_indices(examples) -> tuple:
    case_a, case_b = None, None
    for i, ex in enumerate(examples):
        words, gold = tokens_gold_from_example(ex)
        text = " ".join(words).lower()
        has_misc = "MISC" in gold
        has_per = "PER" in gold
        if case_a is None and has_misc and ("harry" in text or "potter" in text or "tesla" in text):
            case_a = i
        if case_b is None and has_per and has_misc:
            case_b = i
        if case_a is None and has_misc:
            case_a = i
        if case_b is None and has_per:
            case_b = i
    return case_a or 0, case_b or min(1, len(examples) - 1)


def pgmner_preds_for_sample(model, main_mod, args, test_ds, idx, device, labels):
    batch = collate_fn([test_ds[idx]])
    with torch.no_grad():
        logits, _ = model(to_infer_batch(batch, device), only_test=True)
    pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
    id2label = {i: l for i, l in enumerate(labels)}
    return [id2label.get(int(i), "O") for i in pred_ids]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--gpu", type=int, dest="gpu_id")
    args_cli = parser.parse_args()
    ensure_dirs()
    cfg = load_config()
    run_dir = ROOT / cfg["pgmner_run_dir"]
    ckpt = ROOT / cfg["pgmner_ckpt"]
    main_mod, args = load_main_module(int(args_cli.gpu_id), run_dir)
    device = args.device
    labels = get_labels(args.labels)
    _, _, test_examples, test_ds = test_loader(args, labels, main_mod)

    idx_a = int(cfg.get("case_a_idx", -1))
    idx_b = int(cfg.get("case_b_idx", -1))
    if idx_a < 0 or idx_b < 0:
        idx_a, idx_b = find_case_indices(test_examples)

    hvp_path = ROOT / cfg.get("hvpnet_preds", "")
    smn_path = ROOT / cfg.get("smner_preds", "")
    hvp_preds = parse_predictions(hvp_path) if hvp_path.exists() else []
    smn_preds = parse_predictions(smn_path) if smn_path.exists() else []

    model, _ = load_model(main_mod, args, ckpt=ckpt)

    cases = []
    for tag, idx in [("a", idx_a), ("b", idx_b)]:
        ex = test_examples[idx]
        tokens, gold = tokens_gold_from_example(ex)
        pgm = pgmner_preds_for_sample(model, main_mod, args, test_ds, idx, device, labels)
        # align lengths
        n = min(len(tokens), len(gold), len(pgm))
        tokens, gold, pgm = tokens[:n], gold[:n], pgm[:n]
        hvp = (hvp_preds[idx] if idx < len(hvp_preds) else ["O"] * n)[:n]
        smn = (smn_preds[idx] if idx < len(smn_preds) else ["O"] * n)[:n]

        batch = collate_fn([test_ds[idx]])
        ot_batch = forward_ot_batch(model, batch, device)
        plan = partial_plan_aug(model, ot_batch)
        ot_npy = VIZ_DIR / f"case_{tag}_ot.npy"
        np.save(ot_npy, plan)

        img_src = Path(args.data_dir + "_images") / getattr(ex, "img_id", "")
        img_dst = VIZ_DIR / f"case_{tag}.jpg"
        if img_src.exists():
            shutil.copy(img_src, img_dst)
        elif hasattr(ex, "img_path") and Path(ex.img_path).exists():
            shutil.copy(ex.img_path, img_dst)

        cases.append(
            {
                "name": f"Case {tag.upper()}",
                "idx": idx,
                "tokens": tokens,
                "gold": gold,
                "preds": {"HVPNet": hvp, "SMNER": smn, "PGMNER": pgm},
                "image_path": str(img_dst.relative_to(ROOT)),
                "ot_plan": str(ot_npy.relative_to(ROOT)),
                "dustbin_mass_per_token": dustbin_mass_per_token(plan),
            }
        )

        print(f"\n=== Case {tag.upper()} (idx={idx}) ===")
        for model_name in ["HVPNet", "SMNER", "PGMNER"]:
            ok = sum(1 for g, p in zip(gold, cases[-1]["preds"][model_name]) if g == p)
            print(f"  {model_name}: token-acc={ok}/{n}")

    out = VIZ_DIR / "case_study.json"
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
