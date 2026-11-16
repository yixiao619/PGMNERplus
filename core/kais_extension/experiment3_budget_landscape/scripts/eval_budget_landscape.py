#!/usr/bin/env python3
"""Experiment 3: post-hoc transport-budget landscape audit.

This script performs inference-only diagnostics. It does not train and never
calls optimizer.step(). It caches encoder outputs per batch and sweeps only the
inference-time actual transported mass used by PartialSinkhorn.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
import types
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, SequentialSampler


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "kais_extension" / "experiment3_budget_landscape"
RESULTS = EXP_DIR / "results"
FIGURES = EXP_DIR / "figures"
LOGS = EXP_DIR / "logs"

S_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
S_CKPT = 0.594940338282
LOSS_FLAT_RTOL = 1e-4
LOSS_FLAT_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(ROOT))
    p.add_argument("--data-dir", default="data/twitter2017")
    p.add_argument("--checkpoint", default="/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
    p.add_argument("--labels-file", default="kais_extension/configs/labels_twitter2017.txt")
    p.add_argument("--model-name-or-path", default="pretrain/roberta-large")
    p.add_argument("--tokenizer-name", default="pretrain/roberta-large")
    p.add_argument("--vit-name", default="pretrain/clip-vit-base-patch16")
    p.add_argument("--clip-model", default="pretrain/clip-vit-base-patch16")
    p.add_argument("--gpu-id", default="-1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--skip-relevance", action="store_true")
    return p.parse_args()


def ensure_dirs() -> None:
    for d in [EXP_DIR, RESULTS, FIGURES, LOGS, EXP_DIR / "scripts", EXP_DIR / "cache"]:
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pgmner(repo: Path, cli_args: argparse.Namespace):
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.argv = [
        "main-mner-robert.py",
        "--data_dir", cli_args.data_dir,
        "--output_dir", str(RESULTS / "pgmner_eval_side_outputs"),
        "--model_name_or_path", cli_args.model_name_or_path,
        "--tokenizer_name", cli_args.tokenizer_name,
        "--vit_name", cli_args.vit_name,
        "--gpu_id", cli_args.gpu_id,
        "--do_train", "false",
        "--do_eval", "false",
        "--do_predict", "false",
        "--batch_size", str(cli_args.batch_size),
        "--max_span_len", "4",
        "--max_seq_len", "128",
        "--seed", "42",
        "--use_pp_ib",
        "--use_got",
        "--got_mode", "partial_sinkhorn",
        "--ot_sinkhorn_iter", "30",
        "--partial_mass_init", "0.5",
        "--dust_cost", "1.0",
        "--entropy_eps", "0.1",
        "--ot_alpha", "0.05",
        "--auto_img_dims", "true",
        "--use_hf_clip_vision", "true",
    ]
    spec = importlib.util.spec_from_file_location("pgmner_main_exp3", repo / "main-mner-robert.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import main-mner-robert.py")
    pgmner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pgmner)
    pgmner.args.labels = str((repo / cli_args.labels_file).resolve() if not os.path.isabs(cli_args.labels_file) else cli_args.labels_file)
    return pgmner


def load_with_legacy_clip_key_remap(model, state: dict):
    try:
        model.load_state_dict(state, strict=True)
        return "strict"
    except RuntimeError:
        mapped = {}
        changed = 0
        old = "image_model._hf.encoder.vision_model."
        new = "image_model._hf.encoder."
        for k, v in state.items():
            if k.startswith(old):
                mapped[new + k[len(old):]] = v
                changed += 1
            else:
                mapped[k] = v
        if changed == 0:
            raise
        model.load_state_dict(mapped, strict=True)
        return f"legacy_clip_key_remap:{changed}"


def get_partial_ot(model):
    aligner = getattr(model, "semantic_ot_aligner", None)
    partial = getattr(aligner, "_partial_ot", None) if aligner is not None else None
    if partial is None:
        raise RuntimeError("model.semantic_ot_aligner._partial_ot not found")
    return partial


@contextmanager
def actual_s_override(partial, s_value: Optional[float]):
    original = partial.get_s
    actual_holder = {"value": None}
    if s_value is None:
        yield actual_holder
        return
    s_float = float(s_value)

    def patched_get_s(self):
        t = torch.tensor(s_float, device=self.s_logit.device, dtype=torch.float32)
        actual_holder["value"] = float(t.detach().cpu().item())
        return t

    partial.get_s = types.MethodType(patched_get_s, partial)
    try:
        yield actual_holder
    finally:
        partial.get_s = original


def entity_sets_for_sample(all_span_idxs, probs, gold_labels, span_masks) -> Tuple[set, set, List[Dict[str, Any]], List[Dict[str, Any]], List[float]]:
    raw_pred = probs.argmax(dim=-1)
    candidates = []
    for i in range(probs.shape[0]):
        if not bool((span_masks[i] > 0).item()):
            continue
        ws = int(all_span_idxs[i, 0].item())
        we = int(all_span_idxs[i, 1].item())
        rp = int(raw_pred[i].item())
        score = float(probs[i, rp].detach().cpu().item())
        gold = int(gold_labels[i].item())
        candidates.append({"i": i, "ws": ws, "we": we, "rp": rp, "score": score, "gold": gold})
    pred_cands = [c for c in candidates if c["rp"] != 0]
    pred_cands.sort(key=lambda x: -x["score"])
    chosen = []
    for c in pred_cands:
        if any(not (c["we"] < ch["ws"] or ch["we"] < c["ws"]) for ch in chosen):
            continue
        chosen.append(c)
    pred_set = {(c["ws"], c["we"], c["rp"]) for c in chosen}
    gold_set = {(c["ws"], c["we"], c["gold"]) for c in candidates if c["gold"] != 0}
    pred_list = [{"start": a, "end": b, "label_id": lab} for a, b, lab in sorted(pred_set)]
    gold_list = [{"start": a, "end": b, "label_id": lab} for a, b, lab in sorted(gold_set)]
    valid_conf = [float(probs[c["i"], c["rp"]].detach().cpu().item()) for c in candidates]
    return pred_set, gold_set, pred_list, gold_list, valid_conf


def per_sample_ce(probs, gold_labels, span_weights, span_masks) -> List[float]:
    gold = gold_labels.clamp_min(0)
    p_gold = probs.gather(-1, gold.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    ce = -torch.log(p_gold) * span_weights
    mask = span_masks.float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    vals = (ce * mask).sum(dim=1) / denom
    return [float(x) for x in vals.detach().cpu().tolist()]


def run_aligner_classifier(model, batch: Dict[str, torch.Tensor], text_embed_local, vis_embed_local, attn_last):
    mask_t = batch["span_masks"].float()
    out = model.semantic_ot_aligner(
        text_embed_local,
        vis_embed_local,
        mask_t=mask_t,
        mask_v=None,
        attn_t=attn_last if getattr(model, "use_got", False) else None,
        span_token_idxes=batch.get("span_token_idxes"),
        patch_pos=None,
    )
    text_proj, visual_proj, align_text, align_visual, transport_plan, loss_ot, ot_components = out
    text_combined = torch.cat([text_proj, align_text], dim=-1)
    fused_text = model.fusion_text(text_combined)
    fused_text = model.ln_text(fused_text + align_text)
    logits = model.span_classifier(fused_text)
    bias = getattr(model, "_misc_logit_bias", None)
    if bias is not None and getattr(model, "id_misc_focal", -1) >= 0:
        logits = logits.clone()
        logits[:, :, model.id_misc_focal] += float(bias)
    probs = model.softmax(logits)
    plan_mass = transport_plan.sum(dim=(1, 2)).detach().cpu().tolist()
    return probs, transport_plan, loss_ot, ot_components, plan_mass


def evaluate_landscape(pgmner, model, tokenizer, labels, examples, conditions: List[Tuple[str, Optional[float]]], batch_size: int):
    from utils.datasets import SpanNerDataset, collate_fn

    dataset = SpanNerDataset(examples, args=pgmner.args, tokenizer=tokenizer, labels=labels, dev=True)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, collate_fn=collate_fn)
    partial = get_partial_ot(model)
    partial.set_warmup(False)
    model.eval()

    sample_rows: List[Dict[str, Any]] = []
    dataset_acc = {
        cond: {"tp": 0, "pred": 0, "gold": 0, "loss_sum": 0.0, "ot_sum": 0.0, "n": 0}
        for cond, _ in conditions
    }
    validation_records = []
    global_index = 0
    t0 = time.time()

    for batch_index, batch_cpu in enumerate(loader):
        batch = {k: v.to(pgmner.args.device) for k, v in batch_cpu.items()}
        bsz = int(batch["input_ids"].shape[0])
        batch_examples = examples[global_index: global_index + bsz]
        with torch.no_grad():
            spans_rep, txt_rep_g, attn_last = model.span_encoding(**batch)
            if getattr(model, "use_pp_ib", False):
                _mu_t, _lv_t, _zt, _lp_t = model.dist_t(txt_rep_g)
                text_gate = model.dist_t.last_probe["gate"].detach().mean(dim=1).cpu().tolist()
            else:
                text_gate = [None] * bsz
            img_g, img_l = model.image_model(batch["image"])
            if getattr(model, "use_pp_ib", False):
                _mu_v, _lv_v, _zv, _lp_v = model.dist_v(img_g)
                visual_gate = model.dist_v.last_probe["gate"].detach().mean(dim=1).cpu().tolist()
            else:
                visual_gate = [None] * bsz

            for cond_name, s_value in conditions:
                with actual_s_override(partial, s_value) as actual_holder:
                    probs, transport_plan, loss_ot, ot_components, plan_mass = run_aligner_classifier(
                        model, batch, spans_rep, img_l, attn_last
                    )
                if batch_index == 0 and cond_name != "checkpoint_native":
                    validation_records.append({
                        "requested_s": s_value,
                        "actual_s_seen_by_get_s": actual_holder["value"],
                        "abs_error": abs(float(s_value) - float(actual_holder["value"])),
                    })
                losses = per_sample_ce(probs, batch["span_labels"], batch["span_weights"], batch["span_masks"])
                ot_loss_value = float(loss_ot.detach().cpu().item()) if torch.is_tensor(loss_ot) else ""
                for bi, ex in enumerate(batch_examples):
                    pred_set, gold_set, pred_list, gold_list, confs = entity_sets_for_sample(
                        batch["span_word_idxes"][bi], probs[bi], batch["span_labels"][bi], batch["span_masks"][bi]
                    )
                    tp = len(pred_set & gold_set)
                    fp = len(pred_set - gold_set)
                    fn = len(gold_set - pred_set)
                    prec = tp / (tp + fp + 1e-10)
                    rec = tp / (tp + fn + 1e-10)
                    f1 = 2 * prec * rec / (prec + rec + 1e-10)
                    row = {
                        "sample_id": ex.guid,
                        "sample_index": global_index + bi,
                        "image_id": ex.imgid,
                        "text": " ".join(ex.words[: getattr(ex, "orig_n_words", len(ex.words))]),
                        "condition": cond_name,
                        "s": S_CKPT if cond_name == "checkpoint_native" else float(s_value),
                        "gold_entities": json.dumps(gold_list, ensure_ascii=False, sort_keys=True),
                        "predicted_entities": json.dumps(pred_list, ensure_ascii=False, sort_keys=True),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "sample_precision": prec,
                        "sample_recall": rec,
                        "sample_f1": f1,
                        "exact_correct": pred_set == gold_set,
                        "task_loss": losses[bi],
                        "span_ce_loss": losses[bi],
                        "ot_loss_batch_mean": ot_loss_value,
                        "transport_plan_mass": float(plan_mass[bi]),
                        "dust_mass_estimate": float(1.0 - plan_mass[bi]),
                        "confidence_mean_valid_span": float(np.mean(confs)) if confs else "",
                        "confidence_max_valid_span": float(np.max(confs)) if confs else "",
                        "pp_ib_text_gate_mean": "" if text_gate[bi] is None else float(text_gate[bi]),
                        "pp_ib_visual_gate_mean": "" if visual_gate[bi] is None else float(visual_gate[bi]),
                    }
                    sample_rows.append(row)
                    acc = dataset_acc[cond_name]
                    acc["tp"] += tp
                    acc["pred"] += len(pred_set)
                    acc["gold"] += len(gold_set)
                    acc["loss_sum"] += float(losses[bi])
                    acc["ot_sum"] += float(ot_loss_value) if ot_loss_value != "" else 0.0
                    acc["n"] += 1
        global_index += bsz
        if (batch_index + 1) % 10 == 0:
            print(f"processed batches={batch_index + 1}, samples={global_index}, elapsed={time.time() - t0:.1f}s", flush=True)

    dataset_rows = []
    for cond_name, s_value in conditions:
        acc = dataset_acc[cond_name]
        tp = acc["tp"]
        pred = acc["pred"]
        gold = acc["gold"]
        precision = tp / (pred + 1e-10)
        recall = tp / (gold + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        dataset_rows.append({
            "s": S_CKPT if cond_name == "checkpoint_native" else float(s_value),
            "condition": cond_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": acc["loss_sum"] / max(1, acc["n"]),
            "span_ce_loss": acc["loss_sum"] / max(1, acc["n"]),
            "ot_loss": acc["ot_sum"] / max(1, acc["n"]),
            "num_pred": pred,
            "tp": tp,
            "fp": pred - tp,
            "fn": gold - tp,
        })
    return dataset_rows, sample_rows, validation_records


def choose_argmin_with_tie(rows: List[Dict[str, Any]], value_key: str, s_global: float) -> Tuple[float, int, float]:
    vals = np.array([float(r[value_key]) for r in rows], dtype=float)
    best = float(vals.min())
    tol = max(LOSS_FLAT_ATOL, LOSS_FLAT_RTOL * abs(best))
    tied = [i for i, v in enumerate(vals) if abs(float(v) - best) <= tol]
    chosen = min(tied, key=lambda i: abs(float(rows[i]["s"]) - s_global))
    return float(rows[chosen]["s"]), len(tied), best


def choose_argmax_f1(rows: List[Dict[str, Any]], s_global: float) -> Tuple[float, int, float]:
    vals = np.array([float(r["sample_f1"]) for r in rows], dtype=float)
    best = float(vals.max())
    tied = [i for i, v in enumerate(vals) if abs(float(v) - best) <= 1e-12]
    chosen = min(tied, key=lambda i: (abs(float(rows[i]["s"]) - s_global), float(rows[i]["s"])))
    return float(rows[chosen]["s"]), len(tied), best


def oracle_and_heterogeneity(sample_rows: List[Dict[str, Any]], dataset_rows: List[Dict[str, Any]]):
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        by_sample[row["sample_id"]].append(row)

    oracle_rows = []
    transition_rows = []
    loss_regret_rows = []
    for sid, rows in by_sample.items():
        rows_sorted = sorted(rows, key=lambda r: (r["condition"] != "checkpoint_native", float(r["s"])))
        ck = [r for r in rows if r["condition"] == "checkpoint_native"][0]
        grid_rows = [r for r in rows if r["condition"] != "checkpoint_native"]
        sx_loss, n_loss_tied, min_loss = choose_argmin_with_tie(grid_rows + [ck], "task_loss", S_CKPT)
        sx_f1, n_f1_tied, best_f1 = choose_argmax_f1(grid_rows + [ck], S_CKPT)
        pred_serials = [r["predicted_entities"] for r in rows]
        exact_vals = [bool(r["exact_correct"]) for r in rows]
        ck_correct = bool(ck["exact_correct"])
        improve = [float(r["s"]) for r in rows if (not ck_correct and bool(r["exact_correct"]))]
        harm = [float(r["s"]) for r in rows if (ck_correct and not bool(r["exact_correct"]))]
        invariant = len(set(pred_serials)) == 1
        losses = [float(r["task_loss"]) for r in rows]
        min_l, max_l = min(losses), max(losses)
        tol = max(LOSS_FLAT_ATOL, LOSS_FLAT_RTOL * abs(min_l))
        near_flat = (max_l - min_l) <= tol
        oracle_rows.append({
            "sample_id": sid,
            "sample_index": ck["sample_index"],
            "image_id": ck["image_id"],
            "s_x_loss_star": sx_loss,
            "min_loss": min_loss,
            "loss_at_checkpoint_s": float(ck["task_loss"]),
            "oracle_loss_gain": float(ck["task_loss"]) - min_loss,
            "loss_tied_s_count": n_loss_tied,
            "s_x_f1_star": sx_f1,
            "best_sample_f1": best_f1,
            "f1_at_checkpoint_s": float(ck["sample_f1"]),
            "oracle_f1_gain": best_f1 - float(ck["sample_f1"]),
            "f1_tied_s_count": n_f1_tied,
            "checkpoint_prediction_correct": ck_correct,
            "exists_other_s_improves_wrong_to_correct": bool(improve),
            "exists_other_s_harms_correct_to_wrong": bool(harm),
            "improving_s_values": json.dumps(sorted(set(improve))),
            "harming_s_values": json.dumps(sorted(set(harm))),
            "prediction_invariant": invariant,
            "loss_near_flat": near_flat,
            "loss_range": max_l - min_l,
        })
        for r in rows_sorted:
            transition_rows.append({
                "sample_id": sid,
                "condition": r["condition"],
                "s": r["s"],
                "exact_correct": r["exact_correct"],
                "prediction_signature": r["predicted_entities"],
            })

    # Regret for each fixed condition relative to per-sample loss oracle.
    min_by_sample = {r["sample_id"]: float(r["min_loss"]) for r in oracle_rows}
    for drow in dataset_rows:
        cond = drow["condition"]
        cond_rows = [r for r in sample_rows if r["condition"] == cond]
        regret = np.mean([float(r["task_loss"]) - min_by_sample[r["sample_id"]] for r in cond_rows])
        loss_regret_rows.append({
            "s": drow["s"],
            "condition": cond,
            "mean_loss": drow["loss"],
            "mean_excess_loss_vs_instance_oracle": float(regret),
            "note": "POST-HOC ORACLE UPPER BOUND; not deployable performance",
        })

    sx_loss_vals = np.array([float(r["s_x_loss_star"]) for r in oracle_rows], dtype=float)
    sx_f1_vals = np.array([float(r["s_x_f1_star"]) for r in oracle_rows], dtype=float)
    n = len(oracle_rows)
    pred_invariant = np.array([bool(r["prediction_invariant"]) for r in oracle_rows])
    loss_flat = np.array([bool(r["loss_near_flat"]) for r in oracle_rows])
    sensitive = ~pred_invariant
    ck_wrong = [r for r in oracle_rows if not bool(r["checkpoint_prediction_correct"])]
    ck_correct = [r for r in oracle_rows if bool(r["checkpoint_prediction_correct"])]
    hist_counts = Counter(sx_loss_vals.tolist())
    probs = np.array(list(hist_counts.values()), dtype=float) / max(1, n)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    hetero = {
        "n_samples": n,
        "oracle_loss_s_mean": float(np.mean(sx_loss_vals)),
        "oracle_loss_s_std": float(np.std(sx_loss_vals)),
        "oracle_loss_s_iqr": float(np.percentile(sx_loss_vals, 75) - np.percentile(sx_loss_vals, 25)),
        "oracle_loss_s_entropy": entropy,
        "oracle_f1_s_mean": float(np.mean(sx_f1_vals)),
        "oracle_f1_s_std": float(np.std(sx_f1_vals)),
        "prediction_invariant_fraction": float(pred_invariant.mean()),
        "sensitive_fraction": float(sensitive.mean()),
        "sensitive_sample_count": int(sensitive.sum()),
        "loss_near_flat_fraction": float(loss_flat.mean()),
        "avg_loss_tied_s_count": float(np.mean([int(r["loss_tied_s_count"]) for r in oracle_rows])),
        "avg_f1_tied_s_count": float(np.mean([int(r["f1_tied_s_count"]) for r in oracle_rows])),
        "frac_abs_loss_oracle_diff_ge_0_05": float((np.abs(sx_loss_vals - S_CKPT) >= 0.05).mean()),
        "frac_abs_loss_oracle_diff_ge_0_10": float((np.abs(sx_loss_vals - S_CKPT) >= 0.10).mean()),
        "frac_abs_loss_oracle_diff_ge_0_20": float((np.abs(sx_loss_vals - S_CKPT) >= 0.20).mean()),
        "checkpoint_wrong_count": len(ck_wrong),
        "checkpoint_wrong_recoverable_fraction": float(np.mean([bool(r["exists_other_s_improves_wrong_to_correct"]) for r in ck_wrong])) if ck_wrong else 0.0,
        "checkpoint_correct_count": len(ck_correct),
        "checkpoint_correct_harmed_fraction": float(np.mean([bool(r["exists_other_s_harms_correct_to_wrong"]) for r in ck_correct])) if ck_correct else 0.0,
    }
    flat = {
        "loss_tolerance_rule": "max(1e-6, 1e-4 * abs(min_loss))",
        "exact_flat_prediction_fraction": hetero["prediction_invariant_fraction"],
        "near_flat_loss_fraction": hetero["loss_near_flat_fraction"],
        "average_number_of_loss_tied_or_nearly_tied_s_values": hetero["avg_loss_tied_s_count"],
        "average_number_of_f1_tied_s_values": hetero["avg_f1_tied_s_count"],
        "effective_sensitive_sample_count": hetero["sensitive_sample_count"],
        "note": ">70% insensitive must be reported explicitly" if hetero["prediction_invariant_fraction"] > 0.70 else "",
    }
    return oracle_rows, hetero, loss_regret_rows, transition_rows, flat


def compute_relevance(cli_args: argparse.Namespace, examples, model=None) -> List[Dict[str, Any]]:
    rows = []
    if cli_args.skip_relevance:
        return [{
            "sample_id": ex.guid,
            "image_id": ex.imgid,
            "global_clip_similarity": "",
            "entity_token_grounded_fraction": "",
            "entity_patch_mean_max_similarity": "",
            "entity_patch_max_similarity": "",
            "relevance_status": "SKIPPED",
        } for ex in examples]
    try:
        from analysis.dataset_analysis import HFCLIPScorer, extract_entities
        scorer = HFCLIPScorer(cli_args.clip_model, device=torch.device("cpu"))
        image_dir = cli_args.data_dir + "_images"
        for i, ex in enumerate(examples):
            img_path = os.path.join(image_dir, str(ex.imgid) + ".jpg")
            row = {
                "sample_id": ex.guid,
                "sample_index": i,
                "image_id": ex.imgid,
                "global_clip_similarity": "",
                "entity_token_grounded_fraction": "",
                "entity_patch_mean_max_similarity": "",
                "entity_patch_max_similarity": "",
                "num_entity_tokens": 0,
                "num_grounded_entity_tokens_threshold_0_25": 0,
                "relevance_status": "OK",
            }
            if not os.path.isfile(img_path):
                row["relevance_status"] = "MISSING_IMAGE"
                rows.append(row)
                continue
            tokens = ex.words[: getattr(ex, "orig_n_words", len(ex.words))]
            labels = ex.bio_labels[: getattr(ex, "orig_n_words", len(ex.bio_labels))]
            full_text = " ".join(tokens)
            t = scorer.encode_text(full_text)
            v = scorer.encode_image_path(img_path)
            row["global_clip_similarity"] = float((t * v).sum(dim=-1).item())
            entities = extract_entities(tokens, labels)
            if entities:
                patches = scorer.encode_image_patches(img_path, (3, 3))
                max_sims = []
                total_tok = 0
                grounded_tok = 0
                for ent in entities:
                    total_tok += len(ent["token_indices"])
                    ent_feat = scorer.encode_text(ent["text"])
                    sims = (ent_feat @ patches.T).squeeze(0).detach().float().cpu().numpy()
                    mx = float(np.max(sims))
                    max_sims.append(mx)
                    if mx > 0.25:
                        grounded_tok += len(ent["token_indices"])
                row["num_entity_tokens"] = total_tok
                row["num_grounded_entity_tokens_threshold_0_25"] = grounded_tok
                row["entity_token_grounded_fraction"] = grounded_tok / total_tok if total_tok else 0.0
                row["entity_patch_mean_max_similarity"] = float(np.mean(max_sims))
                row["entity_patch_max_similarity"] = float(np.max(max_sims))
            else:
                row["entity_token_grounded_fraction"] = 0.0
                row["entity_patch_mean_max_similarity"] = ""
                row["entity_patch_max_similarity"] = ""
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"relevance samples={i + 1}", flush=True)
    except Exception as exc:  # noqa: BLE001
        rows = [{
            "sample_id": ex.guid,
            "sample_index": i,
            "image_id": ex.imgid,
            "global_clip_similarity": "",
            "entity_token_grounded_fraction": "",
            "entity_patch_mean_max_similarity": "",
            "entity_patch_max_similarity": "",
            "relevance_status": f"UNAVAILABLE:{type(exc).__name__}:{exc}",
        } for i, ex in enumerate(examples)]
    return rows


def relevance_relationship(oracle_rows, relevance_rows):
    rel_by_id = {r["sample_id"]: r for r in relevance_rows}
    merged = []
    for o in oracle_rows:
        rel = rel_by_id.get(o["sample_id"], {})
        row = {**o, **{f"rel_{k}": v for k, v in rel.items() if k != "sample_id"}}
        merged.append(row)

    def numeric_pair(xkey: str, ykey: str):
        xs, ys = [], []
        for r in merged:
            try:
                x = float(r[xkey])
                y = float(r[ykey])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)

    rel_rows = []
    for oracle_key in ["s_x_loss_star", "s_x_f1_star"]:
        for rel_key in ["rel_global_clip_similarity", "rel_entity_token_grounded_fraction", "rel_entity_patch_mean_max_similarity"]:
            xs, ys = numeric_pair(oracle_key, rel_key)
            if len(xs) >= 3 and len(set(xs.tolist())) > 1 and len(set(ys.tolist())) > 1:
                pear = stats.pearsonr(ys, xs)
                spear = stats.spearmanr(ys, xs)
                rel_rows.append({
                    "oracle_budget": oracle_key,
                    "relevance_feature": rel_key,
                    "n": len(xs),
                    "pearson_r": float(pear.statistic),
                    "pearson_p": float(pear.pvalue),
                    "spearman_rho": float(spear.statistic),
                    "spearman_p": float(spear.pvalue),
                })
            else:
                rel_rows.append({
                    "oracle_budget": oracle_key,
                    "relevance_feature": rel_key,
                    "n": len(xs),
                    "pearson_r": "",
                    "pearson_p": "",
                    "spearman_rho": "",
                    "spearman_p": "",
                })

    # Equal-frequency groups by global CLIP similarity.
    valid = []
    for r in merged:
        try:
            sim = float(r["rel_global_clip_similarity"])
            valid.append((sim, r))
        except (TypeError, ValueError, KeyError):
            continue
    valid.sort(key=lambda x: x[0])
    groups = {"Low": [], "Medium": [], "High": []}
    if valid:
        n = len(valid)
        for idx, (_, row) in enumerate(valid):
            g = "Low" if idx < n / 3 else "Medium" if idx < 2 * n / 3 else "High"
            row["relevance_group"] = g
            groups[g].append(row)
    group_rows = []
    for g, rows in groups.items():
        vals = np.array([float(r["s_x_loss_star"]) for r in rows], dtype=float) if rows else np.array([])
        group_rows.append({
            "group": g,
            "n": len(rows),
            "mean_oracle_s": float(np.mean(vals)) if len(vals) else "",
            "median_oracle_s": float(np.median(vals)) if len(vals) else "",
            "std_oracle_s": float(np.std(vals)) if len(vals) else "",
            "iqr_oracle_s": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) else "",
            "sensitive_fraction": float(np.mean([not bool(r["prediction_invariant"]) for r in rows])) if rows else "",
            "recoverable_error_fraction": float(np.mean([bool(r["exists_other_s_improves_wrong_to_correct"]) for r in rows if not bool(r["checkpoint_prediction_correct"])])) if any(not bool(r["checkpoint_prediction_correct"]) for r in rows) else 0.0,
        })

    test_rows = []
    group_vals = [np.array([float(r["s_x_loss_star"]) for r in groups[g]], dtype=float) for g in ["Low", "Medium", "High"]]
    if all(len(v) > 0 for v in group_vals):
        kw = stats.kruskal(*group_vals)
        test_rows.append({"test": "Kruskal-Wallis", "comparison": "Low/Medium/High", "statistic": float(kw.statistic), "p": float(kw.pvalue), "p_holm": float(kw.pvalue)})
        pairs = [("Low", "Medium"), ("Low", "High"), ("Medium", "High")]
        raw = []
        for a, b in pairs:
            u = stats.mannwhitneyu(
                [float(r["s_x_loss_star"]) for r in groups[a]],
                [float(r["s_x_loss_star"]) for r in groups[b]],
                alternative="two-sided",
            )
            raw.append((a, b, float(u.statistic), float(u.pvalue)))
        order = sorted(range(len(raw)), key=lambda i: raw[i][3])
        adj = [None] * len(raw)
        prev = 0.0
        m = len(raw)
        for rank, i in enumerate(order):
            val = min(1.0, raw[i][3] * (m - rank))
            val = max(prev, val)
            prev = val
            adj[i] = val
        for (a, b, stat, p), p_adj in zip(raw, adj):
            test_rows.append({"test": "Mann-Whitney U", "comparison": f"{a} vs {b}", "statistic": stat, "p": p, "p_holm": p_adj})
    return merged, rel_rows, group_rows, test_rows


def make_figures(dataset_rows, oracle_rows, regret_rows, merged_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [r for r in dataset_rows if r["condition"] != "checkpoint_native"]
    grid = sorted(grid, key=lambda r: float(r["s"]))
    xs = [float(r["s"]) for r in grid]
    ys = [float(r["f1"]) for r in grid]
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.axvline(S_CKPT, color="black", linestyle="--", linewidth=1.2, label=f"checkpoint s={S_CKPT:.3f}")
    plt.xlabel("Transport budget s")
    plt.ylabel("Twitter-2017 span F1")
    plt.title("Dataset-Level Budget Sensitivity")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3a_dataset_f1_vs_s.pdf")
    plt.savefig(FIGURES / "fig3a_dataset_f1_vs_s.png", dpi=300)
    plt.close()

    vals = [float(r["s_x_loss_star"]) for r in oracle_rows]
    counts = Counter(vals)
    xvals = sorted(counts)
    plt.figure(figsize=(5.2, 3.4))
    plt.bar([str(f"{x:.2f}") for x in xvals], [counts[x] for x in xvals], color="#4c78a8")
    plt.axvline(len([x for x in xvals if x < S_CKPT]) - 0.5, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("POST-HOC oracle budget s*x")
    plt.ylabel("Number of samples")
    plt.title("Instance-wise Oracle Budget Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3b_oracle_budget_distribution.pdf")
    plt.savefig(FIGURES / "fig3b_oracle_budget_distribution.png", dpi=300)
    plt.close()

    groups = ["Low", "Medium", "High"]
    data = [[float(r["s_x_loss_star"]) for r in merged_rows if r.get("relevance_group") == g] for g in groups]
    plt.figure(figsize=(5.2, 3.4))
    if any(data):
        plt.boxplot(data, labels=groups, showmeans=True)
    plt.ylabel("POST-HOC oracle budget s*x")
    plt.xlabel("CLIP relevance group")
    plt.title("Visual Relevance vs Preferred Budget")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3c_relevance_vs_budget.pdf")
    plt.savefig(FIGURES / "fig3c_relevance_vs_budget.png", dpi=300)
    plt.close()

    reg_grid = sorted([r for r in regret_rows if r["condition"] != "checkpoint_native"], key=lambda r: float(r["s"]))
    plt.figure(figsize=(5.2, 3.4))
    plt.plot([float(r["s"]) for r in reg_grid], [float(r["mean_excess_loss_vs_instance_oracle"]) for r in reg_grid], marker="o", linewidth=1.8, color="#f58518")
    plt.axvline(S_CKPT, color="black", linestyle="--", linewidth=1.2, label=f"checkpoint s={S_CKPT:.3f}")
    plt.xlabel("Fixed global transport budget s")
    plt.ylabel("Mean excess task loss")
    plt.title("Global Budget Regret")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3d_global_budget_regret.pdf")
    plt.savefig(FIGURES / "fig3d_global_budget_regret.png", dpi=300)
    plt.close()


def classifications(hetero, regret_rows, rel_rows):
    sens = hetero["sensitive_fraction"]
    if sens >= 0.30:
        a = "STRONG"
    elif sens >= 0.15:
        a = "MODERATE"
    elif sens > 0.02:
        a = "WEAK"
    else:
        a = "NONE"
    std = hetero["oracle_loss_s_std"]
    diff10 = hetero["frac_abs_loss_oracle_diff_ge_0_10"]
    # Classification B is explicitly about heterogeneity among prediction-sensitive
    # samples. A broad loss-oracle distribution is not enough when predictions are
    # almost entirely invariant to s.
    if hetero.get("sensitive_sample_count", 0) < 5 or sens < 0.02:
        b = "NONE"
    elif std >= 0.12 and diff10 >= 0.30:
        b = "STRONG"
    elif std >= 0.07 and diff10 >= 0.15:
        b = "MODERATE"
    elif std > 0.02:
        b = "WEAK"
    else:
        b = "NONE"
    clip_loss = next((r for r in rel_rows if r["oracle_budget"] == "s_x_loss_star" and r["relevance_feature"] == "rel_global_clip_similarity"), None)
    if clip_loss and clip_loss.get("spearman_rho") != "":
        rho = float(clip_loss["spearman_rho"])
        p = float(clip_loss["spearman_p"])
        if p < 0.05 and rho > 0:
            c = "POSITIVE_ASSOCIATION"
        elif p < 0.05 and rho < 0:
            c = "NEGATIVE_ASSOCIATION"
        else:
            c = "NO_CLEAR_ASSOCIATION"
    else:
        c = "UNCERTAIN"
    ck_reg = next((float(r["mean_excess_loss_vs_instance_oracle"]) for r in regret_rows if r["condition"] == "checkpoint_native"), 0.0)
    if ck_reg >= 0.05:
        d = "LARGE"
    elif ck_reg >= 0.01:
        d = "MODERATE"
    elif ck_reg > 1e-5:
        d = "SMALL"
    else:
        d = "NONE"
    if a in {"STRONG", "MODERATE"} and b in {"STRONG", "MODERATE"} and d in {"LARGE", "MODERATE"}:
        e = "YES"
    elif a != "NONE" and b != "NONE" and d != "NONE":
        e = "BORDERLINE"
    else:
        e = "NO"
    return {"A": a, "B": b, "C": c, "D": d, "E": e}


def write_report(dataset_rows, hetero, regret_rows, rel_rows, group_rows, test_rows, classes):
    ck = next(r for r in dataset_rows if r["condition"] == "checkpoint_native")
    best = max([r for r in dataset_rows if r["condition"] != "checkpoint_native"], key=lambda r: float(r["f1"]))
    f1s = [float(r["f1"]) for r in dataset_rows if r["condition"] != "checkpoint_native"]
    ck_reg = next(r for r in regret_rows if r["condition"] == "checkpoint_native")
    clip = next((r for r in rel_rows if r["oracle_budget"] == "s_x_loss_star" and r["relevance_feature"] == "rel_global_clip_similarity"), {})
    report = f"""# Experiment 3: Instance-wise Transport Budget Landscape

## 1. Research Question

Does a single globally shared transport budget adequately serve all Twitter-2017 test samples, or do samples prefer different inference-time budgets?

All oracle analyses below are **POST-HOC ORACLE** diagnostics using test labels. They are not deployable model performance.

## 2. Experimental Setup

- Checkpoint: `/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth`
- Dataset: Twitter-2017 test split
- Global checkpoint `s`: `{S_CKPT:.12f}`
- Grid: `{S_GRID}`
- No training and no optimizer step.
- Encoder outputs were cached per batch; only GOT-A/fusion/classifier were recomputed across `s`.

## 3. Dataset-Level Sensitivity

- Checkpoint-native F1: `{float(ck['f1']):.6f}`
- Best fixed grid F1: `{float(best['f1']):.6f}` at `s={float(best['s']):.2f}`
- Grid F1 range: `{min(f1s):.6f}` to `{max(f1s):.6f}`

See `results/dataset_level_budget_landscape.csv`.

## 4. Instance-Level Budget Heterogeneity

- Oracle loss-budget mean/std/IQR: `{hetero['oracle_loss_s_mean']:.6f}` / `{hetero['oracle_loss_s_std']:.6f}` / `{hetero['oracle_loss_s_iqr']:.6f}`
- Fraction `|s*_loss - s_global| >= 0.10`: `{hetero['frac_abs_loss_oracle_diff_ge_0_10']:.6f}`
- Sensitive sample count: `{hetero['sensitive_sample_count']}` of `{hetero['n_samples']}`

See `results/sample_oracle_budgets.csv`.

## 5. Flat / Insensitive Samples

- Prediction-invariant fraction: `{hetero['prediction_invariant_fraction']:.6f}`
- Loss-near-flat fraction: `{hetero['loss_near_flat_fraction']:.6f}`
- Average near-tied loss budgets: `{hetero['avg_loss_tied_s_count']:.6f}`

See `results/flat_landscape_analysis.json`.

## 6. Post-hoc Oracle Analysis

The loss oracle chooses `argmin_s` of per-sample span CE diagnostic loss with tie-breaking toward checkpoint `s`. The F1 oracle chooses `argmax_s` sample entity F1 with the specified tie rules.

These are **POST-HOC ORACLE UPPER BOUND** diagnostics.

## 7. Global-Budget Regret

- Checkpoint global regret: `{float(ck_reg['mean_excess_loss_vs_instance_oracle']):.8f}`
- Best post-hoc global budget by mean loss is recorded in `results/global_budget_regret.csv`.

## 8. Relationship with Visual Relevance

- CLIP similarity vs loss-oracle `s*` Spearman rho: `{clip.get('spearman_rho', '')}`
- p-value: `{clip.get('spearman_p', '')}`
- Group statistics: `results/relevance_group_statistics.csv`
- Statistical tests: `results/relevance_statistical_tests.csv`

Correlation is not interpreted as causation.

## 9. Prediction Help/Harm Transitions

- Checkpoint-wrong recoverable fraction: `{hetero['checkpoint_wrong_recoverable_fraction']:.6f}`
- Checkpoint-correct harmed fraction: `{hetero['checkpoint_correct_harmed_fraction']:.6f}`

Transition records are in `results/prediction_transition_counts.csv`.

## 10. Implications for Instance-Adaptive Transport

The result should be used only to decide whether an instance-adaptive `s_x` is worth studying next. It does not establish deployable gains because oracle choices use labels.

## 11. Limitations

- Per-sample task loss is a diagnostic span CE computed from evaluation probabilities and gold spans.
- OT loss is available as batch mean and transport-plan mass statistics, not as a fully decomposed original per-sample training loss.
- CPU-only environment required batch encoder caching.
- Oracle budgets are selected from a coarse grid plus checkpoint-native condition.

## 12. Recommended Decision for Experiment 4

Proceed only if the final classifications below justify it quantitatively.

## Final Decision

A. Does the model show meaningful sample-level sensitivity to s? `{classes['A']}`

B. Is there meaningful heterogeneity in preferred s among sensitive samples? `{classes['B']}`

C. Is preferred s statistically associated with visual relevance? `{classes['C']}`

D. Does an instance-wise oracle provide meaningful room over a fixed global budget? `{classes['D']}`

E. Is instance-adaptive s_x empirically motivated strongly enough to proceed? `{classes['E']}`
"""
    (EXP_DIR / "EXPERIMENT_3_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    cli = parse_args()
    repo = Path(cli.repo_root).resolve()
    pgmner = load_pgmner(repo, cli)
    labels = pgmner.get_labels(pgmner.args.labels)
    config = pgmner.AutoConfig.from_pretrained(
        pgmner.args.config_name if pgmner.args.config_name else pgmner.args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None,
    )
    tokenizer = pgmner.RobertaTokenizer.from_pretrained(
        pgmner.args.tokenizer_name if pgmner.args.tokenizer_name else pgmner.args.model_name_or_path
    )
    model = pgmner.BertSpanNerBN(config, args=pgmner.args, num_labels=len(labels))
    state = torch.load(cli.checkpoint, map_location=pgmner.args.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    load_mode = load_with_legacy_clip_key_remap(model, state)
    model.to(pgmner.args.device)
    model.eval()
    if hasattr(model, "set_partial_ot_warmup"):
        model.set_partial_ot_warmup(False)

    examples = pgmner.load_examples(
        pgmner.args.data_dir,
        mode="test",
        tokenizer=tokenizer,
        visual_suffix_map=pgmner.args.visual_suffix_map,
        aux_knowledge_map=pgmner.resolve_aux_knowledge_map(pgmner.args, "test"),
        vllm_desc_map=pgmner.resolve_vllm_desc_map(pgmner.args, "test"),
        max_seq_len=pgmner.args.max_seq_len,
        max_aux_words=int(getattr(pgmner.args, "max_aux_words", 12)),
        aux_dropout=0.0,
        desc_max_words=int(getattr(pgmner.args, "desc_max_words", 60)),
        vllm_desc_dropout=0.0,
        desc_sep_token=str(getattr(pgmner.args, "desc_sep_token", "</s>")),
    )
    if cli.limit_samples:
        examples = examples[: cli.limit_samples]

    conditions = [("checkpoint_native", None)] + [(f"s_{s:.2f}", s) for s in S_GRID]
    dataset_rows, sample_rows, validation_records = evaluate_landscape(
        pgmner, model, tokenizer, labels, examples, conditions, cli.batch_size
    )
    validation = {
        "status": "success" if all(r["abs_error"] <= 1e-7 for r in validation_records) else "failed",
        "checkpoint_native_s": S_CKPT,
        "records": validation_records,
        "load_mode": load_mode,
    }
    write_json(RESULTS / "s_override_validation.json", validation)
    write_csv(RESULTS / "dataset_level_budget_landscape.csv", dataset_rows)
    write_csv(RESULTS / "sample_level_budget_landscape.csv", sample_rows)

    oracle_rows, hetero, regret_rows, transition_rows, flat = oracle_and_heterogeneity(sample_rows, dataset_rows)
    write_csv(RESULTS / "sample_oracle_budgets.csv", oracle_rows)
    write_json(RESULTS / "budget_heterogeneity_summary.json", hetero)
    write_csv(RESULTS / "global_budget_regret.csv", regret_rows)
    write_csv(RESULTS / "prediction_transition_counts.csv", transition_rows)
    write_json(RESULTS / "flat_landscape_analysis.json", flat)

    relevance_rows = compute_relevance(cli, examples, model=model)
    write_csv(RESULTS / "sample_visual_relevance.csv", relevance_rows)
    merged, rel_rows, group_rows, test_rows = relevance_relationship(oracle_rows, relevance_rows)
    write_csv(RESULTS / "relevance_budget_relationship.csv", rel_rows)
    write_csv(RESULTS / "relevance_group_statistics.csv", group_rows)
    write_csv(RESULTS / "relevance_statistical_tests.csv", test_rows)
    write_csv(RESULTS / "oracle_relevance_merged.csv", merged)

    make_figures(dataset_rows, oracle_rows, regret_rows, merged)
    classes = classifications(hetero, regret_rows, rel_rows)
    write_json(RESULTS / "final_classifications.json", classes)
    write_report(dataset_rows, hetero, regret_rows, rel_rows, group_rows, test_rows, classes)

    summary = {
        "checkpoint_native_f1": next(r for r in dataset_rows if r["condition"] == "checkpoint_native")["f1"],
        "best_fixed_grid": max([r for r in dataset_rows if r["condition"] != "checkpoint_native"], key=lambda r: float(r["f1"])),
        "heterogeneity": hetero,
        "classifications": classes,
    }
    write_json(RESULTS / "experiment3_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
