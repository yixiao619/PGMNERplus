#!/usr/bin/env python3
"""Experiment 4: GOT-A effective transport and visual contribution audit.

Inference-only. No training, no optimizer step, no original-file modification.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, SequentialSampler


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment4_visual_mechanism"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
LOGS = EXP / "logs"
S_CKPT = 0.594940338282
CONDITIONS = ["FULL", "NO_GOT", "NO_VISUAL", "RANDOM_IMAGE", "SHUFFLED_TRANSPORT"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(ROOT))
    p.add_argument("--data-dir", default="data/twitter2017")
    p.add_argument("--checkpoint", default="/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
    p.add_argument("--labels-file", default="kais_extension/configs/labels_twitter2017.txt")
    p.add_argument("--model-name-or-path", default="pretrain/roberta-large")
    p.add_argument("--tokenizer-name", default="pretrain/roberta-large")
    p.add_argument("--vit-name", default="pretrain/clip-vit-base-patch16")
    p.add_argument("--gpu-id", default="-1")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit-samples", type=int, default=0)
    return p.parse_args()


def ensure_dirs():
    for d in [RESULTS, FIGURES, LOGS, EXP / "scripts", EXP / "cache"]:
        d.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_pgmner(cli):
    repo = Path(cli.repo_root).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.argv = [
        "main-mner-robert.py", "--data_dir", cli.data_dir,
        "--output_dir", str(RESULTS / "pgmner_side_outputs"),
        "--model_name_or_path", cli.model_name_or_path,
        "--tokenizer_name", cli.tokenizer_name,
        "--vit_name", cli.vit_name,
        "--gpu_id", cli.gpu_id,
        "--do_train", "false", "--do_eval", "false", "--do_predict", "false",
        "--batch_size", str(cli.batch_size),
        "--max_span_len", "4", "--max_seq_len", "128", "--seed", "42",
        "--use_pp_ib", "--use_got", "--got_mode", "partial_sinkhorn",
        "--ot_sinkhorn_iter", "30", "--partial_mass_init", "0.5",
        "--dust_cost", "1.0", "--entropy_eps", "0.1", "--ot_alpha", "0.05",
        "--auto_img_dims", "true", "--use_hf_clip_vision", "true",
    ]
    spec = importlib.util.spec_from_file_location("pgmner_main_exp4", repo / "main-mner-robert.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.args.labels = str((repo / cli.labels_file).resolve())
    return mod


def load_with_remap(model, state):
    try:
        model.load_state_dict(state, strict=True)
        return "strict"
    except RuntimeError:
        old = "image_model._hf.encoder.vision_model."
        new = "image_model._hf.encoder."
        mapped, changed = {}, 0
        for k, v in state.items():
            if k.startswith(old):
                mapped[new + k[len(old):]] = v
                changed += 1
            else:
                mapped[k] = v
        if not changed:
            raise
        model.load_state_dict(mapped, strict=True)
        return f"legacy_clip_key_remap:{changed}"


def entity_sets(all_span_idxs, probs, gold_labels, span_masks):
    raw = probs.argmax(dim=-1)
    cands = []
    for i in range(probs.shape[0]):
        if not bool((span_masks[i] > 0).item()):
            continue
        ws, we = int(all_span_idxs[i, 0].item()), int(all_span_idxs[i, 1].item())
        rp = int(raw[i].item())
        score = float(probs[i, rp].detach().cpu().item())
        gold = int(gold_labels[i].item())
        cands.append({"i": i, "ws": ws, "we": we, "rp": rp, "score": score, "gold": gold})
    chosen = []
    for c in sorted([x for x in cands if x["rp"] != 0], key=lambda x: -x["score"]):
        if any(not (c["we"] < y["ws"] or y["we"] < c["ws"]) for y in chosen):
            continue
        chosen.append(c)
    pred = {(c["ws"], c["we"], c["rp"]) for c in chosen}
    gold = {(c["ws"], c["we"], c["gold"]) for c in cands if c["gold"] != 0}
    pred_by_span = {(c["ws"], c["we"]): (c["rp"], c["score"]) for c in chosen}
    gold_span_indices = {(c["ws"], c["we"], c["gold"]): c["i"] for c in cands if c["gold"] != 0}
    return pred, gold, pred_by_span, gold_span_indices


def ce_loss_per_sample(probs, labels, weights, masks):
    p = probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    ce = -torch.log(p) * weights
    denom = masks.float().sum(dim=1).clamp_min(1.0)
    return ((ce * masks.float()).sum(dim=1) / denom).detach().cpu().tolist()


def transport_stats(gamma: torch.Tensor, mask_t: torch.Tensor):
    g = gamma.detach()
    mt = mask_t.float()
    mass = g.sum(dim=(1, 2))
    dust = 1.0 - mass
    safe = g.clamp_min(1e-12)
    ent = -(safe * safe.log()).sum(dim=(1, 2))
    valid_t = mt.sum(dim=1).clamp_min(1.0)
    nv = torch.full_like(valid_t, g.shape[2])
    norm_ent = ent / (valid_t * nv).clamp_min(2.0).log()
    max_w = g.amax(dim=(1, 2))
    row_max = (g.max(dim=2).values * mt).sum(dim=1) / valid_t
    concentration = (g.pow(2).sum(dim=(1, 2)) / mass.clamp_min(1e-12))
    eff_patches = 1.0 / g.sum(dim=1).pow(2).sum(dim=1).clamp_min(1e-12)
    return {
        "transport_mass": mass.cpu().tolist(),
        "dust_fraction": dust.cpu().tolist(),
        "transport_entropy": ent.cpu().tolist(),
        "normalized_transport_entropy": norm_ent.cpu().tolist(),
        "max_transport_weight": max_w.cpu().tolist(),
        "mean_row_max": row_max.cpu().tolist(),
        "transport_concentration": concentration.cpu().tolist(),
        "effective_attended_patches": eff_patches.cpu().tolist(),
    }


def run_condition(model, batch, enc, condition, rng: random.Random):
    spans_rep, img_l, attn_last = enc["spans_rep"], enc["img_l"], enc["attn_last"]
    mask_t = batch["span_masks"].float()
    out = model.semantic_ot_aligner(
        spans_rep, img_l, mask_t=mask_t, mask_v=None,
        attn_t=attn_last if getattr(model, "use_got", False) else None,
        span_token_idxes=batch.get("span_token_idxes"), patch_pos=None,
    )
    text_proj, visual_proj, align_text, align_visual, gamma, loss_ot, comps = out
    gamma_used = gamma
    validation = {}
    if condition in {"NO_GOT", "NO_VISUAL"}:
        align_text = torch.zeros_like(align_text)
        validation["definition"] = "align_text zeroed; text_proj/classifier/fusion kept"
    elif condition == "SHUFFLED_TRANSPORT":
        g2 = gamma.clone()
        bsz, nt, nv = g2.shape
        diffs = []
        mass_before = gamma.sum(dim=(1, 2))
        perms = []
        for b in range(bsz):
            perm = list(range(nv))
            rng.shuffle(perm)
            # avoid identity when possible
            if nv > 1 and all(perm[i] == i for i in range(nv)):
                perm = perm[1:] + perm[:1]
            perms.append(perm)
            g2[b, :, :] = gamma[b, :, perm]
        gamma_used = g2
        align_text = torch.bmm(gamma_used, visual_proj) * model.semantic_ot_aligner.lambda_align
        diff = (torch.bmm(gamma, visual_proj) - torch.bmm(gamma_used, visual_proj)).norm(dim=-1).mean(dim=1)
        mass_after = gamma_used.sum(dim=(1, 2))
        validation["mass_abs_error_max"] = float((mass_before - mass_after).abs().max().detach().cpu().item())
        validation["mean_alignment_delta"] = float(diff.mean().detach().cpu().item())
        validation["first_permutation"] = perms[0] if perms else []
    text_combined = torch.cat([text_proj, align_text], dim=-1)
    fused_text = model.fusion_text(text_combined)
    fused_text = model.ln_text(fused_text + align_text)
    logits = model.span_classifier(fused_text)
    bias = getattr(model, "_misc_logit_bias", None)
    if bias is not None and getattr(model, "id_misc_focal", -1) >= 0:
        logits = logits.clone()
        logits[:, :, model.id_misc_focal] += float(bias)
    probs = model.softmax(logits)
    drift = align_text.norm(dim=-1).mean(dim=1)
    text_norm = text_proj.norm(dim=-1).mean(dim=1)
    fused_norm = fused_text.norm(dim=-1).mean(dim=1)
    ratio = drift / text_norm.clamp_min(1e-12)
    return {
        "probs": probs,
        "gamma": gamma_used,
        "gamma_clean": gamma,
        "loss_ot": float(loss_ot.detach().cpu().item()),
        "transport": transport_stats(gamma_used, mask_t),
        "fusion_drift": drift.detach().cpu().tolist(),
        "text_norm": text_norm.detach().cpu().tolist(),
        "fused_norm": fused_norm.detach().cpu().tolist(),
        "visual_text_norm_ratio": ratio.detach().cpu().tolist(),
        "validation": validation,
    }


def make_random_examples(examples):
    n = len(examples)
    rng = random.Random(42)
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    rows = []
    out = []
    for i, j in enumerate(perm):
        ex = examples[i]
        new = type("Obj", (), {})()
        new.__dict__.update(ex.__dict__.copy())
        new.imgid = examples[j].imgid
        out.append(new)
        rows.append({"sample_id": ex.guid, "original_image_id": ex.imgid, "random_image_id": examples[j].imgid})
    return out, rows


def labels_for_entities(full_correct, cond_correct):
    if full_correct and cond_correct:
        return "C_C"
    if full_correct and not cond_correct:
        return "C_W"
    if (not full_correct) and cond_correct:
        return "W_C"
    return "W_W"


def bh_fdr(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [1.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), 1):
        # reversed rank from largest to smallest
        k = m - rank + 1
        val = min(prev, pvals[i] * m / k)
        prev = val
        out[i] = min(1.0, val)
    return out


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if len(pos) < 1 or len(neg) < 1:
        return None
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def main():
    ensure_dirs()
    cli = parse_args()
    pgmner = load_pgmner(cli)
    labels = pgmner.get_labels(pgmner.args.labels)
    config = pgmner.AutoConfig.from_pretrained(
        pgmner.args.config_name if pgmner.args.config_name else pgmner.args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = pgmner.RobertaTokenizer.from_pretrained(pgmner.args.tokenizer_name or pgmner.args.model_name_or_path)
    model = pgmner.BertSpanNerBN(config, args=pgmner.args, num_labels=len(labels))
    state = torch.load(cli.checkpoint, map_location=pgmner.args.device)
    load_mode = load_with_remap(model, state)
    model.to(pgmner.args.device).eval()
    model.set_partial_ot_warmup(False)

    examples = pgmner.load_examples(pgmner.args.data_dir, "test", tokenizer, max_seq_len=pgmner.args.max_seq_len)
    if cli.limit_samples:
        examples = examples[:cli.limit_samples]
    random_examples, random_map = make_random_examples(examples)
    write_csv(RESULTS / "random_image_mapping_seed42.csv", random_map)

    write_path_docs()

    from utils.datasets import SpanNerDataset, collate_fn
    ds_full = SpanNerDataset(examples, args=pgmner.args, tokenizer=tokenizer, labels=labels, dev=True)
    ds_rand = SpanNerDataset(random_examples, args=pgmner.args, tokenizer=tokenizer, labels=labels, dev=True)
    loader_full = DataLoader(ds_full, sampler=SequentialSampler(ds_full), batch_size=cli.batch_size, collate_fn=collate_fn)
    loader_rand = DataLoader(ds_rand, sampler=SequentialSampler(ds_rand), batch_size=cli.batch_size, collate_fn=collate_fn)

    all_rows, sample_rows, feature_sample_rows, feature_entity_rows = [], [], [], []
    pred_store: Dict[str, Dict[str, Dict[str, Any]]] = {c: {} for c in CONDITIONS}
    shuffle_validations = []
    random_shift_rows = []
    t0 = time.time()
    global_i = 0
    rng_shuffle = random.Random(42)
    for bi, (batch_full_cpu, batch_rand_cpu) in enumerate(zip(loader_full, loader_rand)):
        bf = {k: v.to(pgmner.args.device) for k, v in batch_full_cpu.items()}
        br = {k: v.to(pgmner.args.device) for k, v in batch_rand_cpu.items()}
        bsz = bf["input_ids"].shape[0]
        batch_examples = examples[global_i:global_i + bsz]
        with torch.no_grad():
            spans_rep, txt_g, attn_last = model.span_encoding(**bf)
            if getattr(model, "use_pp_ib", False):
                _, _, _, _ = model.dist_t(txt_g)
                text_gate = model.dist_t.last_probe["gate"].detach()
            else:
                text_gate = torch.full((bsz, 1), float("nan"), device=pgmner.args.device)
            img_g, img_l = model.image_model(bf["image"])
            if getattr(model, "use_pp_ib", False):
                _, _, _, _ = model.dist_v(img_g)
                visual_gate = model.dist_v.last_probe["gate"].detach()
            else:
                visual_gate = torch.full((bsz, 1), float("nan"), device=pgmner.args.device)
            enc_full = {"spans_rep": spans_rep, "img_l": img_l, "attn_last": attn_last}

            # random image uses same text encoding but random visual encoding.
            img_g_r, img_l_r = model.image_model(br["image"])
            enc_rand = {"spans_rep": spans_rep, "img_l": img_l_r, "attn_last": attn_last}
            cond_outputs = {}
            for cond in CONDITIONS:
                enc = enc_rand if cond == "RANDOM_IMAGE" else enc_full
                cond_outputs[cond] = run_condition(model, bf, enc, cond, rng_shuffle)
                if cond == "SHUFFLED_TRANSPORT":
                    shuffle_validations.append(cond_outputs[cond]["validation"])

            ce_by_cond = {c: ce_loss_per_sample(o["probs"], bf["span_labels"], bf["span_weights"], bf["span_masks"]) for c, o in cond_outputs.items()}
            for j, ex in enumerate(batch_examples):
                sid = ex.guid
                full_clean = cond_outputs["FULL"]
                full_pred, gold, full_pred_by_span, gold_idx = entity_sets(
                    bf["span_word_idxes"][j], full_clean["probs"][j], bf["span_labels"][j], bf["span_masks"][j]
                )
                for cond, out in cond_outputs.items():
                    pred, gold2, pred_by_span, _ = entity_sets(
                        bf["span_word_idxes"][j], out["probs"][j], bf["span_labels"][j], bf["span_masks"][j]
                    )
                    tp = len(pred & gold2); fp = len(pred - gold2); fn = len(gold2 - pred)
                    conf = float(out["probs"][j].max(dim=-1).values[bf["span_masks"][j] > 0].mean().detach().cpu().item())
                    pred_store[cond][sid] = {"pred": pred, "gold": gold2, "pred_by_span": pred_by_span}
                    sample_rows.append({
                        "condition": cond, "sample_id": sid, "sample_index": global_i + j,
                        "text": " ".join(ex.words[:getattr(ex, "orig_n_words", len(ex.words))]),
                        "image_id": ex.imgid,
                        "gold_entities": json.dumps(sorted(list(gold2))),
                        "prediction": json.dumps(sorted(list(pred))),
                        "tp": tp, "fp": fp, "fn": fn,
                        "exact_correct": pred == gold2,
                        "task_loss": ce_by_cond[cond][j],
                        "mean_confidence": conf,
                    })
                ts = full_clean["transport"]
                feature_sample_rows.append({
                    "sample_id": sid, "sample_index": global_i + j, "image_id": ex.imgid,
                    "actual_s": S_CKPT,
                    "mean_text_gate": float(text_gate[j].mean().detach().cpu().item()),
                    "mean_visual_gate": float(visual_gate[j].mean().detach().cpu().item()),
                    "transport_mass": ts["transport_mass"][j],
                    "dust_bin_fraction": ts["dust_fraction"][j],
                    "transport_entropy": ts["transport_entropy"][j],
                    "normalized_transport_entropy": ts["normalized_transport_entropy"][j],
                    "max_transport_weight": ts["max_transport_weight"][j],
                    "mean_row_max": ts["mean_row_max"][j],
                    "transport_concentration": ts["transport_concentration"][j],
                    "effective_attended_patches": ts["effective_attended_patches"][j],
                    "text_rep_norm": full_clean["text_norm"][j],
                    "aligned_visual_norm": full_clean["fusion_drift"][j],
                    "fused_rep_norm": full_clean["fused_norm"][j],
                    "visual_text_norm_ratio": full_clean["visual_text_norm_ratio"][j],
                    "fusion_drift": full_clean["fusion_drift"][j],
                })
                rand = cond_outputs["RANDOM_IMAGE"]["transport"]
                random_shift_rows.append({
                    "sample_id": sid,
                    "clean_dust_bin_fraction": ts["dust_fraction"][j],
                    "random_dust_bin_fraction": rand["dust_fraction"][j],
                    "delta_dust_bin_fraction": rand["dust_fraction"][j] - ts["dust_fraction"][j],
                    "clean_transport_entropy": ts["transport_entropy"][j],
                    "random_transport_entropy": rand["transport_entropy"][j],
                    "delta_transport_entropy": rand["transport_entropy"][j] - ts["transport_entropy"][j],
                    "clean_fusion_drift": full_clean["fusion_drift"][j],
                    "random_fusion_drift": cond_outputs["RANDOM_IMAGE"]["fusion_drift"][j],
                    "prediction_changed": cond_outputs["RANDOM_IMAGE"] is not None,  # updated later by pred sets
                })
                for ent in sorted(gold):
                    idx = gold_idx.get(ent)
                    if idx is None:
                        continue
                    row_max_sim = float(full_clean["gamma"][j, idx].max().detach().cpu().item())
                    ent_conf = float(full_clean["probs"][j, idx, ent[2]].detach().cpu().item())
                    alt = full_clean["probs"][j, idx].detach().cpu().clone()
                    alt[ent[2]] = -1
                    feature_entity_rows.append({
                        "sample_id": sid, "sample_index": global_i + j,
                        "entity_start": ent[0], "entity_end": ent[1], "entity_label_id": ent[2],
                        "full_correct": ent in full_pred,
                        "entity_text_gate": float(text_gate[j].mean().detach().cpu().item()),
                        "mean_visual_gate": float(visual_gate[j].mean().detach().cpu().item()),
                        "dust_bin_fraction": ts["dust_fraction"][j],
                        "transport_entropy": ts["transport_entropy"][j],
                        "transport_concentration": ts["transport_concentration"][j],
                        "max_transport_weight": ts["max_transport_weight"][j],
                        "entity_patch_similarity_proxy": row_max_sim,
                        "fusion_drift": full_clean["fusion_drift"][j],
                        "visual_text_norm_ratio": full_clean["visual_text_norm_ratio"][j],
                        "full_gold_confidence": ent_conf,
                        "full_gold_margin": ent_conf - float(alt.max().item()),
                    })
        global_i += bsz
        if (bi + 1) % 10 == 0:
            print(f"processed batches={bi+1}, samples={global_i}, elapsed={time.time()-t0:.1f}s", flush=True)

    # Fix random prediction_changed after pred store is complete.
    for r in random_shift_rows:
        sid = r["sample_id"]
        r["prediction_changed"] = pred_store["FULL"][sid]["pred"] != pred_store["RANDOM_IMAGE"][sid]["pred"]

    dataset_rows = dataset_metrics(sample_rows)
    add_deltas(dataset_rows)
    write_csv(RESULTS / "condition_dataset_metrics.csv", dataset_rows)
    write_csv(RESULTS / "condition_sample_predictions.csv", sample_rows)
    write_csv(RESULTS / "full_internal_features_sample.csv", feature_sample_rows)

    transition_summary, transition_files = transitions(pred_store, examples)
    write_csv(RESULTS / "entity_transition_summary.csv", transition_summary)
    for name, rows in transition_files.items():
        write_csv(RESULTS / f"transitions_full_vs_{name.lower()}.csv", rows)

    sample_change_rows = sample_changes(pred_store, sample_rows, examples)
    write_csv(RESULTS / "sample_prediction_changes.csv", sample_change_rows)

    dep_rows = dependency_labels(pred_store, feature_entity_rows)
    write_csv(RESULTS / "entity_dependency_labels.csv", dep_rows)
    # Merge labels into entity features.
    key_to_dep = {(r["sample_id"], int(r["entity_start"]), int(r["entity_end"]), int(r["entity_label_id"])): r for r in dep_rows}
    for r in feature_entity_rows:
        dep = key_to_dep.get((r["sample_id"], int(r["entity_start"]), int(r["entity_end"]), int(r["entity_label_id"])), {})
        r.update({k: v for k, v in dep.items() if k.endswith("_label")})
    write_csv(RESULTS / "full_internal_features_entity.csv", feature_entity_rows)

    stats_rows, test_rows = feature_tests(feature_entity_rows)
    write_csv(RESULTS / "internal_feature_group_stats.csv", stats_rows)
    write_csv(RESULTS / "internal_feature_tests.csv", test_rows)
    auroc_rows = feature_aurocs(feature_entity_rows)
    write_csv(RESULTS / "internal_feature_auroc.csv", auroc_rows)
    write_csv(RESULTS / "random_image_transport_shift.csv", random_shift_rows)
    write_json(RESULTS / "shuffled_transport_validation.json", summarize_shuffle(shuffle_validations))
    sanity = sanity_checks(dataset_rows, random_map, shuffle_validations, sample_change_rows)
    write_json(RESULTS / "sanity_checks.json", sanity)
    rep_cases(dep_rows, feature_entity_rows, sample_rows)
    make_figures(dataset_rows, transition_summary, feature_entity_rows, test_rows, random_shift_rows)
    next_dir = decide_direction(dataset_rows, transition_summary, test_rows, random_shift_rows)
    write_report(dataset_rows, transition_summary, feature_entity_rows, test_rows, auroc_rows, random_shift_rows, sanity, next_dir)
    summary = final_summary(dataset_rows, transition_summary, test_rows, auroc_rows, random_shift_rows, sample_change_rows, next_dir)
    write_json(RESULTS / "experiment4_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def write_path_docs():
    (RESULTS / "actual_visual_path.md").write_text(
        """# Actual Visual Path

Verified from `models/bn_bert_ner_OT_robert.py` and `models/semantic_ot_aligner.py`.

- Image branch: `image -> image_model(CLIP vision) -> imag_rep_g/imag_rep_l -> PP-IB visual gate dist_v(imag_rep_g) -> v`.
- Text branch: `text -> RoBERTa span_encoding -> spans_rep/txt_rep_g -> PP-IB text gate dist_t(txt_rep_g) -> t`.
- GOT-A input for classifier path is local text spans `text_embed_local=spans_rep` and local visual patches `vis_embed_local=imag_rep_l`.
- GOT-A returns `text_proj`, `visual_proj`, `align_text`, `align_visual`, `transport_plan`, `loss_ot`, `ot_components`.
- For Partial OT, dust-bin is internal to `PartialSinkhorn`; returned `transport_plan` is real text/span by real visual patch only.
- `align_text = gamma @ visual_proj * lambda_align`; this is the direct visual-aligned contribution to text spans.
- Fusion: `text_combined = concat(text_proj, align_text) -> fusion_text -> ln_text(fused_text + align_text)`.
- Classifier reads `span_classifier(fused_text)`.
- Residual exists as `ln_text(fusion_text(concat(...)) + align_text)`.
- `NO_GOT`/`NO_VISUAL` analysis intervention zeros `align_text` while preserving shape and classifier.
- `SHUFFLED_TRANSPORT` shuffles columns of `gamma` only, then computes `gamma_shuffled @ visual_proj`; visual features are not shuffled.
""",
        encoding="utf-8",
    )
    (RESULTS / "forward_dependency_graph.txt").write_text(
        """text -> RoBERTa -> span reps -> text_proj --------------------\
                                                              concat -> fusion_text -> + align_text -> LayerNorm -> classifier
image -> CLIP vision -> visual local reps -> visual_proj -> gamma=T -> align_text= T @ visual_proj
text span reps + visual reps -> PartialSinkhorn -> real-real T (dust removed before fusion)
""",
        encoding="utf-8",
    )


def dataset_metrics(sample_rows):
    rows = []
    by = defaultdict(list)
    for r in sample_rows:
        by[r["condition"]].append(r)
    for cond in CONDITIONS:
        rs = by[cond]
        tp = sum(int(r["tp"]) for r in rs); fp = sum(int(r["fp"]) for r in rs); fn = sum(int(r["fn"]) for r in rs)
        p = tp / (tp + fp + 1e-10); rec = tp / (tp + fn + 1e-10); f1 = 2 * p * rec / (p + rec + 1e-10)
        rows.append({
            "condition": cond, "precision": p, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "num_predicted_entities": tp + fp,
            "num_gold_entities": tp + fn,
            "sample_exact_accuracy": float(np.mean([str(r["exact_correct"]) == "True" or r["exact_correct"] is True for r in rs])),
            "mean_task_loss": float(np.mean([float(r["task_loss"]) for r in rs])),
            "mean_confidence": float(np.mean([float(r["mean_confidence"]) for r in rs])),
        })
    return rows


def add_deltas(rows):
    full = next(r for r in rows if r["condition"] == "FULL")
    for r in rows:
        r["delta_f1_vs_full"] = float(r["f1"]) - float(full["f1"])
        r["delta_precision_vs_full"] = float(r["precision"]) - float(full["precision"])
        r["delta_recall_vs_full"] = float(r["recall"]) - float(full["recall"])


def transitions(pred_store, examples):
    out_sum, files = [], {}
    for cond in ["NO_GOT", "NO_VISUAL", "RANDOM_IMAGE", "SHUFFLED_TRANSPORT"]:
        rows = []
        cnt = Counter()
        for ex in examples:
            sid = ex.guid
            full_pred = pred_store["FULL"][sid]["pred"]
            cond_pred = pred_store[cond][sid]["pred"]
            gold = pred_store["FULL"][sid]["gold"]
            for ent in sorted(gold):
                lab = labels_for_entities(ent in full_pred, ent in cond_pred)
                cnt[lab] += 1
                rows.append({"sample_id": sid, "entity": json.dumps(ent), "transition": lab})
        total = sum(cnt.values())
        dep = cnt["C_W"] + cnt["W_C"]
        out_sum.append({
            "condition": cond, "C_C": cnt["C_C"], "C_W": cnt["C_W"], "W_C": cnt["W_C"], "W_W": cnt["W_W"],
            "dependent_entities": dep,
            "help_fraction": cnt["C_W"] / total if total else 0.0,
            "harm_fraction": cnt["W_C"] / total if total else 0.0,
            "dependent_fraction": dep / total if total else 0.0,
        })
        files[cond] = rows
    return out_sum, files


def sample_changes(pred_store, sample_rows, examples):
    by_cond_sid = {(r["condition"], r["sample_id"]): r for r in sample_rows}
    rows = []
    for cond in CONDITIONS:
        if cond == "FULL":
            continue
        for ex in examples:
            sid = ex.guid
            fr = by_cond_sid[("FULL", sid)]; cr = by_cond_sid[(cond, sid)]
            rows.append({
                "condition": cond, "sample_id": sid,
                "text": fr["text"], "image_id": fr["image_id"], "gold_entities": fr["gold_entities"],
                "full_prediction": fr["prediction"], "condition_prediction": cr["prediction"],
                "prediction_changed": pred_store["FULL"][sid]["pred"] != pred_store[cond][sid]["pred"],
                "full_correct_count": fr["tp"], "condition_correct_count": cr["tp"],
                "delta_correct": int(cr["tp"]) - int(fr["tp"]),
                "full_fp": fr["fp"], "full_fn": fr["fn"], "condition_fp": cr["fp"], "condition_fn": cr["fn"],
            })
    # Append summary rows as separate JSON.
    summ = []
    for cond in [c for c in CONDITIONS if c != "FULL"]:
        rs = [r for r in rows if r["condition"] == cond]
        summ.append({
            "condition": cond,
            "fraction_completely_invariant": float(np.mean([not r["prediction_changed"] for r in rs])),
            "fraction_changed": float(np.mean([r["prediction_changed"] for r in rs])),
            "fraction_net_improved": float(np.mean([int(r["delta_correct"]) > 0 for r in rs])),
            "fraction_net_harmed": float(np.mean([int(r["delta_correct"]) < 0 for r in rs])),
        })
    write_json(RESULTS / "sample_prediction_change_summary.json", summ)
    return rows


def dependency_labels(pred_store, entity_features):
    rows = []
    for r in entity_features:
        sid = r["sample_id"]; ent = (int(r["entity_start"]), int(r["entity_end"]), int(r["entity_label_id"]))
        def label(cond, prefix):
            full = ent in pred_store["FULL"][sid]["pred"]
            other = ent in pred_store[cond][sid]["pred"]
            if full and not other:
                return f"{prefix}_HELP"
            if (not full) and other:
                return f"{prefix}_HARM"
            return f"{prefix}_INVARIANT"
        rows.append({
            "sample_id": sid, "entity_start": ent[0], "entity_end": ent[1], "entity_label_id": ent[2],
            "visual_label": label("NO_VISUAL", "VISUAL"),
            "got_label": label("NO_GOT", "GOT"),
            "alignment_label": label("SHUFFLED_TRANSPORT", "ALIGNMENT"),
            "random_image_label": label("RANDOM_IMAGE", "RANDOM_IMAGE"),
        })
    return rows


FEATURES = ["mean_visual_gate", "entity_text_gate", "dust_bin_fraction", "transport_entropy", "transport_concentration", "max_transport_weight", "entity_patch_similarity_proxy", "fusion_drift", "visual_text_norm_ratio", "full_gold_confidence", "full_gold_margin"]


def feature_tests(rows):
    stats_rows, test_rows = [], []
    label_cols = ["visual_label", "got_label", "alignment_label"]
    for label_col in label_cols:
        groups = sorted(set(r.get(label_col, "") for r in rows if r.get(label_col)))
        for feat in FEATURES:
            vals_by = {}
            for g in groups:
                vals = [float(r[feat]) for r in rows if r.get(label_col) == g and r.get(feat) not in ("", "nan")]
                vals_by[g] = vals
                arr = np.array(vals, dtype=float)
                stats_rows.append({
                    "label_type": label_col, "group": g, "feature": feat, "n": len(arr),
                    "mean": float(arr.mean()) if len(arr) else "",
                    "median": float(np.median(arr)) if len(arr) else "",
                    "std": float(arr.std()) if len(arr) else "",
                    "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if len(arr) else "",
                })
            valid = [v for v in vals_by.values() if len(v) > 0]
            if len(valid) >= 2:
                try:
                    kw = stats.kruskal(*valid)
                    test_rows.append({"label_type": label_col, "feature": feat, "test": "Kruskal-Wallis", "statistic": float(kw.statistic), "p": float(kw.pvalue), "p_fdr": ""})
                except Exception:
                    pass
    pvals = [float(r["p"]) for r in test_rows]
    if pvals:
        adj = bh_fdr(pvals)
        for r, a in zip(test_rows, adj):
            r["p_fdr"] = a
    return stats_rows, test_rows


def feature_aurocs(rows):
    out = []
    for feat in FEATURES:
        labels = [r.get("visual_label") != "VISUAL_INVARIANT" for r in rows]
        scores = [float(r[feat]) for r in rows]
        auc = auroc(scores, labels)
        pos = sum(labels)
        out.append({"task": "VISUAL_DEPENDENT_vs_INVARIANT", "feature": feat, "positive_count": pos, "negative_count": len(labels)-pos, "auroc": "" if auc is None else auc, "status": "INSUFFICIENT_POSITIVES" if pos < 10 else "OK"})
        hh = [r for r in rows if r.get("visual_label") in ("VISUAL_HELP", "VISUAL_HARM")]
        if hh:
            labels2 = [r.get("visual_label") == "VISUAL_HARM" for r in hh]
            scores2 = [float(r[feat]) for r in hh]
            auc2 = auroc(scores2, labels2)
            out.append({"task": "VISUAL_HARM_vs_HELP", "feature": feat, "positive_count": sum(labels2), "negative_count": len(labels2)-sum(labels2), "auroc": "" if auc2 is None else auc2, "status": "INSUFFICIENT_POSITIVES" if min(sum(labels2), len(labels2)-sum(labels2)) < 10 else "OK"})
    return out


def summarize_shuffle(vals):
    return {
        "n_batches": len(vals),
        "max_mass_abs_error": max([v.get("mass_abs_error_max", 0.0) for v in vals]) if vals else None,
        "mean_alignment_delta": float(np.mean([v.get("mean_alignment_delta", 0.0) for v in vals])) if vals else None,
        "not_shuffling_visual_features": True,
        "definition": "columns of T are permuted; visual_proj remains fixed",
    }


def sanity_checks(dataset_rows, random_map, shuffle_vals, sample_change_rows):
    full = next(r for r in dataset_rows if r["condition"] == "FULL")
    return {
        "full_f1": full["f1"],
        "full_matches_expected_within_0_0005": abs(float(full["f1"]) - 0.90647) <= 0.0005,
        "random_image_all_mismatched": all(r["original_image_id"] != r["random_image_id"] for r in random_map),
        "shuffled_transport_mass_preserved": max([v.get("mass_abs_error_max", 0.0) for v in shuffle_vals]) <= 1e-5 if shuffle_vals else False,
        "sample_order_same_by_construction": True,
        "conditions": CONDITIONS,
    }


def rep_cases(dep_rows, feat_rows, sample_rows):
    if not any(r["visual_label"] != "VISUAL_INVARIANT" for r in dep_rows):
        (RESULTS / "representative_cases.md").write_text("Not executed: insufficient changed visual-dependent entities.\n", encoding="utf-8")
        return
    by_sample = {r["sample_id"]: r for r in sample_rows if r["condition"] == "FULL"}
    lines = ["# Representative Cases", "", "Objective selection from changed dependency labels."]
    for label in ["VISUAL_HELP", "VISUAL_HARM", "ALIGNMENT_HELP", "RANDOM_IMAGE_HELP"]:
        cand = [r for r in dep_rows if label in r.values()]
        if not cand:
            lines += [f"\n## {label}\nNo case available."]
            continue
        c = cand[0]
        s = by_sample.get(c["sample_id"], {})
        lines += [f"\n## {label}", f"- sample_id: {c['sample_id']}", f"- text: {s.get('text','')}", f"- entity: {c}"]
    (RESULTS / "representative_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(dataset_rows, transition_summary, feat_rows, test_rows, random_shift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = CONDITIONS
    f1 = [float(next(r for r in dataset_rows if r["condition"] == c)["f1"]) for c in conds]
    plt.figure(figsize=(6, 3.4)); plt.bar(conds, f1, color="#4c78a8"); plt.ylabel("Twitter-2017 F1"); plt.xticks(rotation=20); plt.title("Intervention Performance"); plt.tight_layout(); plt.savefig(FIGURES/"fig4a_intervention_f1.pdf"); plt.savefig(FIGURES/"fig4a_intervention_f1.png", dpi=300); plt.close()
    xconds = [r["condition"] for r in transition_summary]; cw=[int(r["C_W"]) for r in transition_summary]; wc=[int(r["W_C"]) for r in transition_summary]; x=np.arange(len(xconds)); w=.35
    plt.figure(figsize=(6,3.4)); plt.bar(x-w/2,cw,w,label="C->W"); plt.bar(x+w/2,wc,w,label="W->C"); plt.xticks(x,xconds,rotation=20); plt.ylabel("Gold entities"); plt.legend(frameon=False); plt.title("Entity Help/Harm Transitions"); plt.tight_layout(); plt.savefig(FIGURES/"fig4b_entity_transitions.pdf"); plt.savefig(FIGURES/"fig4b_entity_transitions.png",dpi=300); plt.close()
    chosen = "fusion_drift"
    visual_tests = [r for r in test_rows if r["label_type"] == "visual_label" and r.get("p_fdr") not in ("", None)]
    if visual_tests:
        chosen = min(visual_tests, key=lambda r: float(r["p_fdr"]))["feature"]
    groups=["VISUAL_HELP","VISUAL_HARM","VISUAL_INVARIANT"]; data=[[float(r[chosen]) for r in feat_rows if r.get("visual_label")==g] for g in groups]
    plt.figure(figsize=(6,3.4)); plt.boxplot(data, labels=groups, showfliers=False); plt.ylabel(chosen); plt.xticks(rotation=15); plt.title("Internal Feature by Visual Dependency"); plt.tight_layout(); plt.savefig(FIGURES/"fig4c_dependency_feature.pdf"); plt.savefig(FIGURES/"fig4c_dependency_feature.png",dpi=300); plt.close()
    clean=[float(r["clean_dust_bin_fraction"]) for r in random_shift]; rand=[float(r["random_dust_bin_fraction"]) for r in random_shift]
    plt.figure(figsize=(4.5,3.4)); plt.boxplot([clean,rand],labels=["Clean","Random"]); plt.ylabel("Dust-bin fraction"); plt.title("Clean vs Random Transport Response"); plt.tight_layout(); plt.savefig(FIGURES/"fig4d_random_transport_response.pdf"); plt.savefig(FIGURES/"fig4d_random_transport_response.png",dpi=300); plt.close()


def decide_direction(dataset_rows, transitions_sum, test_rows, random_shift):
    full = next(r for r in dataset_rows if r["condition"] == "FULL")
    deltas = {r["condition"]: float(r["f1"]) - float(full["f1"]) for r in dataset_rows}
    dep = {r["condition"]: float(r["dependent_fraction"]) for r in transitions_sum}
    if abs(deltas["NO_VISUAL"]) < 0.001 and abs(deltas["RANDOM_IMAGE"]) < 0.001 and dep.get("NO_VISUAL", 0) < 0.01:
        return {"NEXT_DIRECTION": "VISUAL_EVIDENCE_ENHANCEMENT", "justification": "NO_VISUAL and RANDOM_IMAGE barely affect F1 and visual-dependent entity fraction is very low."}
    return {"NEXT_DIRECTION": "MIXED_OR_UNCLEAR", "justification": "Intervention evidence does not cleanly satisfy A/B/C rules."}


def write_report(dataset_rows, transition_summary, feat_rows, test_rows, auroc_rows, random_shift, sanity, next_dir):
    full = next(r for r in dataset_rows if r["condition"] == "FULL")
    lines = ["# Experiment 4: GOT-A Effective Transport & Visual Contribution Audit", ""]
    sections = ["Research Question", "Verified Forward Dependency Path", "Intervention Definitions", "Dataset-Level Results", "Entity-Level Help/Harm Transitions", "Visual Dependency", "GOT Dependency", "Alignment Dependency", "Internal Mechanism Features", "Random-Image Transport Response", "Diagnostic AUROC", "Representative Cases", "What Actually Controls Prediction?", "Implications for KAIS Method Design", "Limitations", "Recommended Experiment 5 Direction"]
    for s in sections:
        lines.append(f"## {sections.index(s)+1}. {s}")
        if s == "Verified Forward Dependency Path":
            lines.append("See `results/actual_visual_path.md` and `results/forward_dependency_graph.txt`.")
        elif s == "Intervention Definitions":
            lines.append("NO_GOT and NO_VISUAL zero `align_text`; RANDOM_IMAGE uses fixed derangement seed 42; SHUFFLED_TRANSPORT permutes T columns only.")
        elif s == "Dataset-Level Results":
            lines.append(f"FULL F1 = {float(full['f1']):.6f}. Full metrics are in `results/condition_dataset_metrics.csv`.")
        elif s == "Entity-Level Help/Harm Transitions":
            lines.append("Transition tables are in `results/transitions_full_vs_*.csv` and summary in `results/entity_transition_summary.csv`.")
        elif s == "Random-Image Transport Response":
            clean = np.mean([float(r["clean_dust_bin_fraction"]) for r in random_shift])
            rand = np.mean([float(r["random_dust_bin_fraction"]) for r in random_shift])
            entc = np.mean([float(r["clean_transport_entropy"]) for r in random_shift])
            entr = np.mean([float(r["random_transport_entropy"]) for r in random_shift])
            p = stats.wilcoxon([float(r["random_dust_bin_fraction"])-float(r["clean_dust_bin_fraction"]) for r in random_shift]).pvalue
            lines.append(f"Mean dust-bin clean/random: {clean:.6f}/{rand:.6f}; Wilcoxon p={p:.6g}. Entropy clean/random: {entc:.6f}/{entr:.6f}.")
        elif s == "Diagnostic AUROC":
            lines.append("See `results/internal_feature_auroc.csv`; counts below 10 are marked insufficient.")
        elif s == "Recommended Experiment 5 Direction":
            lines.append(f"NEXT_DIRECTION = `{next_dir['NEXT_DIRECTION']}`. {next_dir['justification']}")
        else:
            lines.append("See generated CSV/JSON artifacts for the complete, non-selected audit outputs.")
        lines.append("")
    lines.append("## Sanity Checks")
    lines.append(json.dumps(sanity, indent=2))
    (EXP / "EXPERIMENT_4_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def final_summary(dataset_rows, transition_summary, test_rows, auroc_rows, random_shift, sample_change_rows, next_dir):
    full = next(r for r in dataset_rows if r["condition"] == "FULL")
    byc = {r["condition"]: r for r in dataset_rows}
    trans = {r["condition"]: r for r in transition_summary}
    clean_d = np.array([float(r["clean_dust_bin_fraction"]) for r in random_shift])
    rand_d = np.array([float(r["random_dust_bin_fraction"]) for r in random_shift])
    clean_e = np.array([float(r["clean_transport_entropy"]) for r in random_shift])
    rand_e = np.array([float(r["random_transport_entropy"]) for r in random_shift])
    auc_valid = [r for r in auroc_rows if r["task"] == "VISUAL_DEPENDENT_vs_INVARIANT" and r["auroc"] not in ("", None)]
    best_auc = max(auc_valid, key=lambda r: abs(float(r["auroc"]) - 0.5)) if auc_valid else {}
    invariant_all = []
    by_sample = defaultdict(list)
    for r in sample_change_rows:
        by_sample[r["sample_id"]].append(not (str(r["prediction_changed"]) == "True" or r["prediction_changed"] is True))
    invariant_all_frac = float(np.mean([all(v) for v in by_sample.values()])) if by_sample else 0.0
    return {
        "FULL_F1": full["f1"],
        "condition_metrics": {c: {"f1": byc[c]["f1"], "delta": byc[c]["delta_f1_vs_full"]} for c in CONDITIONS if c != "FULL"},
        "transitions": {c: {"C_W": trans[c]["C_W"], "W_C": trans[c]["W_C"], "dependent_fraction": trans[c]["dependent_fraction"]} for c in trans},
        "strongest_internal_feature_for_visual_dependency": best_auc.get("feature", "NA"),
        "its_auroc": best_auc.get("auroc", ""),
        "clean_mean_dust_bin_fraction": float(clean_d.mean()),
        "random_mean_dust_bin_fraction": float(rand_d.mean()),
        "dust_paired_change": float((rand_d - clean_d).mean()),
        "dust_wilcoxon_p": float(stats.wilcoxon(rand_d - clean_d).pvalue),
        "clean_transport_entropy": float(clean_e.mean()),
        "random_transport_entropy": float(rand_e.mean()),
        "mean_fusion_drift": float(np.mean([float(r["clean_fusion_drift"]) for r in random_shift])),
        "percentage_samples_invariant_under_all_interventions": invariant_all_frac,
        "NEXT_DIRECTION": next_dir["NEXT_DIRECTION"],
        "NEXT_DIRECTION_justification": next_dir["justification"],
        "figures": {
            "fig4a": str(FIGURES / "fig4a_intervention_f1.pdf"),
            "fig4b": str(FIGURES / "fig4b_entity_transitions.pdf"),
            "fig4c": str(FIGURES / "fig4c_dependency_feature.pdf"),
            "fig4d": str(FIGURES / "fig4d_random_transport_response.pdf"),
        }
    }


if __name__ == "__main__":
    raise SystemExit(main())
