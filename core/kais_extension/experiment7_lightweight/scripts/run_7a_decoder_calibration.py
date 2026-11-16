#!/usr/bin/env python3
"""Experiment 7A: dev-calibrated span decoding for the ICDM checkpoint."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, RobertaTokenizer

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "kais_extension" / "experiment7_lightweight"
CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
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

LABELS = ["O", "LOC", "MISC", "ORG", "PER"]
ENTITY_LABELS = ["LOC", "MISC", "ORG", "PER"]


def load_main(extra: List[str]):
    path = ROOT / "main-mner-robert.py"
    spec = importlib.util.spec_from_file_location("main_mner_robert_exp7a", path)
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


def build_model(mod, labels):
    from models.bn_bert_ner_OT_robert import BertSpanNerBN
    args = mod.args
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None,
    )
    model = BertSpanNerBN(config, args=args, num_labels=len(labels))
    model.to(args.device)
    missing, unexpected = model.load_state_dict(load_state(CKPT, args.device), strict=False)
    print(f"loaded ICDM checkpoint missing={len(missing)} unexpected={len(unexpected)}")
    return model


def cache_split(mod, model, tokenizer, labels, split: str, out_path: Path) -> Dict[str, torch.Tensor]:
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
    ds = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=8, collate_fn=collate_fn)
    probs, spans, gold, masks = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"cache {split}"):
            tensors = {k: v.to(args.device) for k, v in batch.items()}
            pred, _ = model(tensors, only_test=False)
            probs.append(pred[0].detach().cpu())
            spans.append(tensors["span_word_idxes"].detach().cpu())
            gold.append(tensors["span_labels"].detach().cpu())
            masks.append(tensors["span_masks"].detach().cpu())
    data = {
        "probs": torch.cat(probs, 0),
        "spans": torch.cat(spans, 0),
        "gold": torch.cat(gold, 0),
        "masks": torch.cat(masks, 0),
        "labels": labels,
    }
    torch.save(data, out_path)
    print(f"saved cache {split}: {out_path}")
    return data


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def decode_counts(
    data: Dict[str, torch.Tensor],
    score_type: str = "prob",
    thresholds: Optional[Dict[int, float]] = None,
    overlap_rule: str = "original",
    class_prior: Optional[Dict[int, float]] = None,
) -> Tuple[Tuple[int, int, int], Dict[str, Dict[str, int]]]:
    probs = data["probs"]
    spans = data["spans"]
    gold = data["gold"]
    masks = data["masks"]
    if thresholds is None:
        thresholds = {i: -1e9 for i in range(1, len(LABELS))}
    tp = pred_total = gold_total = 0
    per = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in ENTITY_LABELS}
    raw = probs.argmax(dim=-1)
    for b in range(probs.shape[0]):
        candidates = []
        gold_set = set()
        for i in range(probs.shape[1]):
            if int(masks[b, i]) <= 0:
                continue
            ws, we = int(spans[b, i, 0]), int(spans[b, i, 1])
            gl = int(gold[b, i])
            if gl != 0:
                gold_set.add((ws, we, gl))
            rp = int(raw[b, i])
            if rp == 0:
                continue
            prob = float(probs[b, i, rp])
            margin = prob - float(probs[b, i, 0])
            score = prob if score_type == "prob" else margin
            if score < float(thresholds.get(rp, -1e9)):
                continue
            length = max(1, we - ws + 1)
            if overlap_rule == "original" or overlap_rule == "confidence":
                rank = score
            elif overlap_rule == "confidence_sqrt_len":
                rank = score * math.sqrt(length)
            elif overlap_rule == "confidence_prior":
                rank = score + float((class_prior or {}).get(rp, 0.0))
            else:
                raise ValueError(overlap_rule)
            candidates.append({"i": i, "ws": ws, "we": we, "label": rp, "score": score, "rank": rank})
        candidates.sort(key=lambda x: -x["rank"])
        chosen = []
        for c in candidates:
            if not any(spans_overlap((c["ws"], c["we"]), (ch["ws"], ch["we"])) for ch in chosen):
                chosen.append(c)
        pred_set = {(c["ws"], c["we"], c["label"]) for c in chosen}
        tp += len(pred_set & gold_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)
        for lab_i, lab in enumerate(LABELS):
            if lab_i == 0:
                continue
            ps = {x for x in pred_set if x[2] == lab_i}
            gs = {x for x in gold_set if x[2] == lab_i}
            per[lab]["tp"] += len(ps & gs)
            per[lab]["fp"] += len(ps - gs)
            per[lab]["fn"] += len(gs - ps)
    return (tp, pred_total, gold_total), per


def prf(counts: Tuple[int, int, int]) -> Dict[str, float]:
    tp, pred, gold = counts
    p = tp / (pred + 1e-10)
    r = tp / (gold + 1e-10)
    f = 2 * p * r / (p + r + 1e-10)
    return {"precision": round(p, 5), "recall": round(r, 5), "f1": round(f, 5), "tp": tp, "pred": pred, "gold": gold}


def per_type_metrics(per_counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for lab, c in per_counts.items():
        p = c["tp"] / (c["tp"] + c["fp"] + 1e-10)
        r = c["tp"] / (c["tp"] + c["fn"] + 1e-10)
        f = 2 * p * r / (p + r + 1e-10)
        out[lab] = {"precision": round(p, 5), "recall": round(r, 5), "f1": round(f, 5)}
    return out


def score_quantiles(data: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
    probs = data["probs"]
    masks = data["masks"].bool()
    raw = probs.argmax(dim=-1)
    ent = (raw != 0) & masks
    if not ent.any():
        return {"prob": [], "margin": []}
    prob = probs.gather(-1, raw.unsqueeze(-1)).squeeze(-1)[ent]
    margin = (prob - probs[..., 0][ent])
    qs = torch.tensor([0, .05, .1, .25, .5, .75, .9, .95, 1.0])
    return {
        "prob": [float(x) for x in torch.quantile(prob, qs)],
        "margin": [float(x) for x in torch.quantile(margin, qs)],
    }


def train_dev_prior(data: Dict[str, torch.Tensor]) -> Dict[int, float]:
    # Small log-prior centered around zero, estimated from dev gold labels only.
    gold = data["gold"]
    masks = data["masks"].bool()
    counts = torch.bincount(gold[masks].reshape(-1), minlength=len(LABELS)).float()[1:]
    freqs = (counts + 1.0) / (counts.sum() + len(counts))
    logp = torch.log(freqs)
    logp = logp - logp.mean()
    return {i + 1: float(0.01 * logp[i]) for i in range(len(counts))}


def eval_config(data, score_type, thresholds, overlap_rule, prior=None):
    counts, per = decode_counts(data, score_type=score_type, thresholds=thresholds, overlap_rule=overlap_rule, class_prior=prior)
    m = prf(counts)
    return m, per_type_metrics(per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh_cache", action="store_true")
    args_cli = ap.parse_args()
    for sub in ["scripts", "configs", "logs", "results", "checkpoints", "figures"]:
        (EXP / sub).mkdir(parents=True, exist_ok=True)
    out_dir = EXP / "checkpoints" / "original_icdm"
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_main(["--output_dir", str(out_dir)])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    labels = mod.get_labels(mod.args.labels)
    tokenizer = RobertaTokenizer.from_pretrained(mod.args.tokenizer_name if mod.args.tokenizer_name else mod.args.model_name_or_path)
    cache_dev = EXP / "results" / "cached_dev_predictions.pt"
    cache_test = EXP / "results" / "cached_test_predictions.pt"
    if args_cli.refresh_cache or not cache_dev.exists() or not cache_test.exists():
        model = build_model(mod, labels)
        dev = cache_split(mod, model, tokenizer, labels, "dev", cache_dev)
        test = cache_split(mod, model, tokenizer, labels, "test", cache_test)
    else:
        dev = torch.load(cache_dev, map_location="cpu", weights_only=False)
        test = torch.load(cache_test, map_location="cpu", weights_only=False)
    audit = {
        "candidate_score_original": "argmax class softmax probability p[argmax]",
        "none_handling": "argmax label 0 is O/non-entity and is excluded",
        "confidence_threshold": "none in original decoder",
        "overlap_resolution": "predicted non-O spans sorted by descending p[argmax], greedy non-overlap",
        "same_span_label": "single argmax over all labels including O",
        "tunable_parameters_original": "none",
        "dev_score_quantiles": score_quantiles(dev),
    }
    (EXP / "results" / "decoder_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    rows = []
    per_rows = []
    base_th = {i: -1e9 for i in range(1, len(LABELS))}
    prior = train_dev_prior(dev)
    configs = []
    # Original
    m_dev, per_dev = eval_config(dev, "prob", base_th, "original")
    m_test, per_test = eval_config(test, "prob", base_th, "original")
    configs.append(("original", m_dev["f1"], "prob", base_th, "original", m_test, per_test))

    threshold_grid = [round(x * 0.05, 2) for x in range(0, 11)]
    best_globals = []
    for score_type in ["prob", "margin"]:
        best = None
        for tau in threshold_grid:
            th = {i: tau for i in range(1, len(LABELS))}
            m, _ = eval_config(dev, score_type, th, "original")
            cand = (m["f1"], tau, score_type, th)
            if best is None or cand[0] > best[0]:
                best = cand
        assert best is not None
        m_test, per_test = eval_config(test, best[2], best[3], "original")
        name = f"global_{best[2]}"
        configs.append((name, best[0], best[2], best[3], "original", m_test, per_test))
        best_globals.append(best)
    global_best = max(best_globals, key=lambda x: x[0])

    # Class-specific coordinate search from dev-selected global threshold.
    score_type = global_best[2]
    th = dict(global_best[3])
    base_tau = list(th.values())[0]
    local = sorted(set(round(max(-0.5, min(1.0, base_tau + d)), 2) for d in [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]))
    best_f1 = global_best[0]
    for _round in range(2):
        changed = False
        for cls in range(1, len(LABELS)):
            cls_best = (best_f1, th[cls])
            for tau in local:
                cand_th = dict(th)
                cand_th[cls] = tau
                m, _ = eval_config(dev, score_type, cand_th, "original")
                if m["f1"] > cls_best[0]:
                    cls_best = (m["f1"], tau)
            if cls_best[1] != th[cls]:
                th[cls] = cls_best[1]
                best_f1 = cls_best[0]
                changed = True
        if not changed:
            break
    m_test, per_test = eval_config(test, score_type, th, "original")
    configs.append(("class_threshold", best_f1, score_type, dict(th), "original", m_test, per_test))

    # Overlap rules with class-threshold settings.
    for rule in ["confidence", "confidence_sqrt_len", "confidence_prior"]:
        m_dev, _ = eval_config(dev, score_type, th, rule, prior)
        m_test, per_test = eval_config(test, score_type, th, rule, prior)
        configs.append((f"overlap_{rule}", m_dev["f1"], score_type, dict(th), rule, m_test, per_test))

    for name, dev_f1, st, ths, rule, mt, per_test in configs:
        row = {
            "decoder": name,
            "dev_f1": dev_f1,
            "test_f1": mt["f1"],
            "test_precision": mt["precision"],
            "test_recall": mt["recall"],
            "delta_vs_0.90647": round(mt["f1"] - REF_F1, 5),
            "score_type": st,
            "global_threshold": "" if len(set(ths.values())) > 1 else list(ths.values())[0],
            "PER_threshold": ths.get(4, ""),
            "LOC_threshold": ths.get(1, ""),
            "ORG_threshold": ths.get(3, ""),
            "MISC_threshold": ths.get(2, ""),
            "overlap_rule": rule,
        }
        rows.append(row)
        for lab, mm in per_test.items():
            per_rows.append({"decoder": name, "label": lab, **mm})
    best = max(rows, key=lambda r: float(r["dev_f1"]))
    (EXP / "results" / "best_7a.json").write_text(json.dumps(best, indent=2) + "\n")
    with (EXP / "results" / "decoder_calibration.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with (EXP / "results" / "decoder_entity_type_f1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        writer.writeheader(); writer.writerows(per_rows)
    print("BEST_7A")
    print(json.dumps(best, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
