#!/usr/bin/env python3
"""Official Exp14 scoring, DEV threshold lock, and TEST one-shot evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment14_gmner_integration"
E2E = Path("/data/hfy/VG-Think/E2E-GMNER")
sys.path.insert(0, str(E2E / "src"))
from e2egmner.evaluation.geometry import max_iou, rescale_assistant_text_bboxes  # noqa: E402
from e2egmner.evaluation.metrics import count_correct_eeg, count_correct_gmner, count_correct_mner, prf  # noqa: E402
from e2egmner.evaluation.parsing import EntityTriple, extract_answer_text, parse_triples, triples_to_canon_text  # noqa: E402

THRESHOLDS = [0.3, 0.5, 0.7]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    return p.parse_args()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]


def get_messages(row):
    user = next(m for m in row["messages"] if m["role"] == "user")
    assistant = next(m for m in row["messages"] if m["role"] == "assistant")
    return user["content"], assistant["content"]


def scaled_gold(row, index):
    _, answer = get_messages(row)
    image_rel = row["images"][0]
    with Image.open(E2E / "data" / image_rel) as im:
        w, h = im.size
    scaled = rescale_assistant_text_bboxes(
        answer, orig_w=w, orig_h=h, min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28, factor=28, do_jitter=False,
    )
    if scaled is None:
        raise RuntimeError(f"gold scaling failed at {index}")
    return parse_triples(extract_answer_text(scaled), strict=True, where=f"gold:{index}")


def cache_key(sample_index, entity):
    return f"{sample_index}\t{entity['mention']}\t{entity['official_type']}"


def predictions(sample, qwen, threshold=None):
    triples = []
    details = []
    for entity in sample["entities"]:
        key = cache_key(sample["sample_index"], entity)
        q = qwen.get(key)
        if q is None:
            raise RuntimeError(f"missing Qwen cache key {key}")
        routed_off = threshold is not None and float(entity["groundability_probability"]) < threshold
        regions = None if routed_off else q.get("pred_boxes")
        valid = bool(q.get("parse_valid", False)) or routed_off
        triple = EntityTriple(entity["mention"], entity["official_type"], regions, valid)
        triples.append(triple)
        details.append({**entity, "qwen_raw_generation": q["raw_generation"],
                        "qwen_parsed_kind": q["parsed_kind"], "qwen_boxes": q.get("pred_boxes"),
                        "routed_to_none": routed_off, "final_regions": regions})
    return triples, details


def score_dataset(pred_by_sample, gold_by_sample):
    counts = {k: {"correct": 0, "predicted": 0, "gold": 0} for k in ["mner", "eeg", "gmner"]}
    for preds, golds in zip(pred_by_sample, gold_by_sample):
        for k in counts:
            counts[k]["predicted"] += len(preds); counts[k]["gold"] += len(golds)
        counts["mner"]["correct"] += count_correct_mner(preds, golds)
        counts["eeg"]["correct"] += count_correct_eeg(preds, golds, 0.5)
        counts["gmner"]["correct"] += count_correct_gmner(preds, golds, 0.5)
    result = {}
    for name, c in counts.items():
        p, r, f = prf(c["correct"], c["predicted"], c["gold"])
        result[name] = {**c, "precision": p, "recall": r, "f1": f}
    return result


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i]); out = [0.0] * len(vals); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]: j += 1
        rank = (i + j + 2) / 2
        for k in range(i, j + 1): out[order[k]] = rank
        i = j + 1
    return out


def auroc(y, scores):
    pos = sum(y); neg = len(y) - pos
    if not pos or not neg: return float("nan")
    rs = ranks(scores)
    return (sum(r for r, yy in zip(rs, y) if yy) - pos * (pos + 1) / 2) / (pos * neg)


def aupr(y, scores):
    pos = sum(y)
    if not pos: return float("nan")
    tp = fp = 0; last_r = area = 0.0
    for yy, _ in sorted(zip(y, scores), key=lambda x: x[1], reverse=True):
        if yy: tp += 1
        else: fp += 1
        rec = tp / pos; prec = tp / (tp + fp); area += (rec - last_r) * prec; last_r = rec
    return area


def match_pred_to_gold(preds, golds):
    used = set(); pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if j not in used and p.text == g.text and p.etype == g.etype:
                used.add(j); pairs.append((i, j)); break
    return pairs


def diagnostic_metrics(samples, pred_by_sample, gold_by_sample, threshold):
    labels, scores, decisions, box_hits = [], [], [], []
    by_type = defaultdict(lambda: {"labels": [], "scores": [], "decisions": [], "hits": []})
    matched = 0
    for sample, preds, golds in zip(samples, pred_by_sample, gold_by_sample):
        details = sample["_details"]
        for pi, gi in match_pred_to_gold(preds, golds):
            g = golds[gi]; p = preds[pi]; d = details[pi]
            label = int(g.regions is not None)
            score = float(d["groundability_probability"])
            decision = int(p.regions is not None)
            hit = int(label and decision and max_iou(p.regions, g.regions) >= 0.5)
            labels.append(label); scores.append(score); decisions.append(decision); box_hits.append(hit); matched += 1
            z = by_type[g.etype]; z["labels"].append(label); z["scores"].append(score); z["decisions"].append(decision); z["hits"].append(hit)
    tp = sum(p and y for p, y in zip(decisions, labels)); fp = sum(p and not y for p, y in zip(decisions, labels))
    fn = sum((not p) and y for p, y in zip(decisions, labels)); tn = sum((not p) and (not y) for p, y in zip(decisions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "matched_predicted_entities": matched, "groundable_gold": sum(labels), "ungroundable_gold": len(labels) - sum(labels),
        "groundability_auroc": auroc(labels, scores), "groundability_aupr": aupr(labels, scores),
        "groundability_precision": precision, "groundability_recall": recall, "groundability_f1": f1,
        "none_accuracy": tn / max(tn + fp, 1), "false_grounding_rate": fp / max(fp + tn, 1),
        "groundable_entity_acc_iou_05": sum(box_hits) / max(sum(labels), 1), "threshold": threshold,
    }
    rows = []
    for typ, z in sorted(by_type.items()):
        yy, pp, hh = z["labels"], z["decisions"], z["hits"]
        ttp = sum(a and b for a, b in zip(pp, yy)); ffp = sum(a and not b for a, b in zip(pp, yy)); ffn = sum((not a) and b for a, b in zip(pp, yy))
        p = ttp / (ttp + ffp) if ttp + ffp else 0.0; r = ttp / (ttp + ffn) if ttp + ffn else 0.0
        rows.append({"type": typ, "matched_n": len(yy), "groundable_n": sum(yy),
                     "groundability_precision": p, "groundability_recall": r,
                     "groundability_f1": 2*p*r/(p+r) if p+r else 0.0,
                     "groundable_acc_iou_05": sum(hh) / max(sum(yy), 1)})
    return result, rows


def flatten_row(method, metrics, diag):
    row = {"method": method}
    for key in ["mner", "eeg", "gmner"]:
        for field in ["precision", "recall", "f1", "correct", "predicted", "gold"]:
            row[f"{key}_{field}"] = metrics[key][field]
    row.update(diag)
    return row


def write_csv(path, rows):
    fields = []
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    args = parse_args(); split = args.split
    pg_samples = read_jsonl(EXP / f"results/pgmner_predicted_entities_{split}.jsonl")
    qrows = read_jsonl(EXP / f"results/qwen_q2_cache_{split}.jsonl"); qwen = {r["key"]: r for r in qrows}
    sft = read_jsonl(E2E / f"data/sft/{split}_sft.jsonl")
    if not (len(pg_samples) == len(sft)):
        raise RuntimeError(f"sample count mismatch PGMNER={len(pg_samples)} SFT={len(sft)}")
    golds = [scaled_gold(row, i) for i, row in enumerate(sft)]
    direct_preds, direct_details = [], []
    for sample in pg_samples:
        p, d = predictions(sample, qwen, None); direct_preds.append(p); direct_details.append(d)
    rows = []
    direct_samples = [{**s, "_details": d} for s, d in zip(pg_samples, direct_details)]
    direct_metrics = score_dataset(direct_preds, golds)
    direct_diag, direct_types = diagnostic_metrics(direct_samples, direct_preds, golds, None)
    rows.append(flatten_row("PGMNER+Qwen", direct_metrics, direct_diag))

    if split == "dev":
        gated_payloads = []
        for threshold in THRESHOLDS:
            predset, detailset = [], []
            for sample in pg_samples:
                p, d = predictions(sample, qwen, threshold); predset.append(p); detailset.append(d)
            enriched = [{**s, "_details": d} for s, d in zip(pg_samples, detailset)]
            metrics = score_dataset(predset, golds); diag, types = diagnostic_metrics(enriched, predset, golds, threshold)
            rows.append(flatten_row(f"PGMNER+Groundability+Qwen@{threshold}", metrics, diag))
            gated_payloads.append((threshold, metrics, diag, types, predset, detailset))
        # Predeclared selection: max DEV GMNER F1, then EEG F1, then closest threshold to 0.5.
        best = max(gated_payloads, key=lambda x: (x[1]["gmner"]["f1"], x[1]["eeg"]["f1"], -abs(x[0]-0.5)))
        locked = {"threshold": best[0], "selection_split": "dev", "selection_metric": "official GMNER F1",
                  "tie_break": "official EEG F1, then closest to 0.5", "threshold_candidates": THRESHOLDS,
                  "selected_dev_metrics": best[1]}
        (EXP / "results/locked_groundability_threshold.json").write_text(json.dumps(locked, indent=2) + "\n")
        selected = best
        write_csv(EXP / "results/dev_threshold_selection.csv", rows)
    else:
        locked = json.loads((EXP / "results/locked_groundability_threshold.json").read_text())
        threshold = float(locked["threshold"])
        predset, detailset = [], []
        for sample in pg_samples:
            p, d = predictions(sample, qwen, threshold); predset.append(p); detailset.append(d)
        enriched = [{**s, "_details": d} for s, d in zip(pg_samples, detailset)]
        metrics = score_dataset(predset, golds); diag, types = diagnostic_metrics(enriched, predset, golds, threshold)
        rows.append(flatten_row(f"PGMNER+Groundability+Qwen@{threshold}", metrics, diag))
        selected = (threshold, metrics, diag, types, predset, detailset)
        write_csv(EXP / "results/official_metrics_test.csv", rows)

    threshold, sel_metrics, sel_diag, sel_types, sel_preds, sel_details = selected
    write_csv(EXP / f"results/entity_type_results_{split}.csv", sel_types)
    prediction_path = EXP / f"results/full_predictions_{split}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as out:
        for i, (sample, gold, direct, gated, dd, gd) in enumerate(zip(pg_samples, golds, direct_preds, sel_preds, direct_details, sel_details)):
            out.write(json.dumps({"split": split, "sample_index": i, "sample_id": sample["sample_id"],
                "image_id": sample["image_id"], "text": sample["text"],
                "gold": triples_to_canon_text(gold), "direct_qwen": triples_to_canon_text(direct),
                "groundability_aware": triples_to_canon_text(gated), "locked_threshold": threshold,
                "entities_direct": dd, "entities_groundability_aware": gd}, ensure_ascii=False) + "\n")
    output = {"split": split, "n_samples": len(pg_samples), "n_gold_entities": sum(map(len, golds)),
              "n_predicted_entities": sum(len(x) for x in direct_preds), "direct": {"official": direct_metrics, "diagnostic": direct_diag},
              "groundability_aware": {"threshold": threshold, "official": sel_metrics, "diagnostic": sel_diag}}
    (EXP / f"results/evaluation_{split}.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
