#!/usr/bin/env python3
"""Experiment 12 artifact-level audit for local E2E-GMNER predictions.

This script intentionally does not run the Qwen model. It evaluates the existing
saved E2E-GMNER test predictions as a local pretrained/fine-tuned artifact and
marks the result separately from the requested gold-query grounder protocol.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path("/data/hfy/VG-Think/E2E-GMNER")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.evaluation.geometry import iou, max_iou  # noqa: E402
from e2egmner.evaluation.parsing import extract_answer_text, parse_triples  # noqa: E402


EXP = Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment12_grounder_screening")
RESULTS = EXP / "results"

SFT = ROOT / "data/sft"
PRED_TEST = ROOT / "outputs/exp_cot_paper/eval/test_pred_best_eval.jsonl"
METRICS_TEST = ROOT / "outputs/exp_cot_paper/eval/test_metrics_best_eval.json"
SPATIAL_PRIORS = Path(
    "/home/jyx/yixiao/MINER-My/kais_extension/experiment11_patch_grounding/results/spatial_prior_baselines.csv"
)
EXP10_GROUNDING = Path(
    "/home/jyx/yixiao/MINER-My/kais_extension/experiment10_explicit_grounding/results/grounding_main.csv"
)
EXP11_PATCH = Path(
    "/home/jyx/yixiao/MINER-My/kais_extension/experiment11_patch_grounding/results/patch_grounding_main.csv"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_answer_from_sft(row: dict) -> str:
    return extract_answer_text(row["messages"][-1]["content"])


def parse_answer(answer: str, where: str):
    return parse_triples(answer, strict=False, where=where)


def flatten_groundable(rows: list[dict], split: str) -> list[dict]:
    out = []
    for idx, row in enumerate(rows):
        triples = parse_answer(gold_answer_from_sft(row), f"{split}:{idx}")
        for ent_idx, t in enumerate(triples):
            if t.region_valid and t.regions is not None:
                # Use max-area box for area stratification when multiple GT boxes exist.
                areas = [
                    max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    for b in t.regions
                ]
                out.append(
                    {
                        "sample_index": idx,
                        "entity_index": ent_idx,
                        "text": t.text,
                        "etype": t.etype,
                        "regions": t.regions,
                        "area": max(areas) if areas else 0.0,
                    }
                )
    return out


def quantile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "mean_iou": float("nan"),
            "median_iou": float("nan"),
            "acc_iou_025": float("nan"),
            "acc_iou_05": float("nan"),
            "acc_iou_075": float("nan"),
        }
    return {
        "n": len(values),
        "mean_iou": mean(values),
        "median_iou": median(values),
        "acc_iou_025": sum(v >= 0.25 for v in values) / len(values),
        "acc_iou_05": sum(v >= 0.5 for v in values) / len(values),
        "acc_iou_075": sum(v >= 0.75 for v in values) / len(values),
    }


def best_matching_pred_iou(gold, preds) -> tuple[float, bool]:
    best = 0.0
    matched_entity = False
    for p in preds:
        if p.text != gold.text or p.etype != gold.etype:
            continue
        matched_entity = True
        if not p.region_valid or p.regions is None:
            continue
        best = max(best, max_iou(p.regions, gold.regions or []))
    return best, matched_entity


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def copy_reference_rows() -> list[dict]:
    rows = []
    if SPATIAL_PRIORS.exists():
        with SPATIAL_PRIORS.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test" and row["group"] == "overall":
                    if row["baseline"] in {"B0_full_image", "B2_entity_type_mean_box"}:
                        rows.append(
                            {
                                "method": row["baseline"],
                                "source": str(SPATIAL_PRIORS),
                                "mean_iou": row["mean_iou"],
                                "acc_iou_05": row["acc_iou_05"],
                                "acc_iou_075": row["acc_iou_075"],
                                "small_box_acc_iou_05": "",
                                "medium_box_acc_iou_05": "",
                                "large_box_acc_iou_05": "",
                                "notes": "Experiment 11A spatial-prior baseline",
                            }
                        )
    if EXP10_GROUNDING.exists():
        with EXP10_GROUNDING.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("variant", "").lower().replace("_", "-") in {"entity", "entity-aware"}:
                    rows.append(
                        {
                            "method": "Experiment10_Entity_Aware_regression",
                            "source": str(EXP10_GROUNDING),
                            "mean_iou": row.get("test_mean_iou", ""),
                            "acc_iou_05": row.get("test_acc_iou_05", ""),
                            "acc_iou_075": row.get("test_acc_iou_075", ""),
                            "small_box_acc_iou_05": "",
                            "medium_box_acc_iou_05": "",
                            "large_box_acc_iou_05": "",
                            "notes": "Frozen-feature coordinate regression",
                        }
                    )
    if EXP11_PATCH.exists():
        with EXP11_PATCH.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("config") == "P1-Focal":
                    rows.append(
                        {
                            "method": "Experiment11_P1_Focal_patch",
                            "source": str(EXP11_PATCH),
                            "mean_iou": row.get("test_mean_iou", ""),
                            "acc_iou_05": row.get("test_acc_iou_05", ""),
                            "acc_iou_075": "",
                            "small_box_acc_iou_05": "",
                            "medium_box_acc_iou_05": "",
                            "large_box_acc_iou_05": "",
                            "notes": "Patch-supervised frozen CLIP head",
                        }
                    )
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(SFT / "train_sft.jsonl")
    dev_rows = read_jsonl(SFT / "dev_sft.jsonl")
    test_rows = read_jsonl(SFT / "test_sft.jsonl")
    pred_rows = read_jsonl(PRED_TEST)

    train_dev_groundable = flatten_groundable(train_rows, "train") + flatten_groundable(dev_rows, "dev")
    test_groundable = flatten_groundable(test_rows, "test")
    q1 = quantile([x["area"] for x in train_dev_groundable], 1 / 3)
    q2 = quantile([x["area"] for x in train_dev_groundable], 2 / 3)

    entity_rows = []
    all_ious = []
    matched_ious = []
    by_type = defaultdict(list)
    by_size = defaultdict(list)
    matched_count = 0

    for idx, (gold_sample, pred_sample) in enumerate(zip(test_rows, pred_rows)):
        golds = parse_answer(gold_answer_from_sft(gold_sample), f"test_gold:{idx}")
        preds = parse_answer(pred_sample.get("response_answer_scaled", ""), f"test_pred:{idx}")
        for ent_idx, g in enumerate(golds):
            if not (g.region_valid and g.regions is not None):
                continue
            giou, matched_entity = best_matching_pred_iou(g, preds)
            matched_count += int(matched_entity)
            if matched_entity:
                matched_ious.append(giou)
            all_ious.append(giou)
            area = max(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in g.regions)
            size_bin = "small" if area <= q1 else ("medium" if area <= q2 else "large")
            by_type[g.etype].append(giou)
            by_size[size_bin].append(giou)
            entity_rows.append(
                {
                    "sample_index": idx,
                    "entity_index": ent_idx,
                    "mention": g.text,
                    "type": g.etype,
                    "size_bin": size_bin,
                    "matched_entity_type": int(matched_entity),
                    "best_iou": giou,
                    "acc_iou_025": int(giou >= 0.25),
                    "acc_iou_05": int(giou >= 0.5),
                    "acc_iou_075": int(giou >= 0.75),
                    "protocol": "existing_e2e_artifact_not_oracle_query",
                }
            )

    write_csv(
        RESULTS / "grounder_test_predictions.csv",
        entity_rows,
        [
            "sample_index",
            "entity_index",
            "mention",
            "type",
            "size_bin",
            "matched_entity_type",
            "best_iou",
            "acc_iou_025",
            "acc_iou_05",
            "acc_iou_075",
            "protocol",
        ],
    )

    overall = summarize(all_ious)
    matched = summarize(matched_ious)
    main_rows = [
        {
            "method": "E2E_GMNER_saved_prediction_all_gold_groundable",
            "protocol": "existing_e2e_artifact_not_oracle_query",
            "dev_query_selected": "NOT_APPLICABLE",
            "n": overall["n"],
            "matched_gold_entity_type": matched_count,
            **{k: v for k, v in overall.items() if k != "n"},
            "notes": "Unmatched gold groundable entities counted as IoU=0.",
        },
        {
            "method": "E2E_GMNER_saved_prediction_matched_only",
            "protocol": "existing_e2e_artifact_not_oracle_query",
            "dev_query_selected": "NOT_APPLICABLE",
            "n": matched["n"],
            "matched_gold_entity_type": matched["n"],
            **{k: v for k, v in matched.items() if k != "n"},
            "notes": "Only gold entities whose mention and type were generated are included.",
        },
    ]
    write_csv(
        RESULTS / "grounder_main.csv",
        main_rows,
        [
            "method",
            "protocol",
            "dev_query_selected",
            "n",
            "matched_gold_entity_type",
            "mean_iou",
            "median_iou",
            "acc_iou_025",
            "acc_iou_05",
            "acc_iou_075",
            "notes",
        ],
    )

    size_rows = []
    for group in ["small", "medium", "large"]:
        row = summarize(by_size[group])
        row.update({"method": "E2E_GMNER_saved_prediction", "size_bin": group})
        size_rows.append(row)
    write_csv(
        RESULTS / "box_size_breakdown.csv",
        size_rows,
        ["method", "size_bin", "n", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075"],
    )

    type_rows = []
    for etype in ["PER", "LOC", "ORG", "MISC", "OTHER"]:
        if etype not in by_type:
            continue
        row = summarize(by_type[etype])
        row.update({"method": "E2E_GMNER_saved_prediction", "entity_type": etype})
        type_rows.append(row)
    write_csv(
        RESULTS / "entity_type_breakdown.csv",
        type_rows,
        ["method", "entity_type", "n", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075"],
    )

    # Required but not executable under the available local interface.
    write_csv(
        RESULTS / "query_format_dev.csv",
        [
            {
                "query_format": q,
                "status": "NOT_EXECUTED",
                "reason": "No existing local entity-query grounding inference script was found; available pipeline is end-to-end GMNER generation.",
            }
            for q in ["Q1_mention_only", "Q2_mention_plus_type", "Q3_mention_plus_context"]
        ],
        ["query_format", "status", "reason"],
    )
    write_csv(
        RESULTS / "query_sensitivity.csv",
        [
            {
                "status": "NOT_EXECUTED",
                "reason": "Requires entity-query grounder outputs for multiple gold queries per image; no local script/artifact provides this.",
            }
        ],
        ["status", "reason"],
    )
    write_csv(
        RESULTS / "image_sensitivity.csv",
        [
            {
                "status": "NOT_EXECUTED",
                "reason": "Requires rerunning model on wrong-image oracle-query prompts; not available from saved end-to-end predictions.",
            }
        ],
        ["status", "reason"],
    )

    comparison = copy_reference_rows()
    small = next((r for r in size_rows if r["size_bin"] == "small"), None)
    medium = next((r for r in size_rows if r["size_bin"] == "medium"), None)
    large = next((r for r in size_rows if r["size_bin"] == "large"), None)
    comparison.append(
        {
            "method": "E2E_GMNER_saved_prediction_all_gold_groundable",
            "source": str(PRED_TEST),
            "mean_iou": overall["mean_iou"],
            "acc_iou_05": overall["acc_iou_05"],
            "acc_iou_075": overall["acc_iou_075"],
            "small_box_acc_iou_05": small["acc_iou_05"] if small else "",
            "medium_box_acc_iou_05": medium["acc_iou_05"] if medium else "",
            "large_box_acc_iou_05": large["acc_iou_05"] if large else "",
            "notes": "Existing end-to-end artifact; not oracle-query localization.",
        }
    )
    write_csv(
        RESULTS / "comparison_with_exp11.csv",
        comparison,
        [
            "method",
            "source",
            "mean_iou",
            "acc_iou_05",
            "acc_iou_075",
            "small_box_acc_iou_05",
            "medium_box_acc_iou_05",
            "large_box_acc_iou_05",
            "notes",
        ],
    )

    metrics_payload = json.loads(METRICS_TEST.read_text(encoding="utf-8"))
    summary = {
        "protocol_status": "NO_DIRECT_ORACLE_QUERY_GROUNDER_SCRIPT_FOUND",
        "existing_artifact": str(PRED_TEST),
        "official_test_metrics": metrics_payload.get("metrics", {}),
        "train_dev_area_quantiles_scaled": {"q33": q1, "q67": q2},
        "all_gold_groundable": overall,
        "matched_entity_type_only": matched,
        "matched_gold_entity_type_fraction": matched_count / max(1, len(all_ious)),
        "size_breakdown": {r["size_bin"]: r for r in size_rows},
        "type_breakdown": {r["entity_type"]: r for r in type_rows},
    }
    (RESULTS / "artifact_localization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
