#!/usr/bin/env python3
"""Compute matched spatial-prior baselines for Experiment 13 selected DEV subset."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path("/data/hfy/VG-Think/E2E-GMNER")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.evaluation.geometry import max_iou, rescale_assistant_text_bboxes, smart_resize  # noqa: E402
from e2egmner.evaluation.parsing import extract_answer_text, parse_triples  # noqa: E402


EXP = Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment13_oracle_qwen_grounding")
RESULTS = EXP / "results"
SFT = ROOT / "data/sft"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_size(image_rel: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(ROOT / "data" / image_rel) as im:
        return im.size


def scaled_entities(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        image_rel = row["images"][0]
        ow, oh = image_size(image_rel)
        try:
            scaled = rescale_assistant_text_bboxes(
                row["messages"][-1]["content"],
                orig_w=ow,
                orig_h=oh,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28,
                factor=28,
            )
            nh, nw = smart_resize(oh, ow, factor=28, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
        except ValueError:
            continue
        if scaled is None:
            continue
        for t in parse_triples(extract_answer_text(scaled), strict=False, where="spatial"):
            if t.region_valid and t.regions is not None:
                for b in t.regions:
                    x1, y1, x2, y2 = b
                    out.append(
                        {
                            "type": t.etype,
                            "box": tuple(b),
                            "cx": (x1 + x2) / 2.0,
                            "cy": (y1 + y2) / 2.0,
                            "w": max(0.0, x2 - x1),
                            "h": max(0.0, y2 - y1),
                            "image_size": (nw, nh),
                        }
                    )
    return out


def mean_box_stats(ents: list[dict]) -> dict[str, float]:
    return {
        "cx": mean([e["cx"] for e in ents]),
        "cy": mean([e["cy"] for e in ents]),
        "w": mean([e["w"] for e in ents]),
        "h": mean([e["h"] for e in ents]),
    }


def instantiate_box(stats: dict[str, float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    iw, ih = image_size
    cx, cy, w, h = stats["cx"], stats["cy"], stats["w"], stats["h"]
    return (
        max(0.0, min(float(iw), cx - w / 2.0)),
        max(0.0, min(float(ih), cy - h / 2.0)),
        max(0.0, min(float(iw), cx + w / 2.0)),
        max(0.0, min(float(ih), cy + h / 2.0)),
    )


def summarize(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "mean_iou": mean(vals),
        "median_iou": median(vals),
        "acc_iou_025": sum(v >= 0.25 for v in vals) / len(vals),
        "acc_iou_05": sum(v >= 0.5 for v in vals) / len(vals),
        "acc_iou_075": sum(v >= 0.75 for v in vals) / len(vals),
    }


def main() -> None:
    train_ents = scaled_entities(read_jsonl(SFT / "train_sft.jsonl"))
    global_stats = mean_box_stats(train_ents)
    by_type = defaultdict(list)
    for e in train_ents:
        by_type[e["type"]].append(e)
    type_stats = {k: mean_box_stats(v) for k, v in by_type.items()}

    subset = json.loads((RESULTS / "dev_screening_subset.json").read_text(encoding="utf-8"))
    selected = subset["groundable_subset"]
    baselines = {"full_image_same_subset": [], "train_mean_box_same_subset": [], "type_mean_box_same_subset": []}
    for e in selected:
        iw, ih = e["scaled_size"]
        gt = [tuple(b) for b in e["bbox_scaled"]]
        baselines["full_image_same_subset"].append(max_iou([(0.0, 0.0, float(iw), float(ih))], gt))
        baselines["train_mean_box_same_subset"].append(max_iou([instantiate_box(global_stats, (iw, ih))], gt))
        stats = type_stats.get(e["type"], global_stats)
        baselines["type_mean_box_same_subset"].append(max_iou([instantiate_box(stats, (iw, ih))], gt))

    with (RESULTS / "spatial_prior_matched.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["baseline", "query_format", "n", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, vals in baselines.items():
            row = {"baseline": name, "query_format": "Q2"}
            row.update(summarize(vals))
            w.writerow(row)


if __name__ == "__main__":
    main()
