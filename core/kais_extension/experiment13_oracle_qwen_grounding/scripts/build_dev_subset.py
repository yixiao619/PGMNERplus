#!/usr/bin/env python3
"""Build deterministic DEV screening subsets for Experiment 13."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data/hfy/VG-Think/E2E-GMNER")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.evaluation.geometry import rescale_assistant_text_bboxes, smart_resize  # noqa: E402
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

    path = ROOT / "data" / image_rel
    with Image.open(path) as im:
        return im.size


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def entities(rows: list[dict], split: str) -> list[dict]:
    out = []
    for sample_idx, row in enumerate(rows):
        image_rel = row["images"][0]
        orig_w, orig_h = image_size(image_rel)
        try:
            new_h, new_w = smart_resize(orig_h, orig_w, factor=28, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
        except ValueError:
            continue
        scaled_full = rescale_assistant_text_bboxes(
            row["messages"][-1]["content"],
            orig_w=orig_w,
            orig_h=orig_h,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            factor=28,
        )
        if scaled_full is None:
            continue
        triples = parse_triples(extract_answer_text(scaled_full), strict=False, where=f"{split}:{sample_idx}")
        orig_triples = parse_triples(extract_answer_text(row["messages"][-1]["content"]), strict=False, where=f"{split}:orig:{sample_idx}")
        groundable_count = sum(1 for t in triples if t.region_valid and t.regions is not None)
        for entity_idx, (t, to) in enumerate(zip(triples, orig_triples)):
            area = 0.0
            if t.region_valid and t.regions is not None:
                area = max(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in t.regions)
            out.append(
                {
                    "split": split,
                    "sample_index": sample_idx,
                    "entity_index": entity_idx,
                    "id": row.get("id", sample_idx),
                    "image": image_rel,
                    "text": row["messages"][0]["content"].replace("<image>", "").replace("Text:", "", 1).strip(),
                    "entity": t.text,
                    "type": t.etype,
                    "bbox_scaled": t.regions,
                    "bbox_orig": to.regions if to.region_valid else None,
                    "groundable": bool(t.region_valid and t.regions is not None),
                    "area_scaled": area,
                    "scaled_size": [new_w, new_h],
                    "orig_size": [orig_w, orig_h],
                    "multi_entity_sample": len(triples) >= 2,
                    "multi_groundable_sample": groundable_count >= 2,
                }
            )
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = entities(read_jsonl(SFT / "train_sft.jsonl"), "train")
    dev = entities(read_jsonl(SFT / "dev_sft.jsonl"), "dev")
    train_dev_ground = [e for e in train + dev if e["groundable"]]
    q33 = quantile([e["area_scaled"] for e in train_dev_ground], 1 / 3)
    q67 = quantile([e["area_scaled"] for e in train_dev_ground], 2 / 3)

    def size_bin(e: dict) -> str:
        if e["area_scaled"] <= q33:
            return "small"
        if e["area_scaled"] <= q67:
            return "medium"
        return "large"

    ground = [dict(e, size_bin=size_bin(e)) for e in dev if e["groundable"]]
    unground = [dict(e, size_bin="ungroundable") for e in dev if not e["groundable"]]

    selected = []
    seen = set()
    # Reserve multi-entity samples first for mandatory query sensitivity.
    multi = [e for e in ground if e["multi_entity_sample"]]
    for e in multi[:40]:
        key = (e["sample_index"], e["entity_index"])
        selected.append(e)
        seen.add(key)

    buckets = defaultdict(list)
    for e in ground:
        buckets[(e["size_bin"], e["type"])].append(e)
    bucket_keys = sorted(buckets)
    while len(selected) < min(200, len(ground)):
        progressed = False
        for key in bucket_keys:
            while buckets[key]:
                e = buckets[key].pop(0)
                k = (e["sample_index"], e["entity_index"])
                if k not in seen:
                    selected.append(e)
                    seen.add(k)
                    progressed = True
                    break
            if len(selected) >= min(200, len(ground)):
                break
        if not progressed:
            break

    unground_selected = unground[: min(100, len(unground))]
    payload = {
        "source": str(SFT / "dev_sft.jsonl"),
        "image_root": str(ROOT / "data"),
        "area_quantiles_scaled_train_dev": {"q33": q33, "q67": q67},
        "groundable_subset": selected,
        "ungroundable_subset": unground_selected,
        "counts": {
            "dev_groundable_total": len(ground),
            "dev_ungroundable_total": len(unground),
            "groundable_selected": len(selected),
            "ungroundable_selected": len(unground_selected),
            "multi_entity_selected": sum(1 for e in selected if e["multi_entity_sample"]),
        },
    }
    (RESULTS / "dev_screening_subset.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
