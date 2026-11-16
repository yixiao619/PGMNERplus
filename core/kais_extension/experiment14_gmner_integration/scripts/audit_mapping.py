#!/usr/bin/env python3
"""Read-only mapping and leakage-risk audit for Experiment 14."""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment14_gmner_integration"
SIDE = ROOT / "kais_extension/experiment9_gmner_audit/cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl"
SFT = Path("/data/hfy/VG-Think/E2E-GMNER/data/sft")


def image_ids(path):
    return {line.strip().split(":", 1)[1] for line in Path(path).open(encoding="utf-8") if line.startswith("IMGID:")}


def norm_text(x):
    return re.sub(r"\s+", " ", x.strip().lower())


def main():
    side = [json.loads(x) for x in SIDE.open(encoding="utf-8")]
    by = {s: [r for r in side if r["split"] == s] for s in ["train", "dev", "test"]}
    pg_sets = {}
    for dataset in ["twitter2015", "twitter2017"]:
        for split in ["train", "dev", "test"]:
            path = ROOT / f"data/{dataset}/{split}.txt"
            if path.exists(): pg_sets[(dataset, split)] = image_ids(path)
    output = {"sample_counts": {}, "entity_counts": {}, "mapping_checks": {}, "source_counts": {}, "overlap_with_original_pgmner_splits": {}}
    for split, rows in by.items():
        sft_rows = [json.loads(x) for x in (SFT / f"{split}_sft.jsonl").open(encoding="utf-8")]
        output["sample_counts"][split] = len(rows)
        output["entity_counts"][split] = sum(len(r["entities"]) for r in rows)
        output["mapping_checks"][split] = {
            "same_row_count_as_sft": len(rows) == len(sft_rows),
            "sample_id_matches": sum(r["sample_id"] == s.get("id") for r, s in zip(rows, sft_rows)),
            "image_rel_matches": sum(r["image_rel"] == (s.get("images") or [None])[0] for r, s in zip(rows, sft_rows)),
            "all_entities_text_mapped": all(e.get("matched_in_text") for r in rows for e in r["entities"]),
        }
        output["source_counts"][split] = dict(Counter("twitter2015" if "twitter2015" in r["image_rel"].lower() else "twitter2017" for r in rows))
        ids = {r["image_id"] for r in rows}
        output["overlap_with_original_pgmner_splits"][split] = {
            f"{ds}_{sp}": len(ids & vals) for (ds, sp), vals in pg_sets.items()
        }
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        ia, ib = {r["image_id"] for r in by[a]}, {r["image_id"] for r in by[b]}
        ta, tb = {norm_text(r["text"]) for r in by[a]}, {norm_text(r["text"]) for r in by[b]}
        output.setdefault("cross_split_duplicates", {})[f"{a}_vs_{b}"] = {"image_id": len(ia & ib), "normalized_text": len(ta & tb)}
    output["type_mapping"] = {"PGMNER_MISC": "official_OTHER", "PER": "PER", "LOC": "LOC", "ORG": "ORG"}
    output["span_mapping"] = "Whitespace token spans from Exp9 cache; decoded PGMNER span mention joins the same source words."
    output["bbox_coordinates"] = "Gold source boxes are original-image xyxy; official evaluation rescales to Qwen smart-resize coordinates. Qwen output is already in smart-resize coordinates."
    output["distribution_warning"] = "The paper-full PGMNER checkpoint was trained on Twitter-2017, while GMNER DEV/TEST mix Twitter-2015 and Twitter-2017 and use a different split. Original-split overlaps quantify leakage risk."
    path = EXP / "results/data_mapping_audit.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
