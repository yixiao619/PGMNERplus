#!/usr/bin/env python3
"""Produce source/type breakdowns and deterministic Exp14 case visualizations."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment14_gmner_integration"
E2E = Path("/data/hfy/VG-Think/E2E-GMNER")
sys.path.insert(0, str(E2E / "src"))
from e2egmner.evaluation.geometry import smart_resize  # noqa: E402
from e2egmner.evaluation.metrics import count_correct_eeg, count_correct_gmner, count_correct_mner, prf  # noqa: E402
from e2egmner.evaluation.parsing import parse_triples  # noqa: E402


def read_jsonl(path): return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]


def score(rows, method, indices, typ=None):
    out = {}
    for metric, fn in [("mner", count_correct_mner), ("eeg", count_correct_eeg), ("gmner", count_correct_gmner)]:
        correct = pred = gold = 0
        for i in indices:
            ps = parse_triples(rows[i][method], strict=False, where=f"{method}:{i}")
            gs = parse_triples(rows[i]["gold"], strict=True, where=f"gold:{i}")
            if typ:
                ps = [x for x in ps if x.etype == typ]; gs = [x for x in gs if x.etype == typ]
            correct += fn(ps, gs) if metric == "mner" else fn(ps, gs, 0.5)
            pred += len(ps); gold += len(gs)
        p, r, f = prf(correct, pred, gold)
        out.update({f"{metric}_precision": p, f"{metric}_recall": r, f"{metric}_f1": f,
                    f"{metric}_correct": correct, f"{metric}_pred": pred, f"{metric}_gold": gold})
    return out


def write_csv(path, rows):
    fields = []
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def original_ids(dataset, split):
    path = ROOT / f"data/{dataset}/{split}.txt"; out = set()
    for line in path.open(encoding="utf-8"):
        if line.startswith("IMGID:"): out.add(line.strip().split(":", 1)[1])
    return out


def draw_case(row, sample, name):
    image = Image.open(sample["image_path"]).convert("RGB")
    nh, nw = smart_resize(image.height, image.width, factor=28, min_pixels=256*28*28, max_pixels=1024*28*28)
    image = image.resize((nw, nh)); draw = ImageDraw.Draw(image)
    gold = parse_triples(row["gold"], strict=True, where="case_gold")
    gated = parse_triples(row["groundability_aware"], strict=False, where="case_pred")
    for t in gold:
        for b in t.regions or []: draw.rectangle(b, outline="green", width=4)
    for t in gated:
        for b in t.regions or []: draw.rectangle(b, outline="red", width=4)
    image.save(EXP / f"figures/{name}.png")


def main():
    rows = read_jsonl(EXP / "results/full_predictions_test.jsonl")
    samples = read_jsonl(EXP / "results/pgmner_predicted_entities_test.jsonl")
    idx = list(range(len(rows)))
    type_rows = []
    for method in ["direct_qwen", "groundability_aware"]:
        for typ in ["PER", "LOC", "ORG", "OTHER"]:
            type_rows.append({"method": method, "type": typ, **score(rows, method, idx, typ)})
    write_csv(EXP / "results/entity_type_official_test.csv", type_rows)

    source_groups = defaultdict(list)
    t17_sets = {sp: original_ids("twitter2017", sp) for sp in ["train", "dev", "test"]}
    for i, s in enumerate(samples):
        source_groups[s["source_dataset"]].append(i)
        if s["source_dataset"] == "Twitter2017":
            found = next((sp for sp, ids in t17_sets.items() if s["image_id"] in ids), "unknown")
            source_groups[f"Twitter2017_original_{found}"].append(i)
    source_rows = []
    for method in ["direct_qwen", "groundability_aware"]:
        for group, indices in sorted(source_groups.items()):
            source_rows.append({"method": method, "source_group": group, "samples": len(indices), **score(rows, method, indices)})
    write_csv(EXP / "results/source_breakdown_test.csv", source_rows)

    categories = defaultdict(list)
    for i, row in enumerate(rows):
        gold = parse_triples(row["gold"], strict=True, where=f"gold:{i}")
        direct = parse_triples(row["direct_qwen"], strict=False, where=f"direct:{i}")
        gated = parse_triples(row["groundability_aware"], strict=False, where=f"gated:{i}")
        dm = count_correct_gmner(direct, gold, 0.5); gm = count_correct_gmner(gated, gold, 0.5)
        mm = count_correct_mner(gated, gold)
        if gm > dm: categories["gate_help"].append(i)
        if gm < dm: categories["gate_harm"].append(i)
        if gm > 0: categories["gmner_success"].append(i)
        if mm == 0 and gold: categories["mner_failure"].append(i)
    chosen = []
    for category in ["gate_help", "gate_harm", "gmner_success", "mner_failure"]:
        if categories[category]: chosen.append((category, categories[category][0]))
    # Fifth case is the second deterministic gate-help example when available.
    if len(categories["gate_help"]) > 1: chosen.append(("gate_help_2", categories["gate_help"][1]))
    case_rows = []
    for category, i in chosen:
        draw_case(rows[i], samples[i], f"case_{category}_{i}")
        case_rows.append({"category": category, "sample_index": i, "sample_id": rows[i]["sample_id"],
                          "image_id": rows[i]["image_id"], "text": rows[i]["text"], "gold": rows[i]["gold"],
                          "direct_qwen": rows[i]["direct_qwen"], "groundability_aware": rows[i]["groundability_aware"],
                          "figure": str(EXP / f"figures/case_{category}_{i}.png")})
    (EXP / "results/error_cases.json").write_text(json.dumps({"counts": {k: len(v) for k,v in categories.items()}, "cases": case_rows}, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"source_breakdown": source_rows, "case_counts": {k: len(v) for k,v in categories.items()}}, indent=2))


if __name__ == "__main__": main()
