#!/usr/bin/env python3
"""Oracle entity-query Qwen grounding wrapper for Experiment 13."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


E2E_ROOT = Path("/data/hfy/VG-Think/E2E-GMNER")
SRC = E2E_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.evaluation.geometry import iou, max_iou  # noqa: E402
from e2egmner.evaluation.parsing import extract_answer_text, parse_bbox_regions, parse_triples  # noqa: E402


EXP = Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment13_oracle_qwen_grounding")
RESULTS = EXP / "results"
LOGS = EXP / "logs"
VIS = EXP / "visualizations"

QUERY_FORMATS = {
    "Q1": 'Given this image, locate the visual region corresponding to the entity:\n\n"{entity}"\n\nReturn only:\n[x1,y1,x2,y2]\n\nIf the entity is not visually present, return:\nNone',
    "Q2": 'Given this image, locate the visual region corresponding to:\n\nEntity: "{entity}"\nType: "{etype}"\n\nReturn only:\n[x1,y1,x2,y2]\n\nIf the entity is not visually present, return:\nNone',
    "Q3": 'Given the image and the following post:\n\n"{text}"\n\nLocate the visual region corresponding specifically to the entity:\n\n"{entity}"\n\nReturn only:\n[x1,y1,x2,y2]\n\nIf the entity is not visually present, return:\nNone',
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default=str(RESULTS / "dev_screening_subset.json"))
    p.add_argument("--model_name", default=str(E2E_ROOT / "model/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--checkpoint", default=str(E2E_ROOT / "outputs/exp_cot_paper/best"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--max_smoke_entities", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(args: argparse.Namespace):
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    if getattr(processor, "image_processor", None) is None:
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, args.checkpoint, is_trainable=False)
    model.to(args.device)
    model.eval()
    return model, processor, dtype


def build_prompt(entity: dict, qid: str) -> str:
    return QUERY_FORMATS[qid].format(entity=entity["entity"], etype=entity["type"], text=entity["text"])


def make_messages(prompt: str):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def parse_prediction(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    answer = extract_answer_text(raw)
    s = answer.strip()
    if s.lower() == "none":
        return {"kind": "none", "boxes": None, "valid": True, "answer": answer}
    # Prefer direct bbox parsing, but tolerate generated tags/explanations.
    try:
        boxes, valid = parse_bbox_regions(s, strict=False, where="oracle_pred")
        if valid and boxes:
            return {"kind": "bbox", "boxes": boxes[:1], "valid": True, "answer": answer}
    except Exception:
        pass
    m = re.search(r"\[[^\[\]]*[-+]?\d[^\[\]]*\]", s)
    if m:
        boxes, valid = parse_bbox_regions(m.group(0), strict=False, where="oracle_pred_regex")
        if valid and boxes:
            return {"kind": "bbox", "boxes": boxes[:1], "valid": True, "answer": answer}
    return {"kind": "invalid", "boxes": None, "valid": False, "answer": answer}


@torch.no_grad()
def generate_one(model, processor, args, entity: dict, qid: str) -> dict[str, Any]:
    prompt = build_prompt(entity, qid)
    image = Image.open(E2E_ROOT / "data" / entity["image"]).convert("RGB")
    messages = make_messages(prompt)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    start = time.time()
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )
    elapsed = time.time() - start
    trimmed = [generated[0, inputs["input_ids"].shape[-1] :]]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    parsed = parse_prediction(raw)
    boxes = parsed["boxes"]
    gt = [tuple(b) for b in (entity.get("bbox_scaled") or [])]
    pred_iou = max_iou(boxes, gt) if boxes and gt else 0.0
    return {
        "query_format": qid,
        "sample_index": entity["sample_index"],
        "entity_index": entity["entity_index"],
        "image": entity["image"],
        "entity": entity["entity"],
        "type": entity["type"],
        "size_bin": entity.get("size_bin", ""),
        "groundable": entity.get("groundable", True),
        "prompt": prompt,
        "raw_generation": raw,
        "parsed_kind": parsed["kind"],
        "parse_valid": parsed["valid"],
        "pred_boxes": boxes,
        "gt_boxes": gt,
        "iou": pred_iou,
        "acc_iou_025": pred_iou >= 0.25,
        "acc_iou_05": pred_iou >= 0.5,
        "acc_iou_075": pred_iou >= 0.75,
        "seconds": elapsed,
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    ious = [float(r["iou"]) for r in rows]
    return {
        "n": len(rows),
        "parse_success_rate": sum(bool(r["parse_valid"]) for r in rows) / len(rows),
        "valid_bbox_rate": sum(r["parsed_kind"] == "bbox" for r in rows) / len(rows),
        "none_prediction_rate": sum(r["parsed_kind"] == "none" for r in rows) / len(rows),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "acc_iou_025": sum(v >= 0.25 for v in ious) / len(ious),
        "acc_iou_05": sum(v >= 0.5 for v in ious) / len(ious),
        "acc_iou_075": sum(v >= 0.75 for v in ious) / len(ious),
        "seconds_per_entity": mean(float(r["seconds"]) for r in rows),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            clean = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                clean[k] = v
            w.writerow(clean)


def box_iou_or_zero(a, b) -> float:
    if not a or not b:
        return 0.0
    return max_iou(a, b)


def run_spatial_priors(entities: list[dict], subset_payload: dict, best_q: str) -> None:
    # Use train/dev thresholds and train statistics already stored from Exp12/13 subset construction.
    train_rows = json.loads((Path("/home/jyx/yixiao/MINER-My/kais_extension/experiment12_grounder_screening/results/artifact_localization_summary.json")).read_text())
    _ = train_rows
    # For Experiment 13 matched subset, report simple full-image only and carry type-mean references as unavailable unless recomputed.
    rows = []
    vals = []
    for e in entities:
        w, h = e["scaled_size"]
        pred = [(0.0, 0.0, float(w), float(h))]
        vals.append(max_iou(pred, [tuple(b) for b in e["bbox_scaled"]]))
    s = summarize([{"iou": v, "parse_valid": True, "parsed_kind": "bbox", "seconds": 0.0} for v in vals])
    rows.append({"baseline": "full_image_same_subset", "query_format": best_q, **s})
    write_csv(
        RESULTS / "spatial_prior_matched.csv",
        rows,
        ["baseline", "query_format", "n", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075"],
    )


def load_existing_e2e_by_sample() -> dict[int, list]:
    pred_path = E2E_ROOT / "outputs/exp_cot_paper/dev_pred_epoch3.jsonl"
    if not pred_path.exists():
        return {}
    mapping = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            mapping[idx] = parse_triples(row.get("response_answer_scaled", ""), strict=False, where=f"dev_e2e:{idx}")
    return mapping


def run_e2e_matched_comparison(entities: list[dict]) -> None:
    preds_by_sample = load_existing_e2e_by_sample()
    if not preds_by_sample:
        write_csv(RESULTS / "e2e_matched_comparison.csv", [{"status": "NOT_AVAILABLE"}], ["status"])
        return
    rows = []
    for e in entities:
        preds = preds_by_sample.get(int(e["sample_index"]), [])
        target = type("Obj", (), {"text": e["entity"], "etype": e["type"], "regions": [tuple(b) for b in e["bbox_scaled"]]})
        best = 0.0
        for p in preds:
            if p.text == target.text and p.etype == target.etype and p.regions is not None:
                best = max(best, max_iou(p.regions, target.regions))
        rows.append({"iou": best, "parse_valid": True, "parsed_kind": "bbox" if best > 0 else "none", "seconds": 0.0})
    s = summarize(rows)
    write_csv(
        RESULTS / "e2e_matched_comparison.csv",
        [{"method": "existing_e2e_dev_epoch3_same_subset", **s}],
        ["method", "n", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075"],
    )


def write_visualizations(rows: list[dict]) -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    selected = rows[:2] + sorted(rows, key=lambda r: r["iou"], reverse=True)[:2] + sorted(rows, key=lambda r: r["iou"])[:2]
    seen = set()
    for i, r in enumerate(selected):
        key = (r["sample_index"], r["entity_index"], r["query_format"])
        if key in seen:
            continue
        seen.add(key)
        img = Image.open(E2E_ROOT / "data" / r["image"]).convert("RGB")
        sw, sh = r.get("scaled_size", [img.width, img.height])
        # Draw in scaled space by resizing image to the model coordinate system.
        img = img.resize((int(sw), int(sh)))
        draw = ImageDraw.Draw(img)
        for b in r.get("gt_boxes") or []:
            draw.rectangle(tuple(b), outline="green", width=4)
        for b in r.get("pred_boxes") or []:
            draw.rectangle(tuple(b), outline="red", width=4)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(r["entity"]))[:50]
        img.save(VIS / f"case_{i}_{r['query_format']}_{safe}.png")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(args.subset).read_text(encoding="utf-8"))
    ground = payload["groundable_subset"]
    unground = payload["ungroundable_subset"]
    if args.mode == "smoke":
        ground = ground[: args.max_smoke_entities]
        qids = ["Q1"]
    else:
        qids = ["Q1", "Q2", "Q3"]

    start_all = time.time()
    model, processor, dtype = load_model(args)
    load_seconds = time.time() - start_all
    peak_mem = torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") and torch.cuda.is_available() else 0

    pred_rows = []
    for qid in qids:
        for e in ground:
            row = generate_one(model, processor, args, e, qid)
            row["scaled_size"] = e["scaled_size"]
            pred_rows.append(row)

    with (RESULTS / ("smoke_raw_generations.jsonl" if args.mode == "smoke" else "query_predictions.jsonl")).open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.mode == "smoke":
        smoke = {
            "success": all(r["parse_valid"] for r in pred_rows),
            "n": len(pred_rows),
            "parse_success_rate": sum(bool(r["parse_valid"]) for r in pred_rows) / max(1, len(pred_rows)),
            "valid_bbox_rate": sum(r["parsed_kind"] == "bbox" for r in pred_rows) / max(1, len(pred_rows)),
            "none_rate": sum(r["parsed_kind"] == "none" for r in pred_rows) / max(1, len(pred_rows)),
            "load_seconds": load_seconds,
            "peak_gpu_memory_bytes": int(peak_mem),
            "examples": pred_rows,
        }
        (RESULTS / "smoke_test.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return

    # Query-format summaries and DEV selection.
    summary_rows = []
    by_q = {}
    for qid in qids:
        rows = [r for r in pred_rows if r["query_format"] == qid]
        s = summarize(rows)
        by_q[qid] = s
        summary_rows.append({"query_format": qid, **s})
    write_csv(
        RESULTS / "query_format_results.csv",
        summary_rows,
        ["query_format", "n", "parse_success_rate", "valid_bbox_rate", "none_prediction_rate", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075", "seconds_per_entity"],
    )
    best_q = max(qids, key=lambda q: (by_q[q]["acc_iou_05"], by_q[q]["acc_iou_075"], by_q[q]["mean_iou"]))
    best_rows = [r for r in pred_rows if r["query_format"] == best_q]

    # Box-size and entity-type breakdown for best query.
    size_rows = []
    for size in ["small", "medium", "large"]:
        rows = [r for r in best_rows if r["size_bin"] == size]
        size_rows.append({"query_format": best_q, "size_bin": size, **summarize(rows)})
    write_csv(
        RESULTS / "box_size_breakdown.csv",
        size_rows,
        ["query_format", "size_bin", "n", "parse_success_rate", "valid_bbox_rate", "none_prediction_rate", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075", "seconds_per_entity"],
    )
    type_rows = []
    for etype in sorted(set(r["type"] for r in best_rows)):
        rows = [r for r in best_rows if r["type"] == etype]
        type_rows.append({"query_format": best_q, "entity_type": etype, **summarize(rows)})
    write_csv(
        RESULTS / "entity_type_breakdown.csv",
        type_rows,
        ["query_format", "entity_type", "n", "parse_success_rate", "valid_bbox_rate", "none_prediction_rate", "mean_iou", "median_iou", "acc_iou_025", "acc_iou_05", "acc_iou_075", "seconds_per_entity"],
    )

    # Correct-vs-wrong query sensitivity using best format.
    by_sample = defaultdict(list)
    for e in ground:
        by_sample[e["sample_index"]].append(e)
    sens_rows = []
    for target in ground:
        peers = [e for e in by_sample[target["sample_index"]] if e["entity_index"] != target["entity_index"]]
        if not peers:
            continue
        wrong_e = dict(target)
        wrong_e["entity"] = peers[0]["entity"]
        wrong_e["type"] = peers[0]["type"]
        correct = next(r for r in best_rows if r["sample_index"] == target["sample_index"] and r["entity_index"] == target["entity_index"])
        wrong = generate_one(model, processor, args, wrong_e, best_q)
        pred_change_iou = box_iou_or_zero(correct.get("pred_boxes"), wrong.get("pred_boxes"))
        sens_rows.append(
            {
                "sample_index": target["sample_index"],
                "entity_index": target["entity_index"],
                "target_entity": target["entity"],
                "wrong_entity": wrong_e["entity"],
                "correct_iou": correct["iou"],
                "wrong_iou": wrong["iou"],
                "correct_acc_iou_05": correct["iou"] >= 0.5,
                "wrong_acc_iou_05": wrong["iou"] >= 0.5,
                "correct_gt_wrong_gt_delta": correct["iou"] - wrong["iou"],
                "prediction_box_iou": pred_change_iou,
                "prediction_diff_iou_lt_05": pred_change_iou < 0.5,
            }
        )
    write_csv(
        RESULTS / "query_sensitivity.csv",
        sens_rows,
        ["sample_index", "entity_index", "target_entity", "wrong_entity", "correct_iou", "wrong_iou", "correct_acc_iou_05", "wrong_acc_iou_05", "correct_gt_wrong_gt_delta", "prediction_box_iou", "prediction_diff_iou_lt_05"],
    )

    # Ungroundable None screening with locked best query.
    none_rows = []
    for e in unground:
        row = generate_one(model, processor, args, e, best_q)
        none_rows.append(row)
    none_acc = sum(r["parsed_kind"] == "none" for r in none_rows) / max(1, len(none_rows))
    write_csv(
        RESULTS / "ungroundable_none_results.csv",
        [
            {
                "query_format": best_q,
                "n": len(none_rows),
                "none_accuracy": none_acc,
                "false_grounding_rate": 1.0 - none_acc,
                "parse_success_rate": sum(r["parse_valid"] for r in none_rows) / max(1, len(none_rows)),
            }
        ],
        ["query_format", "n", "none_accuracy", "false_grounding_rate", "parse_success_rate"],
    )

    run_spatial_priors(ground, payload, best_q)
    run_e2e_matched_comparison(ground)
    write_visualizations(best_rows)

    elapsed = time.time() - start_all
    sec = mean([r["seconds"] for r in pred_rows]) if pred_rows else float("nan")
    efficiency = {
        "device": args.device,
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "total_seconds": elapsed,
        "seconds_per_entity_query": sec,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") and torch.cuda.is_available() else 0),
        "screening_groundable_entities": len(ground),
        "screening_queries": len(pred_rows),
        "estimated_full_dev_seconds_for_three_formats": sec * payload["counts"]["dev_groundable_total"] * 3,
    }
    (RESULTS / "efficiency.json").write_text(json.dumps(efficiency, ensure_ascii=False, indent=2), encoding="utf-8")
    final = {"best_query": best_q, "query_summaries": by_q, "efficiency": efficiency}
    (RESULTS / "screening_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
