#!/usr/bin/env python3
"""Run the Exp13-locked Q2 prompt once per PGMNER predicted entity."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/experiment14_gmner_integration"
E2E = Path("/data/hfy/VG-Think/E2E-GMNER")
SRC = E2E / "src"
sys.path.insert(0, str(SRC))
from e2egmner.evaluation.parsing import extract_answer_text, parse_bbox_regions  # noqa: E402

Q2 = ('Given this image, locate the visual region corresponding to:\n\n'
      'Entity: "{entity}"\nType: "{etype}"\n\nReturn only:\n[x1,y1,x2,y2]\n\n'
      'If the entity is not visually present, return:\nNone')


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def parse_prediction(text):
    answer = extract_answer_text((text or "").strip()).strip()
    if answer.lower() == "none":
        return {"kind": "none", "boxes": None, "valid": True, "answer": answer}
    try:
        boxes, valid = parse_bbox_regions(answer, strict=False, where="exp14_q2")
        if valid and boxes:
            return {"kind": "bbox", "boxes": boxes[:1], "valid": True, "answer": answer}
    except Exception:
        pass
    match = re.search(r"\[[^\[\]]*[-+]?\d[^\[\]]*\]", answer)
    if match:
        boxes, valid = parse_bbox_regions(match.group(0), strict=False, where="exp14_q2_regex")
        if valid and boxes:
            return {"kind": "bbox", "boxes": boxes[:1], "valid": True, "answer": answer}
    return {"kind": "invalid", "boxes": None, "valid": False, "answer": answer}


def entity_key(sample_index, entity):
    return f"{sample_index}\t{entity['mention']}\t{entity['official_type']}"


def load_model(device):
    base_path = E2E / "model/Qwen2.5-VL-7B-Instruct"
    lora_path = E2E / "outputs/exp_cot_paper/best"
    processor = AutoProcessor.from_pretrained(lora_path, trust_remote_code=True)
    if getattr(processor, "image_processor", None) is None:
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path, dtype=torch.bfloat16, trust_remote_code=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, lora_path, is_trainable=False)
    model.to(device).eval()
    return model, processor


@torch.no_grad()
def generate(model, processor, args, sample, entity):
    prompt = Q2.format(entity=entity["mention"], etype=entity["official_type"])
    image = Image.open(sample["image_path"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    t0 = time.time()
    generated = model.generate(
        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
        repetition_penalty=1.0, no_repeat_ngram_size=0,
    )
    elapsed = time.time() - t0
    trimmed = [generated[0, inputs["input_ids"].shape[-1]:]]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return prompt, raw, parse_prediction(raw), elapsed


def main():
    args = args_parser()
    source = EXP / f"results/pgmner_predicted_entities_{args.split}.jsonl"
    samples = [json.loads(x) for x in source.open(encoding="utf-8")]
    jobs = [(s, e) for s in samples for e in s["entities"]]
    if args.limit: jobs = jobs[:args.limit]
    output = EXP / f"results/qwen_q2_cache_{args.split}.jsonl"
    done = {}
    if output.exists():
        for line in output.open(encoding="utf-8"):
            row = json.loads(line)
            done[row["key"]] = row
    pending = [(s, e) for s, e in jobs if entity_key(s["sample_index"], e) not in done]
    print(f"split={args.split} total={len(jobs)} cached={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        return
    torch.cuda.reset_peak_memory_stats()
    load_start = time.time()
    model, processor = load_model(args.device)
    load_seconds = time.time() - load_start
    with output.open("a", encoding="utf-8") as out:
        for idx, (sample, entity) in enumerate(pending, 1):
            prompt, raw, parsed, elapsed = generate(model, processor, args, sample, entity)
            row = {
                "key": entity_key(sample["sample_index"], entity), "split": args.split,
                "sample_index": sample["sample_index"], "sample_id": sample["sample_id"],
                "image_id": sample["image_id"], "image_path": sample["image_path"],
                "mention": entity["mention"], "official_type": entity["official_type"],
                "prompt_format": "Q2_ENTITY_TYPE_LOCKED_FROM_EXP13", "prompt": prompt,
                "raw_generation": raw, "parsed_kind": parsed["kind"],
                "parse_valid": parsed["valid"], "pred_boxes": parsed["boxes"],
                "seconds": elapsed,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if idx % 25 == 0 or idx == len(pending):
                print(f"[{args.split}] {idx}/{len(pending)} generated", flush=True)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    rows = [json.loads(x) for x in output.open(encoding="utf-8")]
    summary = {
        "split": args.split, "jobs": len(jobs), "cached_rows": len(rows),
        "valid_rate": sum(bool(r["parse_valid"]) for r in rows) / max(len(rows), 1),
        "bbox_rate": sum(r["parsed_kind"] == "bbox" for r in rows) / max(len(rows), 1),
        "none_rate": sum(r["parsed_kind"] == "none" for r in rows) / max(len(rows), 1),
        "mean_generation_seconds": sum(float(r["seconds"]) for r in rows) / max(len(rows), 1),
        "model_load_seconds": load_seconds, "peak_gpu_memory_gib": peak,
        "base_model": str(E2E / "model/Qwen2.5-VL-7B-Instruct"),
        "lora": str(E2E / "outputs/exp_cot_paper/best"),
    }
    (EXP / f"results/qwen_q2_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
