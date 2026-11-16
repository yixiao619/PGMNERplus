#!/usr/bin/env python3
"""Offline VLLM descriptions for PGMNER+VLLM (cached JSONL per split)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aux_knowledge.generate_aux import find_image, load_done_ids, read_conll_samples
from aux_knowledge.backends import make_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_vllm_desc")


def out_path(data_dir: Path, split: str) -> Path:
    return data_dir / "vllm_desc" / f"{split}.jsonl"


def generate_one(backend, sample: Dict[str, Any], image_path: Optional[str], max_retries: int) -> Dict[str, Any]:
    from aux_knowledge.prompts_vllm_desc import build_desc_messages

    text = sample["text"]
    last_err = None
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            if hasattr(backend, "generate_desc"):
                return backend.generate_desc(text, image_path)
            if backend.__class__.__name__ == "OpenAIVisionBackend":
                messages = build_desc_messages(text, image_path, backend._encode_image)
                resp = backend.client.chat.completions.create(
                    model=backend.model,
                    messages=messages,
                    max_tokens=getattr(backend, "max_tokens", 96),
                    temperature=0.2,
                )
                desc = (resp.choices[0].message.content or "").strip()
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                return {
                    "desc_text": desc,
                    "usage": usage,
                    "model": backend.model,
                    "wall_sec": time.time() - t0,
                }
            out = backend.generate(text, image_path, [], "clean", extra=sample)
            return {
                "desc_text": out.get("aux_text") or out.get("desc_text") or "",
                "usage": out.get("usage") or {},
                "model": out.get("model", "unknown"),
                "wall_sec": time.time() - t0,
            }
        except Exception as e:
            last_err = e
            if hasattr(backend, "generate_desc") and attempt == max_retries - 1:
                log.warning(
                    "sample %s: VLLM failed (%s); text-only fallback",
                    sample.get("sample_id", "?"),
                    e,
                )
                try:
                    return backend.generate_desc(text, None)
                except Exception as e2:
                    last_err = e2
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"VLLM failed after {max_retries} tries: {last_err}")


def iter_samples(data_dir: Path, split: str) -> Iterator[Dict[str, Any]]:
    txt = data_dir / f"{split}.txt"
    if not txt.is_file():
        raise FileNotFoundError(txt)
    for s in read_conll_samples(txt):
        s["split"] = split
        yield s


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate cached VLLM descriptions for PGMNER+VLLM")
    ap.add_argument("--dataset", required=True, help="twitter2017 | twitter2015")
    ap.add_argument("--split", default="", help="train|dev|test or empty for all")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--image_dir", default="")
    ap.add_argument(
        "--backend",
        default="stub",
        choices=["stub", "openai", "qwen", "qwen35", "structured"],
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--model_path",
        default="",
        help="Local HF weights dir (default: QWEN35_PATH or /home/LargeModels/Qwen/Qwen3.5-27B)",
    )
    ap.add_argument("--fresh", action="store_true", help="Regenerate split (ignore cache)")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--qps", type=float, default=1.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir or REPO / "data" / args.dataset)
    img_root = Path(args.image_dir or REPO / "data" / f"{args.dataset}_images")
    splits = [args.split] if args.split else ["train", "dev", "test"]

    backend = make_backend(
        args.backend,
        model=args.model,
        model_path=args.model_path,
    )
    cost_log = REPO / "data" / args.dataset / "vllm_desc" / "generation_cost.jsonl"

    for split in splits:
        outp = out_path(data_dir, split)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if args.fresh and outp.is_file():
            backup = outp.with_suffix(outp.suffix + ".bak")
            outp.rename(backup)
            log.info("fresh: moved %s -> %s", outp, backup)
        done = load_done_ids(outp)
        n_new = 0
        t_split = time.time()

        with open(outp, "a", encoding="utf-8") as wf:
            for sample in iter_samples(data_dir, split):
                sid = sample["sample_id"]
                if sid in done:
                    continue
                if args.max_samples and n_new >= args.max_samples:
                    break
                img = find_image(img_root, sample["imgid"])
                if args.dry_run:
                    desc_text = f"[dry_run] Image+tweet context for {sid}."
                    row = _row(sample, split, desc_text, {}, "dry_run", 0.0)
                else:
                    if args.qps > 0:
                        time.sleep(1.0 / args.qps)
                    gen = generate_one(backend, sample, img, args.max_retries)
                    desc_text = (gen.get("desc_text") or "").strip()
                    row = _row(
                        sample,
                        split,
                        desc_text,
                        gen.get("usage") or {},
                        gen.get("model", args.backend),
                        float(gen.get("wall_sec", 0.0)),
                    )
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                done.add(sid)
                n_new += 1
                if n_new % 50 == 0:
                    log.info("%s %s: wrote %d", args.dataset, split, n_new)

        wall = time.time() - t_split
        summary = {
            "dataset": args.dataset,
            "split": split,
            "n_new": n_new,
            "wall_sec": wall,
            "backend": args.backend,
            "model": args.model,
        }
        cost_log.parent.mkdir(parents=True, exist_ok=True)
        with open(cost_log, "a", encoding="utf-8") as cf:
            cf.write(json.dumps(summary) + "\n")
        log.info("done %s %s: +%d samples in %.1fs -> %s", args.dataset, split, n_new, wall, outp)


def _row(sample, split, desc_text, usage, model, wall_sec) -> Dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "imgid": sample["imgid"],
        "split": split,
        "desc_text": desc_text,
        "text": sample["text"],
        "usage": usage,
        "model": model,
        "wall_sec": wall_sec,
    }


if __name__ == "__main__":
    main()
