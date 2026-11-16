#!/usr/bin/env python3
"""Offline auxiliary knowledge generation for PGMNER (cached JSONL)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from aux_knowledge.backends import make_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_aux")

VALID_MODES = ("clean", "verify", "train_gold_distill", "oracle")
TRAIN_ONLY_MODES = ("train_gold_distill",)


def read_conll_samples(txt_path: Path) -> Iterator[Dict[str, Any]]:
    words: List[str] = []
    bio: List[str] = []
    imgid: Optional[str] = None
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if not line:
                if words and imgid:
                    yield _sample_dict(words, bio, imgid)
                words, bio, imgid = [], [], None
                continue
            if "IMGID" in line:
                imgid = line.split(":")[-1].strip()
                continue
            sp = line.split("\t")
            if len(sp) not in (2, 4):
                continue
            words.append(sp[0])
            bio.append(sp[-1].replace("\n", ""))
    if words and imgid:
        yield _sample_dict(words, bio, imgid)


def _sample_dict(words, bio, imgid) -> Dict[str, Any]:
    from utils.utils_metrics import get_entities

    ents = get_entities(bio)
    gold = [{"type": e[0], "start": e[1], "end": e[2]} for e in ents]
    text = " ".join(words)
    sid = f"{imgid}"
    return {
        "sample_id": sid,
        "imgid": imgid,
        "text": text,
        "words": words,
        "bio_labels": bio,
        "gold_entities": gold,
    }


def find_image(img_root: Path, img_id: str) -> Optional[str]:
    for ext in (".jpg", ".jpeg", ".png", ""):
        p = img_root / (img_id + ext if ext else img_id)
        if p.is_file() and p.stat().st_size > 0:
            return str(p)
    return None


def load_fewshot(path: Optional[str]) -> List[Dict[str, str]]:
    if not path or not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_candidates(path: Optional[str]) -> Dict[str, List[str]]:
    if not path or not Path(path).is_file():
        return {}
    out: Dict[str, List[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["sample_id"])] = r.get("candidates", [])
    return out


def output_path(out_dir: Path, dataset: str, split: str, mode: str, oracle: bool) -> Path:
    prefix = "ORACLE_" if oracle else ""
    return out_dir / f"{prefix}{dataset}_{split}_{mode}.jsonl"


def load_done_ids(path: Path) -> set:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["sample_id"])
    return done


def assert_mode_guards(mode: str, split: str, allow_oracle: bool) -> None:
    if mode in TRAIN_ONLY_MODES and split != "train":
        raise RuntimeError(f"{mode} only allowed on train split, got {split}")
    if mode == "oracle" and not allow_oracle:
        raise RuntimeError(
            "oracle mode requires --allow_oracle true (leakage skyline; not a method result)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="twitter2017 or twitter2015")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--aux_mode", default="clean", choices=VALID_MODES)
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--image_dir", default="")
    ap.add_argument("--out_dir", default=str(REPO / "aux_knowledge"))
    ap.add_argument("--backend", default="stub", help="stub | openai | qwen")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--fewshot_jsonl", default=str(REPO / "aux_knowledge/fewshot.jsonl"))
    ap.add_argument("--candidates_json", default="")
    ap.add_argument("--allow_oracle", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    assert_mode_guards(args.aux_mode, args.split, args.allow_oracle)
    if args.aux_mode == "oracle":
        log.warning("=" * 60)
        log.warning("ORACLE MODE: gold labels in input — SKYLINE ONLY, NOT A METHOD RESULT")
        log.warning("=" * 60)

    data_dir = Path(args.data_dir or REPO / "data" / args.dataset)
    img_dir = Path(args.image_dir or REPO / "data" / f"{args.dataset}_images")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_path(out_dir, args.dataset, args.split, args.aux_mode, args.aux_mode == "oracle")

    txt = data_dir / f"{args.split}.txt"
    if not txt.is_file():
        raise FileNotFoundError(txt)

    few_shot = load_fewshot(args.fewshot_jsonl)
    cand_map = load_candidates(args.candidates_json)
    done = load_done_ids(out_file)
    backend = make_backend(args.backend, model=args.model)

    samples = list(read_conll_samples(txt))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    log.info("dataset=%s split=%s mode=%s backend=%s out=%s n=%d done=%d",
             args.dataset, args.split, args.aux_mode, args.backend, out_file, len(samples), len(done))

    if args.dry_run:
        log.info("dry_run: would process %d samples", len(samples) - len(done))
        return

    min_interval = 1.0 / max(args.qps, 0.1)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    n_ok = 0
    t0 = time.time()

    with open(out_file, "a", encoding="utf-8") as fout:
        for i, sample in enumerate(samples):
            sid = sample["sample_id"]
            if sid in done:
                continue
            img_path = find_image(img_dir, sample["imgid"])
            extra: Dict[str, Any] = {}
            extra["words"] = sample["words"]
            if args.aux_mode == "verify":
                extra["candidates"] = cand_map.get(sid, [])
            if args.aux_mode in ("train_gold_distill", "oracle"):
                extra["gold_entities"] = sample["gold_entities"]

            aux_text = ""
            meta: Dict[str, Any] = {}
            for attempt in range(args.max_retries):
                try:
                    time.sleep(min_interval)
                    r = backend.generate(
                        sample["text"], img_path, few_shot, args.aux_mode, extra
                    )
                    aux_text = (r.get("aux_text") or "").strip()[:500]
                    meta = {k: r.get(k) for k in ("usage", "model") if r.get(k)}
                    for k, v in (meta.get("usage") or {}).items():
                        usage_total[k] = usage_total.get(k, 0) + int(v)
                    break
                except Exception as e:
                    log.warning("retry %d/%d sid=%s err=%s", attempt + 1, args.max_retries, sid, e)
                    time.sleep(2 ** attempt)
            else:
                log.error("failed sid=%s", sid)
                continue

            row = {
                "sample_id": sid,
                "imgid": sample["imgid"],
                "split": args.split,
                "mode": args.aux_mode,
                "aux_text": aux_text,
                "text": sample["text"],
                **meta,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(sid)
            n_ok += 1
            if n_ok % 50 == 0 or n_ok == 1:
                elapsed = time.time() - t0
                rate = n_ok / max(elapsed, 1e-6)
                remain = len(samples) - len(done)
                eta_s = remain / max(rate, 1e-6)
                log.info(
                    "progress %d/%d (%.1f%%) rate=%.2f/s ETA=%.0fs usage=%s",
                    len(done), len(samples), 100 * len(done) / len(samples),
                    rate, eta_s, usage_total,
                )

    log.info("finished wrote=%d total_in_file=%d usage=%s -> %s", n_ok, len(done), usage_total, out_file)


if __name__ == "__main__":
    main()
