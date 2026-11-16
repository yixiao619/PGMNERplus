#!/usr/bin/env python3
"""Search manuscript/artifact text for transported-mass and grounding claims."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "kais_extension" / "experiment2_mass_audit" / "results"
SEARCH_ROOTS = [
    ROOT,
    Path("/home/jyx/yixiao"),
    Path("/data/jyx/MINER-My/outputs"),
]
TEXT_SUFFIXES = {
    ".tex", ".md", ".txt", ".csv", ".py", ".sh", ".ipynb", ".log", ".json", ".out",
}
SKIP_PARTS = {".git", "__pycache__", ".cache", "node_modules"}

GROUNDING_TERMS = [
    "41.2", "38.7", "0.412", "0.387", "grounding ratio", "grounded",
    "matched region", "matched-region", "CLIP", "cosine", "threshold",
    "0.25", "learned s", "transported mass",
]

CLAIM_RE = re.compile(
    r".*(learnable transported mass|learned s|grounding ratio|fraction of grounded tokens|"
    r"stationary point|dataset grounding|partial mass|transported mass).*",
    re.I,
)


def iter_text_files():
    seen = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            try:
                if path.stat().st_size > 20_000_000:
                    continue
            except OSError:
                continue
            yield path


def classify_claim(text: str) -> str:
    low = text.lower()
    if "learnable" in low or "learned" in low:
        return "mathematical_or_empirical"
    if "control" in low or "transported mass" in low or "partial mass" in low:
        return "mechanism"
    if "stationary" in low or "theorem" in low:
        return "theoretical"
    if "grounding ratio" in low or "fraction" in low:
        return "empirical"
    return "unknown"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grounding_hits = []
    manuscript_claims = []
    for path in iter_text_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, 1):
            compact = line.strip()
            if not compact:
                continue
            low = compact.lower()
            matched_terms = [t for t in GROUNDING_TERMS if t.lower() in low]
            if matched_terms:
                grounding_hits.append({
                    "source_file": str(path),
                    "line": idx,
                    "matched_terms": ";".join(matched_terms),
                    "text": compact[:1000],
                })
            if path.suffix.lower() in {".tex", ".md"} and CLAIM_RE.match(compact):
                manuscript_claims.append({
                    "source_file": str(path),
                    "section": "",
                    "original_claim": compact[:1500],
                    "claim_type": classify_claim(compact),
                    "supporting_artifact": "",
                    "support_status": "NOT_DIRECTLY_VERIFIED",
                })

    with (OUT_DIR / "grounding_search_hits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_file", "line", "matched_terms", "text"])
        writer.writeheader()
        writer.writerows(grounding_hits)

    with (OUT_DIR / "manuscript_s_claims.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_file", "section", "original_claim", "claim_type", "supporting_artifact", "support_status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manuscript_claims)


if __name__ == "__main__":
    main()
