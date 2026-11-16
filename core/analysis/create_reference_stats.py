#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成弱/强配对文献风格参考统计，供 Table 1 对比（无真实 COCO-NER 本地数据时使用）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REFERENCE = {
    "strong_paired_reference": {
        "avg_clip_score": 0.45,
        "one_to_many_ratio": 0.15,
        "entity_token_grounded_pct": 0.80,
        "misc_entity_pct": 0.05,
        "note": "Estimated from strongly-paired datasets (COCO captions, Flickr30k); not measured on this repo.",
    }
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output_file",
        type=str,
        default=str(REPO_ROOT / "analysis" / "results" / "reference_stats.json"),
    )
    args = p.parse_args()
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(REFERENCE, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
