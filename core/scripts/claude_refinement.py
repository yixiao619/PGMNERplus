#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可选：Claude 对预测做后处理（需 ANTHROPIC_API_KEY 与 pip install anthropic）。
当前实现为骨架：读取预测与 gold 后调用 API，输出修正建议 JSON。
完整 span 级对齐与指标重算需在工程内接到 span_f1 流程。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", type=str, default="")
    ap.add_argument("--out", type=str, default=str(REPO_ROOT / "outputs/claude_refinement_suggestions.json"))
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY not set; write stub output only.", file=sys.stderr)
        stub = {"note": "Set ANTHROPIC_API_KEY and implement refine loop for your prediction format."}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stub, indent=2), encoding="utf-8")
        return

    try:
        import anthropic
    except ImportError:
        print("pip install anthropic", file=sys.stderr)
        sys.exit(1)

    _ = anthropic.Anthropic(api_key=key)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"message": "Implement batch refinement on token/span predictions."}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
