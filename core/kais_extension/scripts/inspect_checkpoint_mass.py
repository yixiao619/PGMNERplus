#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sd = torch.load(args.checkpoint, map_location="cpu")
    rows = []
    for k, v in sd.items():
        if "s_logit" in k:
            logit = float(v.item())
            sigmoid = 1.0 / (1.0 + math.exp(-logit))
            rows.append(
                {
                    "parameter": k,
                    "s_logit": logit,
                    "sigmoid": sigmoid,
                    "s_min": 0.3,
                    "s_max": 0.9,
                    "s_value": 0.3 + 0.6 * sigmoid,
                    "shape": list(v.shape),
                }
            )
    out = {"checkpoint": args.checkpoint, "mass_parameters": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
