#!/usr/bin/env python3
"""CPU-only VEE module import/shape/backward sanity checks."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "kais_extension" / "experiment5_vee"

from kais_extension.experiment5_vee.models.visual_evidence_enhancer import VisualEvidenceEnhancer


def check_one(vee_type: str) -> dict:
    torch.manual_seed(42)
    hidden = torch.randn(2, 7, 32, requires_grad=True)
    visual = torch.randn(2, 5, 32, requires_grad=True)
    module = VisualEvidenceEnhancer(32, vee_type=vee_type, initial_gate=0.1)
    out = module(hidden, visual)
    loss = out.hidden.square().mean()
    loss.backward()
    grads = {
        n: {
            "finite": bool(torch.isfinite(p.grad).all().item()) if p.grad is not None else None,
            "abs_max": float(p.grad.abs().max().item()) if p.grad is not None else 0.0,
        }
        for n, p in module.named_parameters()
    }
    return {
        "vee_type": vee_type,
        "hidden_shape_ok": list(out.hidden.shape) == [2, 7, 32],
        "delta_shape_ok": list(out.delta.shape) == [2, 7, 32],
        "loss": float(loss.detach().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "backward_success": True,
        "parameter_count": module.added_parameter_count(),
        "has_trainable_parameters": any(True for _ in module.parameters()),
        "vee_gradients_finite": all(v["finite"] for v in grads.values()) if grads else True,
        "vee_gradients_nonzero": any(v["abs_max"] > 0 for v in grads.values()) if grads else True,
        "gradients": grads,
    }


def main() -> int:
    results = [check_one(v) for v in ("none", "mean", "attention", "gated_attention")]
    (EXP / "results").mkdir(parents=True, exist_ok=True)
    (EXP / "results" / "vee_gradient_sanity.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    with (EXP / "results" / "vee_parameter_counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vee_type", "added_parameters"])
        w.writeheader()
        for vee_type in ("none", "mean", "attention", "gated_attention"):
            module = VisualEvidenceEnhancer(1024, vee_type=vee_type, initial_gate=0.1)
            w.writerow({"vee_type": vee_type, "added_parameters": module.added_parameter_count()})

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(r["loss_finite"] and r["vee_gradients_finite"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
