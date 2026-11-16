#!/usr/bin/env python3
"""One-step s_logit gradient sanity check.

This script performs no optimizer step and does not train the model. It first
tries a synthetic PartialSinkhorn-only backward pass because that directly
checks the differentiable path from a loss through Sinkhorn marginals to s_logit
without downloading encoders or running a long job.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "kais_extension" / "experiment2_mass_audit" / "results"
sys.path.insert(0, str(ROOT))

from models.partial_ot import PartialSinkhorn  # noqa: E402


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    result = {
        "status": "NOT_EXECUTED",
        "mode": "synthetic_partial_sinkhorn",
        "optimizer_step": False,
    }
    try:
        module = PartialSinkhorn(eps=0.08, n_iters=8, dust_cost=1.0, partial_mass_init=0.5)
        module.train()
        text = torch.randn(1, 4, 8, requires_grad=True)
        visual = torch.randn(1, 5, 8, requires_grad=True)
        mask_t = torch.ones(1, 4)
        mask_v = torch.ones(1, 5)
        plan, loss = module(text, visual, mask_t=mask_t, mask_v=mask_v)
        loss.backward()
        grad = module.s_logit.grad
        s_value = float(module.get_s().detach().cpu().item())
        result.update({
            "status": "EXECUTED",
            "loss": float(loss.detach().cpu().item()),
            "actual_s": s_value,
            "s_logit": float(module.s_logit.detach().cpu().item()),
            "grad_exists": grad is not None,
            "grad_value": None if grad is None else float(grad.detach().cpu().item()),
            "grad_abs": None if grad is None else abs(float(grad.detach().cpu().item())),
            "grad_finite": False if grad is None else bool(torch.isfinite(grad).all().item()),
            "grad_nonzero": False if grad is None else bool(abs(float(grad.detach().cpu().item())) > 0.0),
            "plan_mass": float(plan.detach().cpu().sum().item()),
            "note": "Synthetic check covers s_logit -> sigmoid -> s -> marginals -> Sinkhorn -> loss. It is not full PGMNER training/evaluation.",
        })
        if not math.isfinite(result["loss"]):
            result["status"] = "FAILED_NONFINITE_LOSS"
    except Exception as exc:  # noqa: BLE001
        result.update({
            "status": "FAILED",
            "error": repr(exc),
            "note": "Synthetic gradient check failed before any optimizer step.",
        })

    (OUT_DIR / "s_gradient_check.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
