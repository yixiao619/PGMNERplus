#!/usr/bin/env python3
"""Fig.3 hyperparameter sweep with resume; writes results/hparam_sweep.jsonl."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fig_common import (
    BETA_VALUES,
    JSONL,
    ROOT,
    SEEDS,
    SINIT_VALUES,
    best_test_f1_from_run,
    completed_keys,
    ensure_dirs,
    run_training,
    write_jsonl_row,
)

# Backfill legacy seed=42 runs (fig3_gpu2 naming)
LEGACY_BETA = {
    0.001: "outputs/t17_fig3_beta0.001_seed42",
    0.003: "outputs/t17_fig3_beta0.003_seed42",
    0.005: "outputs/t17_fig3_beta0.005_seed42",
    0.02: "outputs/t17_fig3_beta0.02_seed42",
    0.05: "outputs/t17_fig3_beta0.05_seed42",
}
LEGACY_SINIT = {
    0.3: "outputs/t17_fig3_mass0.3_dc08_seed42",
    0.4: "outputs/t17_fig3_mass0.4_dc08_seed42",
    0.5: "outputs/t17_fig3_mass0.5_dc08_seed42",
    0.7: "outputs/t17_fig3_mass0.7_dc08_seed42",
    0.8: "outputs/t17_fig3_mass0.8_dc08_seed42",
    0.9: "outputs/t17_fig3_mass0.9_dc08_seed42",
}


def backfill_legacy() -> None:
    done = completed_keys()
    for val, rel in LEGACY_BETA.items():
        key = ("beta_polar", str(val), 42)
        if key in done:
            continue
        out = ROOT / rel
        f1 = best_test_f1_from_run(out)
        if f1 is None:
            continue
        write_jsonl_row(
            JSONL,
            {
                "hparam_name": "beta_polar",
                "value": val,
                "seed": 42,
                "best_f1": f1,
                "output_dir": str(out),
                "source": "legacy_fig3",
            },
        )
        done.add(key)
    anchor = ROOT / "outputs/pp_ib_partial_ot_grid_dc08_m06_seed42"
    f1_anchor = best_test_f1_from_run(anchor)
    if f1_anchor is not None and ("beta_polar", "0.01", 42) not in done:
        write_jsonl_row(
            JSONL,
            {
                "hparam_name": "beta_polar",
                "value": 0.01,
                "seed": 42,
                "best_f1": f1_anchor,
                "output_dir": str(anchor),
                "source": "legacy_anchor",
            },
        )
    for val, rel in LEGACY_SINIT.items():
        key = ("s_init", str(val), 42)
        if key in done:
            continue
        out = ROOT / rel
        f1 = best_test_f1_from_run(out)
        if f1 is None:
            continue
        write_jsonl_row(
            JSONL,
            {
                "hparam_name": "s_init",
                "value": val,
                "seed": 42,
                "best_f1": f1,
                "output_dir": str(out),
                "source": "legacy_fig3",
            },
        )
    if f1_anchor is not None and ("s_init", "0.6", 42) not in completed_keys():
        write_jsonl_row(
            JSONL,
            {
                "hparam_name": "s_init",
                "value": 0.6,
                "seed": 42,
                "best_f1": f1_anchor,
                "output_dir": str(anchor),
                "source": "legacy_anchor",
            },
        )


def sweep(name: str, values, gpu: int, build_extra) -> None:
    done = completed_keys()
    for val in values:
        for seed in SEEDS:
            if (name, str(val), seed) in done:
                print(f"[SKIP] {name}={val} seed={seed}")
                continue
            out = ROOT / "results" / "runs" / f"fig3_{name}_{val}_seed{seed}"
            extra = build_extra(val)
            print(f"[RUN] {name}={val} seed={seed} -> {out}")
            rc = run_training(out, seed, gpu, extra)
            f1 = best_test_f1_from_run(out)
            row = {
                "hparam_name": name,
                "value": val,
                "seed": seed,
                "best_f1": f1,
                "output_dir": str(out),
                "rc": rc,
            }
            write_jsonl_row(JSONL, row)
            if rc == 0 and f1 is not None and f1 >= 0.5:
                done.add((name, str(val), seed))
            print(f"  rc={rc} best_f1={f1}")
            if rc != 0 or f1 is None or f1 < 0.5:
                print(f"  [WARN] will retry on next sweep: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backfill-only", action="store_true")
    parser.add_argument("--skip-beta", action="store_true")
    parser.add_argument("--skip-sinit", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    backfill_legacy()
    if args.backfill_only:
        return
    if not args.skip_beta:
        sweep(
            "beta_polar",
            BETA_VALUES,
            args.gpu,
            lambda v: {"beta_polar": v, "partial_mass_init": 0.6},
        )
    if not args.skip_sinit:
        sweep(
            "s_init",
            SINIT_VALUES,
            args.gpu,
            lambda v: {"partial_mass_init": v, "beta_polar": 0.01},
        )
    print(f"Done. Results in {JSONL}")


if __name__ == "__main__":
    main()
