#!/usr/bin/env python3
"""Inventory transported-mass parameters from PGMNER-style checkpoints."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "kais_extension" / "experiment2_mass_audit" / "results"

SCAN_ROOTS = [
    Path("/data/jyx/MINER-My/outputs"),
    ROOT / "runs" / "pgmner_vllm",
    ROOT / "outputs",
    ROOT / "runs",
]

CKPT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


def is_relevant(path: Path) -> bool:
    text = str(path).lower()
    keys = ["pp_ib", "partial_ot", "partial", "sinkhorn", "pgmner", "got", "mass"]
    return any(k in text for k in keys)


def find_checkpoints() -> List[Path]:
    seen = set()
    checkpoints: List[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in CKPT_SUFFIXES:
                continue
            if not is_relevant(path):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            checkpoints.append(path)
    return sorted(checkpoints, key=lambda p: str(p))


def unwrap_state(obj: Any) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    if isinstance(obj, dict):
        has_optimizer = any(k.lower().startswith("optimizer") or k == "optimizer_state_dict" for k in obj)
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key], f"dict[{key}]", has_optimizer
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj, "raw_state_dict", has_optimizer
        tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items, "dict_tensor_subset", has_optimizer
        return None, "dict_no_tensor_state", has_optimizer
    return None, type(obj).__name__, False


def extract_s(state: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    candidates = []
    for name, value in state.items():
        if "s_logit" not in name:
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            candidates.append((name, float(value.detach().cpu().reshape(-1)[0].item())))
    if not candidates:
        return None, None, None
    # Prefer the GOT-A semantic aligner parameter when multiple scalar s_logit tensors exist.
    candidates.sort(key=lambda x: (0 if "semantic_ot_aligner._partial_ot.s_logit" in x[0] else 1, x[0]))
    name, s_logit = candidates[0]
    s = 0.3 + 0.6 / (1.0 + math.exp(-s_logit))
    return name, s_logit, s


def infer_dataset(path: Path) -> str:
    text = str(path).lower()
    if "twitter2015" in text or "t15" in text or "2015" in text:
        return "Twitter-2015"
    if "twitter2017" in text or "t17" in text or "2017" in text:
        return "Twitter-2017"
    return "unknown"


def infer_seed(path: Path) -> str:
    m = re.search(r"seed[_-]?(\d+)", str(path), flags=re.I)
    return m.group(1) if m else ""


def infer_category(path: Path) -> str:
    text = str(path).lower()
    if "pp_ib_partial_ot_grid_dc10_m05_seed42" in text:
        return "paper-full"
    if "ablation" in text or "fig" in text or "mass" in text or "dc" in text:
        return "ablation"
    if "pgmner_vllm" in text:
        return "VLLM-era"
    return "unknown"


def read_saved_f1(path: Path) -> str:
    directory = path.parent
    patterns = ["*.json", "*.txt", "*.csv", "*.log"]
    f1_hits = []
    for pattern in patterns:
        for file_path in directory.glob(pattern):
            if file_path.name == path.name or file_path.stat().st_size > 5_000_000:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for regex in [
                r"(?:f1|F1|span_f1|best_f1)[\"'=:\s]+([0-9]+(?:\.[0-9]+)?)",
                r"Test score:\s*\([^)]*?f1=([0-9.]+)",
            ]:
                for m in re.finditer(regex, text):
                    f1_hits.append(f"{file_path.name}:{m.group(1)}")
    return "; ".join(f1_hits[:5])


def summarize(rows: List[Dict[str, str]], group_key: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for row in rows:
        if row["status"] != "ok" or row["actual_s"] == "":
            continue
        key = row[group_key] if group_key else "overall"
        buckets.setdefault(key or "unknown", []).append(float(row["actual_s"]))
    stats = {}
    for key, values in buckets.items():
        stats[key] = {
            "n": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": median(values),
        }
    return stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for ckpt in find_checkpoints():
        row = {
            "checkpoint_path": str(ckpt),
            "dataset": infer_dataset(ckpt),
            "seed": infer_seed(ckpt),
            "experiment_run_name": ckpt.parent.name,
            "s_param_name": "",
            "s_logit": "",
            "actual_s": "",
            "saved_f1": "",
            "category": infer_category(ckpt),
            "mtime": "",
            "status": "",
            "state_format": "",
            "has_optimizer_state": "",
            "error": "",
        }
        try:
            row["mtime"] = str(int(ckpt.stat().st_mtime))
            obj = torch.load(str(ckpt), map_location="cpu")
            state, fmt, has_opt = unwrap_state(obj)
            row["state_format"] = fmt
            row["has_optimizer_state"] = str(bool(has_opt))
            if state is None:
                row["status"] = "no_state_dict"
            else:
                name, s_logit, s = extract_s(state)
                if name is None:
                    row["status"] = "no_s_logit"
                else:
                    row["status"] = "ok"
                    row["s_param_name"] = name
                    row["s_logit"] = f"{s_logit:.12g}"
                    row["actual_s"] = f"{s:.12g}"
                    row["saved_f1"] = read_saved_f1(ckpt)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "load_error"
            row["error"] = repr(exc)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else [
        "checkpoint_path", "dataset", "seed", "experiment_run_name", "s_param_name",
        "s_logit", "actual_s", "saved_f1", "category", "mtime", "status",
        "state_format", "has_optimizer_state", "error",
    ]
    with (OUT_DIR / "checkpoint_mass_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "overall": summarize(rows),
        "by_category": summarize(rows, "category"),
        "by_dataset": summarize(rows, "dataset"),
        "by_seed": summarize(rows, "seed"),
        "n_scanned": len(rows),
        "n_with_s": sum(1 for r in rows if r["status"] == "ok"),
    }
    (OUT_DIR / "checkpoint_mass_inventory_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
