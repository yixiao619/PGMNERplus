#!/usr/bin/env python3
"""CPU-only DEV audit of cached PGMNER/GMNER entity and patch features."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP = ROOT / "kais_extension/literature_feasibility_review"
CACHE = ROOT / "kais_extension/experiment10_explicit_grounding/cache/features_dev.pt"
GOT_CSV = ROOT / "kais_extension/experiment9_gmner_audit/results/grounding_entity_level_dev.csv"
B_CKPT = ROOT / "kais_extension/experiment24_literature_guided/runs/E24_B_GROUNDING_SUPERVISED_POT/best_model.pth"


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    out = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return out


def auroc(y, score) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    return float((ranks(score)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def aupr(y, score) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    if not pos:
        return float("nan")
    order = np.argsort(-score)
    yy = y[order]
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((precision * yy).sum() / pos)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-12)


def geometry():
    centers, boxes = [], []
    for j in range(196):
        x, y = j % 14, j // 14
        box = [x / 14, y / 14, (x + 1) / 14, (y + 1) / 14]
        boxes.append(box)
        centers.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return torch.tensor(centers), torch.tensor(boxes)


CENTERS, PATCH_BOXES = geometry()
POS = torch.cat([CENTERS, PATCH_BOXES], dim=-1)


def center_mask(boxes: torch.Tensor) -> torch.Tensor:
    c = CENTERS[None]
    b = boxes[:, None]
    return ((c[..., 0] >= b[..., 0]) & (c[..., 0] <= b[..., 2]) &
            (c[..., 1] >= b[..., 1]) & (c[..., 1] <= b[..., 3]))


def rect_from_top(scores: np.ndarray, fraction: float):
    k = max(1, int(math.ceil(196 * fraction)))
    idx = np.argsort(scores)[-k:]
    boxes = PATCH_BOXES[idx].numpy()
    return [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]


def normalized_entropy(scores: torch.Tensor) -> torch.Tensor:
    p = scores.softmax(-1)
    return -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(scores.shape[-1])


class Exp24BHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d + 6, d)
        self.dust = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, h, v):
        pos = POS.to(v).unsqueeze(0).expand(v.size(0), -1, -1)
        patch = (self.q(h).unsqueeze(1) * self.k(torch.cat([v, pos], -1))).sum(-1) / math.sqrt(v.size(-1))
        return torch.cat([patch, self.dust(h)], dim=-1)


def score_summary(name: str, scores: torch.Tensor, boxes: torch.Tensor):
    masks = center_mask(boxes)
    flat_y = masks.numpy().reshape(-1).astype(np.int64)
    flat_s = scores.numpy().reshape(-1)
    in_values = scores[masks].numpy()
    out_values = scores[~masks].numpy()
    top = scores.argmax(-1)
    pointing = masks[torch.arange(len(scores)), top].float().mean().item()
    prob = scores.softmax(-1)
    inside_mass = (prob * masks).sum(-1)
    area = masks.float().mean(-1)
    entity_differences = []
    valid_entity_aucs = []
    for mask, score in zip(masks, scores):
        if mask.any() and (~mask).any():
            entity_differences.append(float(score[mask].mean() - score[~mask].mean()))
            valid_entity_aucs.append(auroc(mask.numpy(), score.numpy()))
    rows = {
        "method": name,
        "n_groundable": len(scores),
        "patch_positive_prevalence": float(flat_y.mean()),
        "inside_score_mean": float(in_values.mean()),
        "outside_score_mean": float(out_values.mean()),
        "inside_minus_outside": float(in_values.mean() - out_values.mean()),
        "inside_score_median": float(np.median(in_values)),
        "outside_score_median": float(np.median(out_values)),
        "patch_auroc_pooled": auroc(flat_y, flat_s),
        "patch_aupr_pooled": aupr(flat_y, flat_s),
        "mean_within_entity_inside_minus_outside": float(np.mean(entity_differences)),
        "median_within_entity_inside_minus_outside": float(np.median(entity_differences)),
        "mean_entity_patch_auroc": float(np.mean(valid_entity_aucs)),
        "pointing_accuracy": pointing,
        "inside_probability_mass": float(inside_mass.mean()),
        "uniform_area_expectation": float(area.mean()),
        "mass_lift_ratio_of_means": float(inside_mass.mean() / area.mean().clamp_min(1e-12)),
        "score_variance_mean": float(scores.var(-1, unbiased=False).mean()),
        "score_std_mean": float(scores.std(-1, unbiased=False).mean()),
        "normalized_softmax_entropy": float(normalized_entropy(scores).mean()),
    }
    for frac in (0.1, 0.2, 0.3):
        vals = [iou(rect_from_top(s.numpy(), frac), b.numpy()) for s, b in zip(scores, boxes)]
        rows[f"top{int(frac*100)}_mean_iou"] = float(np.mean(vals))
        rows[f"top{int(frac*100)}_acc05"] = float(np.mean(np.asarray(vals) >= 0.5))
    return rows


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "results").mkdir(exist_ok=True)
    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    h = payload["h"].float()
    v = payload["v"].float()
    y = payload["y_ground"].bool()
    boxes = payload["gt_xyxy"].float()

    # Frozen-feature probe only: h is fused_text, not the pre-GOT cost input.
    cosine = torch.einsum("bd,bpd->bp", F.normalize(h, dim=-1), F.normalize(v, dim=-1))
    rows = [score_summary("raw_cosine_fused_span_vs_visual_proj", cosine[y], boxes[y])]

    head = Exp24BHead(h.shape[-1])
    head.load_state_dict(torch.load(B_CKPT, map_location="cpu", weights_only=True))
    head.eval()
    with torch.no_grad():
        b_logits = []
        for start in range(0, len(h), 128):
            b_logits.append(head(h[start:start + 128], v[start:start + 128]))
        b_logits = torch.cat(b_logits)
    rows.append(score_summary("exp24b_bbox_none_supervised_head", b_logits[y, :196], boxes[y]))

    # Uniform is an analytic reference, not a learned or deterministic top-k box predictor.
    masks = center_mask(boxes[y])
    rows.append({
        "method": "uniform_analytic_reference",
        "n_groundable": int(y.sum()),
        "patch_positive_prevalence": float(masks.float().mean()),
        "inside_score_mean": 0.0,
        "outside_score_mean": 0.0,
        "inside_minus_outside": 0.0,
        "inside_score_median": 0.0,
        "outside_score_median": 0.0,
        "patch_auroc_pooled": 0.5,
        "patch_aupr_pooled": float(masks.float().mean()),
        "mean_entity_patch_auroc": 0.5,
        "pointing_accuracy": float(masks.float().mean()),
        "inside_probability_mass": float(masks.float().mean()),
        "uniform_area_expectation": float(masks.float().mean()),
        "mass_lift_ratio_of_means": 1.0,
        "score_variance_mean": 0.0,
        "score_std_mean": 0.0,
        "normalized_softmax_entropy": 1.0,
        "top10_mean_iou": None,
        "top10_acc05": None,
        "top20_mean_iou": None,
        "top20_acc05": None,
        "top30_mean_iou": None,
        "top30_acc05": None,
    })

    with GOT_CSV.open(newline="") as f:
        got = list(csv.DictReader(f))
    cache_keys = [(str(payload["sample_id"][i]), payload["mention"][i], payload["type"][i]) for i in range(len(y))]
    got_keys = [(r["sample_id"], r["mention"], r["type"]) for r in got]
    exact_order = cache_keys == got_keys
    gg = [r for r in got if r["groundable"] == "1"]
    existing_got = {
        "method": "original_gota_transport_exp9_cache",
        "n_groundable": len(gg),
        "patch_positive_prevalence": float(masks.float().mean()),
        "inside_score_mean": None,
        "outside_score_mean": None,
        "inside_minus_outside": None,
        "inside_score_median": None,
        "outside_score_median": None,
        "patch_auroc_pooled": None,
        "patch_aupr_pooled": None,
        "mean_entity_patch_auroc": None,
        "pointing_accuracy": float(np.mean([float(r["pointing_hit"]) for r in gg])),
        "inside_probability_mass": float(np.mean([float(r["inside_mass"]) for r in gg])),
        "uniform_area_expectation": float(np.mean([float(r["box_area_ratio"]) for r in gg])),
        "mass_lift_ratio_of_means": float(np.mean([float(r["inside_mass"]) for r in gg]) / np.mean([float(r["box_area_ratio"]) for r in gg])),
        "score_variance_mean": None,
        "score_std_mean": None,
        "normalized_softmax_entropy": float(np.mean([float(r["norm_transport_entropy"]) for r in gg])),
    }
    for frac in (10, 20, 30):
        vals = np.asarray([float(r[f"iou_top{frac}"]) for r in gg])
        existing_got[f"top{frac}_mean_iou"] = float(vals.mean())
        existing_got[f"top{frac}_acc05"] = float((vals >= 0.5).mean())
    rows.append(existing_got)

    # Groundability from frozen score summaries. Sign is selected only to report the
    # better conceptual orientation; both orientation and raw AUROC are retained.
    entity_features = {
        "raw_cosine_max": cosine.max(-1).values,
        "raw_cosine_top10pct_mean": cosine.topk(20, dim=-1).values.mean(-1),
        "raw_cosine_variance": cosine.var(-1, unbiased=False),
        "raw_cosine_negative_entropy": -normalized_entropy(cosine),
        "exp24b_non_dustbin_probability": 1 - b_logits.softmax(-1)[:, 196],
    }
    ground_rows = []
    yy = y.numpy().astype(np.int64)
    for name, values in entity_features.items():
        score = values.numpy()
        raw_auc = auroc(yy, score)
        orientation = "higher_predicts_groundable" if raw_auc >= 0.5 else "lower_predicts_groundable"
        oriented = score if raw_auc >= 0.5 else -score
        ground_rows.append({
            "signal": name,
            "raw_auroc_higher_is_groundable": raw_auc,
            "best_oriented_auroc": auroc(yy, oriented),
            "best_oriented_aupr": aupr(yy, oriented),
            "orientation": orientation,
            "groundable_mean": float(score[yy == 1].mean()),
            "ungroundable_mean": float(score[yy == 0].mean()),
            "mean_difference_groundable_minus_ungroundable": float(score[yy == 1].mean() - score[yy == 0].mean()),
        })

    fields = sorted({k for row in rows for k in row})
    with (EXP / "results/feature_localization_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    with (EXP / "results/groundability_score_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ground_rows[0]))
        w.writeheader(); w.writerows(ground_rows)
    audit = {
        "split": "GMNER DEV only",
        "entities": len(y),
        "groundable": int(y.sum()),
        "ungroundable": int((~y).sum()),
        "cache_feature_source": payload["feature_source"],
        "cache_and_exp9_row_order_exact": exact_order,
        "original_gota_full_plan_cached": False,
        "limitations": [
            "Raw cosine uses fused_text and visual_proj; it is not the original GOT-A cost matrix.",
            "Exp9 retained entity-level GOT-A summary statistics but not all 196 transport entries, so original GOT-A patch AUROC and score variance cannot be recovered without forward extraction.",
            "Exp24-B is bbox/None supervised and is reported separately from the original unsupervised GOT-A transport.",
        ],
    }
    (EXP / "results/feature_audit_metadata.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"metadata": audit, "localization": rows, "groundability": ground_rows}, indent=2))


if __name__ == "__main__":
    main()
