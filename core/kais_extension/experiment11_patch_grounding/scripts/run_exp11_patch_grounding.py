#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP10 = ROOT / "kais_extension/experiment10_explicit_grounding"
EXP9 = ROOT / "kais_extension/experiment9_gmner_audit"
EXP = ROOT / "kais_extension/experiment11_patch_grounding"
RES = EXP / "results"
CACHE10 = EXP10 / "cache"

spec = importlib.util.spec_from_file_location("exp10", EXP10 / "scripts/run_exp10_grounding.py")
exp10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp10)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_payload(split):
    return torch.load(CACHE10 / f"features_{split}.pt", map_location="cpu")


def patch_geometry():
    centers = []
    boxes = []
    for j in range(196):
        col, row = j % 14, j // 14
        x1, y1 = col / 14.0, row / 14.0
        x2, y2 = (col + 1) / 14.0, (row + 1) / 14.0
        centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        boxes.append([x1, y1, x2, y2])
    centers = torch.tensor(centers, dtype=torch.float32)
    boxes = torch.tensor(boxes, dtype=torch.float32)
    pos = torch.cat([centers, boxes], dim=-1)
    return centers, boxes, pos


CENTERS, PATCH_BOXES, PATCH_POS = patch_geometry()


def box_iou_np(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    aa = np.maximum(0, a[..., 2] - a[..., 0]) * np.maximum(0, a[..., 3] - a[..., 1])
    bb = np.maximum(0, b[..., 2] - b[..., 0]) * np.maximum(0, b[..., 3] - b[..., 1])
    return inter / (aa + bb - inter + 1e-9)


def cxcywh_to_xyxy_np(box):
    cx, cy, w, h = [float(x) for x in box]
    return [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)]


def xyxy_to_cxcywh_np(box):
    x1, y1, x2, y2 = [float(x) for x in box]
    return [(x1 + x2) / 2, (y1 + y2) / 2, max(0, x2 - x1), max(0, y2 - y1)]


def metrics_for_boxes(pred, gt):
    ious = box_iou_np(np.asarray(pred), np.asarray(gt))
    return {
        "mean_iou": float(np.mean(ious)) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "acc_iou_05": float(np.mean(ious >= 0.5)) if len(ious) else 0.0,
        "acc_iou_075": float(np.mean(ious >= 0.75)) if len(ious) else 0.0,
    }


def groundable_indices(payload):
    return [i for i, y in enumerate(payload["y_ground"].tolist()) if int(y) == 1]


def spatial_priors():
    train = load_payload("train")
    dev = load_payload("dev")
    test = load_payload("test")
    tr_idx = groundable_indices(train)
    train_boxes = train["gt_xyxy"][tr_idx].numpy()
    train_c = np.asarray([xyxy_to_cxcywh_np(b) for b in train_boxes])
    mean_c = train_c.mean(axis=0)
    mean_size = [0.5, 0.5, float(train_c[:, 2].mean()), float(train_c[:, 3].mean())]
    type_mean = {}
    for typ in ["PER", "LOC", "ORG", "MISC"]:
        idx = [i for i in tr_idx if train["type"][i] == typ]
        vals = np.asarray([xyxy_to_cxcywh_np(train["gt_xyxy"][i].numpy()) for i in idx])
        type_mean[typ] = vals.mean(axis=0) if len(vals) else mean_c
    rows = []
    for split, payload in [("dev", dev), ("test", test)]:
        idx = groundable_indices(payload)
        gt = payload["gt_xyxy"][idx].numpy()
        types = [payload["type"][i] for i in idx]
        baselines = {
            "B0_full_image": [[0, 0, 1, 1] for _ in idx],
            "B1_train_mean_box": [cxcywh_to_xyxy_np(mean_c) for _ in idx],
            "B2_entity_type_mean_box": [cxcywh_to_xyxy_np(type_mean[t]) for t in types],
            "B3_centered_mean_size": [cxcywh_to_xyxy_np(mean_size) for _ in idx],
        }
        for name, pred in baselines.items():
            for group in ["overall", "PER", "LOC", "ORG", "MISC"]:
                sub = [i for i, t in enumerate(types) if group == "overall" or t == group]
                if not sub:
                    continue
                m = metrics_for_boxes([pred[i] for i in sub], [gt[i] for i in sub])
                rows.append({"split": split, "baseline": name, "group": group, "n": len(sub), **m})
    write_csv(RES / "spatial_prior_baselines.csv", rows)
    return rows


def make_targets(gt_xyxy):
    b = gt_xyxy.float()
    c = CENTERS.to(b.device)
    inside = (
        (c[:, 0].unsqueeze(0) >= b[:, 0].unsqueeze(1))
        & (c[:, 0].unsqueeze(0) <= b[:, 2].unsqueeze(1))
        & (c[:, 1].unsqueeze(0) >= b[:, 1].unsqueeze(1))
        & (c[:, 1].unsqueeze(0) <= b[:, 3].unsqueeze(1))
    )
    return inside.float()


def soft_overlap_targets(gt_xyxy):
    b = gt_xyxy.float()
    p = PATCH_BOXES.to(b.device)
    ix1 = torch.maximum(b[:, 0].unsqueeze(1), p[:, 0].unsqueeze(0))
    iy1 = torch.maximum(b[:, 1].unsqueeze(1), p[:, 1].unsqueeze(0))
    ix2 = torch.minimum(b[:, 2].unsqueeze(1), p[:, 2].unsqueeze(0))
    iy2 = torch.minimum(b[:, 3].unsqueeze(1), p[:, 3].unsqueeze(0))
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    patch_area = (1 / 14.0) ** 2
    return (inter / patch_area).clamp(0, 1)


class GroundablePatchDataset(Dataset):
    def __init__(self, payload):
        self.payload = payload
        self.idx = groundable_indices(payload)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        return {
            "h": self.payload["h"][i].float(),
            "v": self.payload["v"][i].float(),
            "gt_xyxy": self.payload["gt_xyxy"][i].float(),
            "type": self.payload["type"][i],
            "sample_id": self.payload["sample_id"][i],
            "mention": self.payload["mention"][i],
            "mner_correct": self.payload["mner_correct"][i].float(),
        }


def collate(batch):
    return {
        "h": torch.stack([b["h"] for b in batch]),
        "v": torch.stack([b["v"] for b in batch]),
        "gt_xyxy": torch.stack([b["gt_xyxy"] for b in batch]),
        "type": [b["type"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "mention": [b["mention"] for b in batch],
        "mner_correct": torch.stack([b["mner_correct"] for b in batch]),
    }


class PatchHead(nn.Module):
    def __init__(self, dim, use_pos=False):
        super().__init__()
        self.use_pos = use_pos
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim + (6 if use_pos else 0), dim)

    def forward(self, h, v):
        q = self.wq(h)
        if self.use_pos:
            pos = PATCH_POS.to(v.device).unsqueeze(0).expand(v.shape[0], -1, -1)
            v_in = torch.cat([v, pos], dim=-1)
        else:
            v_in = v
        k = self.wk(v_in)
        return (q.unsqueeze(1) * k).sum(-1) / math.sqrt(k.shape[-1])


def focal_loss(logits, target, pos_weight, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    return ((1 - pt).pow(gamma) * bce).mean()


def box_from_probs(probs, rule):
    p = np.asarray(probs, dtype=float)
    if rule.startswith("thr_"):
        thr = float(rule.split("_")[1])
        idx = np.where(p >= thr)[0]
    else:
        frac = float(rule.split("_")[1])
        k = max(1, int(math.ceil(196 * frac)))
        idx = np.argsort(p)[-k:]
    if len(idx) == 0:
        idx = [int(np.argmax(p))]
    xs1 = PATCH_BOXES[idx, 0].numpy()
    ys1 = PATCH_BOXES[idx, 1].numpy()
    xs2 = PATCH_BOXES[idx, 2].numpy()
    ys2 = PATCH_BOXES[idx, 3].numpy()
    return [float(xs1.min()), float(ys1.min()), float(xs2.max()), float(ys2.max())]


def aucs_flat(targets, probs):
    y = targets.reshape(-1).astype(int).tolist()
    s = probs.reshape(-1).astype(float).tolist()
    return exp10.auroc(y, s), exp10.aupr(y, s), float(np.mean(y))


def eval_patch_model(model, loader, device, decode_rule=None):
    model.eval()
    all_probs, all_targets, all_soft, gt_boxes, types, sample_ids, mentions, mner = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            h = batch["h"].to(device)
            v = batch["v"].to(device)
            gt = batch["gt_xyxy"].to(device)
            logits = model(h, v)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(make_targets(gt).cpu())
            all_soft.append(soft_overlap_targets(gt).cpu())
            gt_boxes.extend(batch["gt_xyxy"].numpy().tolist())
            types.extend(batch["type"])
            sample_ids.extend(batch["sample_id"])
            mentions.extend(batch["mention"])
            mner.extend([int(x) for x in batch["mner_correct"].tolist()])
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    soft = torch.cat(all_soft).numpy()
    patch_auroc, patch_aupr, prevalence = aucs_flat(targets, probs)
    top = probs.argmax(axis=1)
    pointing = float(np.mean([targets[i, top[i]] > 0.5 for i in range(len(top))]))
    inside_mass = (probs * targets).sum(axis=1) / (probs.sum(axis=1) + 1e-12)
    area = targets.mean(axis=1)
    soft_area = soft.mean(axis=1)
    mass_lift = inside_mass / (area + 1e-12)
    pnorm = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)
    entropy = -(pnorm * np.log(np.clip(pnorm, 1e-12, 1))).sum(axis=1) / math.log(196)
    out = {
        "patch_auroc": patch_auroc,
        "patch_aupr": patch_aupr,
        "patch_prevalence": prevalence,
        "pointing_accuracy": pointing,
        "inside_mass": float(np.mean(inside_mass)),
        "box_area_ratio": float(np.mean(area)),
        "soft_area_ratio": float(np.mean(soft_area)),
        "mass_lift": float(np.mean(mass_lift)),
        "normalized_entropy": float(np.mean(entropy)),
        "probs": probs,
        "targets": targets,
        "gt_boxes": gt_boxes,
        "types": types,
        "sample_ids": sample_ids,
        "mentions": mentions,
        "mner_correct": mner,
    }
    if decode_rule:
        pred = [box_from_probs(row, decode_rule) for row in probs]
        bm = metrics_for_boxes(pred, gt_boxes)
        out.update({f"box_{k}": v for k, v in bm.items()})
        out["pred_boxes"] = pred
    return out


def select_decode_rule(model, dev_loader, device):
    rules = ["thr_0.3", "thr_0.5", "thr_0.7", "top_0.1", "top_0.2", "top_0.3"]
    scores = []
    for rule in rules:
        m = eval_patch_model(model, dev_loader, device, decode_rule=rule)
        scores.append((m["box_mean_iou"], rule, m))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[0][1], scores


def train_patch_config(name, use_pos, loss_type, train_p, dev_p, test_p, args, device):
    tr_ds, dv_ds, te_ds = GroundablePatchDataset(train_p), GroundablePatchDataset(dev_p), GroundablePatchDataset(test_p)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dv_loader = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    dim = int(train_p["h"].shape[-1])
    model = PatchHead(dim, use_pos=use_pos).to(device)
    # Train foreground/background weights from TRAIN center-inside targets only.
    all_y = make_targets(train_p["gt_xyxy"][groundable_indices(train_p)]).reshape(-1)
    pos = float(all_y.sum().item())
    neg = float(all_y.numel() - pos)
    pos_weight = torch.tensor([min(neg / max(pos, 1.0), 20.0)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    wait = 0
    hist = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            h = batch["h"].to(device)
            v = batch["v"].to(device)
            gt = batch["gt_xyxy"].to(device)
            y = make_targets(gt)
            logits = model(h, v)
            if loss_type == "focal":
                loss = focal_loss(logits, y, pos_weight)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rule, rule_scores = select_decode_rule(model, dv_loader, device)
        dev = eval_patch_model(model, dv_loader, device, decode_rule=rule)
        metric = dev["box_mean_iou"]
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_decode_rule": rule, "dev_mean_iou": metric, "dev_patch_auroc": dev["patch_auroc"], "dev_pointing": dev["pointing_accuracy"], "dev_mass_lift": dev["mass_lift"], "dev_entropy": dev["normalized_entropy"]})
        print(f"[{name}] epoch={epoch} loss={hist[-1]['train_loss']:.4f} rule={rule} dev_iou={metric:.4f} patch_auc={dev['patch_auroc']:.4f} pointing={dev['pointing_accuracy']:.4f}", flush=True)
        if best is None or metric > best["metric"]:
            best = {"metric": metric, "epoch": epoch, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "rule": rule}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best["state"])
    rule, dev_rules = select_decode_rule(model, dv_loader, device)
    dev = eval_patch_model(model, dv_loader, device, decode_rule=rule)
    test = eval_patch_model(model, te_loader, device, decode_rule=rule)
    payload = {"config": name, "use_pos": use_pos, "loss": loss_type, "best_epoch": best["epoch"], "decode_rule": rule, "train_pos_weight": float(pos_weight.item()), "training_time_sec": time.time() - t0, "dev": strip_arrays(dev), "test": strip_arrays(test), "history": hist, "dev_decode_candidates": [{"rule": r, "mean_iou": float(s), "acc_iou_05": float(m["box_acc_iou_05"])} for s, r, m in dev_rules]}
    torch.save({"state_dict": model.state_dict(), "summary": payload}, EXP / "checkpoints" / f"{name}_best.pt")
    return payload, model, dev, test


def strip_arrays(m):
    return {k: v for k, v in m.items() if k not in {"probs", "targets", "gt_boxes", "pred_boxes", "types", "sample_ids", "mentions", "mner_correct"}}


def type_breakdown(name, test_metrics):
    rows = []
    types = test_metrics["types"]
    gt = test_metrics["gt_boxes"]
    pred = test_metrics["pred_boxes"]
    probs = test_metrics["probs"]
    targets = test_metrics["targets"]
    for typ in ["PER", "LOC", "ORG", "MISC"]:
        idx = [i for i, t in enumerate(types) if t == typ]
        if not idx:
            continue
        bm = metrics_for_boxes([pred[i] for i in idx], [gt[i] for i in idx])
        auc, ap, prev = aucs_flat(targets[idx], probs[idx])
        top = probs[idx].argmax(axis=1)
        pointing = float(np.mean([targets[idx][i, top[i]] > 0.5 for i in range(len(idx))]))
        rows.append({"config": name, "type": typ, "n": len(idx), "patch_auroc": auc, "patch_aupr": ap, "patch_prevalence": prev, "pointing_accuracy": pointing, **bm})
    return rows


def load_exp10_groundability(device, test_payload):
    ckpt = torch.load(EXP10 / "checkpoints/entity_best.pt", map_location="cpu")
    dim = int(test_payload["h"].shape[-1])
    model = exp10.GroundingHead(dim, "entity").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = exp10.FeatureDataset(test_payload)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=exp10.collate)
    raw = exp10.eval_model(model, loader, device, threshold=0.45)
    pred = [1 if s >= 0.45 else 0 for s in raw["scores"]]
    return raw, pred


def joint_diagnostic(best_name, best_test, test_payload, device):
    ground_raw, ground_pred = load_exp10_groundability(device, test_payload)
    # best_test is groundable-only. Map predictions by sample_id/mention/type back to all entities.
    loc = {(sid, men, typ): i for i, (sid, men, typ) in enumerate(zip(best_test["sample_ids"], best_test["mentions"], best_test["types"]))}
    pred_box = best_test["pred_boxes"]
    all_rows = []
    ok_ground = []
    ok_joint = []
    ok_joint_mner = []
    for i in range(len(test_payload["y_ground"])):
        y = int(test_payload["y_ground"][i].item())
        gp = int(ground_pred[i])
        mner_ok = int(test_payload["mner_correct"][i].item())
        gcorrect = int(gp == y)
        loc_ok = 1
        if y == 1:
            key = (test_payload["sample_id"][i], test_payload["mention"][i], test_payload["type"][i])
            j = loc.get(key)
            loc_ok = int(j is not None and box_iou_np(np.asarray(pred_box[j]), np.asarray(test_payload["gt_xyxy"][i].numpy())) >= 0.5)
        joint = int(gcorrect and (y == 0 or loc_ok))
        ok_ground.append(gcorrect)
        ok_joint.append(joint)
        ok_joint_mner.append(int(joint and mner_ok))
        all_rows.append({"config": best_name, "groundability_accuracy": float(np.mean(ok_ground)), "diagnostic_joint_groundability_localization": float(np.mean(ok_joint)), "diagnostic_joint_entity_type_grounding": float(np.mean(ok_joint_mner))})
    final = all_rows[-1]
    final.update({"groundability_f1": ground_raw["groundability_f1"], "groundability_auroc": ground_raw["groundability_auroc"], "groundability_aupr": ground_raw["groundability_aupr"]})
    return final


def load_sidecar_by_guid():
    out = {}
    counts = defaultdict(int)
    for line in (EXP9 / "cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        counts[row["split"]] += 1
        out[f"{row['split']}-{counts[row['split']]}"] = row
    return out


def visualize_best(best_name, best_test):
    rows = []
    n = len(best_test["gt_boxes"])
    random.seed(42)
    rand_idx = random.sample(range(n), min(2, n))
    ious = box_iou_np(np.asarray(best_test["pred_boxes"]), np.asarray(best_test["gt_boxes"]))
    succ = np.where(ious >= 0.5)[0]
    fail = np.where(ious < 0.5)[0]
    succ_idx = succ[np.argsort(-ious[succ])[:2]].tolist() if len(succ) else []
    fail_idx = fail[np.argsort(ious[fail])[:2]].tolist() if len(fail) else []
    selected = []
    seen = set()
    for tag, idxs in [("random", rand_idx), ("success", succ_idx), ("failure", fail_idx)]:
        for i in idxs:
            if i in seen:
                continue
            seen.add(i)
            selected.append((tag, i))
    side = load_sidecar_by_guid()
    out_dir = EXP / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, i in selected:
        sid = best_test["sample_ids"][i]
        img = Image.open(side[sid]["image_path"]).convert("RGB").resize((224, 224))
        heat = best_test["probs"][i].reshape(14, 14)
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=180)
        ax.imshow(img)
        ax.imshow(heat / (heat.max() + 1e-12), alpha=0.45, cmap="magma", extent=(0, 224, 224, 0), interpolation="nearest")
        gx1, gy1, gx2, gy2 = [x * 224 for x in best_test["gt_boxes"][i]]
        px1, py1, px2, py2 = [x * 224 for x in best_test["pred_boxes"][i]]
        ax.add_patch(plt.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False, edgecolor="lime", linewidth=2))
        ax.add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False, edgecolor="cyan", linewidth=2))
        top = int(best_test["probs"][i].argmax())
        x1, y1, x2, y2 = [x * 224 for x in PATCH_BOXES[top].tolist()]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1.2))
        ax.set_title(f"{tag} {best_test['mentions'][i]} {best_test['types'][i]} IoU={ious[i]:.3f}", fontsize=8)
        ax.axis("off")
        fig.tight_layout(pad=0.05)
        fname = out_dir / f"{tag}_{sid}_{best_test['mentions'][i].replace('/', '_').replace(' ', '_')}.png"
        fig.savefig(fname)
        plt.close(fig)
        rows.append({"case_type": tag, "sample_id": sid, "mention": best_test["mentions"][i], "type": best_test["types"][i], "iou": float(ious[i]), "file": str(fname)})
    write_json(RES / "visualization_cases.json", rows)


def make_report(spatial_rows, summaries, patch_rows, box_rows, type_rows, joint, best):
    full = next(r for r in spatial_rows if r["split"] == "test" and r["baseline"] == "B0_full_image" and r["group"] == "overall")
    mean = next(r for r in spatial_rows if r["split"] == "test" and r["baseline"] == "B1_train_mean_box" and r["group"] == "overall")
    typ = next(r for r in spatial_rows if r["split"] == "test" and r["baseline"] == "B2_entity_type_mean_box" and r["group"] == "overall")
    ctr = next(r for r in spatial_rows if r["split"] == "test" and r["baseline"] == "B3_centered_mean_size" and r["group"] == "overall")
    b = next(r for r in patch_rows if r["config"] == best)
    best_box = next(r for r in box_rows if r["config"] == best)
    prior_best = max(float(full["mean_iou"]), float(mean["mean_iou"]), float(typ["mean_iou"]), float(ctr["mean_iou"]))
    if b["pointing_accuracy"] > b["box_area_ratio"] + 0.05 and b["mass_lift"] > 1.10 and b["normalized_entropy"] < 0.95 and best_box["mean_iou"] > prior_best + 0.02:
        case = "CASE A — STRONG_LOCALIZATION"
        rec = "Proceed to Experiment 12: full Groundability-Aware GMNER integration."
    elif b["pointing_accuracy"] > b["box_area_ratio"] + 0.05 and b["mass_lift"] > 1.10 and b["normalized_entropy"] < 0.95:
        case = "CASE B — PATCH_SIGNAL_BUT_BOX_WEAK"
        rec = "Keep patch grounding; improve box decoding / region proposal / bbox head."
    elif best_box["mean_iou"] <= prior_best + 0.02 and b["mass_lift"] <= 1.05:
        case = "CASE C — SPATIAL_PRIOR_DOMINATES"
        rec = "Do not use this direct frozen CLIP patch head; investigate stronger pretrained grounding or region proposal features."
    else:
        case = "CASE D — WEAK"
        rec = "Stop this frozen-feature grounding architecture."
    write_json(RES / "best_config.json", {"best_dev_config": best, "final_case": case, "recommended_experiment_12": rec})
    report = f"""# Experiment 11: Spatial-Prior Audit and Patch-Supervised Entity Grounding

## 1. Goal

Experiment 11 separates spatial prior effects from learned entity-region correspondence. The original PGMNER checkpoint remains frozen and unchanged.

## 2. Experiment 11A: Spatial Priors

TEST spatial-prior baselines:

| Baseline | Mean IoU | Acc@0.5 |
|---|---:|---:|
| Full image | {float(full['mean_iou']):.4f} | {float(full['acc_iou_05']):.4f} |
| TRAIN mean box | {float(mean['mean_iou']):.4f} | {float(mean['acc_iou_05']):.4f} |
| Type mean box | {float(typ['mean_iou']):.4f} | {float(typ['acc_iou_05']):.4f} |
| Centered mean-size | {float(ctr['mean_iou']):.4f} | {float(ctr['acc_iou_05']):.4f} |

Experiment 10 Entity-Aware coordinate regression reference: mean IoU 0.5091, Acc@0.5 0.5130.

## 3. Experiment 11B: Patch-Supervised Grounding

The head predicts independent 14x14 patch probabilities with weighted BCE or focal BCE. P0 uses no explicit patch position. P1 concatenates patch coordinates `[x_center,y_center,x_min,y_min,x_max,y_max]` into the key projection.

## 4. Main Results

Detailed rows are saved in `results/patch_grounding_main.csv`, `results/patch_metrics.csv`, and `results/box_metrics.csv`.

BEST_DEV_CONFIG: {best}

BEST TEST patch AUROC: {float(b['patch_auroc']):.4f}
BEST TEST patch AUPR: {float(b['patch_aupr']):.4f}
BEST TEST pointing: {float(b['pointing_accuracy']):.4f}
BEST TEST mass lift: {float(b['mass_lift']):.4f}
BEST TEST entropy: {float(b['normalized_entropy']):.4f}
BEST TEST mean IoU: {float(best_box['mean_iou']):.4f}
BEST TEST Acc@0.5: {float(best_box['acc_iou_05']):.4f}

## 5. Joint Diagnostic

Experiment 10 groundability reference is reused: AUROC 0.7830, AUPR 0.6517.

Diagnostic joint groundability/localization: {float(joint['diagnostic_joint_groundability_localization']):.4f}
Diagnostic entity/type + groundability + grounding: {float(joint['diagnostic_joint_entity_type_grounding']):.4f}

## 6. Decision

FINAL CASE: {case}.

Recommended Experiment 12: {rec}

## 7. Output Files

- `results/spatial_prior_baselines.csv`
- `results/patch_grounding_main.csv`
- `results/patch_metrics.csv`
- `results/box_metrics.csv`
- `results/entity_type_breakdown.csv`
- `results/joint_diagnostic.csv`
- `results/best_config.json`
- `visualizations/*.png`
"""
    (EXP / "EXPERIMENT_11_REPORT.md").write_text(report, encoding="utf-8")
    return case, rec


def run(args):
    seed_all(args.seed)
    RES.mkdir(parents=True, exist_ok=True)
    spatial_rows = spatial_priors()
    train_p, dev_p, test_p = load_payload("train"), load_payload("dev"), load_payload("test")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    configs = [("P0-BCE", False, "bce"), ("P1-BCE", True, "bce"), ("P1-Focal", True, "focal")]
    summaries = []
    models = {}
    tests = {}
    for name, use_pos, loss in configs:
        summary, model, dev, test = train_patch_config(name, use_pos, loss, train_p, dev_p, test_p, args, device)
        summaries.append(summary)
        models[name] = model
        tests[name] = test
    # Optional entropy regularizer is not run automatically unless all are near uniform and weak; record condition only.
    patch_rows, box_rows, type_rows = [], [], []
    for s in summaries:
        name = s["config"]
        patch_rows.append({"config": name, "best_epoch": s["best_epoch"], "decode_rule": s["decode_rule"], **{k: s["test"][k] for k in ["patch_auroc", "patch_aupr", "patch_prevalence", "pointing_accuracy", "inside_mass", "box_area_ratio", "soft_area_ratio", "mass_lift", "normalized_entropy"]}})
        box_rows.append({"config": name, "best_epoch": s["best_epoch"], "decode_rule": s["decode_rule"], **{k.replace("box_", ""): v for k, v in s["test"].items() if k.startswith("box_")}})
        type_rows.extend(type_breakdown(name, tests[name]))
    write_csv(RES / "patch_grounding_main.csv", [
        {"config": s["config"], "best_epoch": s["best_epoch"], "decode_rule": s["decode_rule"], "dev_mean_iou": s["dev"]["box_mean_iou"], "test_mean_iou": s["test"]["box_mean_iou"], "test_acc_iou_05": s["test"]["box_acc_iou_05"], "test_patch_auroc": s["test"]["patch_auroc"], "test_patch_aupr": s["test"]["patch_aupr"], "test_pointing": s["test"]["pointing_accuracy"], "test_mass_lift": s["test"]["mass_lift"], "test_entropy": s["test"]["normalized_entropy"]}
        for s in summaries
    ])
    write_csv(RES / "patch_metrics.csv", patch_rows)
    write_csv(RES / "box_metrics.csv", box_rows)
    write_csv(RES / "entity_type_breakdown.csv", type_rows)
    write_json(RES / "training_summary.json", summaries)
    best = max(summaries, key=lambda s: s["dev"]["box_mean_iou"])["config"]
    joint = joint_diagnostic(best, tests[best], test_p, device)
    write_csv(RES / "joint_diagnostic.csv", [joint])
    visualize_best(best, tests[best])
    case, rec = make_report(spatial_rows, summaries, patch_rows, box_rows, type_rows, joint, best)
    print(json.dumps({"best": best, "case": case, "recommendation": rec}, indent=2, ensure_ascii=False), flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
