#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP9 = ROOT / "kais_extension/experiment9_gmner_audit"
EXP = ROOT / "kais_extension/experiment10_explicit_grounding"

spec = importlib.util.spec_from_file_location("exp10", EXP / "scripts/run_exp10_grounding.py")
exp10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp10)


def load_sidecar_by_guid():
    out = {}
    counts = defaultdict(int)
    p = EXP9 / "cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl"
    for line in p.open(encoding="utf-8"):
        row = json.loads(line)
        split = row["split"]
        counts[split] += 1
        out[f"{split}-{counts[split]}"] = row
    return out


def patch_rect(j, w=224, h=224):
    x = j % 14
    y = j // 14
    return [x * w / 14, y * h / 14, (x + 1) * w / 14, (y + 1) * h / 14]


def to_px(box):
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1 * 224, y1 * 224, x2 * 224, y2 * 224]


def draw_case(row, sidecar, out_path):
    img = Image.open(sidecar[row["sample_id"]]["image_path"]).convert("RGB").resize((224, 224))
    heat = row["attention"].reshape(14, 14)
    heat = heat / (heat.max() + 1e-12)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=180)
    ax.imshow(img)
    ax.imshow(heat, alpha=0.45, cmap="magma", extent=(0, 224, 224, 0), interpolation="nearest")
    gx1, gy1, gx2, gy2 = to_px(row["gt"])
    px1, py1, px2, py2 = to_px(row["pred"])
    ax.add_patch(plt.Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, fill=False, edgecolor="lime", linewidth=2, label="GT"))
    ax.add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1, fill=False, edgecolor="cyan", linewidth=2, label="Pred"))
    tr = patch_rect(int(row["top_patch"]))
    ax.add_patch(plt.Rectangle((tr[0], tr[1]), tr[2] - tr[0], tr[3] - tr[1], fill=False, edgecolor="white", linewidth=1.2))
    ax.set_title(f"{row['case_type']} | {row['mention']} ({row['type']}) | IoU={row['iou']:.3f}", fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.load(EXP / "cache/features_test.pt", map_location="cpu")
    ckpt = torch.load(EXP / "checkpoints/entity_best.pt", map_location="cpu")
    dim = int(data["h"].shape[-1])
    model = exp10.GroundingHead(dim, "entity").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        bs = 128
        for start in range(0, len(data["y_ground"]), bs):
            h = data["h"][start:start + bs].float().to(device)
            v = data["v"][start:start + bs].float().to(device)
            logits, box, attn = model(h, v)
            pred = exp10.cxcywh_to_xyxy(box).cpu()
            iou = exp10.iou_xyxy(pred, data["gt_xyxy"][start:start + bs]).cpu()
            for k in range(h.shape[0]):
                idx = start + k
                if int(data["y_ground"][idx].item()) != 1:
                    continue
                rows.append({
                    "idx": idx,
                    "sample_id": data["sample_id"][idx],
                    "mention": data["mention"][idx],
                    "type": data["type"][idx],
                    "gt": data["gt_xyxy"][idx].tolist(),
                    "pred": pred[k].tolist(),
                    "iou": float(iou[k]),
                    "attention": attn[k].cpu().numpy(),
                    "top_patch": int(attn[k].argmax().item()),
                })
    random_cases = random.sample(rows, 2)
    correct = sorted([r for r in rows if r["iou"] >= 0.5], key=lambda r: r["iou"], reverse=True)[:2]
    failure = sorted([r for r in rows if r["iou"] < 0.5], key=lambda r: r["iou"])[:2]
    selected = []
    seen = set()
    for tag, items in [("random", random_cases), ("correct", correct), ("failure", failure)]:
        for r in items:
            if r["idx"] in seen:
                continue
            seen.add(r["idx"])
            rr = dict(r)
            rr["case_type"] = tag
            selected.append(rr)
    sidecar = load_sidecar_by_guid()
    out_rows = []
    out_dir = EXP / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in selected:
        safe = f"{r['case_type']}_{r['sample_id']}_{r['mention'].replace('/', '_').replace(' ', '_')}.png"
        out_path = out_dir / safe
        draw_case(r, sidecar, out_path)
        clean = {k: v for k, v in r.items() if k != "attention"}
        clean["file"] = str(out_path)
        out_rows.append(clean)
    (EXP / "results/visualization_cases.json").write_text(json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
