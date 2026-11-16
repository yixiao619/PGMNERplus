#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP9 = ROOT / "kais_extension/experiment9_gmner_audit"
EXP = ROOT / "kais_extension/experiment10_explicit_grounding"
RES = EXP / "results"
CACHE = EXP / "cache"
CKPT = "/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth"


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows, fieldnames=None) -> None:
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "extract", "train", "report"], default="all")
    ap.add_argument("--gpu-id", default="0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-extract", action="store_true")
    return ap.parse_args()


def bbox_norm(b, image_path):
    if not b:
        return None
    try:
        w, h = Image.open(image_path).size
    except Exception:
        return None
    x1, y1, x2, y2 = b
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
        vals = [x1, y1, x2, y2]
        return [max(0.0, min(1.0, float(v))) for v in vals]
    scale = 256.0 / min(float(w), float(h))
    rw, rh = float(w) * scale, float(h) * scale
    left = (rw - 224.0) / 2.0
    top = (rh - 224.0) / 2.0
    xx1 = (x1 * scale - left) / 224.0
    yy1 = (y1 * scale - top) / 224.0
    xx2 = (x2 * scale - left) / 224.0
    yy2 = (y2 * scale - top) / 224.0
    lo_x, hi_x = sorted([xx1, xx2])
    lo_y, hi_y = sorted([yy1, yy2])
    lo_x = max(0.0, min(1.0, lo_x))
    hi_x = max(0.0, min(1.0, hi_x))
    lo_y = max(0.0, min(1.0, lo_y))
    hi_y = max(0.0, min(1.0, hi_y))
    if hi_x <= lo_x or hi_y <= lo_y:
        return None
    return [lo_x, lo_y, hi_x, hi_y]


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)
    lo_x = torch.minimum(x1, x2)
    hi_x = torch.maximum(x1, x2)
    lo_y = torch.minimum(y1, y2)
    hi_y = torch.maximum(y1, y2)
    return torch.stack([lo_x, lo_y, hi_x, hi_y], dim=-1)


def iou_xyxy(a, b):
    ix1 = torch.maximum(a[..., 0], b[..., 0])
    iy1 = torch.maximum(a[..., 1], b[..., 1])
    ix2 = torch.minimum(a[..., 2], b[..., 2])
    iy2 = torch.minimum(a[..., 3], b[..., 3])
    iw = (ix2 - ix1).clamp_min(0)
    ih = (iy2 - iy1).clamp_min(0)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]).clamp_min(0) * (a[..., 3] - a[..., 1]).clamp_min(0)
    area_b = (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)
    return inter / (area_a + area_b - inter + 1e-9)


def giou_loss(pred_xyxy, gt_xyxy):
    iou = iou_xyxy(pred_xyxy, gt_xyxy)
    cx1 = torch.minimum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    cy1 = torch.minimum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    cx2 = torch.maximum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    cy2 = torch.maximum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    c_area = (cx2 - cx1).clamp_min(0) * (cy2 - cy1).clamp_min(0) + 1e-9
    area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp_min(0) * (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp_min(0)
    area_g = (gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp_min(0) * (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp_min(0)
    inter = iou * (area_p + area_g) / (1 + iou + 1e-9)
    union = area_p + area_g - inter
    giou = iou - (c_area - union) / c_area
    return (1.0 - giou).mean()


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def auroc(y, score):
    pairs = [(int(a), float(b)) for a, b in zip(y, score)]
    npos = sum(a for a, _ in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.0
    rs = ranks([b for _, b in pairs])
    sum_pos = sum(r for r, (a, _) in zip(rs, pairs) if a == 1)
    return float((sum_pos - npos * (npos + 1) / 2) / (npos * nneg))


def aupr(y, score):
    pairs = sorted([(int(a), float(b)) for a, b in zip(y, score)], key=lambda x: x[1], reverse=True)
    pos = sum(a for a, _ in pairs)
    if pos == 0:
        return 0.0
    tp = 0
    fp = 0
    last_recall = 0.0
    area = 0.0
    for a, _ in pairs:
        if a:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        prec = tp / (tp + fp)
        area += (recall - last_recall) * prec
        last_recall = recall
    return float(area)


def prf_from_scores(y, score, threshold):
    pred = [1 if s >= threshold else 0 for s in score]
    tp = sum(1 for p, g in zip(pred, y) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, y) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, y) if p == 0 and g == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def load_sidecar():
    path = EXP9 / "cache/gmner_pgmner_format/gmner_entity_sidecar.jsonl"
    out = {}
    counts = defaultdict(int)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        split = row["split"]
        counts[split] += 1
        out[(split, f"{split}-{counts[split]}")] = row
    return out


def load_pgmner(args):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "main-mner-robert.py",
        "--data_dir", str(EXP9 / "cache/gmner_pgmner_format"),
        "--output_dir", str(RES / "pgmner_side_outputs"),
        "--model_name_or_path", "pretrain/roberta-large",
        "--tokenizer_name", "pretrain/roberta-large",
        "--vit_name", "pretrain/clip-vit-base-patch16",
        "--gpu_id", args.gpu_id,
        "--do_train", "false",
        "--do_eval", "false",
        "--do_predict", "false",
        "--batch_size", str(args.batch_size),
        "--max_span_len", "4",
        "--max_seq_len", "128",
        "--seed", str(args.seed),
        "--use_pp_ib",
        "--use_got",
        "--got_mode", "partial_sinkhorn",
        "--ot_sinkhorn_iter", "30",
        "--partial_mass_init", "0.5",
        "--dust_cost", "1.0",
        "--entropy_eps", "0.1",
        "--ot_alpha", "0.05",
        "--auto_img_dims", "true",
        "--use_hf_clip_vision", "true",
    ]
    spec = importlib.util.spec_from_file_location("pgmner_main_exp10", ROOT / "main-mner-robert.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.args.labels = str(EXP9 / "cache/gmner_pgmner_format/labels.txt")
    return mod


def load_with_remap(model):
    state = torch.load(CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    old = "image_model._hf.encoder.vision_model."
    new = "image_model._hf.encoder."
    mapped = {new + k[len(old):] if k.startswith(old) else k: v for k, v in state.items()}
    model.load_state_dict(mapped, strict=True)


def extract_features(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    pg = load_pgmner(args)
    labels = pg.get_labels(pg.args.labels)
    config = pg.AutoConfig.from_pretrained(
        pg.args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = pg.RobertaTokenizer.from_pretrained(pg.args.tokenizer_name)
    model = pg.BertSpanNerBN(config, args=pg.args, num_labels=len(labels))
    load_with_remap(model)
    model.to(pg.args.device)
    model.eval()
    model.set_partial_ot_warmup(False)
    for p in model.parameters():
        p.requires_grad_(False)
    side = load_sidecar()
    from utils.datasets import SpanNerDataset, collate_fn

    for split in ["train", "dev", "test"]:
        out_path = CACHE / f"features_{split}.pt"
        if out_path.exists() and not args.force_extract:
            print(f"[extract] reuse {out_path}", flush=True)
            continue
        examples = pg.load_examples(
            pg.args.data_dir,
            mode=split,
            tokenizer=tokenizer,
            visual_suffix_map=pg.args.visual_suffix_map,
            aux_knowledge_map=pg.resolve_aux_knowledge_map(pg.args, split),
            vllm_desc_map=pg.resolve_vllm_desc_map(pg.args, split),
            max_seq_len=pg.args.max_seq_len,
            max_aux_words=int(getattr(pg.args, "max_aux_words", 12)),
            aux_dropout=0.0,
            desc_max_words=int(getattr(pg.args, "desc_max_words", 60)),
            vllm_desc_dropout=0.0,
            desc_sep_token=str(getattr(pg.args, "desc_sep_token", "</s>")),
        )
        ds = SpanNerDataset(examples, args=pg.args, tokenizer=tokenizer, labels=labels, dev=True)
        dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=args.batch_size, collate_fn=collate_fn)
        h_list, v_list, y_ground, gt_box, gt_xyxy, types, mner_ok, sample_ids, mentions = [], [], [], [], [], [], [], [], []
        sample_idx = 0
        with torch.no_grad():
            for bi, batch_cpu in enumerate(dl):
                batch = {k: v.to(pg.args.device) for k, v in batch_cpu.items()}
                spans_rep, txt_rep_g, attn_last = model.span_encoding(**batch)
                if getattr(model, "use_pp_ib", False):
                    model.dist_t(txt_rep_g)
                img_g, img_l = model.image_model(batch["image"])
                if getattr(model, "use_pp_ib", False):
                    model.dist_v(img_g)
                text_proj, visual_proj, align_text, _, _, _, _ = model.semantic_ot_aligner(
                    spans_rep,
                    img_l,
                    mask_t=batch["span_masks"].float(),
                    mask_v=None,
                    attn_t=attn_last,
                    span_token_idxes=batch.get("span_token_idxes"),
                    patch_pos=None,
                )
                fused = model.ln_text(model.fusion_text(torch.cat([text_proj, align_text], dim=-1)) + align_text)
                probs = model.softmax(model.span_classifier(fused))
                spatial_v = visual_proj[:, 1:, :].float().cpu()
                for k, ex in enumerate(examples[sample_idx:sample_idx + batch["input_ids"].shape[0]]):
                    meta = side.get((split, str(ex.guid)), {})
                    spans = batch["span_word_idxes"][k].detach().cpu().numpy()
                    masks = batch["span_masks"][k].detach().cpu().numpy()
                    pred = probs[k].argmax(-1).detach().cpu().numpy()
                    for ent in meta.get("entities", []):
                        if not ent.get("matched_in_text"):
                            continue
                        s = int(ent["token_start"])
                        e = int(ent["token_end"])
                        typ = ent["type"]
                        lab_id = labels.index(typ) if typ in labels else -1
                        span_i = None
                        for ii, (a, b) in enumerate(spans):
                            if masks[ii] > 0 and int(a) == s and int(b) == e:
                                span_i = ii
                                break
                        if span_i is None:
                            continue
                        box = bbox_norm(ent.get("bbox"), meta.get("image_path", "")) if ent.get("bbox") else None
                        h_list.append(fused[k, span_i].detach().float().cpu())
                        v_list.append(spatial_v[k].clone())
                        y_ground.append(1 if box else 0)
                        if box:
                            gt_xyxy.append(torch.tensor(box, dtype=torch.float32))
                            gt_box.append(torch.tensor(xyxy_to_cxcywh(box), dtype=torch.float32))
                        else:
                            gt_xyxy.append(torch.zeros(4))
                            gt_box.append(torch.zeros(4))
                        types.append(typ)
                        mner_ok.append(1 if lab_id >= 0 and int(pred[span_i]) == lab_id else 0)
                        sample_ids.append(str(ex.guid))
                        mentions.append(ent["mention"])
                sample_idx += batch["input_ids"].shape[0]
                print(f"[extract:{split}] batch {bi+1}/{len(dl)} entities={len(y_ground)}", flush=True)
        payload = {
            "h": torch.stack(h_list).half(),
            "v": torch.stack(v_list).half(),
            "y_ground": torch.tensor(y_ground, dtype=torch.float32),
            "gt_box": torch.stack(gt_box).float(),
            "gt_xyxy": torch.stack(gt_xyxy).float(),
            "type": types,
            "mner_correct": torch.tensor(mner_ok, dtype=torch.float32),
            "sample_id": sample_ids,
            "mention": mentions,
            "feature_source": "h=fused_text before original span classifier; v=semantic_ot_aligner visual_proj excluding CLIP class token",
        }
        torch.save(payload, out_path)
        print(f"[extract:{split}] saved {out_path} n={len(y_ground)}", flush=True)


class FeatureDataset(Dataset):
    def __init__(self, payload):
        self.p = payload

    def __len__(self):
        return int(self.p["y_ground"].numel())

    def __getitem__(self, idx):
        return {
            "h": self.p["h"][idx].float(),
            "v": self.p["v"][idx].float(),
            "y_ground": self.p["y_ground"][idx].float(),
            "gt_box": self.p["gt_box"][idx].float(),
            "gt_xyxy": self.p["gt_xyxy"][idx].float(),
            "type": self.p["type"][idx],
            "mner_correct": self.p["mner_correct"][idx].float(),
            "sample_id": self.p["sample_id"][idx],
            "mention": self.p["mention"][idx],
        }


class GroundingHead(nn.Module):
    def __init__(self, dim: int, variant: str):
        super().__init__()
        self.variant = variant
        self.wh = nn.Linear(dim, dim)
        if variant == "entity":
            self.wq = nn.Linear(dim, dim)
            self.wk = nn.Linear(dim, dim)
            self.wv = nn.Linear(dim, dim)
        else:
            self.wv = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ground = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, 1))
        self.box = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, h, v):
        if self.variant == "global":
            z = self.wv(v)
            e = z.mean(dim=1)
            attn = torch.full((v.shape[0], v.shape[1]), 1.0 / v.shape[1], device=v.device)
        else:
            q = self.wq(h).unsqueeze(1)
            k = self.wk(v)
            z = self.wv(v)
            score = (q * k).sum(-1) / math.sqrt(k.shape[-1])
            attn = torch.softmax(score, dim=-1)
            e = torch.bmm(attn.unsqueeze(1), z).squeeze(1)
        r = self.norm(self.wh(h) + e)
        feat = torch.cat([r, e], dim=-1)
        return self.ground(feat).squeeze(-1), torch.sigmoid(self.box(feat)), attn


def collate(batch):
    out = {}
    for key in ["h", "v", "y_ground", "gt_box", "gt_xyxy", "mner_correct"]:
        out[key] = torch.stack([b[key] for b in batch])
    for key in ["type", "sample_id", "mention"]:
        out[key] = [b[key] for b in batch]
    return out


def attention_box_mass(attn, gt_xyxy, y):
    centers = []
    for j in range(196):
        centers.append(((j % 14 + 0.5) / 14.0, (j // 14 + 0.5) / 14.0))
    vals = []
    areas = []
    for row, box, yy in zip(attn.detach().cpu(), gt_xyxy.detach().cpu(), y.detach().cpu()):
        if int(yy.item()) != 1:
            continue
        x1, y1, x2, y2 = box.tolist()
        mask = [(x1 <= x <= x2 and y1 <= yc <= y2) for x, yc in centers]
        vals.append(float(row[mask].sum().item()) if any(mask) else 0.0)
        areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
    return vals, areas


def eval_model(model, loader, device, threshold=0.5):
    model.eval()
    ys, scores, ious, types, mner, entropies, maxps, masses, areas = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            h = batch["h"].to(device)
            v = batch["v"].to(device)
            y = batch["y_ground"].to(device)
            gt_box = batch["gt_box"].to(device)
            gt_xyxy = batch["gt_xyxy"].to(device)
            logits, box, attn = model(h, v)
            prob = torch.sigmoid(logits)
            pred_xyxy = cxcywh_to_xyxy(box)
            iou = iou_xyxy(pred_xyxy, gt_xyxy)
            entropy = (-(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(-1) / math.log(attn.shape[-1])).detach().cpu()
            mass, area = attention_box_mass(attn, gt_xyxy, y)
            ys.extend([int(x) for x in y.cpu().tolist()])
            scores.extend([float(x) for x in prob.cpu().tolist()])
            ious.extend([float(x) for x in iou.cpu().tolist()])
            types.extend(batch["type"])
            mner.extend([int(x) for x in batch["mner_correct"].cpu().tolist()])
            entropies.extend([float(x) for x in entropy.tolist()])
            maxps.extend([float(x) for x in attn.max(-1).values.detach().cpu().tolist()])
            masses.extend(mass)
            areas.extend(area)
    p, r, f1 = prf_from_scores(ys, scores, threshold)
    gidx = [i for i, y in enumerate(ys) if y == 1]
    pred_ground = [1 if s >= threshold else 0 for s in scores]
    joint_ok = []
    for i, y in enumerate(ys):
        if mner[i] != 1:
            joint_ok.append(0)
        elif y == 0:
            joint_ok.append(1 if pred_ground[i] == 0 else 0)
        else:
            joint_ok.append(1 if pred_ground[i] == 1 and ious[i] >= 0.5 else 0)
    return {
        "groundability_auroc": auroc(ys, scores),
        "groundability_aupr": aupr(ys, scores),
        "groundability_precision": p,
        "groundability_recall": r,
        "groundability_f1": f1,
        "threshold": threshold,
        "mean_iou": float(np.mean([ious[i] for i in gidx])) if gidx else 0.0,
        "median_iou": float(np.median([ious[i] for i in gidx])) if gidx else 0.0,
        "acc_iou_05": float(np.mean([ious[i] >= 0.5 for i in gidx])) if gidx else 0.0,
        "acc_iou_075": float(np.mean([ious[i] >= 0.75 for i in gidx])) if gidx else 0.0,
        "joint_metric": float(np.mean(joint_ok)) if joint_ok else 0.0,
        "attention_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "attention_max_prob": float(np.mean(maxps)) if maxps else 0.0,
        "gt_box_attention_mass": float(np.mean(masses)) if masses else 0.0,
        "random_area_expectation": float(np.mean(areas)) if areas else 0.0,
        "ys": ys,
        "scores": scores,
        "ious": ious,
        "types": types,
        "mner": mner,
    }


def choose_threshold(dev_metrics):
    y, score = dev_metrics["ys"], dev_metrics["scores"]
    candidates = sorted(set([i / 100 for i in range(5, 96, 5)] + [0.5]))
    best = (None, -1.0)
    for th in candidates:
        _, _, f1 = prf_from_scores(y, score, th)
        if f1 > best[1]:
            best = (th, f1)
    return float(best[0])


def train_variant(name, train_p, dev_p, test_p, args, device):
    train_ds, dev_ds, test_ds = FeatureDataset(train_p), FeatureDataset(dev_p), FeatureDataset(test_p)
    train_loader = DataLoader(train_ds, batch_size=args.head_batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.head_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.head_batch_size, shuffle=False, collate_fn=collate)
    dim = int(train_p["h"].shape[-1])
    variant = "global" if name == "global" else "entity"
    model = GroundingHead(dim, variant).to(device)
    pos = float(train_p["y_ground"].sum().item())
    neg = float(train_p["y_ground"].numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    wait = 0
    rows = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            h = batch["h"].to(device)
            v = batch["v"].to(device)
            y = batch["y_ground"].to(device)
            gt_box = batch["gt_box"].to(device)
            gt_xyxy = batch["gt_xyxy"].to(device)
            logits, box, _ = model(h, v)
            loss_ground = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            mask = y > 0.5
            if mask.any():
                pred_xyxy = cxcywh_to_xyxy(box[mask])
                loss_box = nn.functional.l1_loss(box[mask], gt_box[mask]) + giou_loss(pred_xyxy, gt_xyxy[mask])
            else:
                loss_box = logits.sum() * 0.0
            loss = loss_ground + loss_box
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        dev_raw = eval_model(model, dev_loader, device, threshold=0.5)
        proxy = 0.5 * dev_raw["groundability_auroc"] + 0.5 * dev_raw["acc_iou_05"]
        row = {"variant": name, "epoch": epoch, "train_loss": float(np.mean(losses)), "dev_proxy": proxy, **{k: v for k, v in dev_raw.items() if k not in {"ys", "scores", "ious", "types", "mner"}}}
        rows.append(row)
        print(f"[train:{name}] epoch={epoch} loss={row['train_loss']:.4f} dev_proxy={proxy:.4f} dev_auc={dev_raw['groundability_auroc']:.4f} dev_acc50={dev_raw['acc_iou_05']:.4f}", flush=True)
        if best is None or proxy > best["proxy"]:
            best = {"proxy": proxy, "epoch": epoch, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best["state"])
    dev_for_th = eval_model(model, dev_loader, device, threshold=0.5)
    threshold = choose_threshold(dev_for_th)
    dev_final = eval_model(model, dev_loader, device, threshold=threshold)
    test_final = eval_model(model, test_loader, device, threshold=threshold)
    out = {
        "variant": name,
        "best_epoch": best["epoch"],
        "best_dev_proxy": best["proxy"],
        "threshold": threshold,
        "training_time_sec": time.time() - t0,
        "dev": {k: v for k, v in dev_final.items() if k not in {"ys", "scores", "ious", "types", "mner"}},
        "test": {k: v for k, v in test_final.items() if k not in {"ys", "scores", "ious", "types", "mner"}},
        "history": rows,
    }
    torch.save({"state_dict": model.state_dict(), "summary": out}, EXP / "checkpoints" / f"{name}_best.pt")
    return out, model, (dev_final, test_final)


def entity_type_breakdown(metrics, threshold):
    rows = []
    by_type = defaultdict(list)
    for i, typ in enumerate(metrics["types"]):
        by_type[typ].append(i)
    for typ, idxs in sorted(by_type.items()):
        y = [metrics["ys"][i] for i in idxs]
        score = [metrics["scores"][i] for i in idxs]
        ious = [metrics["ious"][i] for i in idxs if metrics["ys"][i] == 1]
        p, r, f1 = prf_from_scores(y, score, threshold)
        rows.append({
            "type": typ,
            "n": len(idxs),
            "groundable": sum(y),
            "groundability_auroc": auroc(y, score),
            "groundability_aupr": aupr(y, score),
            "groundability_f1": f1,
            "groundability_precision": p,
            "groundability_recall": r,
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "acc_iou_05": float(np.mean([x >= 0.5 for x in ious])) if ious else 0.0,
        })
    return rows


def train_all(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_p = torch.load(CACHE / "features_train.pt", map_location="cpu")
    dev_p = torch.load(CACHE / "features_dev.pt", map_location="cpu")
    test_p = torch.load(CACHE / "features_test.pt", map_location="cpu")
    summaries = []
    full = {}
    best_models = {}
    full_metrics = {}
    for name in ["global", "entity"]:
        summary, model, metrics = train_variant(name, train_p, dev_p, test_p, args, device)
        summaries.append(summary)
        full[name] = summary
        best_models[name] = model
        full_metrics[name] = metrics
    main_rows = []
    ground_rows = []
    joint_rows = []
    attention_rows = []
    type_rows = []
    for s in summaries:
        name = s["variant"]
        main_rows.append({
            "variant": name,
            "best_epoch": s["best_epoch"],
            "best_dev_proxy": s["best_dev_proxy"],
            "threshold": s["threshold"],
            "dev_groundability_auroc": s["dev"]["groundability_auroc"],
            "test_groundability_auroc": s["test"]["groundability_auroc"],
            "test_groundability_aupr": s["test"]["groundability_aupr"],
            "test_mean_iou": s["test"]["mean_iou"],
            "test_median_iou": s["test"]["median_iou"],
            "test_acc_iou_05": s["test"]["acc_iou_05"],
            "test_acc_iou_075": s["test"]["acc_iou_075"],
            "test_joint_metric": s["test"]["joint_metric"],
        })
        ground_rows.append({"variant": name, **{k: s["test"][k] for k in ["groundability_auroc", "groundability_aupr", "groundability_precision", "groundability_recall", "groundability_f1", "threshold"]}})
        joint_rows.append({"variant": name, "joint_entity_type_grounding_accuracy": s["test"]["joint_metric"], "official_metric_status": "NOT_DIRECTLY_APPLIED_generation_evaluator_available"})
        attention_rows.append({"variant": name, "attention_entropy": s["test"]["attention_entropy"], "attention_max_prob": s["test"]["attention_max_prob"], "gt_box_attention_mass": s["test"]["gt_box_attention_mass"], "random_area_expectation": s["test"]["random_area_expectation"]})
        # Re-evaluate to get per-entity raw vectors for type breakdown.
        loader = DataLoader(FeatureDataset(test_p), batch_size=args.head_batch_size, shuffle=False, collate_fn=collate)
        raw = eval_model(best_models[name], loader, device, threshold=s["threshold"])
        for row in entity_type_breakdown(raw, s["threshold"]):
            row["variant"] = name
            type_rows.append(row)
    write_csv(RES / "grounding_main.csv", main_rows)
    write_csv(RES / "groundability_main.csv", ground_rows)
    write_csv(RES / "joint_gmner_metrics.csv", joint_rows)
    write_csv(RES / "attention_diagnostics.csv", attention_rows)
    write_csv(RES / "entity_type_breakdown.csv", type_rows)
    write_json(RES / "training_summary.json", full)
    make_report(full, main_rows, attention_rows, type_rows)


def make_report(full, main_rows, attention_rows, type_rows):
    g = next(r for r in main_rows if r["variant"] == "global")
    e = next(r for r in main_rows if r["variant"] == "entity")
    ga = next(r for r in attention_rows if r["variant"] == "global")
    ea = next(r for r in attention_rows if r["variant"] == "entity")
    auc = e["test_groundability_auroc"]
    acc = e["test_acc_iou_05"]
    if auc >= 0.65 and acc >= max(0.10, g["test_acc_iou_05"] + 0.02):
        case = "CASE A — PROMISING"
        rec = "Proceed to Experiment 11: full GMNER extension optimization."
    elif acc >= max(0.10, g["test_acc_iou_05"] + 0.02) and auc < 0.60:
        case = "CASE B — GROUNDING_ONLY"
        rec = "Focus on explicit grounding and redesign None detection."
    elif auc >= 0.65 and acc < 0.10:
        case = "CASE C — GROUNDABILITY_ONLY"
        rec = "Retain groundability module but replace coordinate regression with stronger grounding architecture."
    else:
        case = "CASE D — WEAK"
        rec = "Do not make GMNER the main journal extension; return to hard-negative robustness."
    report = f"""# Experiment 10: Explicit Entity Grounding Head Screening

## 1. Goal

This rapid screening tested whether explicit GMNER supervision can learn entity-region grounding on top of frozen PGMNER/CLIP representations. The original PGMNER checkpoint and all original model parameters were frozen.

## 2. Data

The experiment reused the Experiment 9 GMNER-to-PGMNER cache built from `/data/hfy/VG-Think/E2E-GMNER/data/sft/`.

Groundable entities are those with a valid bbox after the same `Resize(256)+CenterCrop(224)` coordinate mapping used in Experiment 9. Ungroundable entities have `bbox=None`.

## 3. Frozen Features

Entity feature `h_i` is the original PGMNER `fused_text` span representation immediately before `span_classifier`. Visual features are `semantic_ot_aligner` `visual_proj` spatial patch features, excluding the CLIP class token, giving 196 patches per sample. Features are cached in `cache/features_*.pt`.

## 4. Official Evaluator

Local official-style E2E-GMNER evaluation code exists under `/data/hfy/VG-Think/E2E-GMNER/src/e2egmner/evaluation/`. It is designed around parsed generated entity triples. This screening reports the diagnostic joint metric requested here and marks the official metric as not directly applied to this frozen-head diagnostic output.

## 5. Variants

Variant A, Global, uses mean pooled visual patch evidence. Variant B, Entity-Aware, uses single-head entity-to-patch attention. Both use the same groundability BCE and bbox L1+GIoU losses.

## 6. Main Results

| Variant | Best Dev Proxy | Test Groundability AUROC | Test AUPR | Test Mean IoU | Test Acc@0.5 | Joint Metric |
|---|---:|---:|---:|---:|---:|---:|
| Global | {g['best_dev_proxy']:.4f} | {g['test_groundability_auroc']:.4f} | {g['test_groundability_aupr']:.4f} | {g['test_mean_iou']:.4f} | {g['test_acc_iou_05']:.4f} | {g['test_joint_metric']:.4f} |
| Entity-Aware | {e['best_dev_proxy']:.4f} | {e['test_groundability_auroc']:.4f} | {e['test_groundability_aupr']:.4f} | {e['test_mean_iou']:.4f} | {e['test_acc_iou_05']:.4f} | {e['test_joint_metric']:.4f} |

Entity-Aware minus Global: AUROC {e['test_groundability_auroc'] - g['test_groundability_auroc']:.4f}, mean IoU {e['test_mean_iou'] - g['test_mean_iou']:.4f}, Acc@0.5 {e['test_acc_iou_05'] - g['test_acc_iou_05']:.4f}, joint {e['test_joint_metric'] - g['test_joint_metric']:.4f}.

## 7. Attention Diagnostics

Global attention entropy: {ga['attention_entropy']:.4f}; GT-box mass: {ga['gt_box_attention_mass']:.4f}; random area expectation: {ga['random_area_expectation']:.4f}.

Entity-aware attention entropy: {ea['attention_entropy']:.4f}; GT-box mass: {ea['gt_box_attention_mass']:.4f}; random area expectation: {ea['random_area_expectation']:.4f}.

## 8. Entity-Type Breakdown

Detailed PER/LOC/ORG/MISC rows are saved in `results/entity_type_breakdown.csv`.

## 9. Decision

FINAL CASE: {case}.

Recommended Experiment 11: {rec}

## 10. Output Files

- `results/grounding_main.csv`
- `results/groundability_main.csv`
- `results/entity_type_breakdown.csv`
- `results/joint_gmner_metrics.csv`
- `results/attention_diagnostics.csv`
- `results/training_summary.json`
"""
    (EXP / "EXPERIMENT_10_REPORT.md").write_text(report, encoding="utf-8")
    write_json(RES / "final_decision.json", {"final_case": case, "recommended_experiment_11": rec})


def main():
    args = parse_args()
    seed_all(args.seed)
    RES.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.mode in {"all", "extract"}:
        extract_features(args)
    if args.mode in {"all", "train"}:
        train_all(args)
    if args.mode == "report":
        full = json.loads((RES / "training_summary.json").read_text())
        rows = list(csv.DictReader((RES / "grounding_main.csv").open()))
        attn = list(csv.DictReader((RES / "attention_diagnostics.csv").open()))
        types = list(csv.DictReader((RES / "entity_type_breakdown.csv").open()))
        for row in rows + attn + types:
            for k, v in list(row.items()):
                try:
                    row[k] = float(v)
                except Exception:
                    pass
        make_report(full, rows, attn, types)


if __name__ == "__main__":
    main()
