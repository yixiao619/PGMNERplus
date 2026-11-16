#!/usr/bin/env python3
"""Frozen GMNER DEV counterfactual extraction for Experiment 25."""
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
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP9 = ROOT / "kais_extension/experiment9_gmner_audit"
EXP10 = ROOT / "kais_extension/experiment10_explicit_grounding"
EXP14 = ROOT / "kais_extension/experiment14_gmner_integration"
EXP24 = ROOT / "kais_extension/experiment24_literature_guided"
OUT = ROOT / "kais_extension/experiment25_reliability_counterfactual"
DATA_DIR = EXP9 / "cache/gmner_pgmner_format"
PGMNER_CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")


class Exp10Head(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.wh = nn.Linear(dim, dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ground = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, 1))
        self.box = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, h, v):
        q, k, z = self.wq(h).unsqueeze(1), self.wk(v), self.wv(v)
        attn = torch.softmax((q * k).sum(-1) / math.sqrt(k.shape[-1]), dim=-1)
        e = torch.bmm(attn.unsqueeze(1), z).squeeze(1)
        r = self.norm(self.wh(h) + e)
        return self.ground(torch.cat([r, e], -1)).squeeze(-1)


class Exp24BHead(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim + 6, dim)
        self.dust = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    @staticmethod
    def positions(device):
        rows = []
        for j in range(196):
            x, y = j % 14, j // 14
            x1, y1, x2, y2 = x / 14, y / 14, (x + 1) / 14, (y + 1) / 14
            rows.append([(x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def forward(self, h, v):
        pos = self.positions(v.device).unsqueeze(0).expand(v.size(0), -1, -1)
        score = (self.q(h).unsqueeze(1) * self.k(torch.cat([v, pos], -1))).sum(-1) / math.sqrt(v.size(-1))
        return torch.cat([score, self.dust(h)], -1)


class CounterfactualDataset(Dataset):
    def __init__(self, base, condition, target_indices):
        self.base, self.condition, self.target_indices = base, condition, target_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = list(self.base[idx])
        if self.condition == "text_only":
            item[10] = torch.zeros_like(item[10])
        elif self.condition in {"random", "hard_negative"}:
            item[10] = self.base[self.target_indices[idx]][10]
        return item


def load_sidecar():
    rows = []
    for line in (DATA_DIR / "gmner_entity_sidecar.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == "dev":
            rows.append(row)
    return rows


def import_pgmner(args):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "main-mner-robert.py", "--data_dir", str(DATA_DIR), "--output_dir", str(OUT / "cache/side"),
        "--model_name_or_path", "pretrain/roberta-large", "--tokenizer_name", "pretrain/roberta-large",
        "--vit_name", "pretrain/clip-vit-base-patch16", "--gpu_id", args.gpu_id,
        "--do_train", "false", "--do_eval", "false", "--do_predict", "false",
        "--batch_size", str(args.batch_size), "--max_span_len", "4", "--max_seq_len", "128",
        "--seed", "42", "--use_pp_ib", "--use_got", "--got_mode", "partial_sinkhorn",
        "--ot_sinkhorn_iter", "30", "--partial_mass_init", "0.5", "--dust_cost", "1.0",
        "--entropy_eps", "0.1", "--ot_alpha", "0.05", "--auto_img_dims", "true",
        "--use_hf_clip_vision", "true",
    ]
    spec = importlib.util.spec_from_file_location("pgmner_main_exp25", ROOT / "main-mner-robert.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.args.labels = str(ROOT / "data/twitter2017/labels.txt")
    return mod


def load_pgmner_state(model):
    state = torch.load(PGMNER_CKPT, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    old, new = "image_model._hf.encoder.vision_model.", "image_model._hf.encoder."
    state = {new + k[len(old):] if k.startswith(old) else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def make_mappings(examples, sidecar):
    n = len(examples)
    random_map = []
    for i in range(n):
        j = (i + 791) % n
        while j == i or sidecar[j]["image_id"] == sidecar[i]["image_id"]:
            j = (j + 1) % n
        random_map.append(j)

    clean = torch.load(EXP10 / "cache/features_dev.pt", map_location="cpu", weights_only=False, mmap=True)
    first = {}
    for i, guid in enumerate(clean["sample_id"]):
        first.setdefault(guid, i)
    guids = [str(ex.guid) for ex in examples]
    idx_by_guid = {g: i for i, g in enumerate(guids)}
    available = [g for g in guids if g in first]
    vectors = torch.stack([clean["v"][first[g]].float().mean(0) for g in available])
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    hard_by_guid = {}
    block = 128
    for start in range(0, len(available), block):
        sim = vectors[start:start + block] @ vectors.T
        for local, src_guid in enumerate(available[start:start + block]):
            src_i = idx_by_guid[src_guid]
            order = torch.argsort(sim[local], descending=True)
            for pos in order.tolist():
                dst_guid = available[pos]
                dst_i = idx_by_guid[dst_guid]
                if dst_i != src_i and sidecar[dst_i]["image_id"] != sidecar[src_i]["image_id"]:
                    hard_by_guid[src_guid] = dst_i
                    break
    hard_map = [hard_by_guid.get(g, random_map[i]) for i, g in enumerate(guids)]
    return random_map, hard_map, len(available)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", default="0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conditions", nargs="+", default=["clean", "text_only", "random", "hard_negative"])
    args = ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    pg = import_pgmner(args)
    labels = pg.get_labels(pg.args.labels)
    config = pg.AutoConfig.from_pretrained(pg.args.model_name_or_path,
        id2label={str(i): x for i, x in enumerate(labels)}, label2id={x: i for i, x in enumerate(labels)})
    tokenizer = pg.RobertaTokenizer.from_pretrained(pg.args.tokenizer_name)
    model = pg.BertSpanNerBN(config, args=pg.args, num_labels=len(labels))
    load_pgmner_state(model)
    model.to(pg.args.device).eval(); model.set_partial_ot_warmup(False)
    for p in model.parameters(): p.requires_grad_(False)

    h10 = Exp10Head().to(pg.args.device).eval()
    h10.load_state_dict(torch.load(EXP10 / "checkpoints/entity_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    h24 = Exp24BHead().to(pg.args.device).eval()
    h24.load_state_dict(torch.load(EXP24 / "runs/E24_B_GROUNDING_SUPERVISED_POT/best_model.pth", map_location="cpu", weights_only=False))
    for head in (h10, h24):
        for p in head.parameters(): p.requires_grad_(False)

    examples = pg.load_examples(pg.args.data_dir, mode="dev", tokenizer=tokenizer,
        visual_suffix_map=pg.args.visual_suffix_map,
        aux_knowledge_map=pg.resolve_aux_knowledge_map(pg.args, "dev"),
        vllm_desc_map=pg.resolve_vllm_desc_map(pg.args, "dev"), max_seq_len=pg.args.max_seq_len,
        max_aux_words=int(getattr(pg.args, "max_aux_words", 12)), aux_dropout=0.0,
        desc_max_words=int(getattr(pg.args, "desc_max_words", 60)), vllm_desc_dropout=0.0,
        desc_sep_token=str(getattr(pg.args, "desc_sep_token", "</s>")))
    sidecar = load_sidecar()
    assert len(examples) == len(sidecar) == 1500
    from utils.datasets import SpanNerDataset, collate_fn
    base = SpanNerDataset(examples, args=pg.args, tokenizer=tokenizer, labels=labels, dev=True)
    random_map, hard_map, hard_covered = make_mappings(examples, sidecar)
    mappings = {"clean": list(range(len(base))), "random": random_map,
                "hard_negative": hard_map, "text_only": list(range(len(base)))}
    mapping_payload = {c: [{"source_guid": str(examples[i].guid), "source_image": sidecar[i]["image_id"],
                            "target_guid": str(examples[j].guid), "target_image": sidecar[j]["image_id"]}
                           for i, j in enumerate(mappings[c])] for c in ("random", "hard_negative")}
    (OUT / "cache/counterfactual_mappings.json").write_text(json.dumps(mapping_payload, indent=2) + "\n")

    run_summary = {"hard_mapping_feature_covered_samples": hard_covered, "conditions": {}}
    for condition in args.conditions:
        out_path = OUT / f"cache/scores_{condition}.csv"
        if out_path.exists():
            print(f"reuse {out_path}", flush=True); continue
        ds = CounterfactualDataset(base, condition, mappings[condition])
        dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=args.batch_size, collate_fn=collate_fn)
        rows, offset, t0 = [], 0, time.time()
        torch.cuda.reset_peak_memory_stats(pg.args.device)
        with torch.no_grad():
            for bi, batch_cpu in enumerate(dl):
                batch = {k: v.to(pg.args.device) for k, v in batch_cpu.items()}
                spans_rep, txt_rep_g, attn_last = model.span_encoding(**batch)
                if model.use_pp_ib: model.dist_t(txt_rep_g)
                img_g, img_l = model.image_model(batch["image"])
                if model.use_pp_ib: model.dist_v(img_g)
                text_proj, visual_proj, align_text, _, _, _, _ = model.semantic_ot_aligner(
                    spans_rep, img_l, mask_t=batch["span_masks"].float(), mask_v=None,
                    attn_t=attn_last, span_token_idxes=batch.get("span_token_idxes"), patch_pos=None)
                fused = model.ln_text(model.fusion_text(torch.cat([text_proj, align_text], -1)) + align_text)
                probs = model.softmax(model.span_classifier(fused)); visual = visual_proj[:, 1:, :]
                bsz = batch["input_ids"].shape[0]
                for k, ex in enumerate(examples[offset:offset + bsz]):
                    meta = sidecar[offset + k]
                    spans = batch["span_word_idxes"][k].detach().cpu().numpy()
                    masks = batch["span_masks"][k].detach().cpu().numpy()
                    for ent in meta.get("entities", []):
                        if not ent.get("matched_in_text"): continue
                        si = next((ii for ii, (a, b) in enumerate(spans) if masks[ii] > 0 and int(a) == int(ent["token_start"]) and int(b) == int(ent["token_end"])), None)
                        if si is None: continue
                        hh, vv = fused[k, si].unsqueeze(0), visual[k].unsqueeze(0)
                        s10 = float(torch.sigmoid(h10(hh, vv))[0])
                        p24 = torch.softmax(h24(hh, vv), -1)
                        s24 = float(1.0 - p24[0, 196])
                        lab = labels.index(ent["type"]) if ent["type"] in labels else -1
                        rows.append({"condition": condition, "sample_id": str(ex.guid), "image_id": meta["image_id"],
                            "mention": ent["mention"], "type": ent["type"], "span_start": ent["token_start"], "span_end": ent["token_end"],
                            "groundable_original": int(bool(ent.get("bbox"))), "exp10_score": s10, "exp24b_score": s24,
                            "mner_correct": int(lab >= 0 and int(probs[k, si].argmax()) == lab)})
                offset += bsz
                print(f"[{condition}] {bi+1}/{len(dl)} entities={len(rows)}", flush=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        run_summary["conditions"][condition] = {"entities": len(rows), "seconds": time.time() - t0,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(pg.args.device) / 1024 ** 2}
    (OUT / "cache/extraction_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
