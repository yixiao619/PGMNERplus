#!/usr/bin/env python3
"""Extract PGMNER predictions and Exp10 groundability scores for Exp14."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path("/home/jyx/yixiao/MINER-My")
EXP9 = ROOT / "kais_extension/experiment9_gmner_audit"
EXP10 = ROOT / "kais_extension/experiment10_explicit_grounding"
EXP14 = ROOT / "kais_extension/experiment14_gmner_integration"
DATA_DIR = EXP9 / "cache/gmner_pgmner_format"
PGMNER_CKPT = Path("/data/jyx/MINER-My/outputs/pp_ib_partial_ot_grid_dc10_m05_seed42/best_model.pth")
GROUND_CKPT = EXP10 / "checkpoints/entity_best.pt"
TYPE_TO_OFFICIAL = {"PER": "PER", "LOC": "LOC", "ORG": "ORG", "MISC": "OTHER"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


class GroundingHead(nn.Module):
    """Exact Exp10 entity-aware head, retained here to avoid source changes."""
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.wh = nn.Linear(dim, dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ground = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, 1))
        self.box = nn.Sequential(nn.Linear(dim * 2, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, h, v):
        q = self.wq(h).unsqueeze(1)
        k = self.wk(v)
        z = self.wv(v)
        attn = torch.softmax((q * k).sum(-1) / math.sqrt(k.shape[-1]), dim=-1)
        e = torch.bmm(attn.unsqueeze(1), z).squeeze(1)
        r = self.norm(self.wh(h) + e)
        return self.ground(torch.cat([r, e], dim=-1)).squeeze(-1)


def load_sidecar(split: str):
    rows = []
    path = DATA_DIR / "gmner_entity_sidecar.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == split:
            rows.append(row)
    return rows


def import_pgmner(run_args):
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.argv = [
        "main-mner-robert.py", "--data_dir", str(DATA_DIR),
        "--output_dir", str(EXP14 / "results/pgmner_side_outputs"),
        "--model_name_or_path", "pretrain/roberta-large", "--tokenizer_name", "pretrain/roberta-large",
        "--vit_name", "pretrain/clip-vit-base-patch16", "--gpu_id", run_args.gpu_id,
        "--do_train", "false", "--do_eval", "false", "--do_predict", "false",
        "--batch_size", str(run_args.batch_size), "--max_span_len", "4", "--max_seq_len", "128",
        "--seed", str(run_args.seed), "--use_pp_ib", "--use_got", "--got_mode", "partial_sinkhorn",
        "--ot_sinkhorn_iter", "30", "--partial_mass_init", "0.5", "--dust_cost", "1.0",
        "--entropy_eps", "0.1", "--ot_alpha", "0.05", "--auto_img_dims", "true",
        "--use_hf_clip_vision", "true",
    ]
    spec = importlib.util.spec_from_file_location("pgmner_main_exp14", ROOT / "main-mner-robert.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    # Classifier indices are defined by the paper checkpoint's training label order.
    mod.args.labels = str(ROOT / "data/twitter2017/labels.txt")
    return mod


def load_pgmner_state(model):
    state = torch.load(PGMNER_CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    old = "image_model._hf.encoder.vision_model."
    new = "image_model._hf.encoder."
    state = {new + k[len(old):] if k.startswith(old) else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def greedy_predictions(span_words, span_masks, probs):
    labels = probs.argmax(-1)
    candidates = []
    for i in range(probs.shape[0]):
        if float(span_masks[i]) <= 0:
            continue
        label = int(labels[i])
        if label == 0:
            continue
        ws, we = [int(x) for x in span_words[i].tolist()]
        candidates.append((float(probs[i, label]), i, ws, we, label))
    candidates.sort(key=lambda x: -x[0])
    chosen = []
    for cand in candidates:
        _, _, ws, we, _ = cand
        if not any(not (we < c[2] or c[3] < ws) for c in chosen):
            chosen.append(cand)
    return chosen


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    pg = import_pgmner(args)
    labels = pg.get_labels(pg.args.labels)
    config = pg.AutoConfig.from_pretrained(
        pg.args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = pg.RobertaTokenizer.from_pretrained(pg.args.tokenizer_name)
    model = pg.BertSpanNerBN(config, args=pg.args, num_labels=len(labels))
    load_pgmner_state(model)
    model.to(pg.args.device).eval()
    model.set_partial_ot_warmup(False)
    for p in model.parameters(): p.requires_grad_(False)

    head_payload = torch.load(GROUND_CKPT, map_location="cpu", weights_only=False)
    head = GroundingHead(1024).to(pg.args.device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()
    for p in head.parameters(): p.requires_grad_(False)

    examples = pg.load_examples(
        pg.args.data_dir, mode=args.split, tokenizer=tokenizer,
        visual_suffix_map=pg.args.visual_suffix_map,
        aux_knowledge_map=pg.resolve_aux_knowledge_map(pg.args, args.split),
        vllm_desc_map=pg.resolve_vllm_desc_map(pg.args, args.split), max_seq_len=pg.args.max_seq_len,
        max_aux_words=int(getattr(pg.args, "max_aux_words", 12)), aux_dropout=0.0,
        desc_max_words=int(getattr(pg.args, "desc_max_words", 60)), vllm_desc_dropout=0.0,
        desc_sep_token=str(getattr(pg.args, "desc_sep_token", "</s>")),
    )
    sidecar = load_sidecar(args.split)
    if args.limit:
        examples, sidecar = examples[:args.limit], sidecar[:args.limit]
    assert len(examples) == len(sidecar), (len(examples), len(sidecar))
    from utils.datasets import SpanNerDataset, collate_fn
    ds = SpanNerDataset(examples, args=pg.args, tokenizer=tokenizer, labels=labels, dev=True)
    dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=args.batch_size, collate_fn=collate_fn)

    out_path = EXP14 / f"results/pgmner_predicted_entities_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_offset = 0
    n_pred = 0
    source_counts = defaultdict(int)
    with out_path.open("w", encoding="utf-8") as out, torch.no_grad():
        for bi, batch_cpu in enumerate(dl):
            batch = {k: v.to(pg.args.device) for k, v in batch_cpu.items()}
            spans_rep, txt_rep_g, attn_last = model.span_encoding(**batch)
            if model.use_pp_ib: model.dist_t(txt_rep_g)
            img_g, img_l = model.image_model(batch["image"])
            if model.use_pp_ib: model.dist_v(img_g)
            text_proj, visual_proj, align_text, _, _, _, _ = model.semantic_ot_aligner(
                spans_rep, img_l, mask_t=batch["span_masks"].float(), mask_v=None,
                attn_t=attn_last, span_token_idxes=batch.get("span_token_idxes"), patch_pos=None,
            )
            fused = model.ln_text(model.fusion_text(torch.cat([text_proj, align_text], -1)) + align_text)
            probs = model.softmax(model.span_classifier(fused))
            visual = visual_proj[:, 1:, :]
            bsz = batch["input_ids"].shape[0]
            for k in range(bsz):
                ex = examples[sample_offset + k]
                meta = sidecar[sample_offset + k]
                if str(ex.imgid) != str(meta["image_id"]):
                    raise RuntimeError(f"sample order mismatch: {ex.imgid} != {meta['image_id']}")
                chosen = greedy_predictions(batch["span_word_idxes"][k], batch["span_masks"][k], probs[k])
                entities = []
                for confidence, span_i, ws, we, label_i in chosen:
                    gprob = float(torch.sigmoid(head(fused[k, span_i].unsqueeze(0), visual[k].unsqueeze(0)))[0])
                    ptype = labels[label_i]
                    entities.append({
                        "span_start": ws, "span_end": we, "mention": " ".join(ex.words[ws:we + 1]),
                        "pgmner_type": ptype, "official_type": TYPE_TO_OFFICIAL[ptype],
                        "entity_confidence": confidence, "groundability_probability": gprob,
                    })
                image_rel = meta["image_rel"]
                source = "Twitter2015" if "twitter2015" in image_rel.lower() else "Twitter2017"
                source_counts[source] += 1
                out.write(json.dumps({
                    "split": args.split, "sample_index": sample_offset + k,
                    "sample_id": meta.get("sample_id"), "guid": str(ex.guid), "image_id": meta["image_id"],
                    "image_rel": image_rel, "image_path": meta["image_path"], "source_dataset": source,
                    "text": meta["text"], "words": ex.words, "entities": entities,
                }, ensure_ascii=False) + "\n")
                n_pred += len(entities)
            sample_offset += bsz
            print(f"[{args.split}] batch {bi+1}/{len(dl)} samples={sample_offset} predicted_entities={n_pred}", flush=True)

    summary = {
        "split": args.split, "samples": len(examples), "predicted_entities": n_pred,
        "source_samples": dict(source_counts), "pgmner_checkpoint": str(PGMNER_CKPT),
        "groundability_checkpoint": str(GROUND_CKPT),
        "groundability_adaptation": "Direct application to decoded predicted-span fused representations; head was trained on gold spans.",
        "output": str(out_path),
    }
    (EXP14 / f"results/pgmner_extraction_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
