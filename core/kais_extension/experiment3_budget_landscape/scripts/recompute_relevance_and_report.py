#!/usr/bin/env python3
"""Recompute CLIP relevance and downstream Experiment 3 summaries.

This does not rerun PGMNER inference. It reads existing Experiment 3 oracle and
landscape CSV files, recomputes label-free CLIP relevance features, then updates
relationship tables, figures, final classifications, and the report.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, RobertaTokenizer


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "kais_extension" / "experiment3_budget_landscape" / "scripts" / "eval_budget_landscape.py"
spec = importlib.util.spec_from_file_location("e3", SCRIPT)
e3 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(e3)


class RobustCLIPScorer:
    def __init__(self, model_path: str):
        mp = str((ROOT / model_path).resolve()) if not os.path.isabs(model_path) else model_path
        self.model = CLIPModel.from_pretrained(mp).eval()
        self.processor = CLIPProcessor.from_pretrained(mp)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        inp = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        out = self.model.text_model(**inp)
        feat = self.model.text_projection(out.pooler_output)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        inp = self.processor(images=images, return_tensors="pt", padding=True)
        out = self.model.vision_model(**inp)
        feat = self.model.visual_projection(out.pooler_output)
        return F.normalize(feat, dim=-1)

    def encode_image_path(self, image_path: str) -> torch.Tensor:
        return self.encode_images([Image.open(image_path).convert("RGB")])

    def encode_image_patches(self, image_path: str, grid=(3, 3)) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        gh, gw = grid
        pw = max(1, w // gw)
        ph = max(1, h // gh)
        patches = []
        for r in range(gh):
            for c in range(gw):
                x0, y0 = c * pw, r * ph
                x1, y1 = min(w, (c + 1) * pw), min(h, (r + 1) * ph)
                if x1 > x0 and y1 > y0:
                    patches.append(img.crop((x0, y0, x1, y1)))
        if not patches:
            patches = [img]
        return self.encode_images(patches)


def read_csv(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_test_examples():
    sys.path.insert(0, str(ROOT))
    from utils.datasets import load_examples

    tok = RobertaTokenizer.from_pretrained(str(ROOT / "pretrain/roberta-large"))
    return load_examples("data/twitter2017", "test", tok, max_seq_len=128)


def extract_entities(tokens, labels):
    entities = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            typ = lab[2:]
            start = i
            parts = [tokens[i]]
            i += 1
            while i < len(labels) and labels[i] == f"I-{typ}":
                parts.append(tokens[i])
                i += 1
            entities.append({"text": " ".join(parts), "token_indices": list(range(start, i)), "type": typ})
        else:
            i += 1
    return entities


def compute_relevance():
    examples = load_test_examples()
    scorer = RobustCLIPScorer("pretrain/clip-vit-base-patch16")
    rows = []
    image_dir = ROOT / "data/twitter2017_images"
    for i, ex in enumerate(examples):
        img_path = image_dir / f"{ex.imgid}.jpg"
        row = {
            "sample_id": ex.guid,
            "sample_index": i,
            "image_id": ex.imgid,
            "global_clip_similarity": "",
            "entity_token_grounded_fraction": "",
            "entity_patch_mean_max_similarity": "",
            "entity_patch_max_similarity": "",
            "num_entity_tokens": 0,
            "num_grounded_entity_tokens_threshold_0_25": 0,
            "relevance_status": "OK",
        }
        if not img_path.is_file():
            row["relevance_status"] = "MISSING_IMAGE"
            rows.append(row)
            continue
        tokens = ex.words[: getattr(ex, "orig_n_words", len(ex.words))]
        labels = ex.bio_labels[: getattr(ex, "orig_n_words", len(ex.bio_labels))]
        full_text = " ".join(tokens)
        try:
            t = scorer.encode_text(full_text)
            v = scorer.encode_image_path(str(img_path))
            row["global_clip_similarity"] = float((t * v).sum(dim=-1).item())
            entities = extract_entities(tokens, labels)
            if entities:
                patches = scorer.encode_image_patches(str(img_path), (3, 3))
                max_sims = []
                total = 0
                grounded = 0
                for ent in entities:
                    total += len(ent["token_indices"])
                    ent_t = scorer.encode_text(ent["text"])
                    sims = (ent_t @ patches.T).squeeze(0).detach().cpu().numpy()
                    mx = float(np.max(sims))
                    max_sims.append(mx)
                    if mx > 0.25:
                        grounded += len(ent["token_indices"])
                row["num_entity_tokens"] = total
                row["num_grounded_entity_tokens_threshold_0_25"] = grounded
                row["entity_token_grounded_fraction"] = grounded / total if total else 0.0
                row["entity_patch_mean_max_similarity"] = float(np.mean(max_sims))
                row["entity_patch_max_similarity"] = float(np.max(max_sims))
            else:
                row["entity_token_grounded_fraction"] = 0.0
        except Exception as exc:  # noqa: BLE001
            row["relevance_status"] = f"IMAGE_ERROR:{type(exc).__name__}:{exc}"
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"relevance recomputed {i + 1}/{len(examples)}", flush=True)
    return rows


def main():
    base = ROOT / "kais_extension/experiment3_budget_landscape/results"
    relevance = compute_relevance()
    e3.write_csv(base / "sample_visual_relevance.csv", relevance)
    oracle = read_csv(base / "sample_oracle_budgets.csv")
    dataset = read_csv(base / "dataset_level_budget_landscape.csv")
    regret = read_csv(base / "global_budget_regret.csv")
    hetero = json.loads((base / "budget_heterogeneity_summary.json").read_text())
    merged, rel_rows, group_rows, test_rows = e3.relevance_relationship(oracle, relevance)
    e3.write_csv(base / "relevance_budget_relationship.csv", rel_rows)
    e3.write_csv(base / "relevance_group_statistics.csv", group_rows)
    e3.write_csv(base / "relevance_statistical_tests.csv", test_rows)
    e3.write_csv(base / "oracle_relevance_merged.csv", merged)
    e3.make_figures(dataset, oracle, regret, merged)
    classes = e3.classifications(hetero, regret, rel_rows)
    e3.write_json(base / "final_classifications.json", classes)
    e3.write_report(dataset, hetero, regret, rel_rows, group_rows, test_rows, classes)
    summary = json.loads((base / "experiment3_summary.json").read_text())
    summary["classifications"] = classes
    e3.write_json(base / "experiment3_summary.json", summary)
    print(json.dumps({"relevance_rows": len(relevance), "classes": classes}, indent=2))


if __name__ == "__main__":
    main()
