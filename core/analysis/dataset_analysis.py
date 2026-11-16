# -*- coding: utf-8 -*-
"""
计算论文 Table 1 风格的数据集指标（与 MINER-My 数据格式对齐）。

指标:
  1. avg_clip_score           — 整句文本 vs 整图 CLIP 余弦相似度
  2. one_to_many_ratio        — 实体中被 ≥2 个 patch 以相似度「点亮」的比例
  3. entity_token_grounded_pct — 实体 token 中 max_patch_sim > 阈值 的比例
  4. misc_entity_pct          — (MISC ∪ OTHER) 实体数 / 全部实体（2015 为 OTHER，2017 为 MISC）

数据格式: 与 utils/datasets.load_examples 一致（空行分句、IMGID:xxx、token\\tBIO）。

CLIP: 默认使用 Hugging Face ``CLIPModel``（本地 ``pretrain/clip-vit-base-patch16``），
与论文 ViT-B/32 对齐，无需单独安装 OpenAI clip 包。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# 仓库根目录（analysis/ 上一级）
REPO_ROOT = Path(__file__).resolve().parents[1]

# Twitter-2015 使用 OTHER，2017 使用 MISC；Table 1「杂类」比例应对齐统计二者
MISC_LIKE_TYPES = frozenset({"MISC", "OTHER"})


def load_conll_miner(data_file: str, image_dir: str) -> list[dict]:
    """解析与 ``utils/datasets.load_examples`` 一致的 txt。"""
    samples = []
    words: list[str] = []
    labels: list[str] = []
    imgid: str | None = None

    def flush():
        nonlocal words, labels, imgid
        if not words or imgid is None:
            words, labels = [], []
            return
        img_path = resolve_image_path(imgid, image_dir)
        samples.append(
            {
                "tokens": list(words),
                "labels": list(labels),
                "image_path": img_path,
                "img_id": imgid,
            }
        )
        words, labels = [], []

    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if line == "":
                flush()
                continue
            if "IMGID" in line:
                imgid = line.split(":")[-1].strip()
                continue
            splits = line.split("\t")
            if len(splits) not in (2, 4):
                continue
            words.append(splits[0])
            labels.append(splits[-1].replace("\n", ""))
    if words and imgid is not None:
        img_path = resolve_image_path(imgid, image_dir)
        samples.append(
            {
                "tokens": list(words),
                "labels": list(labels),
                "image_path": img_path,
                "img_id": imgid,
            }
        )
    return samples


def resolve_image_path(img_id: str, image_dir: str) -> str | None:
    img_id = str(img_id).strip()
    for ext in (".jpg", ".jpeg", ".png", ".gif", ""):
        name = img_id + ext if ext else img_id
        candidate = os.path.join(image_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def extract_entities(tokens: list[str], labels: list[str]) -> list[dict]:
    entities = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            etype = lab[2:]
            span_tok = [tokens[i]]
            span_idx = [i]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                span_tok.append(tokens[j])
                span_idx.append(j)
                j += 1
            entities.append(
                {"text": " ".join(span_tok), "type": etype, "token_indices": span_idx}
            )
            i = j
        else:
            i += 1
    return entities


class HFCLIPScorer:
    def __init__(self, model_path: str, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        mp = model_path
        if not os.path.isabs(mp):
            mp = str(REPO_ROOT / mp)
        print(f"Loading CLIP HF model from {mp} on {device} ...")
        self.model = CLIPModel.from_pretrained(mp).to(device)
        self.processor = CLIPProcessor.from_pretrained(mp)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        inp = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        t = self.model.get_text_features(**inp)
        return F.normalize(t, dim=-1)

    @torch.no_grad()
    def encode_image_path(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        inp = self.processor(images=[img], return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        v = self.model.get_image_features(**inp)
        return F.normalize(v, dim=-1)

    @torch.no_grad()
    def encode_image_patches(self, image_path: str, grid: tuple[int, int]) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        gh, gw = grid
        pw = max(1, W // gw)
        ph = max(1, H // gh)
        patches_pil = []
        for r in range(gh):
            for c in range(gw):
                x0, y0 = c * pw, r * ph
                x1, y1 = min(W, (c + 1) * pw), min(H, (r + 1) * ph)
                if x1 <= x0 or y1 <= y0:
                    continue
                patches_pil.append(img.crop((x0, y0, x1, y1)))
        if not patches_pil:
            inp = self.processor(images=[img], return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            v = self.model.get_image_features(**inp)
            return F.normalize(v, dim=-1)
        inp = self.processor(images=patches_pil, return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        feats = self.model.get_image_features(**inp)
        return F.normalize(feats, dim=-1)

    def text_image_similarity(self, text: str, image_path: str) -> float:
        t = self.encode_text(text)
        v = self.encode_image_path(image_path)
        return (t * v).sum(dim=-1).item()


def compute_metrics(
    samples: list,
    scorer: HFCLIPScorer,
    dataset_name: str,
    grounding_threshold: float,
    one_to_many_threshold: float,
    patch_grid: tuple[int, int],
    max_samples: int | None,
) -> dict:
    valid = [s for s in samples if s.get("image_path") and os.path.isfile(s["image_path"])]
    missing_img = len(samples) - len(valid)
    print(f"\n[{dataset_name}] samples={len(samples)} with_image_file={len(valid)} missing={missing_img}")

    if max_samples:
        valid = valid[: max_samples]
        print(f"  → limited to {len(valid)} samples")

    clip_scores: list[float] = []
    total_entities = 0
    one_to_many_entities = 0
    total_entity_tokens = 0
    grounded_entity_tokens = 0
    entity_type_counts: dict[str, int] = defaultdict(int)
    errors = 0

    for sample in tqdm(valid, desc=dataset_name):
        img_path = sample["image_path"]
        tokens = sample["tokens"]
        labels = sample["labels"]
        entities = extract_entities(tokens, labels)

        try:
            full_text = " ".join(tokens)
            clip_scores.append(scorer.text_image_similarity(full_text, img_path))

            for ent in entities:
                entity_type_counts[ent["type"]] += 1

            if not entities:
                continue

            patch_feats = scorer.encode_image_patches(img_path, patch_grid)
            # (K, d)
            for ent in entities:
                ent_text = ent["text"]
                total_entities += 1
                ntok = len(ent["token_indices"])
                total_entity_tokens += ntok

                t_feat = scorer.encode_text(ent_text)
                sims = (t_feat @ patch_feats.T).squeeze(0)
                sims_np = sims.detach().float().cpu().numpy()

                high = (sims_np > one_to_many_threshold).sum()
                if high >= 2:
                    one_to_many_entities += 1

                if sims_np.max() > grounding_threshold:
                    grounded_entity_tokens += ntok

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] {img_path}: {e}")

    avg_clip = float(np.mean(clip_scores)) if clip_scores else 0.0
    o2m = one_to_many_entities / total_entities if total_entities else 0.0
    grounded_pct = grounded_entity_tokens / total_entity_tokens if total_entity_tokens else 0.0
    typed = sum(entity_type_counts.values())
    misc_like = sum(entity_type_counts.get(t, 0) for t in MISC_LIKE_TYPES)
    misc_pct = misc_like / typed if typed else 0.0
    misc_only_pct = entity_type_counts.get("MISC", 0) / typed if typed else 0.0
    other_only_pct = entity_type_counts.get("OTHER", 0) / typed if typed else 0.0

    return {
        "avg_clip_score": round(avg_clip, 4),
        "one_to_many_ratio": round(o2m, 4),
        "entity_token_grounded_pct": round(grounded_pct, 4),
        # 含 MISC+OTHER，与跨数据集 Table 1 可比
        "misc_entity_pct": round(misc_pct, 4),
        "misc_only_entity_pct": round(misc_only_pct, 4),
        "other_only_entity_pct": round(other_only_pct, 4),
        "n_samples_total": len(samples),
        "n_samples_with_image": len([s for s in samples if s.get("image_path")]),
        "n_samples_analyzed": len(clip_scores),
        "n_entity_tokens": total_entity_tokens,
        "n_entities": total_entities,
        "entity_type_dist": dict(entity_type_counts),
        "n_clip_scores": len(clip_scores),
        "errors": errors,
        "missing_image_file": missing_img,
    }


def discover_json_dataset(data_root: Path, patterns: tuple[str, ...]) -> tuple[list[tuple[str, str]], str] | None:
    """返回 [(json_path, split_name), ...], image_dir 或 None。"""
    found = []
    for pat in patterns:
        p = data_root / pat
        if p.is_file():
            found.append((str(p), pat.split("/")[-1].replace(".json", "")))
    if not found:
        return None
    # 图片目录与 json 同级的 images/ 或父级
    parent = Path(found[0][0]).parent
    for img_sub in ("images", "img", "train2014", "val2014"):
        d = parent / img_sub
        if d.is_dir():
            return found, str(d)
    return found, str(parent)


def load_json_generic(data_file: str, image_dir: str) -> list[dict]:
    with open(data_file, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("samples") or list(raw.values())[0]
    samples = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("sentence") or ""
        tokens = item.get("tokens")
        if tokens is None and text:
            tokens = text.split()
        labels = item.get("labels") or item.get("ner_tags") or item.get("tag") or []
        if labels and len(labels) == 1 and isinstance(labels[0], str):
            labels = labels[0].split()
        if tokens and (not labels or len(labels) != len(tokens)):
            labels = ["O"] * len(tokens)
        img = item.get("image") or item.get("image_id") or item.get("file_name") or ""
        img_path = None
        if img:
            for cand in (
                os.path.join(image_dir, str(img)),
                os.path.join(image_dir, str(img) + ".jpg"),
            ):
                if os.path.isfile(cand):
                    img_path = cand
                    break
        samples.append({"tokens": list(tokens or []), "labels": list(labels or []), "image_path": img_path})
    return samples


def print_table(results: dict, out_stream=sys.stdout):
    paper_values = {
        "Twitter-2015": [0.187, 0.634, 0.412, 0.223],
        "Twitter-2017": [0.193, 0.718, 0.387, 0.316],
        "COCO-NER": [0.341, 0.283, 0.821, 0.084],
        "WikiDiverse": [0.312, 0.319, 0.784, 0.112],
    }
    keys = ["avg_clip_score", "one_to_many_ratio", "entity_token_grounded_pct", "misc_entity_pct"]
    labels = [
        "Avg. CLIP score",
        "One-to-many ratio",
        "% entity tokens grounded",
        "MISC entity proportion",
    ]
    print("\n" + "=" * 72, file=out_stream)
    print("  Table 1 style metrics (measured)", file=out_stream)
    print("=" * 72, file=out_stream)
    header = f"{'Metric':<36}"
    for name in results:
        header += f"  {name[:14]:>14}"
    print(header, file=out_stream)
    print("-" * 72, file=out_stream)
    for key, lab in zip(keys, labels):
        row = f"  {lab:<34}"
        for _name, m in results.items():
            v = m.get(key)
            row += f"  {float(v):>14.4f}" if v is not None else f"  {'----':>14}"
        print(row, file=out_stream)
    print("-" * 72, file=out_stream)
    print("\n  Paper reference (for diff check):", file=out_stream)
    for pname, vals in paper_values.items():
        if pname not in results:
            continue
        m = results[pname]
        print(f"\n  {pname}:", file=out_stream)
        for k, pval in zip(keys, vals):
            av = m.get(k)
            if av is not None:
                d = abs(float(av) - pval)
                flag = "⚠" if d > 0.05 else "✓"
                print(f"    {k}: measured={av:.4f} paper={pval:.4f} |Δ|={d:.4f} {flag}", file=out_stream)


def normalize_dataset_key(name: str) -> str:
    """twitter-2015 → Twitter-2015"""
    n = name.strip()
    low = n.lower()
    mapping = {
        "twitter-2015": "Twitter-2015",
        "twitter2015": "Twitter-2015",
        "twitter-2017": "Twitter-2017",
        "twitter2017": "Twitter-2017",
        "coco-ner": "COCO-NER",
        "cocooner": "COCO-NER",
        "wikidiverse": "WikiDiverse",
    }
    return mapping.get(low, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(REPO_ROOT / "data"))
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "analysis" / "results"))
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="若指定，将 table1 汇总额外写入该 JSON 路径（如 analysis/results/table1_fixed.json）",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="pretrain/clip-vit-base-patch16",
        help="HF CLIP 目录或模型 id",
    )
    parser.add_argument("--grounding_threshold", type=float, default=0.25)
    parser.add_argument("--one_to_many_threshold", type=float, default=0.25)
    parser.add_argument("--patch_grid", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--splits", type=str, default="train,dev,test")
    parser.add_argument(
        "--datasets",
        type=str,
        default="Twitter-2015,Twitter-2017",
        help="逗号分隔: Twitter-2015,Twitter-2017,COCO-NER,WikiDiverse",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    os.makedirs(args.output_dir, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    grid = (args.patch_grid, args.patch_grid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = HFCLIPScorer(args.clip_model, device=device)

    ds_requested = [normalize_dataset_key(x) for x in args.datasets.split(",") if x.strip()]

    DATASETS: dict[str, dict] = {
        "Twitter-2015": {
            "type": "conll",
            "text_dir": data_root / "twitter2015",
            "image_dir": data_root / "twitter2015_images",
        },
        "Twitter-2017": {
            "type": "conll",
            "text_dir": data_root / "twitter2017",
            "image_dir": data_root / "twitter2017_images",
        },
    }

    all_results = {}

    for name in ds_requested:
        if name not in DATASETS:
            # COCO-NER / WikiDiverse: 尝试自动发现
            if name == "COCO-NER":
                disc = discover_json_dataset(
                    data_root, ("coco_ner/train.json", "cocooner/train.json", "COCO-NER/train.json")
                )
            elif name == "WikiDiverse":
                disc = discover_json_dataset(
                    REPO_ROOT,
                    ("data/wikidiverse/train.json", "WikiDiverse/train.json"),
                )
            else:
                print(f"[SKIP] Unknown dataset: {name}")
                continue

            if name in ("COCO-NER", "WikiDiverse"):
                if disc is None:
                    print(f"[SKIP] {name}: no local json found under {data_root}")
                    continue
                files_meta, image_dir = disc
                all_samples: list[dict] = []
                for jpath, _ in files_meta:
                    if os.path.isfile(jpath):
                        all_samples.extend(load_json_generic(jpath, image_dir))
                if not all_samples:
                    print(f"[SKIP] {name}: empty after load")
                    continue
                metrics = compute_metrics(
                    all_samples,
                    scorer,
                    name,
                    args.grounding_threshold,
                    args.one_to_many_threshold,
                    grid,
                    args.max_samples,
                )
                all_results[name] = metrics
                out_f = os.path.join(args.output_dir, f"{name.replace('-', '_')}_metrics.json")
                with open(out_f, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"Saved {out_f}")
                continue

        cfg = DATASETS[name]
        img_dir = str(cfg["image_dir"])
        if not os.path.isdir(img_dir):
            print(f"[WARN] image_dir missing: {img_dir} (symlink target may be unavailable)")
        all_samples = []
        text_dir = cfg["text_dir"]
        for sp in splits:
            fp = text_dir / f"{sp}.txt"
            if not fp.is_file():
                print(f"  [SKIP] no file {fp}")
                continue
            chunk = load_conll_miner(str(fp), img_dir)
            print(f"  {name} {sp}: {len(chunk)} sentences")
            all_samples.extend(chunk)
        if not all_samples:
            print(f"[ERROR] no samples for {name}")
            continue

        metrics = compute_metrics(
            all_samples,
            scorer,
            name,
            args.grounding_threshold,
            args.one_to_many_threshold,
            grid,
            args.max_samples,
        )
        all_results[name] = metrics
        out_f = os.path.join(args.output_dir, f"{name.replace('-', '_')}_metrics.json")
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved {out_f}")

    summary_path = os.path.join(args.output_dir, "table1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if args.output_file:
        outp = Path(args.output_file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(outp.resolve())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Also wrote summary: {out_path}")

    print_table(all_results)
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
