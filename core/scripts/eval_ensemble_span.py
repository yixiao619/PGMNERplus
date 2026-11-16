#!/usr/bin/env python3
"""Score ensemble test_predictions.txt with official span_f1_prune (same as training eval)."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoConfig, RobertaTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine_utils import arg_parse
from eval_metric import span_f1_prune
from models.bn_bert_ner_OT_robert import BertSpanNerBN
from utils.datasets import SpanNerDataset, collate_fn, get_labels, load_examples


def parse_token_preds(path: Path) -> list[list[str]]:
    sents: list[list[str]] = []
    cur: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split()
        if len(parts) >= 3:
            cur.append(parts[2])
    if cur:
        sents.append(cur)
    return sents


def majority_label(tags: list[str]) -> str:
    non_o = [t for t in tags if t != "O"]
    if not non_o:
        return "O"
    return Counter(non_o).most_common(1)[0][0]


def inject_votes(
    span_word_idxes: torch.Tensor,
    span_masks: torch.Tensor,
    token_preds: list[str],
    label2id: dict[str, int],
    n_cls: int,
    device: torch.device,
) -> torch.Tensor:
    bsz, n_span, _ = span_word_idxes.shape
    out = torch.zeros(bsz, n_span, n_cls, device=device)
    for b in range(bsz):
        words = token_preds[b] if b < len(token_preds) else []
        for i in range(n_span):
            if span_masks[b, i].item() <= 0:
                continue
            ws = int(span_word_idxes[b, i, 0].item())
            we = int(span_word_idxes[b, i, 1].item())
            if ws < 0 or we < 0 or ws >= len(words):
                continue
            we = min(we, len(words) - 1)
            lab_name = majority_label(words[ws : we + 1])
            li = int(label2id.get(lab_name, 0))
            out[b, i, :] = 1e-6
            out[b, i, li] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True)
    ap.add_argument("--ref_output_dir", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default="data/twitter2017")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--labels", type=str, default="data/twitter2017/labels.txt")
    cli = ap.parse_args()

    pred_path = Path(cli.predictions)
    ref_dir = Path(cli.ref_output_dir)
    ckpt = ref_dir / "best_model.pth"
    if not ckpt.is_file():
        raise SystemExit(f"Missing {ckpt}")

    args = arg_parse().parse_args([])
    args.data_dir = cli.data_dir
    args.output_dir = str(ref_dir)
    args.gpu_id = cli.gpu_id
    args.use_pp_ib = True
    args.use_got = True
    args.use_ema = True
    args.got_mode = "partial_sinkhorn"
    args.got_ot_backend = "partial_sinkhorn"
    args.partial_mass_init = 0.7
    args.dust_cost = 0.8
    args.beta_polar = 0.01
    args.delta_minus = 0.2
    args.delta_plus = 0.8
    args.ot_alpha = 0.05
    args.entropy_eps = 0.1
    args.ot_sinkhorn_iter = 30
    args.ot_warmup_epochs = 3
    args.ema_decay = 0.999
    args.vit_name = "pretrain/clip-vit-base-patch16"
    args.use_hf_clip_vision = True
    args.auto_img_dims = True
    args.visual_suffix_map = None
    args.max_aux_words = 12
    args.labels = cli.labels
    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    args.eval_batch_size = 8
    args.load_path = str(ckpt)

    token_preds = parse_token_preds(pred_path)
    labels = get_labels(cli.labels)
    label2id = {lab: i for i, lab in enumerate(labels)}

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id=label2id,
    )
    model = BertSpanNerBN(config, args=args, num_labels=len(labels))
    model.load_state_dict(torch.load(args.load_path, map_location="cpu"), strict=False)
    model.to(args.device)
    model.eval()

    examples = load_examples(
        args.data_dir, mode="test", tokenizer=tokenizer,
        visual_suffix_map=args.visual_suffix_map,
        aux_knowledge_map=None,
        max_seq_len=args.max_seq_len,
        max_aux_words=int(getattr(args, "max_aux_words", 12)),
        aux_dropout=0.0,
    )
    if len(examples) != len(token_preds):
        print(f"[warn] examples={len(examples)} pred_sents={len(token_preds)}")

    dataset = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)
    loader = DataLoader(
        dataset, sampler=SequentialSampler(dataset),
        batch_size=args.eval_batch_size, collate_fn=collate_fn,
    )

    all_counts = torch.zeros(3, device=args.device)
    sent_offset = 0
    for batch in tqdm(loader, desc="span-eval"):
        fea = {k: v.to(args.device) for k, v in batch.items()}
        bsz = fea["span_word_idxes"].shape[0]
        batch_preds = token_preds[sent_offset : sent_offset + bsz]
        sent_offset += bsz

        voted_probs = inject_votes(
            fea["span_word_idxes"], fea["span_masks"], batch_preds,
            label2id, len(labels), args.device,
        )
        counts, _ = span_f1_prune(
            fea["span_word_idxes"], voted_probs,
            fea["span_labels"], fea["span_masks"],
        )
        all_counts += counts

    tp, pred_n, gold_n = all_counts.tolist()
    p = tp / (pred_n + 1e-10)
    r = tp / (gold_n + 1e-10)
    f1 = 2 * p * r / (p + r + 1e-10)
    print(f"predictions={pred_path}")
    print(f"ref_ckpt={ckpt}")
    print(f"span_precision={p:.5f} span_recall={r:.5f} span_f1={f1:.5f}")
    print(f"correct={tp:.0f} pred={pred_n:.0f} gold={gold_n:.0f}")


if __name__ == "__main__":
    main()
