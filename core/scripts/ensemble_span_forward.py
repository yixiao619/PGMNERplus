#!/usr/bin/env python3
"""Official span-F1 ensemble: average test softmax from multiple checkpoints, then span_f1_prune."""
from __future__ import annotations

import argparse
import sys
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


def build_args(data_dir: str, ckpt_dir: Path, gpu_id: int, labels_path: str):
    args = arg_parse().parse_args([])
    args.data_dir = data_dir
    args.output_dir = str(ckpt_dir)
    args.gpu_id = gpu_id
    args.labels = labels_path
    args.use_pp_ib = True
    args.use_got = True
    args.use_ema = True
    args.use_hf_clip_vision = True
    args.auto_img_dims = True
    args.vit_name = "pretrain/clip-vit-base-patch16"
    args.visual_suffix_map = None
    args.max_aux_words = 12
    args.beta_polar = 0.01
    args.delta_minus = 0.2
    args.delta_plus = 0.8
    args.ot_alpha = 0.05
    args.entropy_eps = 0.1
    args.ot_sinkhorn_iter = 30
    args.ot_warmup_epochs = 3
    args.ema_decay = 0.999
    args.got_lambda_sem = 0.5
    args.got_struct_mode = "full"
    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    args.eval_batch_size = 8
    return args


def load_model(args, labels, label2id):
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id=label2id,
    )
    model = BertSpanNerBN(config, args=args, num_labels=len(labels))
    ckpt = Path(args.output_dir) / "best_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(args.device)
    model.eval()
    return model, tokenizer


def infer_hparams_from_eval(ckpt_dir: Path) -> tuple[float, float]:
    """Read partial_mass_init / dust_cost from eval_results if present."""
    mass, dc = 0.6, 0.8
    ev = ckpt_dir / "eval_results.txt"
    if not ev.is_file():
        return mass, dc
    text = ev.read_text(errors="ignore")
    # training log often lacks hparams; use dirname hints
    name = ckpt_dir.name
    if "dc10_m05" in name:
        return 0.5, 1.0
    if "dc08_m06" in name:
        return 0.6, 0.8
    if "mass0.7" in name:
        return 0.7, 0.8
    if "mass0.3" in name:
        return 0.3, 0.8
    return mass, dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dirs", type=str, required=True, help="comma-separated output dirs")
    ap.add_argument("--data_dir", default="data/twitter2017")
    ap.add_argument("--labels", default="data/twitter2017/labels.txt")
    ap.add_argument("--gpu_id", type=int, default=0)
    args_cli = ap.parse_args()

    dirs = [Path(d.strip()) for d in args_cli.model_dirs.split(",") if d.strip()]
    for d in dirs:
        if not (d / "best_model.pth").is_file():
            raise SystemExit(f"Missing checkpoint: {d}/best_model.pth")

    labels = get_labels(args_cli.labels)
    label2id = {lab: i for i, lab in enumerate(labels)}

    base_args = build_args(args_cli.data_dir, dirs[0], args_cli.gpu_id, args_cli.labels)
    _, tokenizer = load_model(base_args, labels, label2id)

    examples = load_examples(
        base_args.data_dir, mode="test", tokenizer=tokenizer,
        visual_suffix_map=None, aux_knowledge_map=None,
        max_seq_len=base_args.max_seq_len, max_aux_words=12, aux_dropout=0.0,
    )
    dataset = SpanNerDataset(examples, args=base_args, tokenizer=tokenizer, labels=labels, dev=True)
    loader = DataLoader(
        dataset, sampler=SequentialSampler(dataset),
        batch_size=base_args.eval_batch_size, collate_fn=collate_fn,
    )

    model_specs = []
    for d in dirs:
        a = build_args(args_cli.data_dir, d, args_cli.gpu_id, args_cli.labels)
        mass, dc = infer_hparams_from_eval(d)
        a.partial_mass_init = mass
        a.dust_cost = dc
        a.got_mode = "partial_sinkhorn"
        a.got_ot_backend = "partial_sinkhorn"
        model_specs.append((d, a, mass, dc))
        print(f"queued {d.name} mass={mass} dc={dc}")

    batches = list(loader)
    n_models = len(model_specs)
    prob_sums = [None] * len(batches)

    for d, a, _mass, _dc in model_specs:
        m, _ = load_model(a, labels, label2id)
        for bi, batch in enumerate(tqdm(batches, desc=f"infer {d.name}", leave=False)):
            fea = {k: v.to(base_args.device) for k, v in batch.items()}
            with torch.no_grad():
                predicts, _ = m(fea, only_test=True)
            p = predicts[0].detach()
            if prob_sums[bi] is None:
                prob_sums[bi] = p
            else:
                prob_sums[bi] = prob_sums[bi] + p
        del m
        torch.cuda.empty_cache()

    all_counts = torch.zeros(3, device=base_args.device)
    for bi, batch in enumerate(batches):
        fea = {k: v.to(base_args.device) for k, v in batch.items()}
        prob_avg = prob_sums[bi] / float(n_models)
        counts, _ = span_f1_prune(
            fea["span_word_idxes"], prob_avg,
            fea["span_labels"], fea["span_masks"],
        )
        all_counts += counts

    tp, pred_n, gold_n = all_counts.tolist()
    p = tp / (pred_n + 1e-10)
    r = tp / (gold_n + 1e-10)
    f1 = 2 * p * r / (p + r + 1e-10)
    print(f"models={n_models}")
    print(f"span_precision={p:.5f} span_recall={r:.5f} span_f1={f1:.5f}")
    print(f"correct={tp:.0f} pred={pred_n:.0f} gold={gold_n:.0f}")


if __name__ == "__main__":
    main()
