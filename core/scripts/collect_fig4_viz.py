#!/usr/bin/env python3
"""Collect intermediate tensors for Fig.4 visualization."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fig_common import (
    VIZ_DIR,
    ensure_dirs,
    load_config,
    load_main_module,
    to_infer_batch,
)
from utils.datasets import collate_fn, get_labels, load_examples, SpanNerDataset
from models.partial_ot import PartialSinkhorn, build_ot_backend
from models.semantic_ot_aligner import sinkhorn_coupling

def build_args(main_mod, ckpt_run: Path, gpu_id: int = 0):
    """Backward-compatible wrapper; prefer load_main_module()."""
    _, args = load_main_module(gpu_id, ckpt_run)
    return args


def load_model(main_mod, args, ckpt: Optional[Path] = None):
    labels = get_labels(args.labels)
    config = main_mod.AutoConfig.from_pretrained(args.model_name_or_path)
    model = main_mod.BertSpanNerBN(config, args=args, num_labels=len(labels))
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()
    return model, labels


def dev_loader(args, labels, main_mod):
    tokenizer = main_mod.AutoTokenizer.from_pretrained(args.tokenizer_name)
    examples = load_examples(args.data_dir, mode="dev", tokenizer=tokenizer)
    ds = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)
    loader = DataLoader(
        ds,
        sampler=SequentialSampler(ds),
        batch_size=8,
        collate_fn=collate_fn,
    )
    return loader, tokenizer, examples


def test_loader(args, labels, main_mod):
    tokenizer = main_mod.AutoTokenizer.from_pretrained(args.tokenizer_name)
    examples = load_examples(args.data_dir, mode="test", tokenizer=tokenizer)
    ds = SpanNerDataset(examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)
    loader = DataLoader(
        ds,
        sampler=SequentialSampler(ds),
        batch_size=1,
        collate_fn=collate_fn,
    )
    return loader, tokenizer, examples, ds


def collect_gates(model, loader, device) -> np.ndarray:
    gates: List[np.ndarray] = []

    def _hook(_m, _inp, out):
        gates.append(out.detach().cpu().reshape(-1).numpy())

    h1 = model.dist_t.polar_gate.register_forward_hook(_hook)
    h2 = model.dist_v.polar_gate.register_forward_hook(_hook)
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect gates"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(batch, only_test=True)
    h1.remove()
    h2.remove()
    if not gates:
        return np.array([], dtype=np.float32)
    return np.concatenate(gates).astype(np.float32)


@torch.no_grad()
def partial_plan_aug(model, batch) -> np.ndarray:
    """Return (n+1,m+1) transport for partial OT with dust-bin."""
    aligner = model.semantic_ot_aligner
    ot = aligner._partial_ot
    Z_t = batch["text"]
    Z_v = batch["visual"]
    Z_t_n = F.normalize(Z_t, p=2, dim=-1)
    Z_v_n = F.normalize(Z_v, p=2, dim=-1)
    C_sem = 0.5 * (1.0 - torch.einsum("btd,bvd->btv", Z_t_n, Z_v_n))
    B, n, m = C_sem.shape
    device = C_sem.device
    dtype = C_sem.dtype
    mask_t = batch.get("mask_t")
    mask_v = batch.get("mask_v")
    if mask_t is None:
        mask_t = torch.ones(B, n, device=device, dtype=dtype)
    if mask_v is None:
        mask_v = torch.ones(B, m, device=device, dtype=dtype)
    s = ot.get_s().to(device=device, dtype=dtype)
    if s.dim() == 0:
        s = s.expand(B)
    tau = ot.dust_cost
    invalid_cost = 1e4
    mask_ij = mask_t.unsqueeze(-1) * mask_v.unsqueeze(-2)
    C_work = C_sem + (1.0 - mask_ij) * invalid_cost
    denom_t = mask_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
    denom_v = mask_v.sum(dim=1, keepdim=True).clamp_min(1e-12)
    C_aug = torch.full((B, n + 1, m + 1), tau, dtype=dtype, device=device)
    C_aug[:, :n, :m] = C_work
    C_aug[:, n, m] = 0.0
    a = torch.zeros(B, n + 1, dtype=dtype, device=device)
    a[:, :n] = s.unsqueeze(-1) * mask_t / denom_t
    a[:, n] = 1.0 - s
    b = torch.zeros(B, m + 1, dtype=dtype, device=device)
    b[:, :m] = s.unsqueeze(-1) * mask_v / denom_v
    b[:, m] = 1.0 - s
    T_aug = ot._log_sinkhorn(C_aug, a, b)
    return T_aug[0].cpu().numpy()


@torch.no_grad()
def full_plan(model, batch) -> np.ndarray:
    aligner = model.semantic_ot_aligner
    Z_t = batch["text"]
    Z_v = batch["visual"]
    Z_t_n = F.normalize(Z_t, p=2, dim=-1)
    Z_v_n = F.normalize(Z_v, p=2, dim=-1)
    C_sem = 0.5 * (1.0 - torch.einsum("btd,bvd->btv", Z_t_n, Z_v_n))
    mask_t = batch.get("mask_t")
    mask_v = batch.get("mask_v")
    gamma = sinkhorn_coupling(
        C_sem, aligner.eps, mask_t, mask_v, aligner.sinkhorn_iter
    )
    g = gamma[0].cpu().numpy()
    # pad to (n+1,m+1) with dust row/col zeros for viz consistency
    n, m = g.shape
    out = np.zeros((n + 1, m + 1), dtype=np.float32)
    out[:n, :m] = g
    return out


@torch.no_grad()
def gw_plan(model, batch) -> np.ndarray:
    """GW-style plan via node_sinkhorn backend (no dust-bin)."""
    aligner = model.semantic_ot_aligner
    Z_t = batch["text"]
    Z_v = batch["visual"]
    gw = build_ot_backend("node_sinkhorn", eps=aligner.eps, n_iters=aligner.sinkhorn_iter)
    gw = gw.to(Z_t.device)
    mask_t = batch.get("mask_t")
    mask_v = batch.get("mask_v")
    Z_t_n = F.normalize(Z_t, p=2, dim=-1)
    Z_v_n = F.normalize(Z_v, p=2, dim=-1)
    cos = torch.einsum("btd,bvd->btv", Z_t_n, Z_v_n)
    C_sem = 0.5 * (1.0 - cos)
    gamma, _ = gw(Z_t, Z_v, C_pre=C_sem, mask_t=mask_t, mask_v=mask_v)
    g = gamma[0].cpu().numpy()
    n, m = g.shape
    out = np.zeros((n + 1, m + 1), dtype=np.float32)
    out[:n, :m] = g
    return out


def forward_ot_batch(model, ori_batch, device):
    model.eval()
    batch = {k: v.to(device) for k, v in ori_batch.items()}
    with torch.no_grad():
        spans_rep, txt_rep_g, attn_last = model.span_encoding(**batch)
        mask_t = batch["span_masks"].float()
        vis_g, vis_l = model.image_model(batch["image"])
        text_proj = model.semantic_ot_aligner.text_proj(spans_rep)
        visual_proj = model.semantic_ot_aligner.visual_proj(vis_l)
    return {
        "text": text_proj,
        "visual": visual_proj,
        "mask_t": mask_t,
        "mask_v": None,
        "tokens": batch,
    }


def find_misc_sample(examples, keyword: str = "Harry") -> int:
    for i, ex in enumerate(examples):
        words = getattr(ex, "words", None) or getattr(ex, "text_a", "").split()
        labels = getattr(ex, "label", None) or []
        text = " ".join(words) if words else ""
        if keyword.lower() in text.lower():
            if any("MISC" in str(l) for l in labels):
                return i
    for i, ex in enumerate(examples):
        labels = getattr(ex, "label", None) or []
        if any("MISC" in str(l) for l in labels):
            return i
    return 0


def collect_entity_repr(model, loader, device, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Span-level entity representations before (spans_rep) and after OT fusion (fused)."""
    before, after, y = [], [], []
    id2label = {i: l for i, l in enumerate(labels)}
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect t-SNE"):
            batch_d = {k: v.to(device) for k, v in batch.items()}
            spans_rep, txt_rep_g, attn_last = model.span_encoding(**batch_d)
            mask_t = batch_d["span_masks"].float()
            vis_g, vis_l = model.image_model(batch_d["image"])
            if model.use_pp_ib:
                _, _, t, _ = model.dist_t(txt_rep_g)
            else:
                t = txt_rep_g
            aligner = model.semantic_ot_aligner
            (
                text_proj,
                visual_proj,
                align_text,
                align_visual,
                transport_plan,
                loss_ot,
                ot_components,
            ) = aligner(
                spans_rep,
                vis_l,
                mask_t=mask_t,
                attn_t=attn_last,
                span_token_idxes=batch_d.get("span_token_idxes"),
            )
            fused = torch.cat([text_proj, align_text], dim=-1)
            fused = model.fusion_text(fused)
            fused = model.ln_text(fused + align_text)
            span_labels = batch_d["span_labels"]
            span_masks = batch_d["span_masks"]
            for b in range(span_labels.size(0)):
                for s in range(span_labels.size(1)):
                    if not span_masks[b, s]:
                        continue
                    lab_id = int(span_labels[b, s].item())
                    if lab_id <= 0:
                        continue
                    lab = id2label.get(lab_id, "O")
                    if lab == "O":
                        continue
                    before.append(spans_rep[b, s].cpu().numpy())
                    after.append(fused[b, s].cpu().numpy())
                    y.append(lab)
    if not before:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.array([])
    return (
        np.stack(before),
        np.stack(after),
        np.array(y),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--gpu", type=int, dest="gpu_id")
    args_cli = parser.parse_args()
    ensure_dirs()
    cfg = load_config()
    run_dir = ROOT / cfg["pgmner_run_dir"]
    ckpt = ROOT / cfg["pgmner_ckpt"]
    main_mod, args = load_main_module(int(args_cli.gpu_id), run_dir)
    device = args.device

    loader, tokenizer, dev_examples = dev_loader(args, get_labels(args.labels), main_mod)
    test_loader_, _, test_examples, test_ds = test_loader(
        args, get_labels(args.labels), main_mod
    )

    # (a) gates before / after
    model_before, labels = load_model(main_mod, args, ckpt=None)
    gates_before = collect_gates(model_before, loader, device)
    np.save(VIZ_DIR / "gates_before.npy", gates_before)

    model_after, labels = load_model(main_mod, args, ckpt=ckpt)
    gates_after = collect_gates(model_after, loader, device)
    np.save(VIZ_DIR / "gates_after.npy", gates_after)
    print(f"gate mean after = {gates_after.mean():.4f} (n={len(gates_after)})")

    # (b) transport plans for one test sample
    sample_idx = cfg.get("ot_sample_idx", find_misc_sample(test_examples))
    sample_idx = int(sample_idx)
    batch = test_ds[sample_idx]
    # collate single item
    batch = collate_fn([batch])
    ot_batch = forward_ot_batch(model_after, batch, device)
    plan_full = full_plan(model_after, ot_batch)
    plan_gw = gw_plan(model_after, ot_batch)
    plan_partial = partial_plan_aug(model_after, ot_batch)
    np.save(VIZ_DIR / "plan_full.npy", plan_full)
    np.save(VIZ_DIR / "plan_gw.npy", plan_gw)
    np.save(VIZ_DIR / "plan_partial.npy", plan_partial)
    ex = test_examples[sample_idx]
    words = getattr(ex, "words", None) or []
    sample_tokens = {"idx": sample_idx, "tokens": list(words)}
    (VIZ_DIR / "sample_tokens.json").write_text(
        json.dumps(sample_tokens, ensure_ascii=False, indent=2)
    )
    for name, arr in [("full", plan_full), ("gw", plan_gw), ("partial", plan_partial)]:
        nz = int(np.count_nonzero(arr > 1e-6))
        sparsity = 1.0 - nz / arr.size
        print(f"OT {name}: shape={arr.shape} nonzero={nz} sparsity={sparsity:.3f}")

    # (c) t-SNE entity span representations
    from sklearn.manifold import TSNE

    before, after, y = collect_entity_repr(model_after, loader, device, labels)
    if len(before) >= 5:
        tsne = TSNE(n_components=2, perplexity=min(30, len(before) - 1), random_state=42)
        np.save(VIZ_DIR / "tsne_before.npy", tsne.fit_transform(before))
        tsne2 = TSNE(n_components=2, perplexity=min(30, len(after) - 1), random_state=42)
        np.save(VIZ_DIR / "tsne_after.npy", tsne2.fit_transform(after))
        np.save(VIZ_DIR / "tsne_labels.npy", y)
        try:
            from sklearn.metrics import silhouette_score

            le = {lab: i for i, lab in enumerate(sorted(set(y)))}
            yid = np.array([le[lab] for lab in y])
            if len(set(yid)) > 1:
                sb = silhouette_score(before, yid)
                sa = silhouette_score(after, yid)
                print(f"silhouette before={sb:.4f} after={sa:.4f}")
        except Exception as e:
            print("silhouette skipped:", e)
    else:
        print("Warning: too few entity spans for t-SNE")
    print(f"Fig.4 viz data written to {VIZ_DIR}")


if __name__ == "__main__":
    main()
