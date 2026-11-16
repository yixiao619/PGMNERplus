"""Span-grid metrics: greedy non-overlap prune + micro span F1 (PAR-MNER)."""
from __future__ import annotations

import torch


def _spans_overlap(a: tuple, b: tuple) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken):
    """
    predicts: (B, S, C) softmax probabilities.
    span_label_ltoken: (B, S), 0 = O.
    """
    device = predicts.device
    bsz, n_span, _ = predicts.shape
    pred_label_idx = torch.zeros((bsz, n_span), dtype=torch.long, device=device)
    counts = torch.zeros(3, dtype=torch.float, device=device)
    raw_pred = predicts.argmax(dim=-1)

    for b in range(bsz):
        mask = real_span_mask_ltoken[b] > 0
        candidates = []
        for i in range(n_span):
            if not bool(mask[i].item()):
                continue
            ws = int(all_span_idxs[b, i, 0].item())
            we = int(all_span_idxs[b, i, 1].item())
            rp = int(raw_pred[b, i].item())
            score = float(predicts[b, i, rp].item())
            gold = int(span_label_ltoken[b, i].item())
            candidates.append(
                {"i": i, "ws": ws, "we": we, "rp": rp, "score": score, "gold": gold}
            )

        gold_entities = [c for c in candidates if c["gold"] != 0]
        pred_cands = [c for c in candidates if c["rp"] != 0]
        pred_cands.sort(key=lambda x: -x["score"])
        chosen = []
        for c in pred_cands:
            overlap = False
            for ch in chosen:
                if _spans_overlap((c["ws"], c["we"]), (ch["ws"], ch["we"])):
                    overlap = True
                    break
            if not overlap:
                chosen.append(c)
        for c in candidates:
            pred_label_idx[b, c["i"]] = 0
        for c in chosen:
            pred_label_idx[b, c["i"]] = c["rp"]

        pred_set = {(c["ws"], c["we"], c["rp"]) for c in chosen}
        gold_set = {(c["ws"], c["we"], c["gold"]) for c in gold_entities}
        counts[0] += float(len(pred_set & gold_set))
        counts[1] += float(len(pred_set))
        counts[2] += float(len(gold_set))

    return counts, pred_label_idx
