"""Sanity checks: VLLM description tokens are not scored and not in OT spans."""
from __future__ import annotations

import torch


def assert_vllm_desc_batch(fea: dict, strict: bool = True) -> None:
    """
    - segment_ids==1 tokens must not appear inside any active span token range
    - span_word_idxes must lie within tweet word indices (enforced at data layer;
      here we check span_token_idxes vs last tweet token index from segment_ids)
    """
    seg = fea.get("segment_ids")
    span_tok = fea.get("span_token_idxes")
    span_masks = fea.get("span_masks")
    if seg is None or span_tok is None or span_masks is None:
        return

    bsz = seg.shape[0]
    for b in range(bsz):
        desc_pos = (seg[b] == 1).nonzero(as_tuple=False)
        if desc_pos.numel() > 0:
            last_tweet_tok = int(desc_pos[0].item()) - 1
        else:
            tweet_mask = seg[b] == 0
            if not tweet_mask.any():
                continue
            last_tweet_tok = int(tweet_mask.nonzero(as_tuple=False)[-1].item())
        if last_tweet_tok < 0:
            continue
        active = span_masks[b] > 0
        for i in range(span_tok.shape[1]):
            if not bool(active[i].item()):
                continue
            s, e = int(span_tok[b, i, 0].item()), int(span_tok[b, i, 1].item())
            if s == 0 and e == 0:
                continue
            if strict and (s > last_tweet_tok or e > last_tweet_tok):
                raise AssertionError(
                    f"span {i} token range ({s},{e}) exceeds tweet token bound {last_tweet_tok}"
                )
