"""Tests for PGMNER+VLLM description integration."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from utils.datasets import (
    _append_vllm_desc_pgim,
    load_vllm_desc_jsonl,
)
from utils.vllm_sanity import assert_vllm_desc_batch


def test_load_vllm_desc_jsonl():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dev.jsonl"
        p.write_text(
            json.dumps(
                {
                    "sample_id": "17_06_1",
                    "imgid": "17_06_1",
                    "desc_text": "A stadium photo with team logos.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        m = load_vllm_desc_jsonl(str(p))
        assert m["17_06_1"] == "A stadium photo with team logos."


def test_append_desc_labels_only_on_tweet():
    words = ["Hello", "NYC"]
    bio = ["O", "B-LOC"]
    desc_map = {"img1": "The image shows a city skyline at night."}
    w, b, orig_n = _append_vllm_desc_pgim(words, bio, "img1", desc_map, desc_max_words=20)
    assert orig_n == 2
    assert len(w) > 2
    assert all(lab == "O" for lab in b[orig_n:])
    assert b[:orig_n] == bio


def test_sanity_rejects_span_on_desc_segment():  # noqa: D103
    import torch

    # [CLS]=0, tweet=0, desc=1, [SEP]=0
    segment_ids = torch.tensor([[0, 0, 1, 0]])
    span_token_idxes = torch.tensor([[[1, 2]]])  # spans into desc token
    span_masks = torch.tensor([[1]])
    try:
        assert_vllm_desc_batch(
            {
                "segment_ids": segment_ids,
                "span_token_idxes": span_token_idxes,
                "span_masks": span_masks,
            }
        )
        raise AssertionError("expected AssertionError")
    except AssertionError as e:
        if "expected AssertionError" in str(e):
            raise
