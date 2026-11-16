"""Unit tests for VLLM aux knowledge pipeline."""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aux_knowledge.generate_aux import assert_mode_guards, output_path
from utils.datasets import _append_aux_knowledge_words, load_aux_knowledge_jsonl
from utils.utils_contrastive import InputExample


def test_append_aux_knowledge_orig_span_scope():
    words = ["Hello", "London"]
    bio = ["O", "B-LOC"]
    aux_map = {"img1": "ORG Apple store opens"}
    w, b, orig_n = _append_aux_knowledge_words(words, bio, "img1", aux_map, max_aux_words=12)
    assert orig_n == 2
    assert len(w) > 2
    assert len(w) <= 2 + 12
    assert all(x == "O" for x in b[2:])


def test_entities_only_on_orig_words():
    tok = type("T", (), {"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": ""})()
    words = ["A", "B", "extra", "knowledge"]
    bio = ["B-PER", "O", "O", "O"]
    ex = InputExample("g", "id1", words, bio, tokenizer=tok, orig_n_words=2)
    assert len(ex.entities) == 1
    assert ex.entities[0][1] == 0


def test_oracle_raises_without_flag():
    try:
        assert_mode_guards("oracle", "test", allow_oracle=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "allow_oracle" in str(e)


def test_train_gold_distill_raises_on_test():
    try:
        assert_mode_guards("train_gold_distill", "dev", allow_oracle=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "train" in str(e)


def test_load_aux_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"sample_id": "x1", "aux_text": "PER John"}) + "\n")
        path = f.name
    m = load_aux_knowledge_jsonl(path)
    assert m["x1"] == "PER John"


def test_oracle_output_prefix():
    p = output_path(Path("/tmp"), "twitter2017", "train", "oracle", oracle=True)
    assert p.name.startswith("ORACLE_")
