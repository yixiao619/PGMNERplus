"""Dataset construction and I/O for sample-level groundability prediction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..evaluation.parsing import extract_answer_text, parse_triples

GROUNDABILITY_FIELDS = (
    "id",
    "split",
    "text",
    "image",
    "label",
    "entity_count",
    "grounded_entity_count",
)
TEXT_MARKER_RE = re.compile(r"(?:^|\n)\s*Text:\s*", flags=re.I)


@dataclass(frozen=True)
class GroundabilityRecord:
    id: int | str
    split: str
    text: str
    image: str
    label: int
    entity_count: int
    grounded_entity_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def extract_tweet_text(user_content: str) -> str:
    """Return only the tweet after the final ``Text:`` marker.

    The SFT user message also contains task instructions. Including those in a
    classifier would be harmless but makes leakage checks less precise, so this
    helper deliberately rejects records without a usable tweet.
    """

    if not isinstance(user_content, str):
        raise ValueError("user content must be a string")
    matches = list(TEXT_MARKER_RE.finditer(user_content))
    if not matches:
        raise ValueError("user content does not contain a 'Text:' marker")
    text = user_content[matches[-1].end() :].strip()
    if not text:
        raise ValueError("tweet text is empty")
    return text


def _messages(record: Mapping[str, Any], *, where: str) -> Sequence[Mapping[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{where}: expected at least two messages")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError(f"{where}: every message must be an object")
    return messages


def build_groundability_record(
    record: Mapping[str, Any],
    *,
    split: str,
    fallback_id: int,
    where: str,
) -> GroundabilityRecord:
    messages = _messages(record, where=where)
    user_content = messages[0].get("content")
    assistant_content = messages[-1].get("content")
    if not isinstance(user_content, str) or not isinstance(assistant_content, str):
        raise ValueError(f"{where}: message content must be strings")

    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ValueError(f"{where}: expected exactly one image path")

    triples = parse_triples(
        extract_answer_text(assistant_content),
        strict=True,
        where=f"{where}:answer",
    )
    grounded = sum(bool(triple.region_valid and triple.regions) for triple in triples)
    sample_id = record.get("id", fallback_id)
    if not isinstance(sample_id, (int, str)):
        raise ValueError(f"{where}: id must be an int or string")
    return GroundabilityRecord(
        id=sample_id,
        split=split,
        text=extract_tweet_text(user_content),
        image=images[0],
        label=int(grounded > 0),
        entity_count=len(triples),
        grounded_entity_count=grounded,
    )


def read_groundability_jsonl(path: str | Path) -> list[GroundabilityRecord]:
    records: list[GroundabilityRecord] = []
    for line_no, row in enumerate(read_jsonl(path), 1):
        missing = [field for field in GROUNDABILITY_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}:{line_no}: missing fields {missing}")
        record = GroundabilityRecord(
            id=row["id"],
            split=str(row["split"]),
            text=str(row["text"]),
            image=str(row["image"]),
            label=int(row["label"]),
            entity_count=int(row["entity_count"]),
            grounded_entity_count=int(row["grounded_entity_count"]),
        )
        if record.label not in {0, 1}:
            raise ValueError(f"{path}:{line_no}: label must be 0 or 1")
        if record.entity_count < 0 or not 0 <= record.grounded_entity_count <= record.entity_count:
            raise ValueError(f"{path}:{line_no}: invalid entity counts")
        if record.label != int(record.grounded_entity_count > 0):
            raise ValueError(f"{path}:{line_no}: label/count disagreement")
        records.append(record)
    return records


def resolve_image_path(record: GroundabilityRecord, image_root: str | Path | None) -> Path:
    path = Path(record.image)
    if not path.is_absolute() and image_root is not None:
        path = Path(image_root) / path
    return path.resolve()


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
