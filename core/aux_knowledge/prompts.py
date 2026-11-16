"""Prompt templates for auxiliary knowledge generation."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

TASK_HEAD = (
    "You assist multimodal named entity recognition (MNER) on social media posts. "
    "Entity types: PER (person), LOC (location), ORG (organization), MISC (miscellaneous). "
    "Given the tweet text and image, (1) list candidate named entities with types, "
    "(2) justify each using BOTH text and image evidence, "
    "(3) if the image is unrelated to the text, say so and do not invent visual links. "
    "Be concise (under 80 tokens)."
)


def build_messages(
    text: str,
    image_path: Optional[str],
    few_shot: List[Dict[str, str]],
    mode: str,
    extra: Optional[Dict[str, Any]],
    encode_image: Callable[[str], str],
) -> List[Dict[str, Any]]:
    system = TASK_HEAD
    if mode == "verify":
        system += " Verify or correct the candidate entities below; drop false positives."
    elif mode == "train_gold_distill":
        system += " Teacher mode: refine entity hypotheses (training only)."
    elif mode == "oracle":
        system += " Oracle hints provided (analysis only)."

    user_parts: List[Any] = [{"type": "text", "text": f"Tweet: {text}"}]
    if image_path:
        user_parts.append({"type": "image_url", "image_url": {"url": encode_image(image_path)}})

    if mode == "verify" and extra and extra.get("candidates"):
        user_parts.append(
            {"type": "text", "text": f"Candidates: {extra['candidates']}"}
        )
    if mode in ("train_gold_distill", "oracle") and extra and extra.get("gold_entities"):
        user_parts.append(
            {"type": "text", "text": f"Gold entities (reference): {extra['gold_entities']}"}
        )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for ex in few_shot[:3]:
        messages.append({"role": "user", "content": ex.get("user", "")})
        messages.append({"role": "assistant", "content": ex.get("assistant", "")})
    messages.append({"role": "user", "content": user_parts})
    return messages
