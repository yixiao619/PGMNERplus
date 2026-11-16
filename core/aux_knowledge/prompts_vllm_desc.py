"""PGIM-style VLLM description prompts for PGMNER+VLLM."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

DESC_SYSTEM = (
    "Multimodal NER assistant. Reply with ONLY one or two short English sentences (30-60 tokens). "
    "Describe the image and give brief world knowledge for entity types (PER/LOC/ORG/MISC) "
    "for tweet mentions. Do not invent names absent from the tweet. "
    "No lists, BIO tags, planning, or meta commentary."
)


def build_desc_messages(
    text: str,
    image_path: Optional[str],
    encode_image: Callable[[str], str],
) -> List[Dict[str, Any]]:
    user_parts: List[Any] = [
        {"type": "text", "text": f"Tweet text:\n{text}\n\nWrite the auxiliary description."}
    ]
    if image_path:
        user_parts.append(
            {"type": "image_url", "image_url": {"url": encode_image(image_path)}}
        )
    return [
        {"role": "system", "content": DESC_SYSTEM},
        {"role": "user", "content": user_parts},
    ]
