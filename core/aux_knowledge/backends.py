"""Pluggable VLLM backends for auxiliary knowledge generation."""
from __future__ import annotations

import abc
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class VLLMBackend(abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        text: str,
        image_path: Optional[str],
        few_shot: List[Dict[str, str]],
        mode: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return dict with keys: aux_text, usage (optional), model (optional)."""


class StructuredBackend(VLLMBackend):
    """No API: build concise aux from gold (train) or baseline candidates (all splits)."""

    def generate(self, text, image_path, few_shot, mode, extra=None):
        extra = extra or {}
        parts = []
        if mode in ("train_gold_distill", "oracle") and extra.get("gold_entities"):
            for e in extra["gold_entities"]:
                t = e.get("type", "MISC")
                words = (extra.get("words") or text.split())
                s, end = int(e["start"]), int(e["end"])
                span = " ".join(words[s : end + 1])
                if span and span not in ("@", "#"):
                    parts.append(f"{t} {span}")
        elif extra.get("candidates"):
            parts.extend(str(c) for c in extra["candidates"][:12])
        elif extra.get("words"):
            for w in extra["words"]:
                if len(w) <= 1:
                    continue
                if w.startswith("@"):
                    parts.append(f"PER {w}")
                elif w.startswith("#"):
                    parts.append(f"MISC {w}")
        if not parts and mode == "clean":
            parts.append("PER LOC ORG MISC")
        body = " ".join(parts[:8]) if parts else ""
        return {"aux_text": body, "usage": {}, "model": "structured"}


class StubBackend(VLLMBackend):
    """Deterministic offline backend for CI and cache warmup."""

    def generate(self, text, image_path, few_shot, mode, extra=None):
        ents = (extra or {}).get("candidates") or []
        hint = f" Candidates: {ents}." if ents and mode == "verify" else ""
        aux = (
            f"Entities PER LOC ORG MISC from tweet and image.{hint} "
            f"Text: {text[:60]}."
        )[:320]
        return {"aux_text": aux, "usage": {"prompt_tokens": 0, "completion_tokens": len(aux.split())}, "model": "stub"}


class OpenAIVisionBackend(VLLMBackend):
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 80):
        self.model = model
        self.max_tokens = max_tokens
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.client = OpenAI()

    def _encode_image(self, path: str) -> str:
        data = Path(path).read_bytes()
        ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"

    def generate(self, text, image_path, few_shot, mode, extra=None):
        from .prompts import build_messages

        messages = build_messages(text, image_path, few_shot, mode, extra, self._encode_image)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        aux = (resp.choices[0].message.content or "").strip()
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return {"aux_text": aux, "usage": usage, "model": self.model}


def make_backend(name: str, **kwargs) -> VLLMBackend:
    name = (name or "stub").lower()
    if name in ("structured", "gold", "fast", "local_gold"):
        return StructuredBackend()
    if name in ("qwen35", "qwen3.5", "qwen3_5", "local_qwen"):
        from aux_knowledge.qwen35_vl_backend import LocalQwen35VLBackend

        return LocalQwen35VLBackend(
            model_path=kwargs.get("model_path", ""),
            max_new_tokens=int(kwargs.get("max_new_tokens", 96)),
        )
    if name in ("stub",):
        return StubBackend()
    if name in ("local", "qwen"):
        mp = kwargs.get("model_path", "") or os.environ.get(
            "QWEN35_PATH", "/home/LargeModels/Qwen/Qwen3.5-27B"
        )
        if Path(mp).is_dir():
            from aux_knowledge.qwen35_vl_backend import LocalQwen35VLBackend

            return LocalQwen35VLBackend(model_path=mp)
        return StubBackend()
    if name in ("openai", "gpt4o", "gpt-4o"):
        return OpenAIVisionBackend(model=kwargs.get("model", "gpt-4o-mini"))
    raise ValueError(f"Unknown backend: {name}")
