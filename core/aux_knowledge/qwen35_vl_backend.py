"""Local Qwen3.5-VL inference for PGMNER+VLLM description cache."""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aux_knowledge.prompts_vllm_desc import DESC_SYSTEM

log = logging.getLogger(__name__)

_THINKING_RE = re.compile(
    r"<think>.*?</think>\s*",
    re.DOTALL | re.IGNORECASE,
)


def _strip_thinking(text: str) -> str:
    text = _THINKING_RE.sub("", text or "")
    return text.strip()


_META_PREFIXES = (
    "the user wants",
    "my task is",
    "goal:",
    "input consists",
    "the tweet mentions",
)


def _polish_desc(raw: str) -> str:
    raw = _strip_thinking(raw)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    kept: List[str] = []
    for ln in lines:
        low = ln.lower()
        if any(low.startswith(p) for p in _META_PREFIXES):
            continue
        kept.append(ln)
    if not kept:
        kept = [raw] if raw else []
    text = " ".join(kept)
    # Prefer sentences that describe the scene (common in Qwen outputs).
    for prefix in ("the image shows", "the photo shows", "the picture shows"):
        idx = text.lower().find(prefix)
        if idx >= 0:
            text = text[idx:]
            break
    if len(text) > 480:
        text = text[:480].rsplit(" ", 1)[0] + "."
    return text.strip()


def _valid_image(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.is_file() and p.stat().st_size > 0:
        return str(p)
    return None


def _load_rgb_pil(path: str):
    """Load image as RGB PIL; supports avif/heic via torchvision decoders."""
    from PIL import Image

    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".avif":
        from torchvision.io import decode_avif, read_file

        t = decode_avif(read_file(str(p)))
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            elif t.shape[0] == 4:
                t = t[:3]
            arr = t.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(arr.astype("uint8"), mode="RGB")
    if suf in (".heic", ".heif"):
        from torchvision.io import decode_heic, read_file

        t = decode_heic(read_file(str(p)))
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            elif t.shape[0] == 4:
                t = t[:3]
            arr = t.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(arr.astype("uint8"), mode="RGB")
    return Image.open(p).convert("RGB")


class LocalQwen35VLBackend:
    """HuggingFace Qwen3.5-27B on a single GPU (set CUDA_VISIBLE_DEVICES externally)."""

    _processor = None
    _model = None
    _model_path_loaded = ""

    def __init__(
        self,
        model_path: str = "",
        max_new_tokens: int = 128,
        model_label: str = "Qwen3.5-27B-local",
    ):
        self.model_path = (
            model_path
            or os.environ.get("QWEN35_PATH", "")
            or "/home/LargeModels/Qwen/Qwen3.5-27B"
        )
        self.max_new_tokens = max_new_tokens
        self.model_label = model_label

    def _ensure_loaded(self) -> None:
        if (
            LocalQwen35VLBackend._model is not None
            and LocalQwen35VLBackend._model_path_loaded == self.model_path
        ):
            return
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        mp = self.model_path
        if not Path(mp).is_dir():
            raise FileNotFoundError(f"Qwen3.5 model path not found: {mp}")

        log.info("Loading Qwen3.5 from %s ...", mp)
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(mp, trust_remote_code=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            mp,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        LocalQwen35VLBackend._processor = proc
        LocalQwen35VLBackend._model = model
        LocalQwen35VLBackend._model_path_loaded = mp
        log.info("Qwen3.5 loaded in %.1fs", time.time() - t0)

    def generate_desc(self, text: str, image_path: Optional[str]) -> Dict[str, Any]:
        import torch

        self._ensure_loaded()
        proc = LocalQwen35VLBackend._processor
        model = LocalQwen35VLBackend._model

        user_content: List[Dict[str, Any]] = []
        model_tag = self.model_label
        img_path = _valid_image(image_path)
        if img_path:
            try:
                user_content.append({"type": "image", "image": _load_rgb_pil(img_path)})
            except Exception as e:
                log.warning("image load failed %s: %s; text-only", img_path, e)
                model_tag = f"{self.model_label}-textonly"
        user_content.append(
            {
                "type": "text",
                "text": f"Tweet text:\n{text}\n\nWrite the auxiliary description.",
            }
        )
        messages = [
            {"role": "system", "content": DESC_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        t0 = time.time()
        try:
            inputs = proc.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except Exception as e:
            if img_path and len(user_content) > 1:
                log.warning("multimodal infer failed: %s; retry text-only", e)
                messages[1]["content"] = [user_content[-1]]
                model_tag = f"{self.model_label}-textonly"
                inputs = proc.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    enable_thinking=False,
                )
            else:
                raise
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }
        in_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_ids = out[:, in_len:]
        desc = _polish_desc(proc.batch_decode(new_ids, skip_special_tokens=True)[0])
        n_tok = int(new_ids.shape[1])
        wall = time.time() - t0
        return {
            "desc_text": desc,
            "usage": {"prompt_tokens": in_len, "completion_tokens": n_tok},
            "model": model_tag,
            "wall_sec": wall,
        }

    def generate(
        self,
        text: str,
        image_path: Optional[str],
        few_shot: List[Dict[str, str]],
        mode: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r = self.generate_desc(text, image_path)
        return {
            "aux_text": r["desc_text"],
            "desc_text": r["desc_text"],
            "usage": r["usage"],
            "model": r["model"],
            "wall_sec": r["wall_sec"],
        }
