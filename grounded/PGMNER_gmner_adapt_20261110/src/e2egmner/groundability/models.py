"""Small classifier heads used on frozen CLIP embeddings."""

from __future__ import annotations

import torch
from torch import nn


def normalized_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor.float(), p=2, dim=-1)


def clip_fusion_features(text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    text = normalized_embeddings(text)
    image = normalized_embeddings(image)
    if text.shape != image.shape:
        raise ValueError(f"text/image embedding shape mismatch: {text.shape} vs {image.shape}")
    cosine = (text * image).sum(dim=-1, keepdim=True)
    return torch.cat((text, image, torch.abs(text - image), text * image, cosine), dim=-1)


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def features_for_mode(text: torch.Tensor, image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "text_only":
        return normalized_embeddings(text)
    if mode == "image_only":
        return normalized_embeddings(image)
    if mode == "clip_fusion":
        return clip_fusion_features(text, image)
    raise ValueError(f"unknown CLIP feature mode: {mode}")
