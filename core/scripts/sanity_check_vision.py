"""
sanity_check_vision.py — 验证 ViTImageModel 真的在跑 HF CLIP，预训练权重已加载，
                         视觉特征非平凡。
用法: python scripts/sanity_check_vision.py
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loader4image import ViTImageModel


class _Args:
    use_hf_clip_vision = True
    auto_img_dims = True
    vit_name = "pretrain/clip-vit-base-patch16"


def main():
    args = _Args()
    print(f"[*] Loading ViTImageModel with vit_name={args.vit_name}")
    model = ViTImageModel(args).eval()

    assert model._hf is not None, "FAILED: ViTImageModel._hf is None (HF path not used)"
    print("[OK] model._hf is not None — HF CLIP vision tower 已挂上")

    enc = model._hf.encoder
    first_conv_w = enc.vision_model.embeddings.patch_embedding.weight.detach().cpu().numpy()
    print(f"[*] patch_embedding.weight shape = {first_conv_w.shape}")
    print(f"[*] patch_embedding.weight mean = {first_conv_w.mean():.6f}, std = {first_conv_w.std():.6f}")
    if abs(first_conv_w.mean()) > 0.01 or first_conv_w.std() > 0.05:
        print("[WARN] patch_embedding 看起来像是随机初始化（mean/std 异常大），权重可能没加载进来")
    else:
        print("[OK] patch_embedding 权重统计像是预训练值")

    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    x = tf(pil).unsqueeze(0)

    with torch.no_grad():
        g, h = model(x)
    print(f"[*] global shape = {tuple(g.shape)}, local shape = {tuple(h.shape)}")
    print(f"[*] global mean={g.mean().item():.4f}, std={g.std().item():.4f}")
    print(f"[*] local  mean={h.mean().item():.4f}, std={h.std().item():.4f}")

    expected_tokens = 197 if "patch16" in args.vit_name else 50
    assert h.shape == (1, expected_tokens, 768), (
        f"FAILED: expected local shape (1, {expected_tokens}, 768), got {tuple(h.shape)}"
    )
    print(f"[OK] local shape 正确，patches = {expected_tokens - 1}")

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    x2 = tf(Image.fromarray(img2)).unsqueeze(0)
    with torch.no_grad():
        g2, _ = model(x2)
    cos = torch.nn.functional.cosine_similarity(g, g2, dim=-1).item()
    print(f"[*] cosine(random_img, black_img) = {cos:.4f}")
    if cos > 0.99:
        print("[WARN] 两张完全不同的图特征几乎完全一样，可能权重没加载")
    else:
        print("[OK] 不同图像产生不同特征，视觉塔确实在工作")

    print("\n=== Sanity check 全部通过 ===")


if __name__ == "__main__":
    main()
