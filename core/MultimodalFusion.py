import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import logging
# from tqdm import tqdm
# import os


class MultiModalFusion(nn.Module):
    """
    多模态融合模块
    """

    def __init__(self, emb_dim, fusion_dim=None, num_heads=8):
        super(MultiModalFusion, self).__init__()
        self.emb_dim = emb_dim
        self.fusion_dim = fusion_dim or emb_dim

        # 多头注意力融合
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # 融合投影层
        self.fusion_proj = nn.Sequential(
            nn.Linear(emb_dim * 3, self.fusion_dim),  # 原文本 + 图像感知文本 + 自注意力结果
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, emb_dim)
        )

        # 残差连接的LayerNorm
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, original_text, text_enhanced, image_enhanced):
        """
        Args:
            original_text: [batch_size, t_len, emb_dim] 原始文本表示
            text_enhanced: [batch_size, t_len, emb_dim] 图像感知的文本表示
            image_enhanced: [batch_size, v_len, emb_dim] 文本感知的图像表示
        Returns:
            fused_text: [batch_size, t_len, emb_dim] 最终的文本表示
        """
        # 使用图像感知文本表示作为查询，增强的图像表示作为键值
        attn_output, attn_weights = self.multihead_attn(
            query=text_enhanced,
            key=image_enhanced,
            value=image_enhanced
        )  # [batch_size, t_len, emb_dim]

        # 三路融合：原始文本 + 图像感知文本 + 注意力输出
        concatenated = torch.cat([
            original_text,
            text_enhanced,
            attn_output
        ], dim=-1)  # [batch_size, t_len, emb_dim * 3]

        # 融合投影
        fused_text = self.fusion_proj(concatenated)  # [batch_size, t_len, emb_dim]

        # 残差连接和层归一化
        fused_text = self.layer_norm(fused_text + original_text)

        return fused_text, attn_weights