import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np


class SequenceInfoBottleneck(nn.Module):
    """
    面向序列表征的信息瓶颈计算
    处理文本序列表征T'和图像序列表征V'之间的信息瓶颈

    输入形状:
    - T': [batch_size, text_seq_len, text_feature_dim]
    - V': [batch_size, image_seq_len, image_feature_dim]
    """

    def __init__(self,
                 text_feature_dim,
                 image_feature_dim,
                 bottleneck_dim=128,
                 beta=1.0,
                 sequence_pooling='attention'):
        super().__init__()

        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta  # 信息瓶颈权重参数
        self.sequence_pooling = sequence_pooling

        # 编码器：将序列表征映射到瓶颈表征
        self.text_encoder = self._build_sequence_encoder(text_feature_dim, bottleneck_dim)
        self.image_encoder = self._build_sequence_encoder(image_feature_dim, bottleneck_dim)

        # 解码器：从瓶颈表征重构序列
        self.text_decoder = self._build_sequence_decoder(bottleneck_dim, text_feature_dim)
        self.image_decoder = self._build_sequence_decoder(bottleneck_dim, image_feature_dim)

        # 注意力机制（用于序列池化）
        if sequence_pooling == 'attention':
            self.text_attention = nn.MultiheadAttention(text_feature_dim, num_heads=8, batch_first=True)
            self.image_attention = nn.MultiheadAttention(image_feature_dim, num_heads=8, batch_first=True)

    def _build_sequence_encoder(self, input_dim, output_dim):
        """构建序列编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim * 2)  # 输出均值和方差
        )

    def _build_sequence_decoder(self, input_dim, output_dim):
        """构建序列解码器"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim)
        )

    def sequence_pooling_operation(self, sequence, modality='text'):
        """
        序列池化操作
        Args:
            sequence: [batch_size, seq_len, feature_dim]
            modality: 'text' or 'image'
        Returns:
            pooled: [batch_size, feature_dim]
        """
        if self.sequence_pooling == 'mean':
            return torch.mean(sequence, dim=1)
        elif self.sequence_pooling == 'max':
            return torch.max(sequence, dim=1)[0]
        elif self.sequence_pooling == 'last':
            return sequence[:, -1, :]
        elif self.sequence_pooling == 'attention':
            attention_layer = self.text_attention if modality == 'text' else self.image_attention
            # 自注意力池化
            attn_output, _ = attention_layer(sequence, sequence, sequence)
            return torch.mean(attn_output, dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {self.sequence_pooling}")

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_sequences(self, text_seq, image_seq):
        """
        编码序列到瓶颈表征
        Args:
            text_seq: [batch_size, text_seq_len, text_feature_dim]
            image_seq: [batch_size, image_seq_len, image_feature_dim]
        Returns:
            瓶颈表征的均值和方差
        """
        batch_size = text_seq.size(0)

        # 方法1: 逐位置编码（保持序列结构）
        text_encoded = self.text_encoder(text_seq)  # [batch_size, text_seq_len, bottleneck_dim*2]
        image_encoded = self.image_encoder(image_seq)  # [batch_size, image_seq_len, bottleneck_dim*2]

        # 分离均值和方差
        text_mu, text_logvar = text_encoded.chunk(2, dim=-1)
        image_mu, image_logvar = image_encoded.chunk(2, dim=-1)

        return text_mu, text_logvar, image_mu, image_logvar

    def compute_sequence_mutual_information(self, text_seq, image_seq):
        """
        计算序列间的互信息
        使用序列对齐和注意力机制
        """
        # 编码到瓶颈空间
        text_mu, text_logvar, image_mu, image_logvar = self.encode_sequences(text_seq, image_seq)

        # 采样瓶颈表征
        text_z = self.reparameterize(text_mu, text_logvar)  # [batch_size, text_seq_len, bottleneck_dim]
        image_z = self.reparameterize(image_mu, image_logvar)  # [batch_size, image_seq_len, bottleneck_dim]

        # 计算序列间的交叉注意力
        cross_attention = torch.bmm(text_z, image_z.transpose(1, 2))  # [batch_size, text_seq_len, image_seq_len]
        cross_attention = F.softmax(cross_attention, dim=-1)

        # 加权图像序列表征
        aligned_image_z = torch.bmm(cross_attention, image_z)  # [batch_size, text_seq_len, bottleneck_dim]

        # 计算对齐后的互信息（使用余弦相似度）
        text_z_norm = F.normalize(text_z, p=2, dim=-1)
        aligned_image_z_norm = F.normalize(aligned_image_z, p=2, dim=-1)

        # 逐位置互信息
        pointwise_mi = torch.sum(text_z_norm * aligned_image_z_norm, dim=-1)  # [batch_size, text_seq_len]

        # 序列级互信息（平均）
        sequence_mi = torch.mean(pointwise_mi, dim=1)  # [batch_size]

        return sequence_mi.mean(), text_z, image_z, aligned_image_z

    def compute_kl_divergence(self, mu, logvar):
        """计算KL散度"""
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl_div.mean()

    def forward(self, text_seq, image_seq):
        """
        前向传播计算信息瓶颈损失
        Args:
            text_seq: [batch_size, text_seq_len, text_feature_dim]
            image_seq: [batch_size, image_seq_len, image_feature_dim]
        Returns:
            loss components and bottleneck representations
        """
        # 编码序列
        text_mu, text_logvar, image_mu, image_logvar = self.encode_sequences(text_seq, image_seq)

        # 计算序列间互信息
        mutual_info, text_z, image_z, aligned_image_z = self.compute_sequence_mutual_information(text_seq, image_seq)

        # 计算KL散度（正则化项）
        text_kl = self.compute_kl_divergence(text_mu, text_logvar)
        image_kl = self.compute_kl_divergence(image_mu, image_logvar)

        # 重构损失
        # 使用池化后的表征进行重构
        text_pooled = self.sequence_pooling_operation(text_z, 'text')
        image_pooled = self.sequence_pooling_operation(aligned_image_z, 'image')

        text_recon = self.text_decoder(text_pooled)
        image_recon = self.image_decoder(image_pooled)

        # 计算重构损失（与原始序列的池化版本比较）
        text_target = self.sequence_pooling_operation(text_seq, 'text')
        image_target = self.sequence_pooling_operation(image_seq, 'image')

        text_recon_loss = F.mse_loss(text_recon, text_target)
        image_recon_loss = F.mse_loss(image_recon, image_target)

        # 信息瓶颈损失
        # L = -I(Z;Y) + β*I(X;Z) = -mutual_info + β*(text_kl + image_kl)
        info_bottleneck_loss = -mutual_info + self.beta * (text_kl + image_kl)

        # 总损失
        total_loss = info_bottleneck_loss + text_recon_loss + image_recon_loss

        return {
            'total_loss': total_loss,
            'info_bottleneck_loss': info_bottleneck_loss,
            'mutual_info': mutual_info,
            'text_kl': text_kl,
            'image_kl': image_kl,
            'text_recon_loss': text_recon_loss,
            'image_recon_loss': image_recon_loss,
            'text_z': text_z,
            'image_z': image_z,
            'aligned_image_z': aligned_image_z
        }


def demo_usage():
    """演示如何使用序列信息瓶颈"""

    # 设置参数
    batch_size = 16
    text_seq_len = 20
    image_seq_len = 14  # 例如7x7的图像块序列
    text_feature_dim = 768
    image_feature_dim = 2048
    bottleneck_dim = 128

    # 创建模拟数据
    text_seq = torch.randn(batch_size, text_seq_len, text_feature_dim)
    image_seq = torch.randn(batch_size, image_seq_len, image_feature_dim)

    # 初始化模型
    model = SequenceInfoBottleneck(
        text_feature_dim=text_feature_dim,
        image_feature_dim=image_feature_dim,
        bottleneck_dim=bottleneck_dim,
        beta=1.0,
        sequence_pooling='attention'
    )

    # 前向传播
    results = model(text_seq, image_seq)

    print("=== 序列信息瓶颈计算结果 ===")
    print(f"总损失: {results['total_loss'].item():.4f}")
    print(f"信息瓶颈损失: {results['info_bottleneck_loss'].item():.4f}")
    print(f"互信息: {results['mutual_info'].item():.4f}")
    print(f"文本KL散度: {results['text_kl'].item():.4f}")
    print(f"图像KL散度: {results['image_kl'].item():.4f}")
    print(f"文本重构损失: {results['text_recon_loss'].item():.4f}")
    print(f"图像重构损失: {results['image_recon_loss'].item():.4f}")

    print(f"\n瓶颈表征形状:")
    print(f"文本瓶颈表征: {results['text_z'].shape}")
    print(f"图像瓶颈表征: {results['image_z'].shape}")
    print(f"对齐后图像表征: {results['aligned_image_z'].shape}")

    return model, results


class SequenceInfoBottleneckTrainer:
    """序列信息瓶颈训练器"""

    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train_step(self, text_seq, image_seq):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        text_seq = text_seq.to(self.device)
        image_seq = image_seq.to(self.device)

        results = self.model(text_seq, image_seq)
        loss = results['total_loss']

        loss.backward()
        self.optimizer.step()

        return results

    def evaluate(self, text_seq, image_seq):
        """评估模式"""
        self.model.eval()
        with torch.no_grad():
            text_seq = text_seq.to(self.device)
            image_seq = image_seq.to(self.device)
            results = self.model(text_seq, image_seq)
        return results


if __name__ == "__main__":
    # 运行演示
    model, results = demo_usage()

    # 训练示例
    print("\n=== 训练示例 ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = SequenceInfoBottleneckTrainer(model, optimizer)

    # 模拟训练数据
    for epoch in range(5):
        text_seq = torch.randn(8, 15, 768)
        image_seq = torch.randn(8, 10, 2048)

        train_results = trainer.train_step(text_seq, image_seq)
        print(f"Epoch {epoch + 1}: Loss = {train_results['total_loss'].item():.4f}, "
              f"MI = {train_results['mutual_info'].item():.4f}")