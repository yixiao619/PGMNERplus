import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import warnings
import numpy as np


class OptimalTransportFusion(nn.Module):
    """
    基于最优传输的多模态融合模块
    添加过滤桶
    """

    def __init__(self, text_dim,visual_dim,hidden_dim=768,
                 num_iterations=100, reg_param=1e-3,threshold=1e-3,
                 fusion_method='sinkhorn',temperature=0.07,prompt_ratio=0.3):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.threshold = threshold
        self.fusion_method = fusion_method
        self.temperature = temperature
        self.prompt_ratio = prompt_ratio

        # 特征映射到统一维度
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # self.temperature = nn.Parameter(torch.tensor(0.07))

        self.dropout = nn.Dropout(p=0.2)
        self.layernorm = nn.LayerNorm(hidden_dim)

        
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )


        # 融合后的特征处理
        self.fusion_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        #
        # 可学习的正则化参数
        self.learnable_reg = nn.Parameter(torch.tensor(reg_param))



    def cosine_sim(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        计算两个3D张量之间的余弦相似度矩阵，并进行归一化处理

        参数：
        a: 形状为 (bz, sqa, dim) 的张量
        b: 形状为 (bz, sqb, dim) 的张量
        eps: 防止除以零的小常数

        返回：
        归一化后的余弦相似度矩阵，形状为 (bz, sqa, sqb)
        """
        # L2归一化，确保计算余弦相似度
        text_proj = F.normalize(a, p=2, dim=-1)  # [batch_size, t_len, emb_dim]
        image_proj = F.normalize(b, p=2, dim=-1)  # [batch_size, v_len, emb_dim]

        # 使用torch.bmm进行批量矩阵乘法
        # 计算余弦相似度矩阵
        similarity_matrix = torch.bmm(text_proj, image_proj.transpose(-2, -1))  # [batch_size, t_len, v_len]

        # 应用温度缩放
        # similarity_matrix = similarity_matrix / temperature

        # 转换为成本矩阵：相似度越高，成本越低
        # 使用 (1 - similarity) 作为成本，并确保成本非负
        cost_matrix = 1.0 - similarity_matrix  # [batch_size, t_len, v_len]

        # cost_matrix=similarity_matrix

        # 为了数值稳定性，将成本矩阵缩放到合适的范围
        # cost_matrix = F.relu(cost_matrix)  # 确保非负

        return similarity_matrix,cost_matrix


    def sinkhorn_algorithm(self, cost_matrix,num_iterations=None):
        """
        Sinkhorn算法求解最优传输计划
        Args:
            cost_matrix: [batch_size, t_len, v_len]
        Returns:
            transport_plan: [batch_size, t_len, v_len] 最优传输计划
        """

        if num_iterations is None:
            num_iterations = self.num_iterations

        batch_size, t_len, v_len = cost_matrix.shape
        device = cost_matrix.device

        # reg = torch.clamp(self.learnable_reg, min=1e-3)

        # 初始化
        K = torch.exp(-cost_matrix / self.reg_param)  # [batch_size, t_len, v_len]

        # 均匀分布作为边际分布
        mu = torch.ones(batch_size, t_len, device=device) / t_len  # 文本边际分布
        nu = torch.ones(batch_size, v_len, device=device) / v_len  # 图像边际分布

        # Sinkhorn迭代
        u = torch.ones_like(mu)  # [batch_size, t_len]
        v = torch.ones_like(nu)  # [batch_size, v_len]

        for i in range(num_iterations):
            u_prev = u.clone()

            # 更新u
            u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

            # 更新v
            v = nu / (torch.bmm(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1) + 1e-8)

            # 检查收敛
            if torch.norm(u - u_prev) < self.threshold:
                break

        # 计算最优传输计划
        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [batch_size, t_len, v_len]

        return transport_plan


    def log_sinkhorn_iterations(self,
                                cost_matrix: torch.Tensor,
                                log_mu: torch.Tensor,
                                log_nu: torch.Tensor,
                                num_iterations: int = 50) -> torch.Tensor:
        """
        Perform Sinkhorn Normalization in Log-space for stability.
        Args:
            cost_matrix: (B, M, N) 代价矩阵（已在log空间）
            log_mu:      (B, M)    源边际分布（log空间）
            log_nu:      (B, N)    目标边际分布（log空间）
            num_iterations: Sinkhorn迭代次数
        Returns:
            log transport matrix of shape (B, M, N)
        """
        # cost_matrix 已经是 log 空间，直接使用
        # 注意：调用方负责做 -C/reg 的转换
        u = torch.zeros_like(log_mu)  # (B, M)
        v = torch.zeros_like(log_nu)  # (B, N)

        for _ in range(num_iterations):
            # u 更新：对 N 维度(dim=2) 做 logsumexp
            # v.unsqueeze(1): (B,1,N) broadcast → (B,M,N)
            u = log_mu - torch.logsumexp(cost_matrix + v.unsqueeze(1), dim=2)

            # v 更新：对 M 维度(dim=1) 做 logsumexp
            # u.unsqueeze(2): (B,M,1) broadcast → (B,M,N)
            v = log_nu - torch.logsumexp(cost_matrix + u.unsqueeze(2), dim=1)

        # 最终传输矩阵：log T_ij = log K_ij + u_i + v_j
        return cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)

    def add_prompt_bucket(self, cost_matrix: torch.Tensor,
                          prompt_value: torch.Tensor):
        """
        在代价矩阵中加入 Prompt Bucket，并返回对应的边际分布。
        Args:
            cost_matrix:  (B, M, N) 原始代价矩阵
            prompt_value: (B, 1, 1) bucket 填充值
        Returns:
            extended_matrix: (B, M+1, N+1)
            log_mu:          (B, M+1)  源边际分布
            log_nu:          (B, N+1)  目标边际分布
            norm:            标量，归一化常数 -log(M+N)
        """
        batch_size, m, n = cost_matrix.shape
        device = cost_matrix.device
        one = cost_matrix.new_tensor(1)
        ms = (m * one)  # float tensor，用于 .log()
        ns = (n * one)

        # ---- 构造扩展矩阵 (B, M+1, N+1) ----
        # prompt_col: 源点→桶 的代价列，shape (B, M, 1)
        prompt_col = prompt_value.expand(batch_size, m, 1)
        # prompt_row: 桶→目标点 的代价行，shape (B, 1, N)
        prompt_row = prompt_value.expand(batch_size, 1, n)
        # p_corner:   桶→桶，shape (B, 1, 1)
        p_corner = prompt_value.expand(batch_size, 1, 1)

        # 先拼列：scores | prompt_col → (B, M, N+1)
        left = torch.cat([cost_matrix, prompt_col], dim=2)
        # 再拼行：[prompt_row | p_corner] → (B, 1, N+1)
        right = torch.cat([prompt_row, p_corner], dim=2)
        # 合并 → (B, M+1, N+1)
        extended_matrix = torch.cat([left, right], dim=1)

        # ---- 构造边际分布 ----
        # norm = -log(M+N)，归一化使总质量=1
        norm = -(ms + ns).log()

        # log_mu: 前M行每个质量=1/(M+N)，桶行质量=N/(M+N)
        # log_mu shape: (M+1,) → expand → (B, M+1)
        log_mu = torch.cat([
            norm.expand(m),  # [1/(M+N)] * M
            (ns.log() + norm).unsqueeze(0)  # N/(M+N)，桶行
        ])  # (M+1,)
        log_mu = log_mu.unsqueeze(0).expand(batch_size, -1)  # (B, M+1)

        # log_nu: 前N列每个质量=1/(M+N)，桶列质量=M/(M+N)
        log_nu = torch.cat([
            norm.expand(n),  # [1/(M+N)] * N
            (ms.log() + norm).unsqueeze(0)  # M/(M+N)，桶列
        ])  # (N+1,)
        log_nu = log_nu.unsqueeze(0).expand(batch_size, -1)  # (B, N+1)

        return extended_matrix, log_mu, log_nu, norm

    def optimal_transport_fusion(self, text_features, visual_features):
        """
        使用最优传输进行特征融合
        """
        # 计算传输计划
        if self.fusion_method == 'sinkhorn':
            b, m, _ = text_features.shape
            _, n, _ = visual_features.shape

            # 1. 计算相似度矩阵作为代价矩阵，shape (B, M, N)
            simlarity_matrix,cost_matrix = self.cosine_sim(text_features, visual_features)

            # 2. 转换到 log 空间（Sinkhorn 在 log 空间操作）
            log_cost_matrix = -cost_matrix / self.reg_param  # (B, M, N)

            # 3. 自适应计算 Prompt Bucket 值
            # 取对角线元素（每个样本的匹配分数），找第 prompt_ratio 分位
            diag_scores = torch.diagonal(simlarity_matrix, dim1=-2, dim2=-1)  # (B, M)
            k = max(1, int(self.prompt_ratio * m))
            # kthvalue 在最后一维操作，取第k小的值
            prompt_token_clip, _ = torch.kthvalue(diag_scores, k, dim=-1)  # (B,)
            # 转换到 log 空间并整形为 (B, 1, 1)

            prompt_cost = F.relu(1.0 - prompt_token_clip)  # 转代价，确保非负
            prompt_value = (-prompt_cost / self.reg_param).view(b, 1, 1)  # (B,1,1)


            # 4. 加入 Prompt Bucket，获取扩展矩阵和边际分布
            extended_matrix, log_mu, log_nu, norm = self.add_prompt_bucket(
                log_cost_matrix, prompt_value
            )  # extended_matrix: (B, M+1, N+1)

            # 5. 执行 log-space Sinkhorn
            Z = self.log_sinkhorn_iterations(
                extended_matrix, log_mu, log_nu
            )  # (B, M+1, N+1)

            # 6. 反归一化：乘以 (M+N)，即减去 norm（norm = -log(M+N)）
            Z = Z - norm  # Z + log(M+N)

            # 7. 截去 bucket 行列，还原为 (B, M, N)
            transport_plan = Z[:, :m, :n].exp()
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        
        text_attn_weights = F.softmax(transport_plan / self.temperature, dim=-1)
        text_attended = torch.bmm(text_attn_weights, visual_features)

        # 图像关注文本
        image_attn_weights = F.softmax(transport_plan.transpose(1, 2) / self.temperature, dim=-1)
        image_attended = torch.bmm(image_attn_weights, text_features)



        return text_attended, image_attended,transport_plan

    def forward(self, text_features, visual_features, attention_mask=None):
        """
        前向传播
        Args:
            text_features: [B, N_t, text_dim] 文本特征
            visual_features: [B, N_v, visual_dim] 视觉特征
            attention_mask: [B, N_t] 注意力掩码
        Returns:
            fused_text: [B, N_t, hidden_dim] 融合后的文本特征
            fused_visual: [B, N_v, hidden_dim] 融合后的视觉特征
            transport_plan: [B, N_t, N_v] 传输计划
        """
        # 特征投影到统一空间
        text_proj = self.text_proj(text_features)  # [B, N_t, hidden_dim]
        visual_proj = self.visual_proj(visual_features)  # [B, N_v, hidden_dim]

        fused_text, fused_visual,transport_plan = self.optimal_transport_fusion(
            text_proj,  # 文本特征 [B, N_t, D]
            visual_proj,  # 视觉特征 [B, N_v, D]
        )

        return text_proj,visual_proj,fused_text, fused_visual,transport_plan


