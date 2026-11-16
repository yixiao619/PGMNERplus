import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import warnings
import numpy as np




def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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

    # 为了数值稳定性，将成本矩阵缩放到合适的范围
    cost_matrix = F.relu(cost_matrix)  # 确保非负



    return cost_matrix


class OptimalTransportFusion(nn.Module):
    """
    基于最优传输的多模态融合模块
    替换原有的bottleneck attention机制
    """

    def __init__(self, text_dim,visual_dim,hidden_dim=768,
                 num_iterations=100, reg_param=1e-3,threshold=1e-3,
                 fusion_method='sinkhorn',temperature=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.threshold = threshold
        self.fusion_method = fusion_method
        self.temperature = temperature

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

        # 门控机制
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )


        # # 成本矩阵学习网络
        # self.cost_network = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1)
        # )
        #
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

    def distance(self, x, y, type="L2"):
        len1, len2 = x.size(-2), y.size(-2)
        bsz, dim = x.size(0), x.size(-1)
        tx = x.unsqueeze(dim=-2).expand(bsz, len1, len2, dim)
        ty = y.unsqueeze(dim=-3).expand(bsz, len1, len2, dim)
        if type == "L2":
            dist = torch.linalg.norm(tx - ty, dim=-1)
            return dist
        else:
            sim = F.cosine_similarity(tx, ty, dim=-1)
            return 1. - sim

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

        cost_matrix=similarity_matrix

        # 为了数值稳定性，将成本矩阵缩放到合适的范围
        # cost_matrix = F.relu(cost_matrix)  # 确保非负

        return cost_matrix


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

    def sinkhorn_algorithm_batched(self,
            x: torch.Tensor, y: torch.Tensor, epsilon= 1e-3, niter= 1000, p = 2
    ):
        r"""
        另外一种sinkhorn方法
        Compute the sinkhorn approximation to the Wasserstein distance between two sets of samples using cosine distance (batched version).

        The sinkhorn algorithm adds a small entropy regularization term to the empirical Wasserstein distance.
        Hence this function solves the modified optimal transport problem:
        $ \text{maximize}_{\pi \in \Pi(a, b)} \sum_\limits_{ij} \pi_{ij}c_{ij} +\epsilon\sum\limits_{ij} \log \pi_{ij}
        \text{s.t} \, \pi 1 = a, \pi^T 1 = b
        $
        Where $\{c_{ij}\}$ is the cost matrix based on cosine distances, $\Pi(a, b)$ is the set of joint distributions with marginals $a$ and $b$.
        In the sample-based setting, all weights $a$ and $b$ are equal to $1/n$.

        The cosine distance is computed as: $d_{cosine} = 1 - \cos(\theta)$, where $\cos(\theta) = \frac{x \cdot y}{||x|| ||y||}$

        Args:
            x (torch.Tensor): tensor of samples from one distribution, shape [batch_size, x_len, dim]
            y (torch.Tensor): tensor of samples from another distribution, shape [batch_size, y_len, dim]
            epsilon (float): entropy regularization strength
            niter (int): max number of iterations
            p (int): power transformation applied to cosine distance (for compatibility)

        Returns:
            cost (torch.Tensor): Sinkhorn cost for each batch, shape [batch_size]
            transport (torch.Tensor): Transport plan for each batch, shape [batch_size, x_len, y_len]

        Source: https://personal.math.ubc.ca/~geoff/courses/W2019T1/Lecture13.pdf
        Code adapted from https://github.com/gpeyre/SinkhornAutoDiff
        Modified to use cosine distance as cost metric.
        """
        assert len(x.shape) == 3 and len(y.shape) == 3, "x and y must be 3D tensors"
        assert x.shape[0] == y.shape[0], "Batch sizes must match"
        assert x.shape[2] == y.shape[2], "Feature dimensions must match"

        batch_size, n_x, d = x.shape
        _, n_y, _ = y.shape
        n_warn = 2000
        if max(n_x, n_y) > n_warn:
            warnings.warn(
                f"The Sinkhorn algorithm is O(n^2) in the number of samples and can be slow, "
                f"consider using a different metric or reducing number of samples (n_warn={n_warn})"
            )
        epsilon=torch.clamp(self.learnable_reg, min=1e-3)

        # Compute pairwise cosine distances for each batch
        # x: [batch_size, n_x, d], y: [batch_size, n_y, d]
        # cost_matrix: [batch_size, n_x, n_y]

        # Normalize vectors to unit length for cosine similarity
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)  # [batch_size, n_x, d]
        y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)  # [batch_size, n_y, d]

        # Compute cosine similarity: cos_sim = x_norm @ y_norm^T
        # cos_sim: [batch_size, n_x, n_y]
        # cos_sim = torch.bmm(x_norm, y_norm.transpose(-1, -2))

        # Convert cosine similarity to cosine distance
        # cosine_distance = 1 - cosine_similarity
        # This gives us a distance in [0, 2] where 0 means identical vectors
        cost_matrix = self.distance(x_norm, y_norm)

        # Optional: apply power transformation if p != 2
        if p != 2:
            cost_matrix = torch.pow(cost_matrix.clamp(min=1e-16), p / 2.0)

        # Convert to double for numerical stability
        # cost_matrix = cost_matrix.double()
        K = torch.exp(-cost_matrix / epsilon)

        # Marginal distributions (uniform)
        a = torch.ones(batch_size, n_x,  device=x.device) / n_x  # [batch_size, n_x]
        b = torch.ones(batch_size, n_y, device=x.device) / n_y  # [batch_size, n_y]

        def MC(u, v):
            r"""Modified cost for logarithmic updates on u,v
            $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$

            Args:
                u: [batch_size, n_x]
                v: [batch_size, n_y]
            Returns:
                [batch_size, n_x, n_y]
            """
            return (-cost_matrix + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon

        err = 1e6
        actual_niter = 0
        thresh = 1e-3
        # u = torch.ones(batch_size, n_x,  device=x.device)  # [batch_size, n_x]
        # v = torch.ones(batch_size, n_y,  device=x.device)  # [batch_size, n_y]
        u = torch.zeros(batch_size, n_x, device=x.device)  # [batch_size, n_x]
        v = torch.zeros(batch_size, n_y, device=x.device)  # [batch_size, n_y]

        # Sinkhorn loop
        for actual_niter in range(niter):
            u1 = u.clone()
            v1 = v.clone()

            # Update u
            mc_uv = MC(u, v)  # [batch_size, n_x, n_y]
            u = epsilon * (torch.log(a + 1e-16) - torch.logsumexp(mc_uv, dim=-1)) + u  # [batch_size, n_x]

            # Update v
            mc_uv = MC(u, v)  # [batch_size, n_x, n_y]
            v = epsilon * (torch.log(b + 1e-16) - torch.logsumexp(mc_uv.transpose(-1, -2),
                                                                  dim=-1)) + v  # [batch_size, n_y]

            # Compute error across all batches
            err_u = torch.max((u - u1).abs().sum(dim=-1))  # max across batches and sum across features
            err_v = torch.max((v - v1).abs().sum(dim=-1))  # max across batches and sum across features
            err = torch.max(err_u, err_v).item()

            actual_niter += 1
            if err < thresh:
                break

        if actual_niter == niter:
            warnings.warn(
                f"Sinkhorn algorithm did not converge after {niter} iterations, final error is {err}"
            )

        # Compute final transport plan and cost
        U, V = u, v
        transport = torch.exp(MC(U, V))  # [batch_size, n_x, n_y]
        cost = torch.sum(transport * cost_matrix, dim=(-1, -2))  # [batch_size]

        return cost, transport


    def gromov_wasserstein_alignment(self, text_features, visual_features):
        """
        Gromov-Wasserstein距离进行结构对齐
        该代码有点问题，耗时特别长
        """
        # 计算内部距离矩阵
        text_gram = torch.cdist(text_features, text_features, p=2)  # [B, N_t, N_t]
        visual_gram = torch.cdist(visual_features, visual_features, p=2)  # [B, N_v, N_v]

        B, N_t, N_v = text_features.shape[0], text_features.shape[1], visual_features.shape[1]

        # 初始化传输计划
        transport_plan = torch.ones(B, N_t, N_v, device=text_features.device) / (N_t * N_v)

        for _ in range(10):  # GW迭代次数较少
            # 计算梯度
            grad = torch.zeros_like(transport_plan)

            for i in range(N_t):
                for j in range(N_v):
                    cost_ij = 0
                    for k in range(N_t):
                        for l in range(N_v):
                            cost_ij += transport_plan[:, k, l] * (text_gram[:, i, k] - visual_gram[:, j, l]) ** 2
                    grad[:, i, j] = cost_ij

            # 使用Sinkhorn更新
            # transport_plan = torch.softmax(-grad / self.learnable_reg, dim=-1)
            transport_plan = torch.softmax(-grad.transpose(-2, -1) / self.learnable_reg, dim=-1).transpose(-2, -1)

        return transport_plan

    def optimal_transport_fusion(self, text_features, visual_features):
        """
        使用最优传输进行特征融合
        """
        # 计算传输计划
        if self.fusion_method == 'sinkhorn':
            cost_matrix = self.cosine_sim(text_features, visual_features)
            transport_plan = self.sinkhorn_algorithm(cost_matrix)
            print(f"\n")
            print(f"sinkhorn传输矩阵和: {transport_plan.sum(dim=(1, 2))}")
            print(f"Row marginals (should be close to uniform): {transport_plan.sum(dim=-1)}")
            print(f"Col marginals (should be close to uniform): {transport_plan.sum(dim=-2)}")
        elif self.fusion_method == 'gromov_wasserstein':
            transport_plan = self.gromov_wasserstein_alignment(text_features, visual_features)
            print(f"传输矩阵和: {transport_plan.sum(dim=(1, 2))}")
        elif self.fusion_method == 'sinkhorn_algorithm_batched':
            cost_loss,transport_plan = self.sinkhorn_algorithm_batched(text_features, visual_features)
            # print(f"sinkhorn_algorithm_batched传输矩阵和: {transport_plan.sum(dim=(1, 2))}")
            # print(f"Row marginals (should be close to uniform): {transport_plan.sum(dim=-1)}")
            # print(f"Col marginals (should be close to uniform): {transport_plan.sum(dim=-2)}")
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # =====================基于传输计划进行特征融合(1)============================
        # # 文本 -> 视觉的传输
        # text_to_visual = torch.bmm(transport_plan, visual_features)  # [B, N_t, D]
        # # 视觉 -> 文本的传输
        # visual_to_text = torch.bmm(transport_plan.transpose(-2, -1), text_features)  # [B, N_v, D]

        # ================================使用传输矩阵作为注意力权重(2)=================================
        # 文本关注图像
        text_attn_weights = F.softmax(transport_plan / self.temperature, dim=-1)
        text_attended = torch.bmm(text_attn_weights, visual_features)

        # 图像关注文本
        image_attn_weights = F.softmax(transport_plan.transpose(1, 2) / self.temperature, dim=-1)
        image_attended = torch.bmm(image_attn_weights, text_features)

        #==============================融合策略1================================
        # # 融合文本特征
        # fused_text = torch.cat([text_features, text_to_visual], dim=-1)  # [B, N_t, 2D]
        # text_gate_weights = self.text_gate(fused_text)
        # text_enhanced = text_features + text_gate_weights * text_to_visual
        # # fused_text = self.fusion_transform(fused_text)  # [B, N_t, D]
        #
        # # 融合视觉特征
        # fused_visual = torch.cat([visual_features, visual_to_text], dim=-1)  # [B, N_v, 2D]
        # visual_gate_weights = self.visual_gate(fused_visual)
        # visual_enhanced = visual_features + visual_gate_weights * visual_to_text
        # # fused_visual = self.fusion_transform(fused_visual)  # [B, N_v, D]



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



class WassersteinDistanceLoss(nn.Module):
    """
    Wasserstein距离损失函数
    """

    def __init__(self, reg_param=0.1):
        super().__init__()
        self.reg_param = reg_param

    def forward(self, transport_plan, cost_matrix):
        """
        计算Wasserstein距离损失
        """
        # 传输成本
        transport_cost = torch.sum(transport_plan * cost_matrix, dim=(-2, -1))  # [B,]

        # 熵正则化项
        entropy_reg = -self.reg_param * torch.sum(
            transport_plan * torch.log(transport_plan + 1e-8), dim=(-2, -1)
        )  # [B,]

        wasserstein_loss = transport_cost - entropy_reg
        return wasserstein_loss.mean()




