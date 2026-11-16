"""
Semantic OT / GOT 对齐模块：Sinkhorn 耦合 γ，语义代价 C_sem（cosine），可选 Gromov–Wasserstein 结构项 L_str。

- **语义 OT（Wasserstein 线性项）**：L_sem = ⟨γ, C_sem⟩，与 ⟨γ, C_sem⟩ 为 OT 主项。
- **结构 OT（GW 二次项）**：L_str = Σ_{i,i',j,j'} (A_t[i,i'] - A_v[j,j'])² γ_{ij} γ_{i'j'}（见 forward 中 einsum，等价于 broadcast 展开）。
- **熵正则**：ε · Σ γ (log γ - 1)，与 -ε H(γ) 差常数，梯度等价。
- **GOT 总损失（batch 平均）**：L_got = λ · L_sem + (1-λ) · L_str + ε · L_reg（对应 forward 中 `L_got_batch`）。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .partial_ot import build_ot_backend


def sinkhorn_coupling(
    C_sem: torch.Tensor,
    eps: float,
    mask_t: Optional[torch.Tensor],
    mask_v: Optional[torch.Tensor],
    n_iters: int,
    invalid_cost: float = 1e4,
) -> torch.Tensor:
    """
    由代价矩阵 C_sem 经 K=exp(-C/eps)，Sinkhorn 双缩放得到近似双随机传输 γ（与历史 baseline 一致）。

    C_sem: [B, N_t, N_v]
    mask_t: [B, N_t]  1 有效 0 padding；None 表示全 1
    mask_v: [B, N_v]
    对无效 (i,j)：在 C 上加 invalid_cost，使 exp(-C/eps)≈0。
    """
    B, Nt, Nv = C_sem.shape
    device = C_sem.device
    dtype = C_sem.dtype

    if mask_t is None:
        mask_t = torch.ones(B, Nt, device=device, dtype=dtype)
    else:
        mask_t = mask_t.float()
    if mask_v is None:
        mask_v = torch.ones(B, Nv, device=device, dtype=dtype)
    else:
        mask_v = mask_v.float()

    mask_ij = mask_t.unsqueeze(-1) * mask_v.unsqueeze(-2)
    C = C_sem + (1.0 - mask_ij) * invalid_cost

    log_K = -C / eps
    K = torch.exp(log_K) * mask_ij
    K = K + 1e-16

    # 边际（仅有效位置归一化）
    a = mask_t / (mask_t.sum(dim=1, keepdim=True) + 1e-12)
    b = mask_v / (mask_v.sum(dim=1, keepdim=True) + 1e-12)

    u = torch.ones(B, Nt, device=device, dtype=dtype) / float(Nt)
    v = torch.ones(B, Nv, device=device, dtype=dtype) / float(Nv)

    for _ in range(n_iters):
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
        u = a / (Kv + 1e-12)
        Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
        v = b / (Ktu + 1e-12)

    gamma = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return gamma


def _gromov_wasserstein_structure_loss(
    A_t: torch.Tensor, A_v: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """
    L_str[b] = Σ_{i,i',j,j'} (A_t[i,i'] - A_v[j,j'])² γ_{ij} γ_{i'j'}

    等价于 (A_t², r r^T) + (A_v², c c^T) - 2 ⟨A_t γ A_v^T, γ⟩ 的分解形式；此处用 einsum 与 broadcast 安全实现，无 5D 中间张量。

    A_t: [B, N_t, N_t], A_v: [B, N_v, N_v], gamma: [B, N_t, N_v]
    返回: [B]
    """
    r = gamma.sum(dim=2)  # [B, N_t] 行边际
    c = gamma.sum(dim=1)  # [B, N_v] 列边际
    term1 = torch.einsum("bij,bi,bj->b", A_t * A_t, r, r)
    term2 = torch.einsum("bij,bi,bj->b", A_v * A_v, c, c)
    cross = torch.einsum("bik,bjl,bij,bkl->b", A_t, A_v, gamma, gamma)
    return term1 + term2 - 2.0 * cross


class SemanticOTAligner(nn.Module):
    """
    语义 OT（可选 GOT 结构项）：投影头与 OptimalTransportFusion 对齐，便于公平对比。
    """

    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        hidden_dim: int,
        eps: float = 0.1,
        sinkhorn_iter: int = 10,
        lambda_align: float = 1.0,
        use_structure: bool = False,
        lambda_sem: float = 0.5,
        struct_mode: str = "full",
        got_ot_backend: str = "gw",
        partial_mass_init: float = 0.6,
        dust_cost: float = 0.5,
    ):
        """
        struct_mode（仅 use_structure=True 且后端为 gw 时生效）:
          - "full": A_t 来自注意力图、A_v 来自 patch 网格邻接；
          - "identity": A_t/A_v 为单位阵（无结构消融）；
          - "ones": A_t/A_v 为全 1（无局部结构消融）。

        got_ot_backend（与 use_got 联用时；非 "gw" 时启用 dust-bin Partial Sinkhorn 等，且关闭 GW）:
          - "gw": 语义 + Gromov–Wasserstein（默认）；
          - "partial_sinkhorn" / "node_sinkhorn" / "partial_only" / "entropy_only": 见 models/partial_ot.build_ot_backend。
        """
        super().__init__()
        self.eps = eps
        self.sinkhorn_iter = sinkhorn_iter
        self.lambda_align = lambda_align
        self.use_structure = use_structure
        self.lambda_sem = lambda_sem
        sm = (struct_mode or "full").lower()
        if sm not in ("full", "identity", "ones"):
            raise ValueError(f"struct_mode must be full|identity|ones, got {struct_mode!r}")
        self.struct_mode = sm
        gb = (got_ot_backend or "gw").lower().replace("-", "_")
        if gb in ("gromov", "gromov_wasserstein", "graph_ot"):
            gb = "gw"
        self.got_ot_backend = gb
        self._partial_ot: Optional[nn.Module] = None
        if self.got_ot_backend != "gw":
            self._partial_ot = build_ot_backend(
                self.got_ot_backend,
                eps=self.eps,
                n_iters=self.sinkhorn_iter,
                dust_cost=float(dust_cost),
                partial_mass_init=float(partial_mass_init),
                cost_half=True,
                small_eps=1e-4,
            )
            self.use_structure = False
        # 最近一次 forward 的 γ（CPU detached），供 visualize_gamma 等调试
        self.last_gamma: Optional[torch.Tensor] = None

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

    def _build_text_adj_from_attn(
        self,
        attn_t: torch.Tensor,
        mask_t: Optional[torch.Tensor],
        topk: int = 5,
    ) -> torch.Tensor:
        """
        attn_t: [B, H, N_t, N_t] 已为 span/节点级注意力时，按行 top-k 构图并对称化。
        返回: A_t [B, N_t, N_t]
        """
        attn_mean = attn_t.mean(dim=1)  # [B, N_t, N_t]
        B, Nt, _ = attn_mean.shape
        device = attn_mean.device
        dtype = attn_mean.dtype
        A_t = torch.zeros(B, Nt, Nt, device=device, dtype=dtype)
        if mask_t is None:
            m = torch.ones(B, Nt, device=device, dtype=dtype)
        else:
            m = mask_t.float()
        for b in range(B):
            for i in range(Nt):
                if m[b, i] <= 0:
                    continue
                row = attn_mean[b, i].clone()
                valid = m[b].bool()
                row = row.masked_fill(~valid, float("-inf"))
                k = min(topk, int(valid.sum().item()))
                if k <= 0:
                    continue
                vals, idx = row.topk(k=k, dim=-1)
                A_t[b, i, idx] = vals
        A_t = 0.5 * (A_t + A_t.transpose(-1, -2))
        for b in range(B):
            vb = m[b].bool()
            A_t[b].masked_fill_(~vb.unsqueeze(0), 0.0)
            A_t[b].masked_fill_(~vb.unsqueeze(1), 0.0)
        return A_t

    def _build_span_adj_from_token_attn(
        self,
        attn_t: torch.Tensor,
        span_token_idxes: torch.Tensor,
        mask_t: Optional[torch.Tensor],
        topk: int = 5,
    ) -> torch.Tensor:
        """
        attn_t: [B, H, S, S] 最后一层 token 自注意力；span_token_idxes: [B, N_t, 2] (start, end)。
        用 span 中心 token 上的平均注意力近似 span–span 边权，再每行 top-k 对称化。
        """
        attn_mean = attn_t.mean(dim=1)  # [B, S, S]
        B, Nt, _ = span_token_idxes.shape
        S = attn_mean.size(1)
        device = attn_mean.device
        dtype = attn_mean.dtype
        if mask_t is None:
            m = torch.ones(B, Nt, device=device, dtype=dtype)
        else:
            m = mask_t.float()
        s0 = span_token_idxes[..., 0].clamp(0, S - 1)
        s1 = span_token_idxes[..., 1].clamp(0, S - 1)
        centers = ((s0 + s1) // 2).long()
        idx_i = centers.unsqueeze(2).expand(-1, -1, Nt)
        idx_j = centers.unsqueeze(1).expand(-1, Nt, -1)
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(idx_i)
        A_t = attn_mean[b_idx, idx_i, idx_j]
        mm = m.unsqueeze(2) * m.unsqueeze(1)
        A_t = A_t * mm
        for b in range(B):
            for i in range(Nt):
                if m[b, i] <= 0:
                    continue
                row = A_t[b, i].clone()
                valid = m[b].bool()
                row = row.masked_fill(~valid, float("-inf"))
                k = min(topk, int(valid.sum().item()))
                if k <= 0:
                    continue
                vals, idx = row.topk(k=k, dim=-1)
                new_row = torch.zeros_like(A_t[b, i])
                new_row[idx] = vals
                A_t[b, i] = new_row
        A_t = 0.5 * (A_t + A_t.transpose(-1, -2))
        for b in range(B):
            vb = m[b].bool()
            A_t[b].masked_fill_(~vb.unsqueeze(0), 0.0)
            A_t[b].masked_fill_(~vb.unsqueeze(1), 0.0)
        return A_t

    def _build_image_adj_from_grid(
        self,
        B: int,
        N_v: int,
        h: Optional[int] = None,
        w: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_cls: bool = True,
    ) -> torch.Tensor:
        """
        网格邻接：patch 间曼哈顿距离 ≤ 1 则连边为 1；CLS 行列置 0（与 patch 无结构边）。
        返回 A_v: [B, N_v, N_v]
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        n_patch = N_v - 1 if use_cls else N_v
        if h is None or w is None:
            s = int(round(math.sqrt(max(1, n_patch))))
            if s * s == n_patch:
                h, w = s, s
            else:
                # CLIP ViT-B/32: 7×7=49 patches + CLS
                for hh in (7, 14, 16, 12):
                    if n_patch % hh == 0:
                        h, w = hh, n_patch // hh
                        break
                else:
                    h, w = s, max(1, n_patch // s)
        A = torch.zeros(N_v, N_v, device=device, dtype=dtype)
        p0 = 1 if use_cls else 0
        for i in range(n_patch):
            for j in range(n_patch):
                ri, ci = i // w, i % w
                rj, cj = j // w, j % w
                if abs(ri - rj) + abs(ci - cj) <= 1:
                    A[p0 + i, p0 + j] = 1.0
        if use_cls:
            A[0, :] = 0.0
            A[:, 0] = 0.0
        return A.unsqueeze(0).expand(B, -1, -1).contiguous()

    def set_partial_ot_warmup(self, active: bool) -> None:
        """对 partial_sinkhorn / partial_only 等模块设置 s=1 热身阶段。"""
        if self._partial_ot is None:
            return
        fn = getattr(self._partial_ot, "set_warmup", None)
        if callable(fn):
            fn(bool(active))

    def forward(
        self,
        Z_t: torch.Tensor,
        Z_v: torch.Tensor,
        mask_t: Optional[torch.Tensor] = None,
        mask_v: Optional[torch.Tensor] = None,
        attn_t: Optional[torch.Tensor] = None,
        span_token_idxes: Optional[torch.Tensor] = None,
        patch_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Z_t: [B, N_t, text_dim], Z_v: [B, N_v, visual_dim]

        attn_t: 可选 [B, H, N_t, N_t] 或 [B, H, S, S]（token 级）；后者需传 span_token_idxes。
        patch_pos: 预留 [B, N_v, 2]，未实现时使用网格推断。

        返回 text_proj, visual_proj, align_text, align_visual, gamma, L_ot, ot_components
        - L_ot：语义-only 时为 ⟨γ,C_sem⟩+εL_reg 的 batch 平均；GOT 时为 L_got（同上公式含 L_str）。
        - ot_components：L_sem、L_str、L_reg 的 batch 平均标量（L_str 在无语义结构时为 0）。
        """
        text_proj = self.text_proj(Z_t)
        visual_proj = self.visual_proj(Z_v)

        Z_t_norm = F.normalize(text_proj, p=2, dim=-1)
        Z_v_norm = F.normalize(visual_proj, p=2, dim=-1)
        cos_sim = torch.einsum("btd,bvd->btv", Z_t_norm, Z_v_norm)
        # 语义代价 C_sem（cosine 距离）
        C_sem = 0.5 * (1.0 - cos_sim)

        if self._partial_ot is not None:
            gamma, L_ot = self._partial_ot(
                text_proj,
                visual_proj,
                C_pre=C_sem,
                mask_t=mask_t,
                mask_v=mask_v,
            )
            B, Nt, Nv = C_sem.shape
            device = C_sem.device
            dtype = C_sem.dtype
            if mask_t is None:
                mt = torch.ones(B, Nt, device=device, dtype=dtype)
            else:
                mt = mask_t.float()
            if mask_v is None:
                mv = torch.ones(B, Nv, device=device, dtype=dtype)
            else:
                mv = mask_v.float()
            m_ij = mt.unsqueeze(-1) * mv.unsqueeze(-2)
            L_sem_batch = (gamma * C_sem * m_ij).sum(dim=(1, 2))
            gamma_safe = gamma.clamp(min=1e-12)
            L_reg_batch = (gamma_safe * (gamma_safe.log() - 1.0)).sum(dim=(1, 2))
            L_str_batch = torch.zeros(B, device=device, dtype=dtype)
            align_text = torch.bmm(gamma, visual_proj) * self.lambda_align
            align_visual = torch.bmm(gamma.transpose(1, 2), text_proj)
            ot_components = {
                "L_sem": L_sem_batch.mean(),
                "L_str": L_str_batch.mean(),
                "L_reg": L_reg_batch.mean(),
            }
            self.last_gamma = gamma.detach().cpu()
            return (
                text_proj,
                visual_proj,
                align_text,
                align_visual,
                gamma,
                L_ot,
                ot_components,
            )

        gamma = sinkhorn_coupling(
            C_sem, self.eps, mask_t, mask_v, self.sinkhorn_iter
        )

        L_sem_batch = (gamma * C_sem).sum(dim=(1, 2))
        gamma_safe = gamma.clamp(min=1e-12)
        L_reg_batch = (gamma_safe * (gamma_safe.log() - 1.0)).sum(dim=(1, 2))

        B, Nt, Nv = gamma.shape
        device = gamma.device
        dtype = gamma.dtype

        L_str_batch = torch.zeros(B, device=device, dtype=dtype)
        if self.use_structure:
            # 结构图 A_t / A_v：full=注意力+网格；identity=I；ones=全 1（消融）
            if self.struct_mode == "identity":
                A_t = torch.eye(Nt, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
                A_v = torch.eye(Nv, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            elif self.struct_mode == "ones":
                A_t = torch.ones(B, Nt, Nt, device=device, dtype=dtype)
                A_v = torch.ones(B, Nv, Nv, device=device, dtype=dtype)
            else:
                # full：文本图 A_t
                if attn_t is not None:
                    if attn_t.dim() == 4 and attn_t.shape[2] == Nt:
                        A_t = self._build_text_adj_from_attn(attn_t, mask_t, topk=5)
                    elif attn_t.dim() == 4 and span_token_idxes is not None:
                        A_t = self._build_span_adj_from_token_attn(
                            attn_t, span_token_idxes, mask_t, topk=5
                        )
                    else:
                        A_t = torch.eye(Nt, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
                else:
                    A_t = torch.eye(Nt, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

                # 图像图 A_v（网格；patch_pos 未用时按 N_v 推断）
                if patch_pos is not None:
                    A_v = self._build_image_adj_from_grid(
                        B, Nv, device=device, dtype=dtype, use_cls=True
                    )
                else:
                    A_v = self._build_image_adj_from_grid(
                        B, Nv, device=device, dtype=dtype, use_cls=True
                    )
            if mask_t is not None:
                m = mask_t.float()
                A_t = A_t * (m.unsqueeze(1) * m.unsqueeze(2))
            if mask_v is not None:
                mv = mask_v.float()
                A_v = A_v * (mv.unsqueeze(1) * mv.unsqueeze(2))

            # 结构 OT（GW）二次项 L_str，与 broadcast 公式等价
            L_str_batch = _gromov_wasserstein_structure_loss(A_t, A_v, gamma)

            # L_got = λ L_sem + (1-λ) L_str + ε L_reg（逐 batch 再 mean）
            L_got_batch = (
                self.lambda_sem * L_sem_batch
                + (1.0 - self.lambda_sem) * L_str_batch
                + self.eps * L_reg_batch
            )
            L_ot = L_got_batch.mean()
        else:
            # 语义 OT：L_ot = ⟨γ, C_sem⟩ + ε L_reg
            L_ot = (L_sem_batch + self.eps * L_reg_batch).mean()

        align_text = torch.bmm(gamma, visual_proj) * self.lambda_align
        align_visual = torch.bmm(gamma.transpose(1, 2), text_proj)

        ot_components = {
            "L_sem": L_sem_batch.mean(),
            "L_str": L_str_batch.mean(),
            "L_reg": L_reg_batch.mean(),
        }
        self.last_gamma = gamma.detach().cpu()
        return text_proj, visual_proj, align_text, align_visual, gamma, L_ot, ot_components
