# -*- coding: utf-8 -*-
"""
Partial OT + 熵正则（dust-bin 增广 + log-domain Sinkhorn）

对应论文式 (11) 一类「带熵正则的部分最优传输」：可学习总传输质量 s∈[0.3,0.9]，
未匹配质量由 dust-bin 吸收。不含 NumPy/SciPy 的黑箱 OT 求解器，仅 PyTorch 基础运算。

说明
----
- ``PartialSinkhorn``：变体 **E**（Partial mass + 熵正则）。
- ``NodeSinkhorn``：变体 **C** / node-level 全质量 Sinkhorn（边际均匀、无 dust-bin）。
- ``PartialOTNoEntropy``：变体 **B**（极小 eps 近似无熵）。
- 代价矩阵默认与 ``SemanticOTAligner`` 一致：C = 0.5 * (1 - cos)，输入已可为任意 Z。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialSinkhorn(nn.Module):
    """
    Dust-bin 增广的部分最优传输 + 熵正则（Sinkhorn）。

    返回
    ----
    T : (B, n, m)   传输计划（不含 dust 行列）
    loss : 标量 ⟨T, C⟩ 的 batch 平均（C 为进入 Sinkhorn 前的 n×m 代价，不含增广 dust 块）
    """

    def __init__(
        self,
        eps: float = 0.1,
        dust_cost: float = 0.5,
        n_iters: int = 30,
        partial_mass_init: float = 0.6,
        cost_half: bool = True,
    ):
        super().__init__()
        self.eps = float(eps)
        self.dust_cost = float(dust_cost)
        self.n_iters = int(n_iters)
        self.cost_half = bool(cost_half)
        self._ot_warmup = False

        p0 = max(0.05, min(0.95, float(partial_mass_init)))
        init_logit = math.log(p0 / (1.0 - p0))
        self.s_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def set_warmup(self, active: bool) -> None:
        self._ot_warmup = bool(active)

    def get_s(self) -> torch.Tensor:
        """可学习标量 s，映射到 [0.3, 0.9]；warmup 时为 1.0（全质量）。"""
        if self._ot_warmup:
            return torch.ones((), device=self.s_logit.device, dtype=torch.float32)
        raw = torch.sigmoid(self.s_logit)
        return 0.3 + 0.6 * raw

    def _cosine_cost(self, z_text: torch.Tensor, z_vis: torch.Tensor) -> torch.Tensor:
        z_t = F.normalize(z_text, dim=-1)
        z_v = F.normalize(z_vis, dim=-1)
        cos = torch.bmm(z_t, z_v.transpose(1, 2))
        if self.cost_half:
            return 0.5 * (1.0 - cos)
        return 1.0 - cos

    def forward(
        self,
        z_text: torch.Tensor,
        z_vis: torch.Tensor,
        C_pre: Optional[torch.Tensor] = None,
        mask_t: Optional[torch.Tensor] = None,
        mask_v: Optional[torch.Tensor] = None,
        invalid_cost: float = 1e4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_text: (B, n, d), z_vis: (B, m, d)
        C_pre:  可选 (B, n, m) 直接作为代价（跳过余弦）
        mask_t / mask_v: 可选 (B, n/m)，1 有效 0 padding；与 SemanticOTAligner 一致
        """
        B, n, _ = z_text.shape
        m = z_vis.size(1)
        device = z_text.device
        dtype = z_text.dtype
        s = self.get_s().to(device=device, dtype=dtype)
        if C_pre is not None:
            C_core = C_pre
        else:
            C_core = self._cosine_cost(z_text, z_vis)

        if mask_t is None:
            mask_t = torch.ones(B, n, device=device, dtype=dtype)
        else:
            mask_t = mask_t.to(device=device, dtype=dtype).float()
        if mask_v is None:
            mask_v = torch.ones(B, m, device=device, dtype=dtype)
        else:
            mask_v = mask_v.to(device=device, dtype=dtype).float()
        mask_ij = mask_t.unsqueeze(-1) * mask_v.unsqueeze(-2)
        C_work = C_core + (1.0 - mask_ij) * invalid_cost

        denom_t = mask_t.sum(dim=1, keepdim=True).clamp_min(1e-12)
        denom_v = mask_v.sum(dim=1, keepdim=True).clamp_min(1e-12)

        tau = self.dust_cost
        C_aug = torch.full(
            (B, n + 1, m + 1), tau, dtype=C_work.dtype, device=C_work.device
        )
        C_aug[:, :n, :m] = C_work
        C_aug[:, n, m] = 0.0

        # 边际：有效 token 分摊 s，dust bin 吸收 (1-s)
        a = torch.zeros(B, n + 1, dtype=C_work.dtype, device=device)
        a[:, :n] = s * mask_t / denom_t
        a[:, n] = 1.0 - s

        b = torch.zeros(B, m + 1, dtype=C_work.dtype, device=device)
        b[:, :m] = s * mask_v / denom_v
        b[:, m] = 1.0 - s

        T_aug = self._log_sinkhorn(C_aug, a, b)
        T = T_aug[:, :n, :m]
        loss = (T * C_core * mask_ij).sum(dim=(1, 2)).mean()
        return T, loss

    def _log_sinkhorn(
        self, C: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Log-domain Sinkhorn，C (B,N,M)，a (B,N)，b (B,M)。"""
        eps = self.eps
        log_a = torch.log(a.clamp_min(1e-12))
        log_b = torch.log(b.clamp_min(1e-12))
        log_K = -C / eps

        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)

        for _ in range(self.n_iters):
            v = log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)
            u = log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)

        log_T = log_K + u.unsqueeze(2) + v.unsqueeze(1)
        return torch.exp(log_T)


class NodeSinkhorn(nn.Module):
    """全边际均匀、标准熵正则 Sinkhorn（变体 A/C 中的「full mass」node-level OT）。"""

    def __init__(
        self, eps: float = 0.1, n_iters: int = 30, cost_half: bool = True
    ):
        super().__init__()
        self.eps = float(eps)
        self.n_iters = int(n_iters)
        self.cost_half = bool(cost_half)

    def set_warmup(self, active: bool) -> None:
        del active  # 全质量 OT 无需热身

    def get_s(self) -> torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float32)

    def forward(
        self,
        z_text: torch.Tensor,
        z_vis: torch.Tensor,
        C_pre: Optional[torch.Tensor] = None,
        mask_t: Optional[torch.Tensor] = None,
        mask_v: Optional[torch.Tensor] = None,
        invalid_cost: float = 1e4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n, _ = z_text.shape
        m = z_vis.size(1)
        device = z_text.device
        dtype = z_text.dtype
        if C_pre is not None:
            C_core = C_pre
        else:
            z_t = F.normalize(z_text, dim=-1)
            z_v = F.normalize(z_vis, dim=-1)
            cos = torch.bmm(z_t, z_v.transpose(1, 2))
            C_core = 0.5 * (1.0 - cos) if self.cost_half else (1.0 - cos)

        if mask_t is None:
            mask_t = torch.ones(B, n, device=device, dtype=dtype)
        else:
            mask_t = mask_t.to(device=device, dtype=dtype).float()
        if mask_v is None:
            mask_v = torch.ones(B, m, device=device, dtype=dtype)
        else:
            mask_v = mask_v.to(device=device, dtype=dtype).float()
        mask_ij = mask_t.unsqueeze(-1) * mask_v.unsqueeze(-2)
        C_work = C_core + (1.0 - mask_ij) * invalid_cost

        a = mask_t / (mask_t.sum(dim=1, keepdim=True) + 1e-12)
        b = mask_v / (mask_v.sum(dim=1, keepdim=True) + 1e-12)
        log_a = torch.log(a.clamp_min(1e-12))
        log_b = torch.log(b.clamp_min(1e-12))
        log_K = -C_work / self.eps
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(self.n_iters):
            v = log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)
            u = log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
        T = torch.exp(log_K + u.unsqueeze(2) + v.unsqueeze(1))
        loss = (T * C_core * mask_ij).sum(dim=(1, 2)).mean()
        return T, loss


class PartialOTNoEntropy(nn.Module):
    """变体 B：极小 eps 近似无熵正则，仍保留可学习 s（PartialSinkhorn 内部）。"""

    def __init__(
        self,
        eps: float = 1e-4,
        n_iters: int = 50,
        partial_mass_init: float = 0.6,
        dust_cost: float = 0.5,
        cost_half: bool = True,
    ):
        super().__init__()
        self.inner = PartialSinkhorn(
            eps=eps,
            dust_cost=dust_cost,
            n_iters=n_iters,
            partial_mass_init=partial_mass_init,
            cost_half=cost_half,
        )

    def forward(
        self,
        z_text: torch.Tensor,
        z_vis: torch.Tensor,
        C_pre: Optional[torch.Tensor] = None,
        mask_t: Optional[torch.Tensor] = None,
        mask_v: Optional[torch.Tensor] = None,
        invalid_cost: float = 1e4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inner(
            z_text,
            z_vis,
            C_pre=C_pre,
            mask_t=mask_t,
            mask_v=mask_v,
            invalid_cost=invalid_cost,
        )

    def get_s(self) -> torch.Tensor:
        return self.inner.get_s()

    def set_warmup(self, active: bool) -> None:
        self.inner.set_warmup(active)


class EntropyOTNoPartial(nn.Module):
    """变体 C：全质量 + 熵正则（与 NodeSinkhorn 等价接口）。"""

    def __init__(self, eps: float = 0.1, n_iters: int = 30, cost_half: bool = True):
        super().__init__()
        self.inner = NodeSinkhorn(eps=eps, n_iters=n_iters, cost_half=cost_half)

    def forward(
        self,
        z_text: torch.Tensor,
        z_vis: torch.Tensor,
        C_pre: Optional[torch.Tensor] = None,
        mask_t: Optional[torch.Tensor] = None,
        mask_v: Optional[torch.Tensor] = None,
        invalid_cost: float = 1e4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inner(
            z_text,
            z_vis,
            C_pre=C_pre,
            mask_t=mask_t,
            mask_v=mask_v,
            invalid_cost=invalid_cost,
        )

    def set_warmup(self, active: bool) -> None:
        self.inner.set_warmup(active)


def build_ot_backend(name: str, **kw) -> nn.Module:
    """工厂：got_ot_backend 名 -> 模块。"""
    n = (name or "gw").lower().replace("-", "_")
    if n in ("partial_sinkhorn", "full", "e", "proposed"):
        return PartialSinkhorn(
            eps=kw.get("eps", 0.1),
            dust_cost=kw.get("dust_cost", 0.5),
            n_iters=kw.get("n_iters", 30),
            partial_mass_init=kw.get("partial_mass_init", 0.6),
            cost_half=kw.get("cost_half", True),
        )
    if n in ("node_sinkhorn", "node", "a_sinkhorn", "entropy_full"):
        return NodeSinkhorn(
            eps=kw.get("eps", 0.1),
            n_iters=kw.get("n_iters", 30),
            cost_half=kw.get("cost_half", True),
        )
    if n in ("partial_only", "partial_no_ent", "b"):
        return PartialOTNoEntropy(
            eps=kw.get("small_eps", 1e-4),
            n_iters=kw.get("n_iters", 50),
            partial_mass_init=kw.get("partial_mass_init", 0.6),
            dust_cost=kw.get("dust_cost", 0.5),
            cost_half=kw.get("cost_half", True),
        )
    if n in ("sinkhorn", "entropy_only", "c", "balanced"):
        return EntropyOTNoPartial(
            eps=kw.get("eps", 0.1),
            n_iters=kw.get("n_iters", 30),
            cost_half=kw.get("cost_half", True),
        )
    raise ValueError(f"unknown partial OT backend: {name!r}")


def build_ot_module(got_mode: str, **kw) -> nn.Module:
    """与 Prompt 一致的工厂别名（got_mode → partial_ot 后端）。"""
    return build_ot_backend(got_mode or "node_sinkhorn", **kw)
