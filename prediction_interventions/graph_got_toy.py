#!/usr/bin/env python3
"""
GOT toy：由同一基准图 + 置换 perm 构造文本图 / 图像图（带小噪声），
真实对齐为 text i ↔ image j 且 j = inv_perm[i]（理想 γ 沿该置换的“斜对角”）。

两次训练共用 H_base, A_base, perm；run_ot 每次重新初始化 log_gamma 与 optimizer。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 超参（与需求一致）----------
n = 10
d = 32
lambda_sem_semonly = 1.0
lambda_sem_got = 0.5
eps = 0.1
lr = 0.5
num_steps = 500
noise_scale = 0.05


def run_ot(
    H_t: torch.Tensor,
    H_v: torch.Tensor,
    A_t: torch.Tensor,
    A_v: torch.Tensor,
    lambda_sem: float,
    eps: float,
    lr: float,
    num_steps: int,
    log_every: int = 50,
) -> "np.ndarray":
    """
    可学习 transport：gamma = softmax(log_gamma)（全局归一化）。
    L_got = λ * L_sem + (1-λ) * L_str + L_reg
    """
    device = H_t.device
    nloc = H_t.shape[0]
    log_gamma = nn.Parameter(torch.zeros(nloc, nloc, device=device, dtype=H_t.dtype))
    optimizer = torch.optim.Adam([log_gamma], lr=lr)

    for step in range(num_steps):
        gamma = torch.softmax(log_gamma.view(-1), dim=0).view(nloc, nloc)

        # 语义：cosine 距离
        H_t_norm = F.normalize(H_t, p=2, dim=-1)
        H_v_norm = F.normalize(H_v, p=2, dim=-1)
        cos_sim = H_t_norm @ H_v_norm.T
        C_sem = 0.5 * (1.0 - cos_sim)
        L_sem = (gamma * C_sem).sum()

        # 结构：GW 型，用 A_t, A_v
        diff = A_t.unsqueeze(2).unsqueeze(3) - A_v.unsqueeze(0).unsqueeze(1)
        weight = gamma.unsqueeze(1).unsqueeze(3) * gamma.unsqueeze(0).unsqueeze(2)
        L_str = (diff**2 * weight).sum()

        gamma_safe = gamma.clamp(min=1e-12)
        L_reg = eps * (gamma_safe * (gamma_safe.log() - 1.0)).sum()

        L_got = lambda_sem * L_sem + (1.0 - lambda_sem) * L_str + L_reg

        optimizer.zero_grad()
        L_got.backward()
        optimizer.step()

        if step % log_every == 0 or step == num_steps - 1:
            print(
                f"step={step:4d}  L_got={L_got.item():.6f}  "
                f"L_sem={L_sem.item():.6f}  L_str={L_str.item():.6f}  L_reg={L_reg.item():.6f}"
            )

    with torch.no_grad():
        gamma_final = torch.softmax(log_gamma.view(-1), dim=0).view(nloc, nloc)
    return gamma_final.detach().cpu().numpy()


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    # ---------- 基准图 + 置换 ----------
    H_base = torch.randn(n, d, device=device, dtype=dtype)
    M = torch.rand(n, n, device=device, dtype=dtype)
    A_base = (M + M.T) / 2.0
    A_base = (A_base - A_base.min()) / (A_base.max() - A_base.min() + 1e-12)
    A_base.fill_diagonal_(0.0)

    perm = torch.randperm(n)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n)

    # ---------- 文本图 / 图像图（真对齐：gamma[i, inv_perm[i]] 应大）----------
    H_t = H_base + noise_scale * torch.randn_like(H_base)
    A_t = A_base.clone()

    H_v = H_base[perm] + noise_scale * torch.randn_like(H_base)
    A_v = A_base[perm][:, perm]

    print("perm =", perm.tolist())
    print("inv_perm (text i -> image j):", inv_perm.tolist())
    print()

    print("=== Run 1: semantic only (lambda_sem = 1.0) ===")
    gamma_sem_only = run_ot(
        H_t, H_v, A_t, A_v,
        lambda_sem=lambda_sem_semonly,
        eps=eps,
        lr=lr,
        num_steps=num_steps,
        log_every=50,
    )

    print("\n=== Run 2: semantic + structure (lambda_sem = 0.5), reinit log_gamma ===")
    gamma_got = run_ot(
        H_t, H_v, A_t, A_v,
        lambda_sem=lambda_sem_got,
        eps=eps,
        lr=lr,
        num_steps=num_steps,
        log_every=50,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(gamma_sem_only, cmap="viridis", aspect="auto")
    axes[0].set_title("Gamma (semantic only, lambda=1.0)")
    axes[0].set_xlabel("image node j")
    axes[0].set_ylabel("text node i")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(gamma_got, cmap="viridis", aspect="auto")
    axes[1].set_title("Gamma (semantic+structure, lambda=0.5)")
    axes[1].set_xlabel("image node j")
    axes[1].set_ylabel("text node i")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    out_path = "gamma_compare_perm.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
