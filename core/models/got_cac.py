# GOT-CAC：在原有 CAC（Prompt Bucket + log-space Sinkhorn）上叠加线性化 Gromov–Wasserstein 结构项。
# 复用 OT.OptimalTransportFusion 的 cosine_sim / add_prompt_bucket / log_sinkhorn_iterations，不修改 OT.py。
# PGMNER改进：+GOT-A-PP 传输计划二值化倾向正则 Omega_pp（见 forward_got_cac 末）
import torch
import torch.nn.functional as F


def _intra_cosine_batch(z: torch.Tensor) -> torch.Tensor:
    """z: [B, L, d] -> 域内余弦相似度 [B, L, L]"""
    zn = F.normalize(z, p=2, dim=-1)
    return torch.bmm(zn, zn.transpose(1, 2))


def _sparsify_got_structure(
    C: torch.Tensor,
    mode: str,
    k: int,
    threshold: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    由稠密余弦相似度构造 GOT-A 结构矩阵 C_t / C_v（仅改构图，不改 L_C = C_t T C_v^T 形式）。
    - full: 保持稠密（与历史行为一致）
    - knn: 每行保留 top-k，再行归一化
    - threshold: sim < threshold 置0，再行归一化
    """
    m = (mode or "full").lower()
    if m == "full":
        return C
    if m == "threshold":
        M = C * (C >= float(threshold)).to(dtype=C.dtype)
        row = M.sum(dim=-1, keepdim=True).clamp_min(eps)
        return M / row
    if m == "knn":
        B, L, _ = C.shape
        kk = max(1, min(int(k), L))
        _, idx = torch.topk(C, k=kk, dim=-1)
        mask = torch.zeros_like(C)
        mask.scatter_(-1, idx, 1.0)
        M = C * mask
        row = M.sum(dim=-1, keepdim=True).clamp_min(eps)
        return M / row
    raise ValueError(f"got_graph_mode must be full|knn|threshold, got {mode!r}")


def omega_pp_transport(
    T: torch.Tensor,
    top_k: int = 3,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    PGMNER改进 +GOT-A-PP: Omega_pp(T)=sum_{i,j} ReLU( -(T_ij-δ_n)(T_ij-δ_p) )，鼓励质量避开中间含糊区。

    T: [B, m, n] 传输计划 (m=span, n=patch)；
    δ_pos=1/(n*top_k), δ_neg=ε/(n*max(m-top_k,1)) 与实现说明一致（需求文档）。
    返回标量（batch 平均）。
    """
    _, m, n = T.shape
    kk = max(1, int(top_k))
    d_pos = 1.0 / (n * kk)
    d_neg = float(epsilon) / (n * max(m - kk, 1))
    p = (T - d_neg) * (T - d_pos)
    return F.relu(-p).sum(dim=(1, 2)).mean()


def forward_got_cac(
    ot,
    text_features: torch.Tensor,
    visual_features: torch.Tensor,
    lambda_gw: float,
    outer_iter: int,
    sinkhorn_iter: int = None,
    graph_mode: str = "full",
    graph_k: int = 5,
    graph_threshold: float = 0.5,
    use_got_pp: bool = False,
    got_pp_top_k: int = 3,
    got_pp_epsilon: float = 0.01,
):
    """
    与 OptimalTransportFusion.forward 对齐的返回值，并额外返回 got_loss。

    Args:
        ot: OptimalTransportFusion 实例
        text_features / visual_features: 与 ot.forward 输入相同（未投影的 span / patch 特征）
    """
    text_proj = ot.text_proj(text_features)
    visual_proj = ot.visual_proj(visual_features)

    b, m, _ = text_proj.shape
    _, n, _ = visual_proj.shape
    device = text_proj.device
    dtype = text_proj.dtype

    sim, cost = ot.cosine_sim(text_proj, visual_proj)
    T = torch.ones(b, m, n, device=device, dtype=dtype) / float(m * n)

    reg = ot.reg_param
    temp = ot.temperature
    ni = int(sinkhorn_iter) if sinkhorn_iter is not None else 50

    C_work = cost
    for _ in range(max(1, int(outer_iter))):
        C_t = _sparsify_got_structure(
            _intra_cosine_batch(text_proj),
            graph_mode,
            graph_k,
            graph_threshold,
        )
        C_v = _sparsify_got_structure(
            _intra_cosine_batch(visual_proj),
            graph_mode,
            graph_k,
            graph_threshold,
        )
        # 线性化 GW：L_C = C_t T C_v^T（GOT 常用近似，避免四阶张量）
        L_C = torch.bmm(torch.bmm(C_t, T), C_v.transpose(1, 2))
        C_work = cost + float(lambda_gw) * L_C

        log_cost_matrix = -C_work / reg

        diag_scores = torch.diagonal(sim, dim1=-2, dim2=-1)
        k = max(1, int(ot.prompt_ratio * m))
        prompt_token_clip, _ = torch.kthvalue(diag_scores, k, dim=-1)
        prompt_cost = F.relu(1.0 - prompt_token_clip)
        prompt_value = (-prompt_cost / reg).view(b, 1, 1)

        extended_matrix, log_mu, log_nu, norm = ot.add_prompt_bucket(
            log_cost_matrix, prompt_value
        )
        Z = ot.log_sinkhorn_iterations(
            extended_matrix, log_mu, log_nu, num_iterations=ni
        )
        Z = Z - norm
        T = Z[:, :m, :n].exp()

    transport_plan = T
    got_loss = (transport_plan * C_work).sum(dim=(1, 2)).mean()
    # PGMNER改进 +GOT-A-PP: 不增加 Sinkhorn 步数，仅对最终 T 加可导分段线性惩罚
    if use_got_pp:
        got_pp_omega = omega_pp_transport(
            transport_plan, top_k=got_pp_top_k, epsilon=got_pp_epsilon
        )
    else:
        got_pp_omega = transport_plan.new_zeros(())

    text_attn_weights = F.softmax(transport_plan / temp, dim=-1)
    text_attended = torch.bmm(text_attn_weights, visual_proj)
    image_attn_weights = F.softmax(transport_plan.transpose(1, 2) / temp, dim=-1)
    image_attended = torch.bmm(image_attn_weights, text_proj)

    return (
        text_proj,
        visual_proj,
        text_attended,
        image_attended,
        transport_plan,
        got_loss,
        got_pp_omega,
    )
