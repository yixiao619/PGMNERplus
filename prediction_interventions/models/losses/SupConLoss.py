import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class SupConR(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConR, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        '''
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        '''

        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count
        labels = labels.squeeze()

        distance = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))

        mask = distance.unsqueeze(1) - distance.unsqueeze(2)
        mask[mask >= 0] = 1
        mask[mask < 0] = 0
        mask = mask.repeat(anchor_count, anchor_count, anchor_count)

        diag_mask = torch.eye(batch_size * anchor_count).unsqueeze(1).repeat(1, anchor_count * batch_size, 1).bool().to(device)
        mask.masked_fill_(diag_mask, 0)
        negative_samples = torch.zeros((batch_size, batch_size, batch_size))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),  # 2bs,2bs
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        I = torch.eye(batch_size, dtype=torch.float32).to(device)
        I = I.repeat(anchor_count, contrast_count)  # 2N, 2N

        logits_mask = torch.scatter(
            torch.ones_like(I),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.permute(1, 0)
        exp_logits = exp_logits.unsqueeze(0).repeat(contrast_count * batch_size, 1, 1)  # 2N,2N,2N
        negative_samples = exp_logits * mask
        # print(negative_samples.shape)
        # print(mask.shape)
        log_prob = logits - torch.log(negative_samples.sum(2))
        # log(a/b) = loga - logb

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (logits_mask * log_prob).sum(1) / logits_mask.sum(1)

        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        use_misc_weighted_scl: bool = False,
        scl_misc_neg_weight: float = 2.0,
        id_misc: int = -1,
        id_org: int = -1,
        id_loc: int = -1,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        # MTC / SCL：加强 MISC 与 ORG、LOC 之间的负对权重（关闭时与原版完全一致）
        self.use_misc_weighted_scl = use_misc_weighted_scl
        self.scl_misc_neg_weight = float(scl_misc_neg_weight)
        self.id_misc = id_misc
        self.id_org = id_org
        self.id_loc = id_loc

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:  # 如果labels不为none,则按照labels生成mask,否则按照给定的mask或者生成一个单位对角矩阵
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        features= F.normalize(features,dim=-1)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) 
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0   # 可以设置成一个超参，后面再调整
        )
        mask = mask * logits_mask

        # compute log_prob：分母为加权 sum_j exp(sim_ij/tau)（MISC<->ORG/LOC 负对权重大于 1）
        exp_logits = torch.exp(logits) * logits_mask
        if (
            self.use_misc_weighted_scl
            and self.id_misc >= 0
            and self.id_org >= 0
            and self.id_loc >= 0
        ):
            M = logits.shape[0]
            N = batch_size
            flat = labels.view(-1).long()
            idx = torch.arange(M, device=device)
            row_lab = flat[idx[:, None] % N]
            col_lab = flat[idx[None, :] % N]
            w = torch.ones((M, M), device=device, dtype=logits.dtype)
            wm = torch.tensor(
                self.scl_misc_neg_weight, device=device, dtype=logits.dtype
            )
            org_or_loc = (col_lab == self.id_org) | (col_lab == self.id_loc)
            misc_neg = (row_lab == self.id_misc) & org_or_loc
            rev_misc = (
                ((row_lab == self.id_org) | (row_lab == self.id_loc))
                & (col_lab == self.id_misc)
            )
            w = torch.where(misc_neg | rev_misc, wm, w)
            denom = (exp_logits * w).sum(1, keepdim=True).clamp(min=1e-12)
            log_prob = logits - torch.log(denom)
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-12)  # 避免除以0
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # print('(mask * log_prob).sum(1):', (mask * log_prob).sum(1))
        # print('mask.sum(1):', mask.sum(1))
        # print('non-negtive number:', (labels == 1).sum())
        # print('negtive number:', (labels == 0).sum())

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




# 进一步优化的版本
class SupConLoss_Enhanced(nn.Module):
    """增强版有监督对比学习损失"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07,
                 normalize=True, reduction='mean'):
        super(SupConLoss_Enhanced, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        device = features.device

        # 输入验证
        if len(features.shape) == 2:
            # 如果输入是[bsz, feature_dim]，添加view维度
            features = features.unsqueeze(1)  # [bsz, 1, feature_dim]
        elif len(features.shape) < 2 or len(features.shape) > 3:
            raise ValueError('Features should be [bsz, feature_dim] or [bsz, n_views, feature_dim]')

        # 特征归一化（推荐）
        if self.normalize:
            features = F.normalize(features, dim=-1)

        batch_size, n_views = features.shape[:2]

        # 处理mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 展开特征
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = n_views
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 扩展mask并去除自对比
        mask = mask.repeat(anchor_count, n_views)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算损失
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 处理没有正样本的情况
        mask_sum = mask.sum(1)
        valid_samples = mask_sum > 0

        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = torch.zeros_like(mask_sum)
        mean_log_prob_pos[valid_samples] = (mask * log_prob).sum(1)[valid_samples] / mask_sum[valid_samples]

        # 损失计算
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid_samples]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




def supervised_contrastive_loss(features, labels, temperature=0.07, base_temperature=0.07):
    """
    有监督对比学习损失函数
    参考：https://arxiv.org/abs/2004.11362

    Args:
        features: [N, E] 标准化的特征向量
        labels: [N] 标签
        temperature: 温度参数
        base_temperature: 基础温度参数

    Returns:
        loss: 有监督对比损失
    """
    device = features.device
    batch_size = features.shape[0]

    # 计算相似度矩阵
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T), temperature
    )  # [N, N]

    # 数值稳定性：减去最大值
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # 创建掩码
    labels = labels.contiguous().view(-1, 1)  # [N, 1]
    mask = torch.eq(labels, labels.T).float().to(device)  # [N, N] 相同标签为1

    # 去除对角线（自己与自己的相似度）
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0.01
    )
    mask = mask * logits_mask

    # 计算log概率
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # 计算正样本的平均log概率
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # 损失
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.mean()

    return loss




# 测试和比较函数
def test_supcon_losses():
    """测试不同版本的SupCon损失函数"""
    print("=== SupCon损失函数测试 ===\n")

    # 测试数据
    batch_size, n_views, feature_dim = 8, 2, 128
    features = torch.randn(batch_size, n_views, feature_dim)
    features = F.normalize(features, dim=-1)  # 归一化
    labels = torch.randint(0, 3, (batch_size,))  # 3个类别

    print(f"输入形状: features {features.shape}, labels {labels.shape}")
    print(f"标签分布: {torch.bincount(labels)}")

    # # 测试原始版本
    # try:
    #     loss_original = SupConLoss_Original()(features, labels)
    #     print(f"原始版本损失: {loss_original.item():.4f}")
    # except Exception as e:
    #     print(f"原始版本错误: {e}")
    #
    # # 测试修正版本
    # loss_corrected = SupConLoss_Corrected()(features, labels)
    # print(f"修正版本损失: {loss_corrected.item():.4f}")

    # 测试增强版本
    loss_enhanced = SupConLoss_Enhanced()(features, labels)
    print(f"增强版本损失: {loss_enhanced.item():.4f}")

    # 测试边界情况
    print("\n=== 边界情况测试 ===")

    # 1. 单一类别
    single_class_labels = torch.zeros(batch_size, dtype=torch.long)
    loss_single = SupConLoss_Enhanced()(features, single_class_labels)
    print(f"单一类别损失: {loss_single.item():.4f}")

    # 2. 每个样本不同类别
    unique_labels = torch.arange(batch_size)
    loss_unique = SupConLoss_Enhanced()(features, unique_labels)
    print(f"唯一类别损失: {loss_unique.item():.4f}")

    # 3. 二维输入（无view维度）
    features_2d = features[:, 0, :]  # [batch_size, feature_dim]
    loss_2d = SupConLoss_Enhanced()(features_2d, labels)
    print(f"二维输入损失: {loss_2d.item():.4f}")


