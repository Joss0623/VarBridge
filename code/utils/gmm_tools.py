import torch
import torch.nn as nn
import numpy as np
import math

class GMMWeightedHuberLoss(nn.Module):
    def __init__(self, gmm_priors_path, total_epochs=55, delta=1.0):
        super(GMMWeightedHuberLoss, self).__init__()
        # 1. 加载 GMM 统计先验
        self.priors = np.load(gmm_priors_path, allow_pickle=True)
        self.total_epochs = total_epochs
        self.num_channels = len(self.priors)
        # 2. 基础 Loss 换成 Huber，delta=1.0 是标准设置
        self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)

    def forward(self, y_pred, y_true, epoch):
        # 计算基于 GMM 机理的动态权重
        weights, alpha = self._get_weights(y_true, epoch)
        
        # 计算基础 Huber Loss
        base_loss = self.loss_fn(y_pred, y_true)
        # 应用权重
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean(), weights.mean(), alpha

    def _get_weights(self, y_true, epoch):
        y_np = y_true.detach().cpu().numpy()
        bs, seq, dim = y_np.shape
        prob_density = np.zeros_like(y_np)

        effective_dim = min(dim, self.num_channels)

        # 概率密度计算：识别代谢状态的“平凡性”
        for c in range(effective_dim):
            p = self.priors[c]
            c_val = y_np[:, :, c]
            c_prob = np.zeros_like(c_val)
            for k in range(len(p['weights'])):
                norm_pdf = (1.0 / (p['stds'][k] * np.sqrt(2 * np.pi) + 1e-6)) * \
                           np.exp(-0.5 * ((c_val - p['means'][k]) / (p['stds'][k] + 1e-6))**2)
                c_prob += p['weights'][k] * norm_pdf
            prob_density[:, :, c] = c_prob

        # NLL (负对数似然) 反映了点相对于全局分布的“距离”
        distances = -np.log(prob_density + 1e-9)
        # 局部 Z-Score 归一化：将距离信号对齐到统一量级
        distances_norm = np.abs((distances - distances.mean()) / (distances.std() + 1e-6))

        # Alpha 调度：前期专注高频规律，后期专注罕见拐点
        alpha = 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))
        
        # 参考你提供的逻辑进行权重分配
        forward_weight = np.exp(-0.1 * distances_norm)
        reversed_weight = 1 + 0.3 * np.log1p(distances_norm)
        
        final_w = alpha * forward_weight + (1 - alpha) * reversed_weight
        
        # 对整个 Batch 的权重进行归一化，均值为 1.0 保证梯度稳定
        final_w = final_w / (np.mean(final_w) + 1e-6)
        
        return torch.tensor(final_w, dtype=torch.float32).to(y_true.device), alpha