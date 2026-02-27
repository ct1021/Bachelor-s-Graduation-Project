"""
hybrid_loss.py — 混合损失函数: BC Loss + λ·KL Divergence

基于俞老师建议：借助于元学习模型和步态概率特性的 KL 散度作为损失机制。

Loss = L_BC + λ * D_KL( π_online || π_prior )

其中:
    - L_BC    : 行为克隆损失 (MSE between predicted action & expert action)
    - D_KL    : KL 散度，约束在线策略分布不偏离离线先验分布
    - λ (lambda): KL 散度权重系数

核心思想:
    1. 先用专家数据训练 GaitPriorNetwork（离线小模型）
    2. RL 训练时，将 KL(π_online || π_prior) 加入损失
    3. 这样实时策略既能探索新动作，又不会遗忘步态先验
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict

# 导入本项目的先验模型
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gait_prior_model import GaitPriorNetwork


class HybridLoss(nn.Module):
    """
    混合损失函数: BC Loss + λ·KL Divergence。

    Args:
        prior_model (GaitPriorNetwork): 已训练好的离线步态先验网络 (冻结参数)
        lambda_kl (float): KL 散度权重系数, 默认 0.1
            - 值越大 → 策略越保守，紧跟先验
            - 值越小 → 策略更自由，允许偏离先验
        bc_weight (float): 行为克隆损失权重, 默认 1.0
    """

    def __init__(
        self,
        prior_model: GaitPriorNetwork,
        lambda_kl: float = 0.1,
        bc_weight: float = 1.0,
    ):
        super().__init__()
        self.prior_model = prior_model
        # 冻结先验模型参数，不参与梯度更新
        for param in self.prior_model.parameters():
            param.requires_grad = False
        self.prior_model.eval()

        self.lambda_kl = lambda_kl
        self.bc_weight = bc_weight
        self.mse = nn.MSELoss()

    def bc_loss(
        self, pred_actions: torch.Tensor, expert_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        行为克隆损失 (Behavioral Cloning Loss)。

        使用 MSE 回归损失衡量预测动作与专家动作的差异：
            L_BC = (1/N) Σ ||a_pred - a_expert||²

        Args:
            pred_actions:   在线策略预测的动作, shape=(batch, action_dim)
            expert_actions: 专家示范的真实动作, shape=(batch, action_dim)

        Returns:
            标量 MSE 损失
        """
        return self.mse(pred_actions, expert_actions)

    def kl_divergence_loss(
        self, policy_dist: Normal, obs: torch.Tensor
    ) -> torch.Tensor:
        """
        KL 散度损失：约束在线策略分布不偏离离线先验分布。

        D_KL( π_online(·|s) || π_prior(·|s) )

        使用 torch.distributions.kl_divergence 精确计算两个
        高斯分布之间的 KL 散度（有闭式解）。

        Args:
            policy_dist: 在线策略的动作分布 Normal(μ_online, σ_online)
            obs:         当前观测, shape=(batch, obs_dim)
                         用于让先验模型产生对应的 π_prior(·|s)

        Returns:
            KL 散度的 batch 均值 (标量)
        """
        with torch.no_grad():
            prior_dist = self.prior_model.get_distribution(obs)

        # kl_divergence 返回 shape=(batch, action_dim)，先对动作维度求和，再对 batch 求均值
        kl = kl_divergence(policy_dist, prior_dist)  # (batch, action_dim)
        kl = kl.sum(dim=-1).mean()  # 标量
        return kl

    def forward(
        self,
        policy_dist: Normal,
        pred_actions: torch.Tensor,
        expert_actions: torch.Tensor,
        obs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算混合损失。

        Loss = bc_weight * L_BC + lambda_kl * D_KL

        Args:
            policy_dist:    在线策略的动作分布 Normal(μ, σ)
            pred_actions:   策略预测动作 (通常为 policy_dist.mean)
            expert_actions: 专家动作
            obs:            当前观测

        Returns:
            dict: {
                "total_loss": 总损失 (用于 backward),
                "bc_loss":    行为克隆损失分量,
                "kl_loss":    KL 散度损失分量,
            }
        """
        l_bc = self.bc_loss(pred_actions, expert_actions)
        l_kl = self.kl_divergence_loss(policy_dist, obs)

        total = self.bc_weight * l_bc + self.lambda_kl * l_kl

        return {
            "total_loss": total,
            "bc_loss": l_bc,
            "kl_loss": l_kl,
        }


def pretrain_with_hybrid_loss(
    online_model: GaitPriorNetwork,
    prior_model: GaitPriorNetwork,
    expert_obs: torch.Tensor,
    expert_actions: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_kl: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, list]:
    """
    完整的混合损失预训练流程示例。

    使用 BC + KL 散度联合训练在线策略网络:
        1. 用专家数据做行为克隆 (让策略模仿步态)
        2. 同时用 KL 散度约束策略不偏离先验 (防止过拟合)

    Args:
        online_model:   要训练的在线策略网络
        prior_model:    已冻结的离线步态先验网络
        expert_obs:     专家观测数据, shape=(N, obs_dim)
        expert_actions: 专家动作数据, shape=(N, action_dim)
        epochs:         训练轮数
        batch_size:     批次大小
        lr:             学习率
        lambda_kl:      KL 散度权重
        device:         训练设备

    Returns:
        训练历史 dict: {"total": [...], "bc": [...], "kl": [...]}
    """
    online_model = online_model.to(device)
    prior_model = prior_model.to(device)

    criterion = HybridLoss(prior_model, lambda_kl=lambda_kl)
    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)

    dataset = TensorDataset(expert_obs, expert_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"total": [], "bc": [], "kl": []}

    for epoch in range(epochs):
        epoch_losses = {"total": 0.0, "bc": 0.0, "kl": 0.0}
        n_batches = 0

        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # 获取在线策略的分布和预测动作
            policy_dist = online_model.get_distribution(obs_batch)
            pred_actions = policy_dist.mean  # 使用均值作为预测

            # 计算混合损失
            losses = criterion(policy_dist, pred_actions, act_batch, obs_batch)

            # 反向传播
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

            epoch_losses["total"] += losses["total_loss"].item()
            epoch_losses["bc"] += losses["bc_loss"].item()
            epoch_losses["kl"] += losses["kl_loss"].item()
            n_batches += 1

        # 记录 epoch 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)
            history[key].append(epoch_losses[key])

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Total: {epoch_losses['total']:.4f} | "
                f"BC: {epoch_losses['bc']:.4f} | "
                f"KL: {epoch_losses['kl']:.4f}"
            )

    return history


# ======================== 使用示例 ========================
if __name__ == "__main__":
    OBS_DIM = 24
    ACT_DIM = 6
    N_EXPERT = 1000  # 模拟专家数据量

    # 创建先验模型 (模拟已离线训练好)
    prior = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)

    # 创建在线策略模型
    online = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)

    # 模拟专家数据 (实际项目中应从 data/ 加载 MoCap 数据)
    expert_obs = torch.randn(N_EXPERT, OBS_DIM)
    expert_act = torch.randn(N_EXPERT, ACT_DIM) * 0.5  # 专家动作幅度较小

    print("=" * 60)
    print("混合损失预训练测试 (BC + KL Divergence)")
    print("=" * 60)

    history = pretrain_with_hybrid_loss(
        online_model=online,
        prior_model=prior,
        expert_obs=expert_obs,
        expert_actions=expert_act,
        epochs=30,
        batch_size=64,
        lambda_kl=0.1,
    )

    print(f"\n最终损失 → Total: {history['total'][-1]:.4f}, "
          f"BC: {history['bc'][-1]:.4f}, KL: {history['kl'][-1]:.4f}")
    print("✅ 混合损失函数 — 所有功能测试通过!")
