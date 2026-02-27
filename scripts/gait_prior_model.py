"""
gait_prior_model.py — 离线步态先验策略网络 (Offline Gait Prior)

基于俞老师建议：训练一个离线的小模型作为步态先验，
再结合运行过程实时数据进行更新。

本模块实现一个轻量级 MLP 策略网络，输出高斯动作分布的
均值(μ)和标准差(σ)，供后续 KL 散度约束使用。
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple
import os


class GaitPriorNetwork(nn.Module):
    """
    轻量级 MLP 步态先验策略网络。
    
    架构: obs_dim → 128 → 64 → action_dim (μ, log_σ)
    
    用途:
        1. 先用专家步态数据（MoCap / BC）离线训练本网络
        2. 训练完成后冻结参数，作为 KL 散度正则项的 "anchor"
        3. RL 在线训练时，约束实时策略不偏离此先验分布
    
    Args:
        obs_dim (int):    观测空间维度
        action_dim (int): 动作空间维度
        hidden1 (int):    第一隐藏层神经元数 (默认 128)
        hidden2 (int):    第二隐藏层神经元数 (默认 64)
        log_std_min (float): log_σ 下界裁剪 (防止方差过小)
        log_std_max (float): log_σ 上界裁剪 (防止方差过大)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # ---- 共享特征提取层 ----
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        # ---- 均值头 (μ) ----
        self.mean_head = nn.Linear(hidden2, action_dim)

        # ---- 对数标准差头 (log σ) ----
        self.log_std_head = nn.Linear(hidden2, action_dim)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化，提升训练稳定性。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            obs: 观测张量, shape = (batch, obs_dim)

        Returns:
            mean:    动作均值 μ,  shape = (batch, action_dim)
            log_std: 对数标准差,  shape = (batch, action_dim)
        """
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # 裁剪 log_std 防止数值不稳定
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """
        获取给定观测下的动作概率分布 (高斯分布)。

        这是 KL 散度计算的核心方法：
            KL( π_online || π_prior ) 中的 π_prior 即来自此方法。

        Args:
            obs: 观测张量, shape = (batch, obs_dim)

        Returns:
            Normal 分布对象, 可用于 sample / log_prob / entropy 等操作
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def predict_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        预测动作（用于推理 / 数据收集）。

        Args:
            obs: 观测张量
            deterministic: True 时返回均值，False 时采样

        Returns:
            action: 动作张量, shape = (batch, action_dim)
        """
        dist = self.get_distribution(obs)
        if deterministic:
            return dist.mean
        return dist.sample()

    def save(self, path: str):
        """保存模型权重。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✅ 先验模型已保存至: {path}")

    def load(self, path: str, device: str = "cpu"):
        """加载模型权重。"""
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"✅ 先验模型已从 {path} 加载")
        return self


# ======================== 使用示例 ========================
if __name__ == "__main__":
    # 假设: 观测维度=24 (双足机器人典型值), 动作维度=6 (6个关节)
    OBS_DIM = 24
    ACT_DIM = 6
    BATCH = 32

    model = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟前向传播
    dummy_obs = torch.randn(BATCH, OBS_DIM)
    mean, log_std = model(dummy_obs)
    print(f"均值 shape: {mean.shape}, log_std shape: {log_std.shape}")

    # 获取概率分布
    dist = model.get_distribution(dummy_obs)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    print(f"采样动作 shape: {action.shape}, log_prob shape: {log_prob.shape}")

    # 保存 & 加载
    model.save("models/gait_prior_test.pt")
    model2 = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)
    model2.load("models/gait_prior_test.pt")
    print("✅ 先验模型 — 所有功能测试通过!")
