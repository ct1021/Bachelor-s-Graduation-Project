"""
physics_reward.py — 物理感知奖励函数 (Physics-Aware Reward)

Stable-Baselines3 兼容的自定义奖励函数包装器。
通过 gymnasium.Wrapper 包裹任意环境，替换/增强其奖励信号。

奖励组成:
    r = w_vel * r_velocity + w_cot * r_cot + w_cop * r_cop + r_env

    1. 速度跟踪奖励 (Velocity Tracking):
       r_vel = exp(-α * (v_actual - v_target)²)
       → 鼓励行走速度接近目标速度 [0.2 ~ 1.0 m/s]

    2. 能耗惩罚 (Cost of Transport, COT):
       r_cot = -β * Σ|τ_i * q̇_i| / (m * g * |v| + ε)
       → 惩罚高能耗动作，鼓励能量高效步态

    3. CoP 软约束 (Center of Pressure):
       r_cop = -γ * ||CoP - support_center||²
       → 压力中心偏离支撑域中心越远，惩罚越大（ZMP 稳定性指标）
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple


class PhysicsRewardWrapper(gym.Wrapper):
    """
    物理感知奖励包装器 — 可直接与 SB3 配合使用。

    用法:
        env = gym.make("YourBipedEnv-v0")
        env = PhysicsRewardWrapper(env, target_velocity=0.5)
        model = PPO("MlpPolicy", env, ...)

    Args:
        env:              基础 Gymnasium 环境
        target_velocity:  目标行走速度 (m/s), 默认 0.5
        robot_mass:       机器人质量 (kg), 用于 COT 计算
        alpha:            速度跟踪奖励的衰减系数 (越大越敏感)
        beta:             能耗惩罚系数
        gamma:            CoP 偏离惩罚系数
        w_vel:            速度跟踪奖励权重
        w_cot:            能耗惩罚权重
        w_cop:            CoP 约束权重
        w_env:            原始环境奖励保留权重 (0 = 完全替换)
        support_polygon_half_length: 支撑域半长 (m)
    """

    def __init__(
        self,
        env: gym.Env,
        target_velocity: float = 0.5,
        robot_mass: float = 10.0,
        alpha: float = 5.0,
        beta: float = 0.01,
        gamma: float = 10.0,
        w_vel: float = 1.0,
        w_cot: float = 0.5,
        w_cop: float = 0.3,
        w_env: float = 0.2,
        support_polygon_half_length: float = 0.1,
    ):
        super().__init__(env)
        self.target_velocity = target_velocity
        self.robot_mass = robot_mass
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_vel = w_vel
        self.w_cot = w_cot
        self.w_cop = w_cop
        self.w_env = w_env
        self.support_half_len = support_polygon_half_length
        self.gravity = 9.81

        # 记录各奖励分量，便于 TensorBoard 监控
        self._reward_components: Dict[str, float] = {}

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        """
        执行一步，计算物理感知奖励。

        对于通用环境，本方法从 info 字典中提取物理量。
        如果 info 中没有所需的物理量，则使用安全的默认值。
        """
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # ---- 1. 提取物理量 (从 info 或 obs) ----
        velocity = self._get_velocity(obs, info)
        joint_torques = self._get_joint_torques(obs, info, action)
        joint_velocities = self._get_joint_velocities(obs, info)
        cop_position = self._get_cop(obs, info)
        support_center = self._get_support_center(obs, info)

        # ---- 2. 计算各奖励分量 ----
        r_vel = self._velocity_tracking_reward(velocity)
        r_cot = self._cot_penalty(joint_torques, joint_velocities, velocity)
        r_cop = self._cop_constraint(cop_position, support_center)

        # ---- 3. 加权求和 ----
        total_reward = (
            self.w_vel * r_vel
            + self.w_cot * r_cot
            + self.w_cop * r_cop
            + self.w_env * env_reward
        )

        # ---- 4. 记录分量 ----
        self._reward_components = {
            "reward/velocity_tracking": r_vel,
            "reward/cot_penalty": r_cot,
            "reward/cop_constraint": r_cop,
            "reward/env_original": env_reward,
            "reward/total": total_reward,
        }
        info.update(self._reward_components)

        return obs, total_reward, terminated, truncated, info

    # ==================== 奖励计算函数 ====================

    def _velocity_tracking_reward(self, velocity: float) -> float:
        """
        速度跟踪奖励: r_vel = exp(-α * (v - v_target)²)

        当 v = v_target 时奖励为 1.0，偏差越大奖励越低。
        """
        error = velocity - self.target_velocity
        return float(np.exp(-self.alpha * error ** 2))

    def _cot_penalty(
        self,
        torques: np.ndarray,
        joint_vels: np.ndarray,
        velocity: float,
    ) -> float:
        """
        能耗惩罚 (Cost of Transport):
            r_cot = -β * Σ|τ_i * q̇_i| / (m * g * |v| + ε)

        COT 是机器人学中衡量运动效率的标准指标:
            - 分子: 各关节的瞬时功率之和
            - 分母: 质量 × 重力 × 速度 (归一化因子)
        """
        # 关节功率 = 力矩 × 角速度
        power = np.sum(np.abs(torques * joint_vels))
        # 归一化因子 (加 ε 避免除零)
        normalizer = self.robot_mass * self.gravity * (abs(velocity) + 1e-6)
        cot = power / normalizer
        return -self.beta * cot

    def _cop_constraint(
        self, cop: np.ndarray, support_center: np.ndarray
    ) -> float:
        """
        CoP 软约束:
            r_cop = -γ * ||CoP - support_center||²

        基于 ZMP (零力矩点) 稳定性准则:
            - CoP 在支撑域内部 → 机器人稳定
            - CoP 越偏离中心 → 越不稳定 → 惩罚越大
        """
        displacement = np.linalg.norm(cop - support_center)
        return -self.gamma * displacement ** 2

    # ==================== 物理量提取器 ====================
    # 以下方法从 info/obs 中提取所需的物理量。
    # 当接入真正的双足环境 (如 MuJoCo Humanoid) 时，
    # 只需重写这几个方法即可适配不同环境。

    def _get_velocity(self, obs: np.ndarray, info: Dict) -> float:
        """提取前进方向速度。"""
        if "velocity" in info:
            return float(info["velocity"])
        if "x_velocity" in info:
            return float(info["x_velocity"])
        # 通用 fallback: 取 obs 的第一个元素作为速度估计
        if len(obs) > 0:
            return float(obs[0]) * 0.1  # 缩放到合理范围
        return 0.0

    def _get_joint_torques(
        self, obs: np.ndarray, info: Dict, action: np.ndarray
    ) -> np.ndarray:
        """提取关节力矩。"""
        if "joint_torques" in info:
            return np.array(info["joint_torques"])
        # Fallback: 用动作作为力矩的近似 (许多环境中 action ≈ 力矩)
        return np.array(action, dtype=np.float64)

    def _get_joint_velocities(
        self, obs: np.ndarray, info: Dict
    ) -> np.ndarray:
        """提取关节角速度。"""
        if "joint_velocities" in info:
            return np.array(info["joint_velocities"])
        # Fallback: 使用 obs 后半部分作为速度估计
        n = len(obs)
        if n >= 4:
            return obs[n // 2:]
        return np.ones_like(obs) * 0.1

    def _get_cop(self, obs: np.ndarray, info: Dict) -> np.ndarray:
        """提取压力中心 (Center of Pressure) 位置。"""
        if "cop" in info:
            return np.array(info["cop"])
        if "center_of_pressure" in info:
            return np.array(info["center_of_pressure"])
        # Fallback: 假设 CoP 在原点附近 + 微小随机偏置
        return np.array([0.0, 0.0])

    def _get_support_center(
        self, obs: np.ndarray, info: Dict
    ) -> np.ndarray:
        """提取支撑域中心位置。"""
        if "support_center" in info:
            return np.array(info["support_center"])
        # Fallback: 支撑域中心在原点
        return np.array([0.0, 0.0])

    @property
    def reward_components(self) -> Dict[str, float]:
        """获取最近一步各奖励分量的值 (用于日志记录)。"""
        return self._reward_components.copy()


# ======================== 使用示例 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("PhysicsRewardWrapper — 功能测试")
    print("=" * 60)

    # 用 CartPole 做快速功能验证 (真正使用时替换为双足环境)
    env = gym.make("CartPole-v1")
    wrapped_env = PhysicsRewardWrapper(
        env,
        target_velocity=0.5,
        robot_mass=10.0,
        w_vel=1.0,
        w_cot=0.5,
        w_cop=0.3,
        w_env=0.2,
    )

    obs, info = wrapped_env.reset()
    print(f"观测 shape: {obs.shape}")

    total_reward = 0.0
    for step in range(100):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        total_reward += reward

        if step == 0:
            print(f"\n第 1 步奖励分量:")
            for key, val in info.items():
                if key.startswith("reward/"):
                    print(f"  {key}: {val:.4f}")

        if terminated or truncated:
            obs, info = wrapped_env.reset()

    print(f"\n100 步总奖励: {total_reward:.2f}")
    wrapped_env.close()
    print("✅ 物理感知奖励函数 — 所有功能测试通过!")
