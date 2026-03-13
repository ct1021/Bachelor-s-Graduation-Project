"""
sim2sim_dr_eval.py — Sim2Sim 域随机化 (Domain Randomization) 测试
========================================================================

本脚本用于读取刚刚训练好的 PPO 模型 (g1_ppo_500k.zip)，
并在注入了【物理参数随机扰动】的 MuJoCo 环境中进行推理验证，
以测试策略的抗干扰鲁棒性，这是迈向真实物理世界 (Sim2Real) 的关键一步。

扰动项包含：
- 地面摩擦力随机变化 (±30%)
- 机器人主体各个连杆质量随机变化 (±20%)
"""

import os
import sys
import numpy as np
import argparse

# 确保能找到 envs 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from envs.g1_env import G1WalkEnv
import mujoco

class G1DomainRandomizationEnv(G1WalkEnv):
    def __init__(self, render_mode=None, rand_level=0.2):
        super().__init__(render_mode=render_mode)
        self.rand_level = rand_level
        self.original_mass = self.model.body_mass.copy()
        self.original_friction = self.model.geom_friction.copy()
        
    def reset(self, seed=None, options=None):
        # 1. 质量随机扰动 (Mass Randomization)
        # 针对每个 body 增减随机百分比的质量
        mass_noise = 1.0 + np.random.uniform(-self.rand_level, self.rand_level, size=self.original_mass.shape)
        self.model.body_mass[:] = self.original_mass * mass_noise
        
        # 2. 摩擦力随机扰动 (Friction Randomization)
        # 地面摩擦力或脚底摩擦力的随机
        fric_noise = 1.0 + np.random.uniform(-self.rand_level, self.rand_level, size=self.original_friction.shape)
        self.model.geom_friction[:] = self.original_friction * fric_noise
        
        print(f"[Sim2Sim] Domain Randomization applied (Level: ±{self.rand_level*100:.0f}%)")
        return super().reset(seed=seed, options=options)


def evaluate_robustness(model_path, num_episodes=5, rand_level=0.2):
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model at {model_path}. Error: {e}")
        return

    print("=" * 60)
    print(f" Sim2Sim Domain Randomization Evaluation")
    print(f" Model: {os.path.basename(model_path)}")
    print(f" Perturbation Level: ±{rand_level*100}% (Mass & Friction)")
    print("=" * 60)

    # 我们使用渲染模式来同时观察跌倒或抖动情况
    env = G1DomainRandomizationEnv(render_mode="human" if sys.platform.startswith("win") else None, rand_level=rand_level)
    
    total_rewards = []
    survival_steps = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        steps = 0
        
        for _ in range(1000): # max 1000 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            ep_reward += reward
            steps += 1
            
            if env.render_mode == "human":
                env.render()
                
            if terminated or truncated:
                break
                
        total_rewards.append(ep_reward)
        survival_steps.append(steps)
        print(f"Episode {ep+1}: Survived {steps} steps | Reward: {ep_reward:.2f}")

    print("-" * 60)
    print(f"Average Survival: {np.mean(survival_steps):.1f} steps")
    print(f"Average Reward:   {np.mean(total_rewards):.2f}")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/g1_ppo_500k.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.2, help="±20% physical noise")
    args = parser.parse_args()
    
    model_filepath = os.path.join(PROJECT_ROOT, args.model)
    evaluate_robustness(model_filepath, args.episodes, args.noise)
