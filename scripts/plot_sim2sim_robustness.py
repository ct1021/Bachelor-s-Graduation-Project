"""
plot_sim2sim_robustness.py
========================================================================
系统性地验证 PPO 算法对物理参数摄动（质量、摩擦力）的鲁棒性，
并生成用于毕业论文或中期检查报告的可视化图表。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.sim2sim_dr_eval import G1DomainRandomizationEnv

def run_dr_experiment(model_path, noise_levels, episodes_per_level=10):
    print("加载模型:", model_path)
    model = PPO.load(model_path)
    
    results = {}
    
    for noise in noise_levels:
        print(f"\n[Testing] 注入环境噪声: ±{noise*100:.0f}%")
        env = G1DomainRandomizationEnv(render_mode=None, rand_level=noise)
        survival_steps = []
        
        for ep in range(episodes_per_level):
            obs, _ = env.reset()
            steps = 0
            for _ in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                steps += 1
                if term or trunc:
                    break
            survival_steps.append(steps)
            
        avg_survival = np.mean(survival_steps)
        std_survival = np.std(survival_steps)
        results[noise] = (avg_survival, std_survival)
        print(f" -> 平均存活步数: {avg_survival:.1f} ± {std_survival:.1f}")
        env.close()
        
    return results

def plot_results(results, save_path):
    noises = list(results.keys())
    means = [results[n][0] for n in noises]
    stds = [results[n][1] for n in noises]
    
    x_labels = [f"±{n*100:.0f}%" for n in noises]
    x_pos = np.arange(len(noises))
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.8, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    
    plt.title('Sim2Sim Domain Randomization: Policy Robustness', fontsize=14, fontweight='bold')
    plt.xlabel('Physical Noise Level (Mass & Friction)', fontsize=12)
    plt.ylabel('Average Survival Steps', fontsize=12)
    plt.xticks(x_pos, x_labels)
    plt.ylim(0, max(means) + max(stds) + 10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上写具体的步数
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] 鲁棒性验证图表已保存至: {save_path}")

if __name__ == "__main__":
    model_file = os.path.join(PROJECT_ROOT, "models", "g1_ppo_500k.zip")
    
    # 噪声档位：无噪声, 10% 变化, 20% 变化, 30% 变化
    noise_tiers = [0.0, 0.1, 0.2, 0.3]
    output_img = os.path.join(PROJECT_ROOT, "docs", "figures", "sim2sim_robustness.png")
    
    # 每次测试跑 10 个 episode
    eval_results = run_dr_experiment(model_file, noise_tiers, episodes_per_level=10)
    plot_results(eval_results, output_img)
