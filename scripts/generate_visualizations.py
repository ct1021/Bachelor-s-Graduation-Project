# -*- coding: utf-8 -*-
"""
generate_visualizations.py -- Generate result charts for monthly report

Produces:
  1. Model architecture diagram (text-based)
  2. Hybrid loss training curve (BC + KL convergence)
  3. Physics reward components comparison
  4. URDF validation summary chart
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set Chinese font support or fallback
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig1_architecture_diagram():
    """Generate GaitPriorNetwork architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("GaitPriorNetwork Architecture (12,236 params)", fontsize=16, fontweight="bold", pad=20)

    # Layer boxes
    layers = [
        (1.5, 3, 1.8, 1.0, "Input\nobs_dim=24", "#4A90D9"),
        (3.8, 3, 1.8, 1.0, "Hidden 1\n128 + ReLU", "#5BA85B"),
        (6.1, 3, 1.8, 1.0, "Hidden 2\n64 + ReLU", "#5BA85B"),
        (8.5, 4.2, 1.4, 0.8, "Mean Head\n\u03bc (dim=6)", "#E8913A"),
        (8.5, 2.0, 1.4, 0.8, "Log-Std Head\nlog\u03c3 (dim=6)", "#D94A4A"),
    ]

    for x, y, w, h, text, color in layers:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white",
                              linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

    # Arrows
    arrow_style = dict(arrowstyle="-|>", color="#333333", lw=2)
    ax.annotate("", xy=(2.9, 3.5), xytext=(2.4, 3.5), arrowprops=arrow_style)
    ax.annotate("", xy=(5.2, 3.5), xytext=(4.7, 3.5), arrowprops=arrow_style)
    ax.annotate("", xy=(7.8, 4.6), xytext=(7.0, 3.7), arrowprops=arrow_style)
    ax.annotate("", xy=(7.8, 2.4), xytext=(7.0, 3.3), arrowprops=arrow_style)

    # Output label
    ax.text(5, 0.8, "Output: Normal(\u03bc, exp(log\u03c3))  \u2192  KL Divergence Computation",
            ha="center", fontsize=12, style="italic",
            bbox=dict(boxstyle="round", facecolor="#F0F0F0", edgecolor="#CCCCCC"))

    path = os.path.join(OUTPUT_DIR, "fig1_architecture.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")
    return path


def fig2_training_curves():
    """Generate hybrid loss training curves using actual training."""
    from scripts.gait_prior_model import GaitPriorNetwork
    from scripts.hybrid_loss import pretrain_with_hybrid_loss

    OBS_DIM, ACT_DIM = 24, 6
    prior = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)
    online = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)

    # Generate synthetic expert data
    np.random.seed(42)
    torch.manual_seed(42)
    expert_obs = torch.randn(500, OBS_DIM)
    expert_act = torch.randn(500, ACT_DIM) * 0.3

    history = pretrain_with_hybrid_loss(
        online_model=online,
        prior_model=prior,
        expert_obs=expert_obs,
        expert_actions=expert_act,
        epochs=80,
        batch_size=64,
        lambda_kl=0.1,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["total"]) + 1)

    # Left: All losses
    ax1.plot(epochs, history["total"], "b-", linewidth=2.5, label="Total Loss", alpha=0.9)
    ax1.plot(epochs, history["bc"], "g--", linewidth=2, label="BC Loss (MSE)", alpha=0.8)
    ax1.plot(epochs, [k * 0.1 for k in history["kl"]], "r:", linewidth=2, label="\u03bb\u00b7KL Loss (\u03bb=0.1)", alpha=0.8)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Hybrid Loss Convergence (BC + KL)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))

    # Right: KL divergence alone
    ax2.plot(epochs, history["kl"], "r-", linewidth=2.5, alpha=0.9)
    ax2.fill_between(epochs, history["kl"], alpha=0.15, color="red")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("KL Divergence", fontsize=12)
    ax2.set_title("KL(\u03c0_online || \u03c0_prior) Convergence", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))

    path = os.path.join(OUTPUT_DIR, "fig2_training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")
    return path


def fig3_reward_components():
    """Generate physics reward components demo chart."""
    import gymnasium as gym
    from scripts.physics_reward import PhysicsRewardWrapper

    env = gym.make("CartPole-v1")
    wrapped = PhysicsRewardWrapper(env, target_velocity=0.5, robot_mass=10.0)

    vel_rewards, cot_rewards, cop_rewards, total_rewards = [], [], [], []

    obs, _ = wrapped.reset()
    for _ in range(200):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        vel_rewards.append(info.get("reward/velocity_tracking", 0))
        cot_rewards.append(info.get("reward/cot_penalty", 0))
        cop_rewards.append(info.get("reward/cop_constraint", 0))
        total_rewards.append(info.get("reward/total", 0))
        if terminated or truncated:
            obs, _ = wrapped.reset()
    wrapped.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    steps = range(len(vel_rewards))

    ax = axes[0, 0]
    ax.plot(steps, vel_rewards, color="#4A90D9", linewidth=1.5, alpha=0.8)
    ax.fill_between(steps, vel_rewards, alpha=0.2, color="#4A90D9")
    ax.set_title("Velocity Tracking Reward", fontsize=13, fontweight="bold")
    ax.set_ylabel("r_vel = exp(-\u03b1\u00b7(v-v*)^2)", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, cot_rewards, color="#E8913A", linewidth=1.5, alpha=0.8)
    ax.fill_between(steps, cot_rewards, alpha=0.2, color="#E8913A")
    ax.set_title("COT Energy Penalty", fontsize=13, fontweight="bold")
    ax.set_ylabel("r_cot = -\u03b2\u00b7\u03a3|\u03c4\u00b7q'|/(mg|v|)", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, cop_rewards, color="#D94A4A", linewidth=1.5, alpha=0.8)
    ax.fill_between(steps, cop_rewards, alpha=0.2, color="#D94A4A")
    ax.set_title("CoP Stability Constraint", fontsize=13, fontweight="bold")
    ax.set_ylabel("r_cop = -\u03b3\u00b7||CoP-center||^2", fontsize=10)
    ax.set_xlabel("Step", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, total_rewards, color="#5BA85B", linewidth=2, alpha=0.9)
    ax.fill_between(steps, total_rewards, alpha=0.2, color="#5BA85B")
    ax.set_title("Total Physics-Aware Reward", fontsize=13, fontweight="bold")
    ax.set_ylabel("r_total (weighted sum)", fontsize=10)
    ax.set_xlabel("Step", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("PhysicsRewardWrapper - Reward Components (200 steps)", fontsize=15, fontweight="bold", y=1.01)
    path = os.path.join(OUTPUT_DIR, "fig3_reward_components.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")
    return path


def fig4_urdf_report():
    """Generate URDF validation visual summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Joint config bar chart
    ax = axes[0]
    joints = ["L-Hip", "L-Knee", "L-Ankle", "R-Hip", "R-Knee", "R-Ankle"]
    lower = [-1.57, 0.0, -0.78, -1.57, 0.0, -0.78]
    upper = [1.57, 2.60, 0.78, 1.57, 2.60, 0.78]
    colors = ["#4A90D9", "#4A90D9", "#4A90D9", "#E8913A", "#E8913A", "#E8913A"]

    y_pos = np.arange(len(joints))
    ranges = [u - l for l, u in zip(lower, upper)]

    bars = ax.barh(y_pos, ranges, left=lower, color=colors, height=0.6, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(joints, fontsize=11)
    ax.set_xlabel("Joint Angle (rad)", fontsize=11)
    ax.set_title("Joint Configuration (6 DoF)", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    left_patch = mpatches.Patch(color="#4A90D9", label="Left Leg")
    right_patch = mpatches.Patch(color="#E8913A", label="Right Leg")
    ax.legend(handles=[left_patch, right_patch], loc="lower right", fontsize=10)

    # Right: Mass distribution pie chart
    ax = axes[1]
    parts = ["Torso\n5.0 kg", "L-Upper\n1.5 kg", "L-Lower\n1.0 kg", "L-Foot\n0.5 kg",
             "R-Upper\n1.5 kg", "R-Lower\n1.0 kg", "R-Foot\n0.5 kg"]
    masses = [5.0, 1.5, 1.0, 0.5, 1.5, 1.0, 0.5]
    pie_colors = ["#4A90D9", "#5BA85B", "#7CC87C", "#A8DCA8",
                  "#E8913A", "#F0AA5E", "#F5C589"]
    explode = [0.05] * 7

    wedges, texts, autotexts = ax.pie(masses, labels=parts, autopct="%1.0f%%",
                                       colors=pie_colors, explode=explode,
                                       textprops={"fontsize": 9},
                                       startangle=90)
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(9)
    ax.set_title("Mass Distribution (Total: 11.0 kg)", fontsize=14, fontweight="bold")

    path = os.path.join(OUTPUT_DIR, "fig4_urdf_report.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")
    return path


def fig5_pipeline_overview():
    """Generate project pipeline overview diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Technical Pipeline: Imitation Learning + RL Fine-tuning", fontsize=15, fontweight="bold", pad=15)

    # Pipeline stages
    stages = [
        (1.5, 2.5, "MoCap Data\n(CMU / Lab)", "#9B59B6", "Phase P1"),
        (4.0, 2.5, "IK Retargeting\n(Human\u2192Robot)", "#3498DB", "Phase P1"),
        (6.5, 2.5, "BC Pre-train\n(Warm-start)", "#2ECC71", "Phase P2"),
        (9.0, 2.5, "PPO Fine-tune\n(+ KL Loss)", "#E67E22", "Phase 3"),
        (11.5, 2.5, "Sim2Real\n(Domain Rand.)", "#E74C3C", "Phase 4"),
    ]

    for x, y, text, color, phase in stages:
        box = FancyBboxPatch((x - 1.0, y - 0.6), 2.0, 1.2,
                              boxstyle="round,pad=0.12",
                              facecolor=color, edgecolor="white",
                              linewidth=2, alpha=0.88)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(x, y - 1.0, phase, ha="center", fontsize=9, color=color, fontweight="bold")

    # Status indicators
    status_box = FancyBboxPatch((0.3, 3.8), 2.2, 0.6, boxstyle="round,pad=0.08",
                                 facecolor="#2ECC71", alpha=0.15, edgecolor="#2ECC71", linewidth=1.5)
    ax.add_patch(status_box)
    ax.text(1.4, 4.1, "DONE: Offline Prior + KL Loss + Physics Reward", fontsize=9, color="#2ECC71", fontweight="bold")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=2.5)
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 1.0
        x2 = stages[i+1][0] - 1.0
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5), arrowprops=arrow_kw)

    path = os.path.join(OUTPUT_DIR, "fig5_pipeline.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] {path}")
    return path


if __name__ == "__main__":
    print("=" * 50)
    print("Generating Visualizations for Monthly Report")
    print("=" * 50)

    fig1_architecture_diagram()
    fig2_training_curves()
    fig3_reward_components()
    fig4_urdf_report()
    fig5_pipeline_overview()

    print("\n" + "=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 50)
