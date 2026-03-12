"""
train_g1_ppo.py — PPO + 模仿学习训练脚本
==========================================
基于 Stable-Baselines3 的 PPO 训练骨架。
奖励函数内置 Tracking Reward 项，体现 "模仿学习 + 强化学习" 融合特色。

用法:
    # 快速冒烟测试 (1 万步)
    python scripts/train_g1_ppo.py --timesteps 10000

    # 正式训练 (50 万步, ~1-2h on RTX 2070)
    python scripts/train_g1_ppo.py --timesteps 500000

    # 训练后渲染视频
    python scripts/train_g1_ppo.py --timesteps 500000 --render

技术路线:
    Sim (MuJoCo G1) → Sim2Sim (PyBullet 验证) → Sim2Real (实机部署)
"""

import argparse
import os
import sys
import time
import numpy as np

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")


def make_env():
    """创建 G1 训练环境"""
    from envs.g1_env import G1WalkEnv
    env = G1WalkEnv(render_mode=None)
    return env


def train(args):
    """PPO 训练主流程"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, BaseCallback
    )
    
    print("=" * 60)
    print("  G1 PPO + Imitation Learning Training")
    print("=" * 60)
    
    # ---- 目录准备 ----
    log_dir = os.path.join(PROJECT_ROOT, "logs", "g1_ppo")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # ---- 环境 ----
    print("\n[1/4] Creating environment...")
    env = DummyVecEnv([make_env])
    env = VecMonitor(env, log_dir)
    
    eval_env = DummyVecEnv([make_env])
    
    print(f"  Obs space:  {env.observation_space.shape}")
    print(f"  Act space:  {env.action_space.shape}")
    
    # ---- PPO 模型 ----
    print("\n[2/4] Initializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {total_params:,}")
    print(f"  Device: {model.device}")
    
    # ---- Callbacks ----
    callbacks = []
    
    # 定期保存 checkpoint
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000, args.timesteps // 10),
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="g1_ppo",
    )
    callbacks.append(checkpoint_cb)
    
    # 定期评估
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=log_dir,
        eval_freq=max(5000, args.timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
    )
    callbacks.append(eval_cb)
    
    # ---- 训练 ----
    print(f"\n[3/4] Training for {args.timesteps:,} timesteps...")
    print(f"  TensorBoard: tensorboard --logdir={log_dir}")
    print(f"  Estimated time: ~{args.timesteps / 250000:.1f} hours on RTX 2070")
    print("-" * 60)
    
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    
    # ---- 保存 ----
    model_path = os.path.join(model_dir, f"g1_ppo_{args.timesteps // 1000}k")
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print(f"[4/4] Training complete!")
    print(f"  Duration: {elapsed / 60:.1f} min")
    print(f"  Model saved: {model_path}.zip")
    print(f"  TensorBoard logs: {log_dir}")
    print("=" * 60)
    
    env.close()
    eval_env.close()
    
    return model_path


def render_trained_policy(model_path, output_path=None):
    """用训练好的策略渲染仿真视频"""
    from stable_baselines3 import PPO
    
    try:
        import imageio
    except ImportError:
        print("[WARN] imageio not installed. Skipping video render.")
        return
    
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "docs", "figures", "g1_trained.mp4")
    
    print(f"\n[Render] Loading model: {model_path}")
    model = PPO.load(model_path)
    
    from envs.g1_env import G1WalkEnv
    env = G1WalkEnv(render_mode="rgb_array")
    
    frames = []
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if terminated or truncated:
            break
    
    if frames:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = imageio.get_writer(output_path, fps=30, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[Render] Saved: {output_path} ({len(frames)} frames)")
        print(f"[Render] Total reward: {total_reward:.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="G1 PPO + IL Training")
    parser.add_argument("--timesteps", "-t", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--render", action="store_true",
                       help="Render trained policy after training")
    parser.add_argument("--render-only", type=str, default=None,
                       help="Skip training, render from existing model path")
    args = parser.parse_args()
    
    if args.render_only:
        render_trained_policy(args.render_only)
    else:
        model_path = train(args)
        if args.render:
            render_trained_policy(model_path)
    
    print("\nNext steps:")
    print("  1. View TensorBoard: tensorboard --logdir=logs/g1_ppo/")
    print("  2. Render video: python scripts/train_g1_ppo.py --render-only models/g1_ppo_500k")
    print("  3. Sim2Sim: Verify policy in PyBullet environment")


if __name__ == "__main__":
    main()
