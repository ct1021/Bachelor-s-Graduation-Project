"""
train_g1_ppo.py — V7 GPU 加速 PPO 训练脚本 (支持 BC 预训练权重注入)
===================================================================
核心改进 (V7, 在 V6 基础上):
1. SubprocVecEnv 多进程并行仿真（8x 数据采集加速）
2. 扩大神经网络容量（[512, 256, 128]，独立 Actor/Critic）
3. PPO 超参数适配并行训练（更大 batch, 更多 n_steps）
4. GPU 策略网络推理 + 多核 CPU 物理仿真
5. **[NEW] BC 预训练权重注入**: --bc-pretrain 参数加载行为克隆权重到 Actor

用法:
    # 快速冒烟测试 (100K 步, 单环境)
    python scripts/train_g1_ppo.py --timesteps 100000 --num-envs 1

    # BC→RL 两阶段训练（推荐）
    python scripts/train_g1_ppo.py --bc-pretrain models/bc_pretrained.pt --timesteps 30000000 --num-envs 8

    # GPU 加速训练 (30M 步, 8 并行环境, ~1.5h on RTX 2070)
    python scripts/train_g1_ppo.py --timesteps 30000000 --num-envs 8

    # 训练后渲染视频
    python scripts/train_g1_ppo.py --render-only models/g1_ppo_30000k
"""

import argparse
import multiprocessing
import os
import sys
import time
import numpy as np
import torch

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    # 解决 SubprocVecEnv 在 Linux 上的 ConnectionResetError:
    # 新版 Python (≥3.12) 默认 start_method='spawn', MuJoCo 环境需要 'fork' 或 'forkserver'
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass  # 已经设置过


def make_env(rank=0, seed=0):
    """创建 G1 训练环境（支持多进程）"""
    def _init():
        from envs.g1_env import G1WalkEnv
        env = G1WalkEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init


def load_bc_pretrain(model, bc_path):
    """
    将 BC 预训练权重注入 PPO 的 Actor 网络。

    权重映射 (BC → SB3 PPO MlpPolicy):
      mlp.0.weight/bias  →  mlp_extractor.policy_net.0.weight/bias  (47→512)
      mlp.2.weight/bias  →  mlp_extractor.policy_net.2.weight/bias  (512→256)
      mlp.4.weight/bias  →  mlp_extractor.policy_net.4.weight/bias  (256→128)
      action_head.weight/bias  →  action_net.weight/bias            (128→12)

    注意: Critic 网络保持随机初始化，仅注入 Actor 权重。
    """
    print(f"\n[BC→RL] 加载 BC 预训练权重: {bc_path}")
    bc_state = torch.load(bc_path, map_location="cpu", weights_only=True)

    # 权重键名映射
    KEY_MAP = {
        "mlp.0.weight": "mlp_extractor.policy_net.0.weight",
        "mlp.0.bias":   "mlp_extractor.policy_net.0.bias",
        "mlp.2.weight": "mlp_extractor.policy_net.2.weight",
        "mlp.2.bias":   "mlp_extractor.policy_net.2.bias",
        "mlp.4.weight": "mlp_extractor.policy_net.4.weight",
        "mlp.4.bias":   "mlp_extractor.policy_net.4.bias",
        "action_head.weight": "action_net.weight",
        "action_head.bias":   "action_net.bias",
    }

    policy_state = model.policy.state_dict()
    injected = 0
    for bc_key, sb3_key in KEY_MAP.items():
        if bc_key in bc_state and sb3_key in policy_state:
            if bc_state[bc_key].shape == policy_state[sb3_key].shape:
                policy_state[sb3_key] = bc_state[bc_key]
                injected += 1
                print(f"  ✅ {bc_key} → {sb3_key} {list(bc_state[bc_key].shape)}")
            else:
                print(f"  ⚠️ 形状不匹配: {bc_key} {list(bc_state[bc_key].shape)} vs {sb3_key} {list(policy_state[sb3_key].shape)}")
        else:
            print(f"  ❌ 缺失: bc={bc_key in bc_state}, sb3={sb3_key in policy_state}")

    model.policy.load_state_dict(policy_state)
    print(f"  注入完成: {injected}/8 个权重张量")
    print(f"  Critic 网络保持随机初始化 ✅")
    return model


def train(args):
    """PPO 训练主流程（V7 GPU 加速版，支持 BC 预训练）"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback,
    )
    
    num_envs = args.num_envs
    
    print("=" * 60)
    print("  G1 PPO V6 — GPU Accelerated Training")
    print("=" * 60)
    
    # ---- 目录准备 ----
    log_dir = os.path.join(PROJECT_ROOT, "logs", "g1_ppo")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # ---- 并行环境 ----
    print(f"\n[1/4] Creating {num_envs} parallel environments...")
    if num_envs > 1:
        try:
            env = SubprocVecEnv([make_env(rank=i, seed=42) for i in range(num_envs)])
            print("  SubprocVecEnv created successfully")
        except (ConnectionResetError, EOFError, BrokenPipeError) as e:
            print(f"  ⚠️ SubprocVecEnv failed: {e}")
            print(f"  Falling back to DummyVecEnv with {num_envs} envs...")
            env = DummyVecEnv([make_env(rank=i, seed=42) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(rank=0, seed=42)])
    env = VecMonitor(env, log_dir)
    
    eval_env = DummyVecEnv([make_env(rank=99, seed=123)])
    
    print(f"  Env type:   {'SubprocVecEnv' if num_envs > 1 else 'DummyVecEnv'}")
    print(f"  Num envs:   {num_envs}")
    print(f"  Obs space:  {env.observation_space.shape}")
    print(f"  Act space:  {env.action_space.shape}")
    
    # ---- PPO 模型（V6 扩容版） ----
    print("\n[2/4] Initializing PPO model (V6 expanded)...")
    
    # 根据并行环境数调整超参数
    n_steps_per_env = 4096 if num_envs >= 4 else 2048
    batch_size = 256 if num_envs >= 4 else 64
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,              # 稍高的学习率，配合更大网络和并行数据
        n_steps=n_steps_per_env,         # 每个环境采集 4096 步再更新
        batch_size=batch_size,           # 更大的 mini-batch
        n_epochs=5,                      # 减少重复使用次数，防止过拟合旧数据
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,                   # 重新开启熵奖励，鼓励探索（并行环境更安全）
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,                  # 稍放宽 KL 约束（并行数据更稳定）
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",                   # 自动选择 GPU
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 256, 128],      # Actor: 更宽更深的3层网络
                vf=[512, 256, 128],      # Critic: 独立的3层网络
            ),
            log_std_init=-0.5,           # 初始 std ≈ 0.61，适度探索
        ),
    )

    # ---- BC 预训练权重注入 ----
    if args.bc_pretrain:
        if os.path.exists(args.bc_pretrain):
            model = load_bc_pretrain(model, args.bc_pretrain)
        else:
            print(f"\n⚠️ BC 权重文件未找到: {args.bc_pretrain}，跳过预训练注入")
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Network:    Actor [512, 256, 128] + Critic [512, 256, 128]")
    print(f"  Total params: {total_params:,}")
    print(f"  Device:     {model.device}")
    print(f"  n_steps:    {n_steps_per_env} × {num_envs} envs = {n_steps_per_env * num_envs:,} steps/update")
    print(f"  batch_size: {batch_size}")
    
    # ---- Callbacks ----
    callbacks = []
    
    # 定期保存 checkpoint（按实际总步数计算）
    checkpoint_freq = max(100000, args.timesteps // 10)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, checkpoint_freq // num_envs),
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="g1_ppo",
    )
    callbacks.append(checkpoint_cb)
    
    # 定期评估
    eval_freq = max(10000, args.timesteps // 20)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=log_dir,
        eval_freq=max(1, eval_freq // num_envs),
        n_eval_episodes=5,
        deterministic=True,
    )
    callbacks.append(eval_cb)
    
    # ---- 训练 ----
    effective_fps_est = 600 * num_envs  # 估算
    est_hours = args.timesteps / effective_fps_est / 3600
    print(f"\n[3/4] Training for {args.timesteps:,} timesteps...")
    print(f"  Estimated FPS:  ~{effective_fps_est:,} (×{num_envs} parallel)")
    print(f"  Estimated time: ~{est_hours:.1f} hours")
    print(f"  TensorBoard:    tensorboard --logdir={log_dir}")
    print("-" * 60)
    
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    actual_fps = args.timesteps / elapsed
    
    # ---- 保存 ----
    model_path = os.path.join(model_dir, f"g1_ppo_{args.timesteps // 1000}k")
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print(f"[4/4] Training complete!")
    print(f"  Duration:    {elapsed / 60:.1f} min")
    print(f"  Actual FPS:  {actual_fps:.0f}")
    print(f"  Speedup:     ~{actual_fps / 600:.1f}x vs single-env baseline")
    print(f"  Model saved: {model_path}.zip")
    print(f"  TensorBoard: {log_dir}")
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
    parser = argparse.ArgumentParser(description="G1 PPO V7 — GPU Accelerated Training (BC→RL)")
    parser.add_argument("--timesteps", "-t", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--num-envs", "-n", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--render", action="store_true",
                       help="Render trained policy after training")
    parser.add_argument("--render-only", type=str, default=None,
                       help="Skip training, render from existing model path")
    parser.add_argument("--bc-pretrain", type=str, default=None,
                       help="BC 预训练权重路径 (e.g. models/bc_pretrained.pt)")
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
    print("  3. Evaluate: python scripts/evaluate_100m_walk.py")


if __name__ == "__main__":
    main()
