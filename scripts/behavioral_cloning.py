"""
behavioral_cloning.py — G1 行为克隆 (IL) 预训练
=================================================
Phase 2: 利用步态参考数据进行监督学习预训练，
生成与 SB3 PPO MlpPolicy Actor 网络兼容的权重文件。

核心思路:
  1. 从 cmu_walking_reference.npz 构造 (观测, 动作) 监督学习对
  2. 观测格式与 g1_env.py 的 _get_obs() 一致 (47维)
  3. 动作 = 下一帧关节角偏移 / ACTION_SCALE，归一化到 [-1, 1]
  4. 训练 Actor 网络 [512, 256, 128] → 12维动作输出
  5. 保存权重，可被 train_g1_ppo.py --bc-pretrain 加载

用法:
    python scripts/behavioral_cloning.py                    # 默认 100 epochs
    python scripts/behavioral_cloning.py --epochs 5         # 快速冒烟测试
    python scripts/behavioral_cloning.py --visualize        # 训练后可视化预测
"""

import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("❌ PyTorch 未安装。请执行: pip install torch")
    sys.exit(1)


# ============================================================
# 常量 — 与 g1_env.py 保持一致
# ============================================================
NUM_LEG_JOINTS = 12
ACTION_SCALE = 0.5       # g1_env.py 中 action * ACTION_SCALE = 角度偏移
OBS_DIM = 47             # g1_env.py 的观测空间维度
CONTROL_FREQ = 100       # Hz
DEFAULT_STANDING_HEIGHT = 0.793   # G1 站立高度


# ============================================================
# 数据集
# ============================================================
class GaitDataset(Dataset):
    """
    从步态参考轨迹构造 (observation, action) 监督学习对。

    观测格式 (47维, 与 g1_env._get_obs() 对齐):
      [0]     root_z          = 站立高度 (0.793)
      [1:4]   root_vel        = (stride/cycle, 0, 0)  前进速度
      [4:6]   roll, pitch     = (0, 0)  直立
      [6:9]   angular_vel     = (0, 0, 0)
      [9:21]  leg_joint_pos   = ref_joint_pos[t]  (12)
      [21:33] leg_joint_vel   = (ref[t+1]-ref[t]) * CONTROL_FREQ  (12)
      [33:35] phase_sin/cos   = sin/cos(2π * phase)  (2)
      [35:47] ref_target      = ref_joint_pos[t]  (12)

    动作 (12维):
      action = (ref[t+1] - ref[t]) / ACTION_SCALE
      (即下一帧的关节角偏移，归一化到 [-1, 1])
    """

    def __init__(self, ref_path):
        data = np.load(ref_path)
        ref_pos = data["joint_positions"]   # (T, 12)
        base_pos = data["base_positions"]   # (T, 3)
        cycle_time = float(data.get("cycle_time", 1.0))

        T = ref_pos.shape[0]
        # 只用 T-1 帧（需要 t+1 构造动作标签）
        N = T - 1

        observations = np.zeros((N, OBS_DIM), dtype=np.float32)
        actions = np.zeros((N, NUM_LEG_JOINTS), dtype=np.float32)

        dt = 1.0 / CONTROL_FREQ
        forward_speed = 0.65 / cycle_time  # stride_length / cycle_time

        for t in range(N):
            phase = (t % int(cycle_time * CONTROL_FREQ)) / (cycle_time * CONTROL_FREQ)

            # 构造观测
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            obs[0] = DEFAULT_STANDING_HEIGHT                    # root_z
            obs[1] = forward_speed                              # vx
            obs[2] = 0.0                                        # vy
            obs[3] = 0.0                                        # vz
            obs[4] = 0.0                                        # roll
            obs[5] = 0.0                                        # pitch
            obs[6:9] = 0.0                                      # wx, wy, wz
            obs[9:21] = ref_pos[t]                              # leg_joint_pos
            obs[21:33] = (ref_pos[t + 1] - ref_pos[t]) * CONTROL_FREQ  # leg_joint_vel
            obs[33] = np.sin(2 * np.pi * phase)                 # phase_sin
            obs[34] = np.cos(2 * np.pi * phase)                 # phase_cos
            obs[35:47] = ref_pos[t]                             # ref_target

            observations[t] = obs

            # 构造动作标签: 下一帧角度偏移 / ACTION_SCALE → [-1, 1]
            delta = ref_pos[t + 1] - ref_pos[t]
            actions[t] = np.clip(delta / ACTION_SCALE, -1.0, 1.0)

        self.observations = torch.from_numpy(observations)
        self.actions = torch.from_numpy(actions)

        print(f"[GaitDataset] ✅ 构造完成: {N} 样本, obs={OBS_DIM}维, act={NUM_LEG_JOINTS}维")
        print(f"  动作范围: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"  动作均值: {actions.mean():.6f}, 标准差: {actions.std():.6f}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# ============================================================
# Actor 网络 — 与 SB3 MlpPolicy 的 pi 网络完全对齐
# ============================================================
class BCActorNetwork(nn.Module):
    """
    行为克隆 Actor 网络。
    架构: obs(47) → [512, 256, 128] → action(12)

    与 SB3 PPO MlpPolicy 对齐:
      - mlp_extractor.policy_net: Linear(47→512) + Linear(512→256) + Linear(256→128)
      - action_net: Linear(128→12)
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=NUM_LEG_JOINTS,
                 hidden_sizes=(512, 256, 128)):
        super().__init__()

        # 构建 MLP 层（对应 SB3 的 mlp_extractor.policy_net）
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        self.mlp = nn.Sequential(*layers)

        # 动作输出层（对应 SB3 的 action_net）
        self.action_head = nn.Linear(prev_size, action_dim)

    def forward(self, obs):
        features = self.mlp(obs)
        return self.action_head(features)


# ============================================================
# 训练
# ============================================================
def train_bc(args):
    ref_path = os.path.join(PROJECT_ROOT, "data", "cmu_walking_reference.npz")
    if not os.path.exists(ref_path):
        print(f"❌ 步态参考数据未找到: {ref_path}")
        print("   请先运行: python scripts/retarget_cmu_to_g1.py")
        sys.exit(1)

    # 数据集
    dataset = GaitDataset(ref_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"\n[训练集] {train_size} 样本, [验证集] {val_size} 样本")
    print(f"[Batch] {args.batch_size}, [Epochs] {args.epochs}, [LR] {args.lr}")

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCActorNetwork()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] BCActorNetwork [512, 256, 128] → {NUM_LEG_JOINTS}")
    print(f"  参数量: {total_params:,}")
    print(f"  设备: {device}")

    # 优化器 + 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # 训练循环
    best_val_loss = float("inf")
    output_path = os.path.join(PROJECT_ROOT, "models", "bc_pretrained.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n" + "=" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Status':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # --- 训练 ---
        model.train()
        train_loss_sum = 0.0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = model(obs_batch)
            loss = criterion(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * obs_batch.size(0)

        train_loss = train_loss_sum / train_size

        # --- 验证 ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                pred = model(obs_batch)
                loss = criterion(pred, act_batch)
                val_loss_sum += loss.item() * obs_batch.size(0)
        val_loss = val_loss_sum / val_size

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 保存最佳模型
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            status = "✅ best"

        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs or status:
            print(f"{epoch:6d} | {train_loss:12.8f} | {val_loss:12.8f} | {current_lr:10.6f} | {status}")

    print("=" * 60)
    print(f"\n✅ 训练完成！最佳验证损失: {best_val_loss:.8f}")
    print(f"   权重保存到: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")

    # 打印权重 key（方便后续与 SB3 对接验证）
    state_dict = torch.load(output_path, map_location="cpu", weights_only=True)
    print(f"\n   权重 keys:")
    for k, v in state_dict.items():
        print(f"     {k}: {list(v.shape)}")

    return output_path


# ============================================================
# 可视化预测 vs 真实
# ============================================================
def visualize_predictions(model_path):
    """对比 BC 模型预测动作 vs 真实参考动作"""
    ref_path = os.path.join(PROJECT_ROOT, "data", "cmu_walking_reference.npz")
    dataset = GaitDataset(ref_path)

    model = BCActorNetwork()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        pred_actions = model(dataset.observations).numpy()
    true_actions = dataset.actions.numpy()

    # 逐关节 MSE
    joint_names = [
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee",
        "L_ankle_pitch", "L_ankle_roll",
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee",
        "R_ankle_pitch", "R_ankle_roll",
    ]

    print("\n" + "=" * 50)
    print("  逐关节预测误差 (MSE)")
    print("=" * 50)
    for j, name in enumerate(joint_names):
        mse = np.mean((pred_actions[:, j] - true_actions[:, j]) ** 2)
        corr = np.corrcoef(pred_actions[:, j], true_actions[:, j])[0, 1]
        print(f"  {name:20s}: MSE={mse:.8f}, r={corr:.4f}")

    total_mse = np.mean((pred_actions - true_actions) ** 2)
    print(f"\n  总体 MSE: {total_mse:.8f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        fig.suptitle("BC Predictions vs Ground Truth (200 frames)", fontsize=14)
        T_show = min(200, len(true_actions))

        for j in range(12):
            ax = axes[j // 4, j % 4]
            ax.plot(true_actions[:T_show, j], "b-", alpha=0.7, label="GT")
            ax.plot(pred_actions[:T_show, j], "r--", alpha=0.7, label="Pred")
            ax.set_title(joint_names[j], fontsize=9)
            ax.set_ylim(-1, 1)
            if j == 0:
                ax.legend(fontsize=7)

        plt.tight_layout()
        fig_path = os.path.join(PROJECT_ROOT, "docs", "figures", "bc_predictions.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=150)
        print(f"\n  📊 对比图保存到: {fig_path}")
    except ImportError:
        print("\n  ⚠️ matplotlib 未安装，跳过可视化图表生成")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="G1 行为克隆 (IL) 预训练")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练 epoch 数 (default: 100)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch 大小 (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率 (default: 1e-3)")
    parser.add_argument("--visualize", action="store_true",
                        help="训练后生成预测对比图")
    args = parser.parse_args()

    model_path = train_bc(args)

    if args.visualize:
        visualize_predictions(model_path)

    print("\n后续步骤:")
    print("  1. 查看权重: python -c \"import torch; print(torch.load('models/bc_pretrained.pt').keys())\"")
    print("  2. PPO 微调: python scripts/train_g1_ppo.py --bc-pretrain models/bc_pretrained.pt")


if __name__ == "__main__":
    main()
