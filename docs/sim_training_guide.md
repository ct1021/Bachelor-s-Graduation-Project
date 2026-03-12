# G1 MuJoCo 仿真训练完整执行指南

> **目标**：Linux 工作站跑通 G1 仿真 → 产出 Reward 曲线 + 步态视频
> **截止**：2026年3月15日前完成 Step 1-3

---

## 总体技术路线

```
Phase A: Sim (当前)         Phase B: Sim2Sim          Phase C: Sim2Real
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MuJoCo G1       │    │ MuJoCo ↔ PyBullet│    │ MuJoCo → 实机   │
│ 真实开源动作捕获 │ →  │ 跨引擎策略验证   │ →  │ 域随机化+部署   │
│ PPO + IL 训练   │    │ 进一步泛化与调优 │    │ CycloneDDS 通信 │
│ TensorBoard     │    │ 环境反馈引入     │    │ 实机调优        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 为什么使用人体步态数据?

1. **自然度**：人体步态经亿万年进化, 是能量效率与稳定性的最优解
2. **安全性**：约束 RL 探索空间, 避免初期产生极端关节动作
3. **ZMP 保证**：正常行走天然满足 CoP 稳定性条件
4. **冷启动**：从人体步态初始化, 样本效率提升 10x+
5. **数据路线**：直接加载 `dance1_subject2.csv` MoCap 数据

---

## 环境准备 (一次性)

```bash
ssh ct@100.117.36.59
conda activate biped_rl
cd ~/Bachelor-s-Graduation-Project
pip install imageio imageio-ffmpeg tqdm
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import mujoco; print('MuJoCo:', mujoco.__version__)"
```

## Step 1: 渲染开源轨迹 Demo 视频 (~2 min)

> 此步骤将自动加载已内置的 29自由度 G1 模型与开源参考轨迹（`robot_sdk/unitree/...`）。

```bash
python scripts/render_g1_demo.py --duration 5
# 输出: docs/figures/g1_open_source_demo.mp4
```

> 如 EGL 报错: `export MUJOCO_GL=osmesa` 后重试

## Step 2: PPO 训练

```bash
# 冒烟测试
python scripts/train_g1_ppo.py --timesteps 10000

# 正式训练 (~1-2h)
nohup python scripts/train_g1_ppo.py --timesteps 500000 --render > train.log 2>&1 &
tensorboard --logdir=logs/g1_ppo/ --host 0.0.0.0 --port 6006
```

## Step 3: Sim2Sim (3月下旬)

MuJoCo 策略 → PyBullet 交叉验证: 速度跟踪误差 < 10%

## Step 4: Sim2Real (4-5月)

域随机化 → CycloneDDS + unitree_sdk2 → 实机部署

---

## 故障排查

| 问题 | 解决 |
|------|------|
| `EGL not available` | `export MUJOCO_GL=osmesa` 或 `apt install libegl1-mesa-dev` |
| `CUDA OOM` | 减小 `batch_size` |
| `ModuleNotFoundError: envs` | `export PYTHONPATH=.` |
| `imageio-ffmpeg missing` | `pip install imageio-ffmpeg` |
