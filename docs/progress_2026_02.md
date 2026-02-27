# 2月项目进展文档

> 更新日期: 2026-02-27

## 一、本月核心工作：落实导师"离线小模型 + KL 散度"建议

根据俞老师反馈 —— *"训练一个离线的小模型，再结合运行过程实时数据进行更新，可以借助于元学习模型和步态概率特性的 KL 散度作为损失机制"* —— 完成了以下 4 个核心模块的开发与验证。

---

### 1. 离线步态先验网络 (`scripts/gait_prior_model.py`)

**目的**：作为"离线小模型"，用专家步态数据预训练，提供步态先验分布。

- **架构**：3 层 MLP → `obs_dim → 128 → 64 → action_dim × 2`
- **输出**：高斯动作分布的均值(μ)和标准差(σ)
- **核心方法**：`get_distribution(obs)` 返回 `torch.distributions.Normal`
- 支持 `save()` / `load()` 便捷方法

**技术亮点**：
- Xavier 均匀初始化提升训练稳定性
- log_std 裁剪 `[-5, 2]` 防止数值不稳定

### 2. 混合损失函数 (`scripts/hybrid_loss.py`)

**目的**：实现 `Loss = BC_Loss + λ × KL_Divergence`，约束在线策略不偏离步态先验。

- **BC Loss**：MSE 回归损失 `||a_pred - a_expert||²`
- **KL Divergence**：`torch.distributions.kl_divergence(π_online, π_prior)` 闭式计算
- **HybridLoss 类**：可配置 λ 权重（默认 0.1）
- **完整训练流程**：`pretrain_with_hybrid_loss()` 函数提供端到端示例

**技术亮点**：
- 先验模型参数自动冻结，不参与梯度更新
- 支持 DataLoader 批量训练，每 10 个 epoch 输出损失分量

### 3. 物理感知奖励函数 (`scripts/physics_reward.py`)

**目的**：SB3 兼容的自定义奖励包装器，引入物理先验约束。

| 奖励分量 | 公式 | 作用 |
|---------|------|-----|
| 速度跟踪 | `r = exp(-α·(v-v*)²)` | 鼓励 0.2-1.0 m/s 行走速度 |
| 能耗惩罚 (COT) | `r = -β·Σ|τ·q̇| / (mgv)` | 惩罚高能耗动作 |
| CoP 软约束 | `r = -γ·||CoP-center||²` | ZMP 稳定性保障 |

- 通过 `gymnasium.Wrapper` 包裹环境，可直接与 `PPO("MlpPolicy", env)` 配合
- 所有奖励分量自动注入 `info` 字典，支持 TensorBoard 监控

### 4. 模型自检脚本 (`scripts/test_load_model.py`)

**验证结果**：4 项测试全部通过 ✅

| 测试项 | 状态 |
|-------|------|
| GaitPriorNetwork 前向传播 | PASS |
| HybridLoss 混合损失计算 | PASS |
| PhysicsRewardWrapper 环境包装 | PASS |
| SB3 已保存模型加载 | PASS |

---

## 二、项目文件结构（更新后）

```
gradually-work/
├── data/                      # 预留 MoCap 步态数据
├── docs/
│   └── environment_setup.md   # 环境搭建记录
├── envs/                      # 双足 URDF 模型
├── logs/                      # TensorBoard 日志
├── models/
│   └── ppo_cartpole_balance.zip
├── scripts/
│   ├── __init__.py            # [新增] 包初始化
│   ├── gait_prior_model.py    # [新增] 离线步态先验网络
│   ├── hybrid_loss.py         # [新增] BC + KL 混合损失
│   ├── physics_reward.py      # [新增] 物理感知奖励函数
│   └── test_load_model.py     # [新增] 全模块自检
├── train_balance.py           # CartPole PPO 基线
└── requirements.txt
```

---

## 三、下一步计划

1. **双足 URDF 导入**：将自研机器人模型导入 `envs/`，进行关节自由度校准
2. **MoCap 数据处理**：编写 IK 重定向脚本，生成专家步态数据
3. **预训练先验模型**：用专家数据通过 `pretrain_with_hybrid_loss()` 训练 GaitPriorNetwork
4. **PPO + 物理奖励训练**：将 PhysicsRewardWrapper 应用到双足环境，执行 RL 微调
5. **Sim2Real 域随机化**：在训练中引入环境参数随机化，提升策略鲁棒性
