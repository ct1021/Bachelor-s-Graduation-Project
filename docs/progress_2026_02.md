# 2026年2月份工作汇报

## （一）上月计划完成情况 (Review)

### 1. 双足模型物理适配
**部分完成。** 已搭建完整的 URDF 验证与静态重力测试工具链（`scripts/validate_urdf.py`），可自动完成关节 DoF 校准、质量统计和碰撞体检测。由于本月春节假期尚未赴深圳确认实验室具体硬件参数，改用简化 6-DoF 双足模型（参照宇树 G1 比例）作为开发占位模型进行验证，并在 PyBullet 中通过了重力落体测试。**待下月到深圳后导入宇树 G1 官方 URDF（23-DoF）进行完整适配。**

### 2. 步态数据 Retargeting 模块开发
**延期至 3 月。** 受限于春节假期和尚未确定最终 MoCap 数据来源（实验室自采 vs CMU 公开数据集），已完成技术调研和方案设计，待下月与博士师兄确认后推进 IK 脚本开发。

### 3. 模仿学习（IL）初步实现
**框架搭建完成。** 已提前实现了行为克隆（BC）算法的核心代码框架，包括混合损失函数（`hybrid_loss.py`）中的 `pretrain_with_hybrid_loss()` 完整训练流程。因专家数据尚未就绪，使用合成数据验证了训练管线的正确性。

---

## （二）本月进展 (Progress)

### 1. 落实导师"离线小模型 + KL 散度"指导建议

根据俞老师上月反馈 —— *"训练一个离线的小模型，再结合运行过程实时数据进行更新，可以借助于元学习模型和步态概率特性的 KL 散度作为损失机制"* —— 设计并实现了以下核心模块：

#### (1) 离线步态先验网络（`scripts/gait_prior_model.py`）
- 搭建了基于 PyTorch 的轻量级 MLP 策略网络，作为"离线小模型"步态先验
- 网络架构：obs_dim → 128 → 64 → action_dim×2，输出高斯动作分布的均值(μ)和标准差(σ)
- 模型参数量仅 **12,236** 个，满足"小模型"定位
- 提供 `get_distribution(obs)` 接口返回 `torch.distributions.Normal`，直接支撑 KL 散度计算

#### (2) 混合损失函数（`scripts/hybrid_loss.py`）
- 实现导师要求的损失机制：**Loss = BC_Loss + λ × KL_Divergence**
- BC_Loss：MSE 回归损失，衡量预测动作与专家动作的偏差
- KL_Divergence：通过 `torch.distributions.kl_divergence` **精确闭式计算**两个高斯分布之间的 KL 散度，约束实时策略分布不偏离先验分布
- λ 权重可配置（默认 0.1），支持在"保守跟随先验"与"自由探索"之间灵活调节
- 验证结果：BC Loss = 0.4607, KL Loss = 2.141, Total Loss = 0.6748，梯度回传正常

#### (3) 物理感知奖励函数（`scripts/physics_reward.py`）
- 设计了 Stable-Baselines3 兼容的 `PhysicsRewardWrapper` 自定义奖励包装器
- 三大物理约束奖励分量：

| 分量 | 公式 | 指标意义 |
|------|------|---------|
| 速度跟踪 | r = exp(-α·(v-v*)²) | 鼓励 0.2~1.0 m/s 目标行走速度 |
| 能耗惩罚 (COT) | r = -β·Σ\|τ·q̇\| / (mg\|v\|) | 成本传输比，惩罚高能耗动作 |
| CoP 软约束 | r = -γ·\|\|CoP-center\|\|² | 基于 ZMP 判据的稳定性保障 |

- 所有分量自动注入 info 字典，可通过 TensorBoard 实时监控各项物理指标

### 2. 双足模型物理环境搭建

#### (1) 简化双足 URDF 模型（`envs/simple_biped.urdf`）
- 参照宇树 G1 人形机器人比例，生成 6-DoF 简化双足模型
- 结构：躯干(5kg) + 左右腿各 3 关节（hip/knee/ankle），总质量 11kg

#### (2) URDF 静态验证工具（`scripts/validate_urdf.py`）
- 利用 PyBullet 自动完成：关节配置扫描、质量统计、自由落体重力测试、地面碰撞检测
- 验证报告结果摘要：

```
Active DOF: 6 (全部 revolute 类型)
Total Mass: 11.000 kg
Gravity Test: 从 1.0m 自由落体 → 最终 z=0.764m, 4 个地面接触点
```

### 3. 工程化规范提升
- 模块化设计：所有新增代码遵循单一职责原则，scripts/ 目录可作为 Python 包直接 import
- 集成测试：通过 `test_load_model.py` 实现 4 项自动化自检，确保各模块可独立运行
- 代码上传：全部代码已推送至 GitHub 仓库 Bachelor-s-Graduation-Project

---

## （三）下月工作计划 (Next Steps)

### 1. 宇树 G1 模型完整适配（需赴深圳后确认）
- 从 GitHub 获取宇树 G1 官方 URDF（23-DoF 版本：`g1_23dof.urdf`），导入 envs/ 目录
- 利用已开发的 `validate_urdf.py` 进行关节限位校准和碰撞体验证
- 搭建 G1 专用 Gymnasium 环境，对接 PhysicsRewardWrapper 物理奖励

### 2. MoCap 步态数据获取与 Retargeting

> **需与博士师兄确认的关键问题：**
> - 实验室是否有自采的人体 MoCap 数据？格式为何（BVH/C3D/CSV）？
> - 若无自采数据，是否采用 CMU MoCap 公开数据集（Subject 35 步行序列）？
> - G1 机器人的人体骨骼→关节映射方案是否有现成参考？

- 开发 BVH 解析器 + IK 运动学逆解脚本，完成人体轨迹→G1 关节角度空间映射
- 输出标准化训练数据对 (obs, action)，存入 data/ 目录

### 3. 行为克隆（BC）预训练 — Warm-start 阶段
- 利用 retargeting 后的专家数据，通过 `pretrain_with_hybrid_loss()` 训练 GaitPriorNetwork
- 验证预训练策略在简化模型上的站立/迈步表现
- 保存预训练权重至 `models/bc_pretrained.pt`，为后续 PPO 微调提供初始化

---

## 组会学习记录 (Group Meeting Log)

*(本月因春节假期暂无新增组会记录)*
