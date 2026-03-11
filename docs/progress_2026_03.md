# 2026年3月份工作汇报

## （一）上月计划完成情况 (Review)

### 1. 宇树 G1 模型完整适配
**进行中。** 已到达深圳校区，获取到宇树 G1 实体机器人和 LimX TRON1B-520 双足机器人。已从 GitHub 拉取官方开源 SDK 和仿真工具，包括 G1 的 23-DoF/29-DoF MuJoCo 模型（MJCF 格式）。下一步将使用已有的 `validate_urdf.py` 对官方模型进行完整适配。

### 2. MoCap 步态数据获取与 Retargeting
**待确认。** 上月遗留的关键问题（MoCap 数据来源、骨骼映射方案）需与博士师兄当面确认。

### 3. 行为克隆（BC）预训练
**框架就绪。** BC 训练管线（`hybrid_loss.py` 中的 `pretrain_with_hybrid_loss()`）已就绪，待专家数据到位后即可开始训练。

---

## （二）本月进展 (Progress)

### 1. 赴深圳校区，接触实体机器人

已于 3 月到达深圳校区，身边有：
- **宇树 G1 人形机器人** — 全尺寸人形双足，23/29-DoF
- **LimX Dynamics TRON1B-520** — 模块化双足机器人，支持 Point-Foot/Sole/Wheeled 三种足端

### 2. 开源项目引入与环境搭建

从 GitHub 拉取了两家厂商的官方开源仓库至项目目录 `robot_sdk/`：

#### 宇树 G1 相关（3 个仓库）

| 仓库 | 说明 |
|------|------|
| `unitree_sdk2_python` | 官方 Python SDK，基于 CycloneDDS，支持高层/底层运动控制 |
| `unitree_mujoco` | MuJoCo 仿真环境 + G1 官方 MJCF 模型（23/29-DoF） |
| `unitree_rl_mjlab` | 基于 MuJoCo 的 RL 训练框架，直接支持 G1 速度跟踪和动作模仿训练 |

#### LimX TRON1 相关（2 个仓库）

| 仓库 | 说明 |
|------|------|
| `tron1-mujoco-sim` | TRON1 MuJoCo 仿真工具 + 底层控制 SDK |
| `tron1-rl-deploy-python` | Python 端 RL 部署算法 |

### 3. 编写机器人操作入门教程

编写了综合中文教程文档 `docs/robot_guide.md`，内容包括：
- **Part A**: 宇树 G1 快速入门（硬件连接、SDK 安装、基本控制示例、MuJoCo 仿真、RL 训练框架）
- **Part B**: LimX TRON1 快速入门（硬件概述、SDK 配置、仿真操作、RL 部署流程）
- **Part C**: 与毕设项目的集成路径（模型替换、RL 管线对接、Sim2Real 路径）

### 4. Linux 工作站部署方案制定与教程增强

由于 G1/TRON1 实体控制和 Sim2Real 部署均需 Linux 环境，利用实验室闲置主机（i9 + RTX 2070 Ti 8GB）规划了双机协作开发方案：
- **架构设计**：Windows 笔记本（编码/仿真/文档）↔ SSH/Tailscale ↔ Linux 工作站（实体控制/部署）↔ 机器人
- **远程访问**：方案对比（SSH 局域网直连 / Tailscale 异地穿透 / VS Code Remote-SSH 开发）
- **数据同步**：Git（代码）+ scp/rsync（大文件/模型权重）
- 将 `robot_guide.md` 从原始 354 行重写为 ~500 行手把手级教程，新增 **Part 0: Linux 工作站搭建**（Ubuntu 安装、GPU 驱动、Conda 环境、SSH 免密、Tailscale 配置），增强各 Part 的首次使用验证清单和故障排查

---

## （三）下月工作计划 (Next Steps)

### 1. G1 官方模型验证与 Gym 环境搭建
- 使用 `validate_urdf.py` 对 G1 23-DoF 模型进行关节限位校准和碰撞体验证
- 搭建 G1 专用 Gymnasium 环境，对接 `PhysicsRewardWrapper`
- 在 MuJoCo 中运行 G1 仿真验证

### 2. 与博士师兄确认数据问题
> **需当面确认的关键问题：**
> - 实验室是否有自采的人体 MoCap 数据？格式？
> - 若无自采数据，是否采用 CMU MoCap 公开数据集？
> - G1 机器人的人体骨骼→关节映射方案

### 3. MoCap 数据处理与 Retargeting
- 开发 BVH 解析器 + IK 运动学逆解脚本
- 完成人体轨迹→G1 关节角度空间的映射

### 4. BC 预训练启动（Warm-start）
- 利用 retargeting 后的专家数据训练 GaitPriorNetwork
- 保存预训练权重到 `models/bc_pretrained.pt`

---

## 组会学习记录 (Group Meeting Log)

*(本月待补充)*
