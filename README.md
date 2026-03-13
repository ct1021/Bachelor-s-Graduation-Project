# 毕业设计：双足机器人强化学习与实机验证 (Bachelor's Graduation Project)

本项目主要专注于宇树 G1 与 LimX TRON1 等双足机器人的全身控制与强化学习验证，从 MuJoCo 仿真环境中的模型训练，到跨引擎的 Sim2Sim 泛化能力验证，最终目标是实现真实世界的 Sim2Real 物理样机部署。

## 核心任务与方向
- **当前核心节点**: **中期答辩冲刺**。目前已成功走通了基于真实人体 MoCap 数据冷启动的 `PPO` 强化学习训练，模型具备基本行走能力。
- **近两日冲刺任务**: 补全 **Sim2Sim** 域随机化 (Domain Randomization) 实验（对质量、摩擦力注入物理噪声）来证明策略框架的鲁棒性，随后进入物理样机硬件网络联调。

## 开发文档导航 (Documentation Architecture)

整个项目的规划、教程与进度统一分类整理于 `docs/` 目录下：

### 📈 1. 进度追踪与汇报 (`docs/progress/`)
用于存放定期的进度汇总和官方答辩物料：
- `当前阶段进展总结.md`: 最新阶段的整体速览。
- `progress_2026_03.md` / `progress_2026_02.md`: 按月度的细化工作汇报。
- `中期检查表-填写内容-卓计2201-程童.md`: 应对学校中期答辩填报的正式文案。

### 🗺️ 2. 计划与日程编排 (`docs/plans/`)
用于存放跨度较长的开发 Phase 或短期冲刺 Schedule：
- `phase3_sim2real_schedule.md`: **正在执行**的三期计划：Sim2Real 与实机部署详细行事历（重点关注！周五任务来源）。
- `Biped Robot RL Phase 2.md`: 二期计划：基于模仿学习的数据驱动方法探索（已基本完成）。

### 🛠️ 3. 技术教程与手册 (`docs/guides/`)
用于存放平台搭建、SDK 联调和网络配置操作指南：
- `robot_guide.md`: 综合型机器人操作入门教程（含 G1/TRON1 的基本跑通测试）。
- `sim_training_guide.md`: 仿真开发训练指南。
- `Linux Workstation Setup.md`: 双机协同开发的 Linux 主机初始化向导。
- `unitree_sdk2_networking_guide.md`: G1 有线网络通信直连配置。
- `environment_setup.md`: Python 开发运行环境依赖指南。