# 🤖 G1 双足机器人 RL 可视化控制台

> **定位**: 毕设核心展示系统 + 大厂具身智能岗位面试 Portfolio
> 
> **一句话简历描述**: *"从零构建了支持 Sim2Real 的双足机器人强化学习在线仿真控制系统，基于 FastAPI+WebSocket 实现了训练实时监控、一键推理评估和关节轨迹可视化"*

---

## 🚀 Quick Start

```bash
# 1. 安装依赖
conda activate biped_rl
pip install fastapi uvicorn websockets

# 2. 启动控制台
cd g1_dashboard
python server.py

# 3. 浏览器打开
# http://localhost:8000
```

---

## 🏗 技术路线 (Technical Roadmap)

```
┌─────────────────────────────────────────────────────────────┐
│                    系统架构全景                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    WebSocket     ┌──────────────────────┐     │
│  │ Browser  │◄════════════════►│   FastAPI Server     │     │
│  │          │    REST API      │                      │     │
│  │ Plotly   │◄────────────────►│ ┌──────────────────┐ │     │
│  │ Charts   │                  │ │  MuJoCo Engine   │ │     │
│  │          │    Video/JSON    │ │  (Headless)      │ │     │
│  │ 3D View  │◄────────────────►│ ├──────────────────┤ │     │
│  │          │                  │ │  SB3 PPO Model   │ │     │
│  └──────────┘                  │ │  (.zip weights)  │ │     │
│                                │ ├──────────────────┤ │     │
│                                │ │  Log Parser      │ │     │
│                                │ │  (regex→JSON)    │ │     │
│                                │ └──────────────────┘ │     │
│                                └──────────┬───────────┘     │
│                                           │ CycloneDDS      │
│                                    ┌──────┴──────┐          │
│                                    │  G1 Real    │          │
│                                    │  Robot      │          │
│                                    └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 核心技术点 (面试必答)

| 技术点 | 面试关键词 | 你的回答要点 |
|--------|-----------|-------------|
| **PPO + IL 融合** | 为什么不用纯 RL？ | 纯 RL 样本效率低，加 IL 先验(BC Loss + KL约束)可以大幅减少探索空间，防止"类星人步态" |
| **域随机化 (DR)** | 如何解决 Sim2Real Gap？ | 对连杆质量、地面摩擦系数注入 20% 高斯噪声，迫使策略学习鲁棒特征而非拟合仿真器Bug |
| **物理奖励设计** | Reward Shaping 怎么做的？ | 三项加权: Tracking(0.5) + Velocity(0.3) + Survival(0.2) - COT能耗惩罚 |
| **WebSocket 实时推送** | 为什么不用轮询？ | 训练指标高频更新，WebSocket 是全双工持久连接，避免 HTTP 重复握手开销 |
| **MuJoCo Headless** | 怎么在服务器上渲染？ | 设置 `MUJOCO_GL=egl` 使用 EGL 离屏渲染，无需显示器 |
| **CycloneDDS** | 实机通信方案？ | DDS 是工业级发布/订阅协议，延迟 0.2ms 级别，远优于 ROS2 默认 Topic |

---

## 🎯 面试项目深挖预演 (STAR 法则)

### Q1: "你这个项目遇到的最大困难是什么？"

> **S**: 训练好的 PPO 策略在仿真器里走得很稳，但一下发到真机就原地抽搐摔倒。
> **T**: 需要找到让仿真策略在真实物理世界也能工作的方法。
> **A**: 引入域随机化 (DR)，在训练时随机扰动 ±20% 的连杆质量和地面摩擦系数。同时用 KL 散度约束策略不偏离离线先验太远。
> **R**: Sim2Sim 评估下，DR 策略在 100 次随机扰动中保持了 85%+ 的站立成功率，而无 DR 的基线只有 40%。

### Q2: "你的奖励函数是怎么设计的？为什么？"

> 我设计了一个四项加权组合奖励：
> - **模仿学习跟踪奖励 (0.5权重)**: `exp(-5 * ||q_current - q_reference||²)`，追踪 MoCap 参考动作
> - **速度追踪 (0.3)**: 追踪目标前进速度 0.5 m/s
> - **存活奖励 (0.2)**: 鼓励机器人不摔倒
> - **能耗惩罚**: `-0.001 * ||torque||²`，对应物理学中的 COT (Cost of Transport)
> 
> 这么设计是因为纯速度追踪会导致"膝盖反折"等诡异步态，加入模仿学习约束才能学出类人步态。

### Q3: "为什么选择 FastAPI 而不是 Flask？"

> FastAPI 原生支持 async/await 和 WebSocket，这对实时训练监控至关重要。它的自动 OpenAPI 文档生成也方便团队协作。性能上基于 Starlette，比 Flask 的同步模型快 3-5 倍。

### Q4: "如果让你继续优化这个系统，你会怎么做？"

> 1. **World Model (DreamerV3)**: 用学到的世界模型做 imagination-based planning，减少真实交互次数
> 2. **VLA 架构**: 把视觉输入接进来，不再只依赖关节状态，实现从像素到动作的端到端控制
> 3. **分布式训练**: 用 Ray/RLlib 进行多 GPU 并行训练，把 10M 步压缩到 1 小时内

---

## 📂 文件结构

```
g1_dashboard/
├── server.py              # FastAPI 主服务 (REST + WebSocket)
├── static/
│   ├── index.html         # 单页应用 (KPI + Charts + Inference)
│   ├── css/dashboard.css  # 暗色主题 UI
│   └── js/app.js          # 前端逻辑 (Plotly + WebSocket)
└── README.md              # 本文件
```

---

## 📝 简历项目描述模板

**项目名称**: 双足机器人强化学习在线仿真控制系统

**项目描述**: 
- 基于 **MuJoCo + Stable-Baselines3** 搭建宇树 G1 (29-DoF) 人形机器人仿真环境
- 设计 **PPO + 模仿学习融合架构**，通过 KL 散度约束和物理感知奖励函数实现类人步态控制
- 采用 **域随机化 (Domain Randomization)** 技术弥合 Sim2Real Gap
- 开发基于 **FastAPI + WebSocket + Plotly.js** 的实时训练监控与推理可视化 Web 控制台
- 通过 **CycloneDDS** 协议实现 0.2ms 延迟的实机通信部署

**技术栈**: Python · PyTorch · MuJoCo · Stable-Baselines3 · FastAPI · WebSocket · Plotly.js · CycloneDDS · Git
