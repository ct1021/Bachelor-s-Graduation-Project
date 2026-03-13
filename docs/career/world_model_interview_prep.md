# 大厂算法岗实习（世界模型/大语言模型）面试突击战备指南

> **用户画像设定**：计算机科学（卓越班）本科应届毕业生，即将完成“基于强化学习与模仿学习融合的高精度（23-DoF）人形机器人步态控制研究”。具备扎实的深度强化学习（PPO）、运动学正逆解推导及 Python 工程落地能力（PyBullet, MuJoCo, DDS 实机通讯）。
> **目标岗位定位**：头部大厂/独角兽（如 字节跳动、腾讯、阿里、智源、月之暗面、MiniMax）—— **大模型算法工程师、世界模型(World Model) 研究员、基础模型(Foundation Models)及强化学习工程师**。

## 🎯 突击打卡矩阵 (3个月高强度备战)

### 【板块一：硬核算法与代码穿透 (LeetCode & LC Top 100)】
大厂算法岗的“敲门砖”，代码手撕必须达到肌肉记忆的程度。
*   **必刷题库**：LeetCode 前 200 题、剑指 Offer、尤其是动态规划（DP）、树图搜索（DFS/BFS）、二分查找与排序。
*   **每日节奏**：保底 3 题（1 Easy, 2 Medium），周末突击 1 Hard。
*   **面试考点**：能不能在 20 分钟内 Bug-free 撕出最优解，并准确说出时间/空间复杂度（Big O）。

### 【板块二：强化学习与决策智能深入 (RL & RLHF)】
你已经具备了极好的 PPO 应用基础，现在需要往“对齐理论”（Alignment）方向靠拢。
*   **核心攻坚**：深入理解 InstructGPT/ChatGPT 背后的 RLHF 范式。
*   **灵魂拷问**：
    *   PPO 与 TRPO 的本质推导区别？为什么大模型对齐都爱用 PPO？
    *   DPO (Direct Preference Optimization) 算法推导：如何抛弃 Reward Model 直接优化策略？跟 PPO 相比的优劣？
    *   Offline RL 理论，CQL 和 IQL 是怎么处理 OOD (Out-of-Distribution) 动作评估问题的？

### 【板块三：大语言模型底座与架构 (Transformer & LLM Pretraining)】
算法岗的核心护城河，必须脱离调包侠，手写底层注意力机制。
*   **核心攻坚**：
    *   Transformer 架构推导（Q, K, V 的物理意义，缩放点积为啥要除以 $\sqrt{d}$）。
    *   MHA (Multi-Head Attention), MQA, GQA 三者的结构区别与推理显存占用情况（KV Cache）。
    *   位置编码技术：RoPE (旋转位置编码) 的数学推导，ALiBi。
*   **实践作业**：用纯 PyTorch 不依赖 `nn.Transformer` 手写一个带 Mask 的 Decoder-only 模型（类 LLaMA 架构）。

### 【板块四：通用世界模型与视频生成引擎 (World Models / Video Gen)】
这是你未来最想做的领域，也是当前 Sora, Sora-like, V-JEPA 最火爆的前沿。
*   **必读神文**：
    *   Yann LeCun 的 *A Path Towards Autonomous Machine Intelligence* (JEPA 联合嵌入预测架构)。
    *   OpenAI 的 *Sora 技术报告* (DiT: Diffusion Transformer 的空间/时间潜空间压缩逻辑)。
    *   *World Models* (Ha & Schmidhuber, 2018) -> 强化学习中利用 RNN/VAE 重建环境先验的奠基石。
*   **灵魂拷问**：
    *   生成式世界模型（Sora、Gen-3）与基于联合嵌入的预测模型（V-JEPA）在提取状态表征上有何区别？
    *   Transformer 如何把视频切成 Spacetime Patches 并做压缩表征？
    *   你的毕设（Gait Prior Network 先验网络）如何启发你去设计一个端到端的世界模拟系统（World Simulator）？

---

## 💡 AI Agent 面试官系统（给接班 Agent 的强制指令）

> **⚠️ 下方为系统Prompt锚点，未来的辅导 Agent 启动时应立刻捕获**

1.  **角色**：字节跳动/腾讯 AI Lab 高级算法研究员（面试官/Mentor）。
2.  **职责**：每天向用户“程童”推送一道大厂高频算法题（核心逻辑解答）或一份顶会论文精读分解；每周末进行一小时的“追问式（Deep Dive）”模拟面试。
3.  **语言风格**：犀利、直指痛点、不讲废话、强逻辑。当发现基础概念薄弱时，强制要求手推公式或写伪代码。
4.  **项目升华策略**：面试官 Agent 会不断诘问用户：“你在 G1 双足机器人上的 ‘KL 散度’与‘离线小模型’设计，如果搬到千亿参数的大模型分布式强化训练（比如 RLHF）中，通信开销该如何解决？” 用极端场景压榨出求职者的工程潜力。
