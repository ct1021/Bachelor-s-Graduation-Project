# 机器人操作入门指南（完整版）

> 本文档为毕设项目配套教程，涵盖 **Linux 工作站搭建**、**宇树 G1 人形机器人**和 **LimX Dynamics TRON1B-520 双足机器人**的完整操作入门。
> 所有开源仓库已克隆到 `robot_sdk/` 目录下。

---

## 目录

- [Part 0: Linux 工作站搭建 & 远程开发](#part-0-linux-工作站搭建--远程开发)
- [Part A: 宇树 G1 完整操作教程](#part-a-宇树-g1-完整操作教程)
- [Part B: LimX TRON1 完整操作教程](#part-b-limx-tron1-完整操作教程)
- [Part C: 与毕设项目集成 & 双机工作流](#part-c-与毕设项目集成--双机工作流)
- [附录：常见问题 & 故障排查](#附录常见问题--故障排查)

---

# Part 0: Linux 工作站搭建 & 远程开发

## 为什么需要 Linux 工作站？

| 操作 | Windows 笔记本 | Linux 工作站 |
|------|:---:|:---:|
| MuJoCo 仿真查看模型 | ✅ | ✅ |
| RL 训练代码编写/调试 | ✅ | ✅ |
| RL 仿真训练 (GPU) | ✅ | ✅ |
| G1 SDK 实体控制 | ❌ | ✅ |
| TRON1 SDK (`limxsdk`) | ❌ | ✅ |
| CycloneDDS 实时通信 | ❌ | ✅ |
| Sim2Real 部署 (`policy.onnx`) | ❌ | ✅ |
| ROS2 (TRON1 部署需要) | ❌ | ✅ |

**结论**：仿真开发可以在 Windows 上做，但所有涉及实体机器人的操作都需要 Linux。

## 整体架构

```
┌─────────────────────────┐          ┌───────────────────────────┐
│  Windows 笔记本 (随身)   │          │  Linux 工作站 (深圳实验室) │
│                         │  SSH /   │                           │
│ · VS Code 编码          │ Tailscale│ · G1/TRON1 SDK 实体控制    │
│ · MuJoCo 仿真预览       │◄────────►│ · CycloneDDS 通信          │
│ · Git 管理代码          │  rsync   │ · RL 训练 (RTX 2070 Ti)    │
│ · 文档/月报撰写         │  scp     │ · ROS2 (TRON1)             │
│ · VS Code Remote-SSH   │  Git     │ · Sim2Real 部署             │
└─────────────────────────┘          └──────────┬────────────────┘
                                                │ 网线 192.168.123.x
                                         ┌──────┴──────┐
                                         │  G1 / TRON1  │
                                         │  实体机器人   │
                                         └─────────────┘
```

---

## 0.1 Ubuntu 22.04 安装（手把手）

### 准备工具
- U 盘（≥8GB）
- [Rufus](https://rufus.ie/) 或 [balenaEtcher](https://etcher.balena.io/)（制作启动盘）
- [Ubuntu 22.04.x LTS ISO](https://releases.ubuntu.com/22.04/)（下载桌面版 Desktop）

### Step 1：制作启动 U 盘

1. 在 Windows 上下载 Rufus，插入 U 盘
2. 打开 Rufus → 选择下载好的 Ubuntu ISO → 点击"开始"
3. 等待写入完成（约 5-10 分钟）

### Step 2：BIOS 设置

1. 重启主机，开机时连续按 `F2` / `Del` / `F12`（按主板品牌不同）
2. 在 BIOS 中：
   - **Boot Order**：将 USB 设为第一启动项
   - **Secure Boot**：关闭（Disabled）
   - 如有 **CSM/Legacy**：设为 UEFI Only
3. 保存并退出（`F10`）

### Step 3：安装 Ubuntu

1. 从 U 盘启动后选择 **Install Ubuntu**
2. 语言选**中文（简体）**，键盘选**英语(美国)**
3. 安装类型选择：
   - 如果整块硬盘给 Linux：选"清除整个磁盘并安装 Ubuntu"
   - 如果要保留 Windows 双系统：选"其他选项"手动分区：
     ```
     /boot/efi  →  512MB  (EFI 系统分区)
     swap       →  16GB   (交换分区，与 RAM 等大)
     /          →  剩余全部 (ext4, 根分区)
     ```
4. 设置用户名和密码（**记住密码，SSH 登录需要**）
5. 等待安装完成，拔掉 U 盘，重启

### Step 4：首次配置

重启进入 Ubuntu 后，打开终端 (`Ctrl+Alt+T`)：

```bash
# 1. 更新系统
sudo apt update && sudo apt upgrade -y

# 2. 安装基础开发工具
sudo apt install -y build-essential cmake git curl wget vim \
    openssh-server net-tools htop

# 3. 安装 NVIDIA 驱动（你的 RTX 2070 Ti 需要）
sudo apt install -y nvidia-driver-535
sudo reboot  # 重启生效

# 4. 验证 GPU
nvidia-smi
# 预期输出：看到 RTX 2070 Ti, 8GB 显存, 驱动版本 535.x
```

### Step 5：安装 Miniconda + 创建 biped_rl 环境

```bash
# 1. 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 2. 创建与 Windows 端相同的 conda 环境
conda create -n biped_rl python=3.10 -y
conda activate biped_rl

# 3. 安装核心依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mujoco gymnasium stable-baselines3 tensorboard
pip install numpy scipy matplotlib

# 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import mujoco; print('MuJoCo OK')"
# 预期输出：CUDA: True  MuJoCo OK
```

---

## 0.2 远程访问方案

### 方案对比

| 方案 | 适用场景 | 配置难度 | 优点 | 缺点 |
|------|---------|:---:|------|------|
| **SSH（局域网直连）** | 同一实验室/校园网 | ⭐ | 零额外配置，速度最快 | 离开局域网不可用 |
| **Tailscale** | 异地远程 | ⭐⭐ | 免公网IP，P2P直连，加密 | 需注册账号 |
| **frp 内网穿透** | 有云服务器时 | ⭐⭐⭐ | 自建可控 | 配置复杂，需公网服务器 |
| **XRDP 远程桌面** | 需要 GUI 操作 | ⭐⭐ | 能看到完整桌面 | 卡顿，带宽消耗大 |

> **推荐组合**：SSH（在实验室时）+ Tailscale（离开深圳后备用）

---

### 方案 A：SSH 直连（局域网 — 最常用）

**Linux 工作站端**（只需配置一次）：
```bash
# SSH 服务已在 Step 4 中安装 (openssh-server)
# 确认已启动：
sudo systemctl enable ssh
sudo systemctl start ssh

# 查看 Linux 工作站 IP 地址：
ip addr show
# 记下类似 192.168.x.x 的地址，假设是 192.168.1.100
```

**Windows 笔记本端**：
```powershell
# 测试连接（替换为实际用户名和IP）
ssh yourname@192.168.1.100
# 输入密码后看到 Linux 终端 = 成功！
```

**配置免密登录（强烈推荐）**：
```powershell
# Windows 端生成密钥（如果还没有的话）
ssh-keygen -t ed25519
# 一路回车即可

# 将公钥传到 Linux 端
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh yourname@192.168.1.100 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 再次连接，不再需要密码：
ssh yourname@192.168.1.100
```

---

### 方案 B：Tailscale（离开深圳后远程访问）

Tailscale 能让两台设备即使不在同一网络也能直接互连，像局域网一样。

**Step 1：两端都安装 Tailscale**

Linux 工作站：
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# 会弹出一个 URL，用浏览器打开并登录/注册
```

Windows 笔记本：
- 下载安装 [Tailscale Windows 客户端](https://tailscale.com/download/windows)
- 登录同一个账号

**Step 2：获取 Tailscale IP**
```bash
# Linux 端
tailscale ip -4
# 输出类似：100.64.0.2
```

**Step 3：通过 Tailscale SSH 连接**
```powershell
# Windows 端（用 Tailscale IP）
ssh yourname@100.64.0.2
```

> ✅ 现在即使你回到广州/其他城市，也能远程访问深圳工作站！

---

### 方案 C：VS Code Remote-SSH（推荐日常开发方式）

这是**最舒适的开发方式**——在 Windows 上用 VS Code 写代码，实际代码运行在 Linux 上。

**Step 1：安装扩展**
- 在 VS Code 中安装 **Remote - SSH** 扩展（搜索 `ms-vscode-remote.remote-ssh`）

**Step 2：配置连接**
1. `Ctrl+Shift+P` → 输入 `Remote-SSH: Open SSH Configuration File`
2. 选择 `C:\Users\CT1021\.ssh\config`，添加：

```
Host lab-linux
    HostName 192.168.1.100
    User yourname
    IdentityFile ~/.ssh/id_ed25519

# 如果用 Tailscale 远程访问
Host lab-linux-remote
    HostName 100.64.0.2
    User yourname
    IdentityFile ~/.ssh/id_ed25519
```

**Step 3：连接**
1. 左下角绿色图标 → `Connect to Host` → 选 `lab-linux`
2. 首次连接会自动在 Linux 端安装 VS Code Server（约 1 分钟）
3. 连接成功后，你可以：
   - 在 VS Code 内直接编辑 Linux 上的文件
   - 在 VS Code 终端中直接运行 Linux 命令
   - 用 VS Code 调试器远程调试 Python 代码

---

## 0.3 数据同步方案

### Git（代码同步——首选）

两台机器都 clone 同一个 GitHub 仓库，通过 push/pull 同步：

```bash
# Linux 端
cd ~
git clone https://github.com/ct1021/Bachelor-s-Graduation-Project.git
cd Bachelor-s-Graduation-Project
conda activate biped_rl
```

**工作流**：
1. 在 Windows 上写代码 → `git push`
2. 在 Linux 上 `git pull` → 运行实体控制
3. Linux 上有新结果 → `git push` → Windows 上 `git pull` 查看

### rsync（大文件同步——模型权重/训练日志）

模型权重和训练日志不适合放 Git（文件太大），用 rsync：

```powershell
# Windows → Linux：把训练好的模型传到 Linux 部署
scp models/bc_pretrained.pt yourname@192.168.1.100:~/Bachelor-s-Graduation-Project/models/

# Linux → Windows：把实验结果拉回来
scp yourname@192.168.1.100:~/Bachelor-s-Graduation-Project/logs/*.csv ./logs/

# 批量同步整个目录
# 需要在 Windows 上安装 rsync (通过 WSL 或 Git Bash)
rsync -avz yourname@192.168.1.100:~/Bachelor-s-Graduation-Project/logs/ ./logs/
```

---

# Part A: 宇树 G1 完整操作教程

## A1. 硬件概述

宇树 G1 是一款人形双足机器人，支持两种配置：

| 配置 | 关节数 | 结构 |
|------|:---:|------|
| **23-DoF** | 23 | 每腿 6 关节 + 腰部 1 关节 + 每臂 5 关节 |
| **29-DoF** | 29 | 每腿 6 关节 + 腰部 3 关节 + 每臂 7 关节（含手腕） |

### G1 23-DoF 关节名称表

| 索引 | 关节名 | 说明 |
|:---:|--------|------|
| 0 | L_LEG_HIP_PITCH | 左腿髋关节-俯仰 |
| 1 | L_LEG_HIP_ROLL | 左腿髋关节-翻滚 |
| 2 | L_LEG_HIP_YAW | 左腿髋关节-偏航 |
| 3 | L_LEG_KNEE | 左膝关节 |
| 4 | L_LEG_ANKLE_PITCH | 左踝关节-俯仰 |
| 5 | L_LEG_ANKLE_ROLL | 左踝关节-翻滚 |
| 6-11 | R_LEG_* | 右腿（同左腿对称） |
| 12 | TORSO | 腰部旋转 |
| 13-16 | L_SHOULDER_* / L_ELBOW_* | 左臂 |
| 17-22 | R_SHOULDER_* / R_ELBOW_* | 右臂 |

> 💡 完整关节表见 `robot_sdk/unitree/unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md`

---

## A2. Windows 仿真 Quick Start（5 分钟验证）

> 🎯 **目标**：在你的 Windows 笔记本上用 MuJoCo 打开 G1 模型，确认环境可用。

### 前置条件检查

```powershell
# 确认 conda 环境
conda activate biped_rl

# 确认 MuJoCo 已安装
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
# ✅ 预期输出：MuJoCo version: 3.x.x
# ❌ 如果报错 ModuleNotFoundError：运行 pip install mujoco
```

### Step 1：查看 G1 模型

```powershell
conda activate biped_rl
cd "E:\Microsoft VS Code\programs\gradually-work"

python -c "
import mujoco
import mujoco.viewer
model = mujoco.MjModel.from_xml_path('robot_sdk/unitree/unitree_mujoco/unitree_robots/g1/scene_29dof.xml')
data = mujoco.MjData(model)
print('模型加载成功!')
print(f'  关节数 (nq): {model.nq}')
print(f'  自由度 (nv): {model.nv}')
print(f'  执行器数: {model.nu}')
mujoco.viewer.launch(model, data)
"
```

**预期结果**：
- 终端输出关节数、自由度信息
- 弹出 MuJoCo 可视化窗口，显示 G1 机器人站在地面上
- 可以用鼠标拖拽旋转视角，滚轮缩放
- 按空格键开始/暂停仿真（G1 会因重力倒下，这是正常的）

### Step 2：验证仿真交互

```powershell
python -c "
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('robot_sdk/unitree/unitree_mujoco/unitree_robots/g1/scene_23dof.xml')
data = mujoco.MjData(model)

# 打印所有关节名
print('=== G1 23-DoF 关节列表 ===')
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f'  Joint {i}: {name}')

# 步进仿真 1 秒
print(f'\n仿真步长: {model.opt.timestep}s')
steps = int(1.0 / model.opt.timestep)
for _ in range(steps):
    mujoco.mj_step(model, data)
print(f'仿真 1 秒后质心高度: {data.qpos[2]:.3f}m')
print('仿真验证完成!')
"
```

**预期结果**：打印 23 个关节名，仿真 1 秒后质心高度下降（重力影响）。

### ✅ Windows 仿真验证清单

完成以上两步后，勾选确认：

- [ ] MuJoCo 版本正确输出
- [ ] G1 29-DoF 模型可视化窗口正常显示
- [ ] G1 23-DoF 关节列表打印正确（23 个关节）
- [ ] 仿真步进无报错

---

## A3. SDK 安装详解

### 路径一：Windows 仿真开发（你的笔记本）

只需要安装 Python SDK，**不需要** CycloneDDS：

```powershell
conda activate biped_rl
cd "E:\Microsoft VS Code\programs\gradually-work\robot_sdk\unitree\unitree_sdk2_python"
pip install -e .
```

验证：
```powershell
python -c "import unitree_sdk2py; print('SDK 安装成功!')"
# ✅ 预期输出：SDK 安装成功!
# ❌ 如果报错 cyclonedds 相关：这在 Windows 上是正常的，仿真不需要 DDS
```

> ⚠️ Windows 上 SDK 主要用于**查看代码结构和仿真开发**。真正控制实体机器人要在 Linux 上。

### 路径二：Linux 实体控制（深圳工作站）

需要完整安装 CycloneDDS + SDK：

```bash
# === 在 Linux 工作站上操作 ===
conda activate biped_rl

# 1. 编译安装 CycloneDDS（实体通信的基础）
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
# 预期：编译完成无报错，~/cyclonedds/install/ 下有 lib/ 和 include/

# 2. 设置环境变量（每次开机自动生效）
echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
source ~/.bashrc

# 3. 安装 Python SDK
cd ~/Bachelor-s-Graduation-Project/robot_sdk/unitree/unitree_sdk2_python
pip install -e .

# 4. 验证
python -c "import unitree_sdk2py; print('SDK 安装成功!')"
# ✅ 预期输出：SDK 安装成功!（无任何 warning）
```

**如果遇到问题**：
```bash
# CycloneDDS 编译失败？检查依赖：
sudo apt install -y cmake gcc g++ python3-dev

# pip install -e . 失败？检查环境变量：
echo $CYCLONEDDS_HOME
# 应输出：/home/yourname/cyclonedds/install
```

---

## A4. 实体 G1 控制（Linux 工作站 → 机器人）

> ⚠️ **安全须知（首次操作必读）**：
> 1. 首次连接时，G1 应处于**吊装状态**（用吊绳悬挂，双脚离地）
> 2. 确保周围无人/无障碍物
> 3. **先用高层控制**熟悉，再尝试底层关节控制
> 4. 底层控制前必须通过 APP 关闭 sport_mode，否则会指令冲突！

### Step 1：网线连接

1. 用网线连接 Linux 工作站与 G1 机器人的网口
2. 配置网卡 IP：

```bash
# 查看网卡名称
ip link show
# 找到物理网卡名（如 enp2s0、eth0、eno1 等，不是 lo）

# 配置 IP 地址（替换 enp2s0 为你的网卡名）
sudo ip addr add 192.168.123.222/24 dev enp2s0
sudo ip link set enp2s0 up

# 验证能 ping 通机器人
ping 192.168.123.161
# ✅ 预期：收到回复（G1 默认 IP 是 192.168.123.161）
# ❌ 如果 ping 不通：检查网线连接、网卡名是否正确
```

### Step 2：读取机器人状态（只读，安全）

```bash
conda activate biped_rl
cd ~/Bachelor-s-Graduation-Project

# 查看 G1 高层状态信息（位置、速度、IMU 数据等）
python robot_sdk/unitree/unitree_sdk2_python/example/g1/high_level/g1_high_level_example.py enp2s0
# 替换 enp2s0 为你的实际网卡名

# 预期输出：持续打印机器人的位置、姿态、速度等信息
# 按 Ctrl+C 停止
```

### Step 3：高层运动控制

```bash
# 高层运动控制测试
python robot_sdk/unitree/unitree_sdk2_python/example/g1/high_level/g1_sportmode_example.py enp2s0
```

可用的内置动作：
```python
test.StandUpDown()       # 站立/趴下（最安全的首次测试）
test.VelocityMove()      # 速度控制行走
test.BalanceAttitude()   # 姿态平衡控制
test.TrajectoryFollow()  # 轨迹跟踪
test.SpecialMotions()    # 特殊动作
```

> 💡 **建议首次测试顺序**：StandUpDown → BalanceAttitude → VelocityMove

### Step 4：底层关节控制

```bash
# ⚠️ 先通过宇树 APP 关闭 sport_mode！
# APP → 设置 → 运动服务 → 关闭

python robot_sdk/unitree/unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py enp2s0
# 此时你可以直接控制每个关节的力矩/位置
```

---

## A5. RL 训练框架（unitree_rl_mjlab）

### 安装

```bash
# 在 Windows 或 Linux 上都可以
conda activate biped_rl
cd robot_sdk/unitree/unitree_rl_mjlab
pip install -e .
```

> 📖 详细依赖见 `robot_sdk/unitree/unitree_rl_mjlab/doc/setup_zh.md`

### 训练 G1 速度跟踪策略

```bash
# 基础训练（调小 num-envs 适配 8GB 显存）
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=2048

# 预期：每隔几秒打印训练进度（reward、episode length 等）
# 训练完成后模型保存在 logs/rsl_rl/g1_velocity/<日期>/
```

### 动作模仿训练（需要 MoCap 数据）

```bash
# 1. 转换动作数据格式
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 --output-fps 50

# 2. 训练
python scripts/train.py Unitree-G1-Tracking \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=2048
```

### 查看训练效果

```bash
python scripts/play.py Unitree-G1-Flat \
  --checkpoint_file=logs/rsl_rl/g1_velocity/<日期>/model_<iter>.pt
# 会弹出 MuJoCo 窗口，显示训练后的 G1 行走效果
```

### Sim2Real 部署到实体 G1（Linux 工作站操作）

```bash
# 1. 确保 G1 吊装、网线已连、sport_mode 已关闭
# 2. 按 L2+R2 进入调试模式

# 3. 导出 ONNX 模型（训练完成后）
# 将 policy.onnx 放入以下目录：
cp policy.onnx robot_sdk/unitree/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/vo/exported/

# 4. 编译部署程序
cd robot_sdk/unitree/unitree_rl_mjlab/deploy/robots/g1
mkdir -p build && cd build
cmake .. && make

# 5. 运行！
./g1_ctrl --network=enp2s0  # 替换为你的网卡名
```

---

# Part B: LimX TRON1 完整操作教程

## B1. 硬件概述

LimX Dynamics TRON1B-520 是一款模块化双足机器人，支持**三合一足端切换**：

| 模式 | 环境变量值 | 适用场景 |
|------|-----------|---------|
| **Point-Foot（点脚）** | `PF_TRON1A` | 简单腿部控制研究 |
| **Sole（平底脚）** | `SF_TRON1A` | 人形行走实验 |
| **Wheeled（轮式）** | `WF_TRON1A` | 全地形移动（≥5 m/s） |

**核心参数**：尺寸 ≤ 392×420×845 mm | 重量 ≤ 20 kg | 负载 额定 10 kg | 电池 48V/5Ah, 续航 ≥2h

---

## B2. Windows 仿真 Quick Start

> ⚠️ TRON1 的 MuJoCo 仿真在 Windows 上可以运行，但 `limxsdk` SDK 只支持 Linux。
> Windows 上只能做**纯仿真预览**，不能控制仿真器中的机器人。

```powershell
conda activate biped_rl
cd "E:\Microsoft VS Code\programs\gradually-work"

# 查看 TRON1 MuJoCo 模型（如果有 mujoco viewer）
python -c "
import mujoco
import mujoco.viewer
import os

# 查找模型文件
sim_dir = 'robot_sdk/limx/tron1-mujoco-sim'
for f in os.listdir(sim_dir):
    if f.endswith('.xml'):
        print(f'找到模型: {f}')
"
```

---

## B3. Linux 完整操作（仿真 + 实控）

### Step 1：安装 limxsdk

```bash
# === 在 Linux 工作站上操作 ===
conda activate biped_rl

# 安装运动控制库（选择对应架构）
# x86_64 主机：
pip install robot_sdk/limx/tron1-mujoco-sim/limxsdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl

# 验证
python -c "import limxsdk; print('limxsdk 安装成功!')"
```

### Step 2：设置机器人类型

```bash
# 根据实际安装的足端选择（先问师兄确认机器人是哪种足端）
export ROBOT_TYPE=PF_TRON1A   # 点脚
# export ROBOT_TYPE=SF_TRON1A  # 平底脚
# export ROBOT_TYPE=WF_TRON1A  # 轮式

# 持久化
echo 'export ROBOT_TYPE=PF_TRON1A' >> ~/.bashrc
source ~/.bashrc
```

### Step 3：MuJoCo 仿真

```bash
# 终端 1：启动仿真器
python robot_sdk/limx/tron1-mujoco-sim/simulator.py
# 预期：弹出 MuJoCo 窗口显示 TRON1 模型

# 终端 2（新开）：运行 RL 控制
conda activate biped_rl
export ROBOT_TYPE=PF_TRON1A
python robot_sdk/limx/tron1-rl-deploy-python/main.py
# 预期：TRON1 在仿真中开始行走

# （可选）终端 3：虚拟手柄
./robot_sdk/limx/tron1-mujoco-sim/robot-joystick/robot-joystick
```

### Step 4：实体 TRON1 控制

> 实体控制流程需参考 LimX 官方文档，基本步骤类似仿真但连接实体机器人。
> 官方文档：[limxdynamics.com](https://www.limxdynamics.com)
> GitHub：[github.com/limxdynamics](https://github.com/limxdynamics)

---

# Part C: 与毕设项目集成 & 双机工作流

## C1. 用 G1 官方模型替换简化模型

当前项目使用 `envs/simple_biped.urdf`（6-DoF 简化模型），可升级为 G1 官方模型：

```bash
# G1 官方 MuJoCo 模型（MJCF 格式，非 URDF）位于：
robot_sdk/unitree/unitree_mujoco/unitree_robots/g1/g1_23dof.xml  # 23-DoF
robot_sdk/unitree/unitree_mujoco/unitree_robots/g1/g1_29dof.xml  # 29-DoF
```

**集成步骤**：
1. 用 `scripts/validate_urdf.py` 对 G1 模型进行关节校验
2. 修改 `PhysicsRewardWrapper` 参数适配 G1 真实质量和关节配置
3. 搭建 G1 专用 Gymnasium 环境

## C2. RL 训练管线对接

```
已有毕设管线                    unitree_rl_mjlab
─────────────────            ─────────────────
GaitPriorNetwork  ←──对接──→  策略网络训练
HybridLoss (BC+KL) ←─参考─→  动作模仿训练
PhysicsRewardWrapper ←─复用─→ 奖励函数设计
```

## C3. 双机工作流推荐

### 日常开发流程

| 步骤 | 在哪里做 | 命令/操作 |
|------|---------|----------|
| 1. 写/改代码 | Windows (VS Code) | 直接编辑 or VS Code Remote-SSH |
| 2. 仿真测试 | Windows | `python -c "import mujoco..."` |
| 3. 提交代码 | Windows | `git add . && git commit && git push` |
| 4. 同步到 Linux | Linux (SSH) | `cd ~/Bachelor-s-Graduation-Project && git pull` |
| 5. 实体控制测试 | Linux (SSH) | `python example/g1/high_level/...` |
| 6. 训练 RL | Linux (SSH) | `python scripts/train.py Unitree-G1-Flat` |
| 7. 同步结果 | Linux → Windows | `git push` 或 `scp` 大文件 |

### 大文件传输

```bash
# 训练日志/模型权重不要放 Git，用 scp 传输：
# Linux → Windows:
scp yourname@192.168.1.100:~/project/logs/model.pt ./models/

# Windows → Linux:
scp ./models/bc_pretrained.pt yourname@192.168.1.100:~/project/models/
```

## C4. 项目目录结构总览

```
gradually-work/
├── data/               # 步态数据（MoCap 等）
├── docs/               # 文档与月报
│   ├── robot_guide.md  # ★ 本文档
│   └── progress_*.md   # 月报
├── envs/               # URDF/MJCF 模型
│   └── simple_biped.urdf
├── models/             # 训练模型权重
├── scripts/            # 毕设核心脚本
├── robot_sdk/          # ★ 机器人 SDK
│   ├── unitree/
│   │   ├── unitree_sdk2_python/   # Python SDK (DDS 通信)
│   │   ├── unitree_mujoco/        # MuJoCo 仿真 + G1 MJCF 模型
│   │   └── unitree_rl_mjlab/      # RL 训练框架
│   └── limx/
│       ├── tron1-mujoco-sim/      # TRON1 MuJoCo 仿真
│       └── tron1-rl-deploy-python/ # TRON1 RL 部署
├── train_balance.py    # PPO 基线训练
└── requirements.txt    # 依赖清单
```

---

# 附录：常见问题 & 故障排查

## Q1: Windows vs Linux 能做什么？

| 操作 | Windows | Linux |
|------|:---:|:---:|
| MuJoCo 仿真 G1/TRON1 | ✅ | ✅ |
| 编写/调试 RL 代码 | ✅ | ✅ |
| URDF/MJCF 验证 | ✅ | ✅ |
| G1 实体高层控制 | ❌ | ✅ |
| G1 实体底层关节控制 | ❌ | ✅ |
| TRON1 SDK (limxsdk) | ❌ | ✅ |
| Sim2Real 部署 | ❌ | ✅ |

## Q2: G1 ping 不通怎么办？

```bash
# 1. 确认网线物理连接
# 2. 确认网卡名正确
ip link show

# 3. 确认 IP 配置
ip addr show enp2s0  # 替换为你的网卡名
# 应看到 192.168.123.222/24

# 4. 如果没有 IP，重新配置
sudo ip addr add 192.168.123.222/24 dev enp2s0
sudo ip link set enp2s0 up

# 5. 确认机器人已开机
ping 192.168.123.161
```

## Q3: CycloneDDS 编译失败？

```bash
# 确保依赖完整
sudo apt install -y cmake gcc g++ python3-dev

# 检查 cmake 版本 >= 3.16
cmake --version

# 如果版本太低
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:kitware/ppa
sudo apt update && sudo apt install -y cmake
```

## Q4: MuJoCo viewer 打不开（Linux 无图形界面）？

```bash
# 方案 1：通过 XRDP 远程桌面连接（能看 GUI）
sudo apt install -y xrdp xfce4
sudo systemctl enable xrdp

# 方案 2：用 X11 转发（需要 Windows 装 X server）
# Windows 安装 VcXsrv，然后：
ssh -X yourname@192.168.1.100
# 此时运行 MuJoCo viewer 窗口会转发到 Windows 上

# 方案 3：离屏渲染（不用 GUI，保存视频）
python -c "
import mujoco
import mediapy
model = mujoco.MjModel.from_xml_path('scene_29dof.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)
frames = []
for _ in range(100):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    frames.append(renderer.render())
mediapy.write_video('g1_sim.mp4', frames, fps=30)
"
```

## Q5: G1 和 TRON1 核心区别？

| 特性 | 宇树 G1 | LimX TRON1 |
|------|---------|------------|
| 类型 | 全尺寸人形 | 模块化双足 |
| DoF | 23/29 | 取决于足端模式 |
| SDK 语言 | C++/Python (DDS) | C++/Python |
| RL 框架 | unitree_rl_mjlab (MuJoCo) | Isaac Gym / Isaac Lab |
| Sim2Real | policy.onnx → C++ 部署 | Python 端直接部署 |
| 开发 OS | Ubuntu 20.04/22.04 | Ubuntu (ROS2 Iron) |
| Windows 仿真 | ✅ MuJoCo | ⚠️ 仅查看模型 |
