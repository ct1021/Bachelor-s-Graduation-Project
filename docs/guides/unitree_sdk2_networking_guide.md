# Unitree SDK 2 联调与 CycloneDDS 组网格式指南 (G1)

进行 Sim2Real 实机部署前，第一步是确保 Linux 工作站能够与 G1 原厂开发板（Raspberry Pi / Jetson）进行高频实时稳定通信。Unitree SDK 2 摒弃了 UDP，全面拥抱了更强鲁棒性的 **CycloneDDS (Data Distribution Service)** 技术。

## 1. 物理拓扑要求
- **直连 / 交换机推荐**：请准备一根千兆以太网线，将 Linux 工作站网口连接到交换机，再从交换机分出一根网线连接到 G1 机器人背部 / 顶部的调试网口。
- **无线警告**：绝对不要在强化学习 Sim2Real 测试中使用 Wi-Fi 传算法，无线的高丢包率和延迟波动会导致机器人产生不可预知的危险抽搐。

## 2. IP 地址配置 (IPv4)
Unitree 官方出厂配置通常要求处于 `192.168.123.xxx` 网段。

* 假设 G1 机器人的 IP 为 `192.168.123.161`（具体请查阅随车手册，或使用 Nmap 扫网段记录确认设备名）。
* 你必须将**你的 Linux 工作站本地网卡**强制配置一个同网段的静态 IP，比如 `192.168.123.222`，子网掩码 `255.255.255.0`。

```bash
# 在 Linux 终端上临时修改以太网卡 IP (假设网卡名为 eth0 / enp3s0)
sudo ifconfig enp3s0 192.168.123.222 netmask 255.255.255.0
```
验证通信通道是否打通：
```bash
ping 192.168.123.161
```
*(要求延迟必须长期稳定在 1ms 以下)*

## 3. CycloneDDS 组网格式与环境变量
SDK2 依赖于 CycloneDDS 在局域网自动发现节点。如果你的电脑有多块网卡（比如同时连着校园网 WiFi 和连接 G1 的直连网线），DDS 很可能会走错网卡导致找不到机器人。

**强制绑定网卡（组网核心格式）**：
需将你用于连接机器人的有线网卡名称（如 `enp3s0`）通过 XML 组网格式写入 Linux 环境变量中。

```bash
# 替换 enp3s0 为你实际的网卡名称
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enp3s0"/></Interfaces></General></Domain></CycloneDDS>'
```
*提示：建议将这行配置写入 `~/.bashrc`，避免每次开终端都要重新输入。*

## 4. 验证 SDK 2 通信
在 Linux 安装 Unitree SDK2 的 Python 绑定：
```bash
pip install unitree_sdk2py
```

执行一段极其简单的心跳包测试代码，验证是否能读到机器人的状态：
```python
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.sport.sport_client import SportClient

# 初始化 DDS 通道 (会自动根据 CYCLONEDDS_URI 发现网卡)
ChannelFactoryInitialize(0, "enp3s0") # 如果上一步环境变量未起效可以尝试手动指定

client = SportClient()
client.SetTimeout(10.0)
client.Init()

# 尝试获取机器人基本状态
while True:
    print("等待获取 G1 状态...")
    time.sleep(1)
```

## 下一步 (Sim2Real 衔接)
一旦 SDK 能读取状态，你就可以通过调用 `client.Move(vx, vy, vyaw)` (高级指令模式) 或直接下发 29 个关节力矩 (底层模式) 来让这 50 完步的 PPO 权重真正跑在钢铁上！但在那之前，**请务必确保已按照本周的安排将灵巧手拆卸完毕。**
