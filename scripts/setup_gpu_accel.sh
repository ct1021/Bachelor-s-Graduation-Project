#!/bin/bash
# ============================================================
# setup_gpu_accel.sh — GPU 加速训练环境安装
# ============================================================
# Phase 1: Isaac Lab + unitree_rl_lab（首选）
# 备选:    MuJoCo MJX + JAX（如果 Isaac Lab 装不上）
#
# 前置: setup_3080.sh 已完成（包括 NVIDIA 驱动）
#
# 用法: ssh robotics@100.74.55.63 (或 ct@100.117.36.59)
#       cd ~/Bachelor-s-Graduation-Project
#       bash scripts/setup_gpu_accel.sh
# ============================================================

set -e

echo "============================================================"
echo "  GPU 加速训练环境安装"
echo "  $(date)"
echo "============================================================"

# 检查 NVIDIA 驱动
if ! nvidia-smi &>/dev/null; then
    echo "❌ NVIDIA 驱动未安装！请先运行 setup_3080.sh"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME ($GPU_MEM)"

# ============================================================
# 方案 A: Isaac Lab（首选）
# ============================================================
echo ""
echo "=== 方案 A: 尝试安装 Isaac Lab ==="
echo ""

ISAAC_OK=false

# 创建独立 conda 环境
if ! conda env list | grep -q "isaaclab"; then
    echo "[1/4] 创建 isaaclab conda 环境..."
    conda create -n isaaclab python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate isaaclab

# 安装 PyTorch (CUDA 12.1)
echo "[2/4] 安装 PyTorch (cu121)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 检查磁盘空间（Isaac Lab + Isaac Sim 需要大量空间）
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -d ' G')
echo "  可用磁盘: ${AVAIL_GB} GB"

if [ "$AVAIL_GB" -lt 30 ]; then
    echo "  ⚠️ 磁盘空间不足 30GB，Isaac Lab (含 Isaac Sim) 可能装不下"
    echo "  → 自动切换到方案 B (MJX)"
else
    echo "[3/4] 克隆 Isaac Lab..."
    cd ~
    if [ ! -d "IsaacLab" ]; then
        git clone https://github.com/isaac-sim/IsaacLab.git
    fi

    echo "[4/4] 安装 Isaac Lab..."
    cd IsaacLab

    # 尝试安装（可能失败，因为需要 Isaac Sim/Omniverse）
    if pip install -e "source/extensions/omni.isaac.lab" 2>/dev/null; then
        echo "  ✅ Isaac Lab 安装成功！"

        # 安装 unitree_rl_lab
        cd ~
        if [ ! -d "unitree_rl_lab" ]; then
            git clone https://github.com/unitreerobotics/unitree_rl_lab.git
        fi
        cd unitree_rl_lab
        if [ -f "./unitree_rl_lab.sh" ]; then
            chmod +x unitree_rl_lab.sh
            ./unitree_rl_lab.sh -i
        else
            pip install -e .
        fi

        # 安装 unitree_ros (URDF/MJCF 资产)
        cd ~
        if [ ! -d "unitree_ros" ]; then
            git clone https://github.com/unitreerobotics/unitree_ros.git
        fi

        # 验证
        echo ""
        echo "  验证 Isaac Lab G1 环境..."
        if ./unitree_rl_lab.sh -l 2>/dev/null | grep -q "G1"; then
            echo "  ✅ 找到 Unitree-G1 环境！"
            ISAAC_OK=true
        else
            echo "  ⚠️ 未找到 G1 环境，请检查 unitree_rl_lab 配置"
        fi
    else
        echo "  ❌ Isaac Lab 安装失败（可能缺少 Isaac Sim/Omniverse）"
        echo "  → 自动切换到方案 B (MJX)"
    fi
fi

# ============================================================
# 方案 B: MuJoCo MJX + JAX（备选）
# ============================================================
if [ "$ISAAC_OK" = false ]; then
    echo ""
    echo "=== 方案 B: 安装 MuJoCo MJX + JAX ==="
    echo ""

    # 使用 biped_rl 环境（已有 MuJoCo）
    conda activate biped_rl

    echo "[1/3] 安装 JAX (CUDA 12)..."
    pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    echo "[2/3] 安装 MuJoCo MJX..."
    pip install mujoco-mjx

    echo "[3/3] 验证 MJX..."
    python -c "
import jax
import mujoco
from mujoco import mjx
print(f'JAX devices: {jax.devices()}')
print(f'MuJoCo version: {mujoco.__version__}')
print('MJX import: OK')
print()

# 尝试加载 G1 模型到 MJX
import os
model_path = os.path.expanduser('~/Bachelor-s-Graduation-Project/robot_sdk/unitree/unitree_mujoco/unitree_robots/g1/scene_29dof.xml')
if os.path.exists(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(model)
    print(f'G1 MJX model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}')
    print('✅ MJX GPU 加速就绪！')
else:
    print(f'⚠️ G1 模型未找到: {model_path}')
"

    echo ""
    echo "  ============================================"
    echo "  MJX 方案说明:"
    echo "  - GPU 并行仿真: 可达 100K+ FPS"
    echo "  - 需要编写 JAX 并行环境包装器"
    echo "  - 直接复用现有的 g1_29dof.xml"
    echo "  - 参见: scripts/mjx_g1_env.py (待编写)"
    echo "  ============================================"
fi

# ============================================================
# 最终状态汇报
# ============================================================
echo ""
echo "============================================================"
echo "  GPU 加速环境安装完成"
echo "============================================================"
if [ "$ISAAC_OK" = true ]; then
    echo "  方案: Isaac Lab + unitree_rl_lab ✅"
    echo "  训练命令:"
    echo "    conda activate isaaclab"
    echo "    python -m unitree_rl_lab.train --task Unitree-G1-29dof-Velocity --num_envs 2048"
else
    echo "  方案: MuJoCo MJX + JAX ✅"
    echo "  训练命令:"
    echo "    conda activate biped_rl"
    echo "    python scripts/train_g1_mjx.py --num-envs 2048 --bc-pretrain models/bc_pretrained.pt"
    echo ""
    echo "  ⚠️ 需要编写 scripts/train_g1_mjx.py (MJX GPU 并行训练脚本)"
fi
echo "============================================================"
