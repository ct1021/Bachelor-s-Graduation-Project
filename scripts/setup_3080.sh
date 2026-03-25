#!/bin/bash
# ============================================================
# setup_3080.sh — RTX 3080 机器一键环境配置
# ============================================================
# 用法: ssh robotics@100.74.55.63
#       bash setup_3080.sh
#
# 注意: 安装 NVIDIA 驱动后需要重启，脚本会提示
# ============================================================

set -e  # 遇到错误立即停止

echo "============================================================"
echo "  RTX 3080 环境配置脚本"
echo "  $(date)"
echo "============================================================"

# ============================================================
# 0-1. APT 换清华源
# ============================================================
echo ""
echo "[1/7] APT 换清华源..."
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
sudo sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
sudo apt update && sudo apt upgrade -y
echo "  ✅ APT 源已切换到清华镜像"

# ============================================================
# 0-2. 安装系统依赖
# ============================================================
echo ""
echo "[2/7] 安装系统依赖..."
sudo apt install -y build-essential cmake curl vim htop tmux \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev \
  git wget unzip
echo "  ✅ 系统依赖安装完成"

# ============================================================
# 0-3. 安装 NVIDIA 驱动
# ============================================================
echo ""
echo "[3/7] 安装 NVIDIA 驱动..."
if nvidia-smi &>/dev/null; then
    echo "  ✅ NVIDIA 驱动已安装:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "  ⚠️  NVIDIA 驱动未安装，开始安装..."
    sudo apt install -y ubuntu-drivers-common
    ubuntu-drivers devices
    sudo ubuntu-drivers autoinstall
    echo ""
    echo "  ============================================"
    echo "  ⚠️  驱动安装完成！必须重启！"
    echo "  请执行: sudo reboot"
    echo "  重启后再次运行本脚本继续配置"
    echo "  ============================================"
    exit 0
fi

# ============================================================
# 0-4. pip/conda 换清华源
# ============================================================
echo ""
echo "[4/7] pip/conda 换清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 2>/dev/null || true
conda config --set show_channel_urls yes 2>/dev/null || true
echo "  ✅ pip/conda 源已配置"

# ============================================================
# 0-5. 克隆项目
# ============================================================
echo ""
echo "[5/7] 克隆项目..."
cd ~
if [ -d "Bachelor-s-Graduation-Project" ]; then
    echo "  项目目录已存在，执行 git pull..."
    cd Bachelor-s-Graduation-Project && git pull
else
    git clone https://github.com/ct1021/Bachelor-s-Graduation-Project.git
    cd Bachelor-s-Graduation-Project
fi
echo "  ✅ 项目就绪: $(pwd)"

# ============================================================
# 0-6. 创建 biped_rl 环境
# ============================================================
echo ""
echo "[6/7] 创建 biped_rl conda 环境..."
if conda env list | grep -q "biped_rl"; then
    echo "  biped_rl 环境已存在，跳过创建"
else
    conda create -n biped_rl python=3.10 -y
fi

# 安装依赖（在子 shell 中激活环境）
eval "$(conda shell.bash hook)"
conda activate biped_rl

pip install stable-baselines3[extra] mujoco tensorboard imageio pandas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo "  ✅ biped_rl 环境就绪"

# ============================================================
# 0-7. 生成步态参考数据
# ============================================================
echo ""
echo "[7/7] 生成步态参考数据..."
cd ~/Bachelor-s-Graduation-Project
python scripts/retarget_cmu_to_g1.py
echo "  ✅ 步态数据生成完成"

# ============================================================
# 验证
# ============================================================
echo ""
echo "============================================================"
echo "  环境验证"
echo "============================================================"
echo -n "  nvidia-smi: "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -n "  Python:     " && python --version
echo -n "  MuJoCo:     " && python -c "import mujoco; print('OK')"
echo -n "  PyTorch:    " && python -c "import torch; print(f'OK, CUDA={torch.cuda.is_available()}')"
echo -n "  SB3:        " && python -c "import stable_baselines3; print('OK')"
echo ""
echo "============================================================"
echo "  ✅ 3080 环境配置完成！"
echo ""
echo "  下一步:"
echo "    1. 运行 BC 预训练:    python scripts/behavioral_cloning.py"
echo "    2. BC→RL 训练:        python scripts/train_g1_ppo.py --bc-pretrain models/bc_pretrained.pt --timesteps 30000000 --num-envs 8"
echo "    3. 100m 评测:         python scripts/evaluate_100m_walk.py"
echo "============================================================"
