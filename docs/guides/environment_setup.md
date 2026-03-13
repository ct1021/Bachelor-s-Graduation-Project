# 环境配置说明

## Python 虚拟环境

本项目使用 Conda 虚拟环境进行管理。

### 激活环境

```bash
conda activate biped_rl
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
gradually-work/
├── data/          # 数据存储目录
├── docs/          # 项目文档
├── envs/          # 环境配置文件
├── scripts/       # 脚本文件
├── models/        # 模型文件
├── logs/          # 日志文件
└── requirements.txt  # Python依赖包
```

## 注意事项

- 在运行任何 Python 脚本前，请确保已激活 `biped_rl` 环境
- 所有新的依赖包都应该添加到 `requirements.txt` 中
