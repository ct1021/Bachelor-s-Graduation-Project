"""
G1 双足机器人强化学习可视化控制台 — FastAPI 后端
=================================================
功能:
  1. /api/status         — 读取远程训练日志, 返回最新指标
  2. /api/inference       — 加载 PPO 模型, MuJoCo 推理 500 步, 返回关节数据+视频
  3. /api/experiments     — 返回历史实验对比数据
  4. /ws/live-training    — WebSocket 实时推送训练日志
  
简历包装关键词: FastAPI · WebSocket · MuJoCo · Stable-Baselines3 · Sim2Real
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "g1_ppo"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

# 确保项目根目录在路径中
sys.path.insert(0, str(PROJECT_ROOT))

# ── FastAPI App ──
app = FastAPI(
    title="G1 Biped Robot RL Dashboard",
    description="基于模仿学习与强化学习融合的双足机器人步态控制 — 可视化控制台",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静态文件 ──
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")


# ═══════════════════════════════════════════════════
# REST API
# ═══════════════════════════════════════════════════

@app.get("/")
async def index():
    """主页"""
    return FileResponse(str(DASHBOARD_DIR / "static" / "index.html"))


@app.get("/api/status")
async def training_status():
    """读取最新训练日志, 解析关键指标"""
    log_files = [
        PROJECT_ROOT / "train_g1_ppo_sprint.log",
        LOGS_DIR / "PPO_1" / "progress.csv",
    ]
    
    latest_metrics = {
        "total_timesteps": 0,
        "ep_rew_mean": 0,
        "fps": 0,
        "approx_kl": 0,
        "iterations": 0,
        "time_elapsed": 0,
        "entropy_loss": 0,
        "value_loss": 0,
        "policy_gradient_loss": 0,
        "is_training": False,
    }
    
    # 尝试解析最新的训练输出日志
    sprint_log = PROJECT_ROOT / "train_g1_ppo_sprint.log"
    if sprint_log.exists():
        try:
            text = sprint_log.read_text(encoding="utf-8", errors="ignore")
            # 解析 SB3 的表格输出格式
            patterns = {
                "total_timesteps": r"total_timesteps\s*\|\s*([\d.e+\-]+)",
                "ep_rew_mean": r"ep_rew_mean\s*\|\s*([\d.e+\-]+)",
                "fps": r"fps\s*\|\s*([\d.e+\-]+)",
                "approx_kl": r"approx_kl\s*\|\s*([\d.e+\-]+)",
                "iterations": r"iterations\s*\|\s*([\d.e+\-]+)",
                "time_elapsed": r"time_elapsed\s*\|\s*([\d.e+\-]+)",
                "entropy_loss": r"entropy_loss\s*\|\s*([\d.e+\-]+)",
                "value_loss": r"value_loss\s*\|\s*([\d.e+\-]+)",
                "policy_gradient_loss": r"policy_gradient_loss\s*\|\s*([\d.e+\-]+)",
            }
            
            for key, pattern in patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    latest_metrics[key] = float(matches[-1])
            
            latest_metrics["is_training"] = True
        except Exception as e:
            latest_metrics["error"] = str(e)
    
    return JSONResponse(latest_metrics)


@app.get("/api/training-history")
async def training_history():
    """返回训练过程的完整时序数据, 用于前端绘制曲线"""
    sprint_log = PROJECT_ROOT / "train_g1_ppo_sprint.log"
    history = {"timesteps": [], "rewards": [], "kl": [], "fps": []}
    
    if sprint_log.exists():
        text = sprint_log.read_text(encoding="utf-8", errors="ignore")
        
        # 按分隔线切块
        blocks = text.split("-" * 30)
        for block in blocks:
            ts_match = re.search(r"total_timesteps\s*\|\s*([\d.e+\-]+)", block)
            rew_match = re.search(r"ep_rew_mean\s*\|\s*([\d.e+\-]+)", block)
            kl_match = re.search(r"approx_kl\s*\|\s*([\d.e+\-]+)", block)
            fps_match = re.search(r"fps\s*\|\s*([\d.e+\-]+)", block)
            
            if ts_match and rew_match:
                history["timesteps"].append(float(ts_match.group(1)))
                history["rewards"].append(float(rew_match.group(1)))
                history["kl"].append(float(kl_match.group(1)) if kl_match else 0)
                history["fps"].append(float(fps_match.group(1)) if fps_match else 0)
    
    return JSONResponse(history)


@app.get("/api/experiments")
async def list_experiments():
    """列出所有已保存的模型检查点"""
    experiments = []
    
    # 扫描 models 目录
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.glob("*.zip")):
            experiments.append({
                "name": f.stem,
                "path": str(f),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
            })
    
    # 扫描 checkpoints
    ckpt_dir = MODELS_DIR / "checkpoints"
    if ckpt_dir.exists():
        for f in sorted(ckpt_dir.glob("*.zip"))[-5:]:  # 最近5个
            experiments.append({
                "name": f"checkpoint/{f.stem}",
                "path": str(f),
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
            })
    
    return JSONResponse({"experiments": experiments})


@app.post("/api/inference")
async def run_inference(model_name: str = "g1_ppo_500k"):
    """加载指定模型, 在 MuJoCo 中推理 500 步, 返回关节轨迹数据"""
    model_path = MODELS_DIR / f"{model_name}.zip"
    if not model_path.exists():
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            return JSONResponse({"error": f"Model not found: {model_name}"}, status_code=404)
    
    try:
        from stable_baselines3 import PPO
    except ImportError:
        return JSONResponse({
            "error": "stable_baselines3 未安装。请切换到 biped_rl 虚拟环境: conda activate biped_rl && pip install stable-baselines3"
        }, status_code=503)
    
    try:
        from envs.g1_env import G1WalkEnv
    except ImportError as e:
        return JSONResponse({
            "error": f"G1 环境加载失败: {e}. 请确保 mujoco 和 envs 模块可用。"
        }, status_code=503)
    
    try:
        model = PPO.load(str(model_path))
        env = G1WalkEnv(render_mode=None)
        obs, _ = env.reset()
        
        trajectory = {
            "timesteps": [],
            "rewards": [],
            "root_z": [],
            "root_vx": [],
            "joint_positions": [],  # 每步29维
            "total_reward": 0,
            "total_steps": 0,
            "terminated_reason": "max_steps",
        }
        
        total_reward = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            trajectory["timesteps"].append(step)
            trajectory["rewards"].append(float(reward))
            trajectory["root_z"].append(float(obs[0]))  # Torso Z
            trajectory["root_vx"].append(float(obs[1]))  # Torso VX
            trajectory["joint_positions"].append([float(x) for x in obs[3:15]])  # 前12个腿部关节
            
            if terminated:
                trajectory["terminated_reason"] = "fallen"
                break
            if truncated:
                trajectory["terminated_reason"] = "timeout"
                break
        
        trajectory["total_reward"] = float(total_reward)
        trajectory["total_steps"] = step + 1
        
        env.close()
        return JSONResponse(trajectory)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════
# WebSocket 实时推送
# ═══════════════════════════════════════════════════

@app.websocket("/ws/live-training")
async def ws_live_training(websocket: WebSocket):
    """WebSocket: 每 2 秒推送一次最新训练指标"""
    await websocket.accept()
    
    sprint_log = PROJECT_ROOT / "train_g1_ppo_sprint.log"
    last_size = 0
    
    try:
        while True:
            if sprint_log.exists():
                current_size = sprint_log.stat().st_size
                if current_size != last_size:
                    last_size = current_size
                    # 解析最新指标
                    text = sprint_log.read_text(encoding="utf-8", errors="ignore")
                    metrics = {}
                    patterns = {
                        "total_timesteps": r"total_timesteps\s*\|\s*([\d.e+\-]+)",
                        "ep_rew_mean": r"ep_rew_mean\s*\|\s*([\d.e+\-]+)",
                        "fps": r"fps\s*\|\s*([\d.e+\-]+)",
                        "approx_kl": r"approx_kl\s*\|\s*([\d.e+\-]+)",
                        "iterations": r"iterations\s*\|\s*([\d.e+\-]+)",
                    }
                    for key, pattern in patterns.items():
                        matches = re.findall(pattern, text)
                        if matches:
                            metrics[key] = float(matches[-1])
                    
                    await websocket.send_json(metrics)
            
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


# ═══════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print(f"\n{'='*60}")
    print(f"  G1 Biped Robot RL Dashboard")
    print(f"  Open: http://localhost:8000")
    print(f"{'='*60}\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
