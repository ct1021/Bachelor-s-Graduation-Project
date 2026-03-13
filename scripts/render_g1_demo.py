"""
render_g1_demo.py — G1 MuJoCo Off-screen Rendering Demo
========================================================
加载已有的 G1 MJCF 模型（29-DoF），并读取开源动作捕捉数据（dance1_subject2.csv），
通过 MuJoCo 离屏渲染器导出 MP4 视频，展示真实的动作跟踪效果。

用法:
    python scripts/render_g1_demo.py --duration 10
"""

import argparse
import os
import sys
import numpy as np

import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_mocap_data(csv_path):
    """
    读取 unitree_rl_mjlab 中的动作捕捉 CSV 数据。
    CSV 包含 36 列:
      [0:3]   - Base Position (x, y, z)
      [3:7]   - Base Rotation (x, y, z, w)
      [7:36]  - 29个关节的位角 (Joint Positions)
    """
    print(f"[Data] Loading MoCap data from: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=",")
    
    base_pos = data[:, 0:3]
    base_rot = data[:, 3:7]
    joint_pos = data[:, 7:36]
    
    return base_pos, base_rot, joint_pos


def render_video(model_path, csv_path, output_path, duration=10.0, fps=50, width=640, height=480):
    try:
        import imageio
    except ImportError:
        print("[ERROR] Please install imageio: pip install imageio imageio-ffmpeg")
        sys.exit(1)
        
    print(f"[Model] Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 获取数据
    base_pos, base_rot, joint_pos = load_mocap_data(csv_path)
    total_frames = joint_pos.shape[0]
    
    # 确保控制数量和数据匹配
    nu = model.nu
    data_nu = joint_pos.shape[1]
    
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = 2.5
    camera.elevation = -15
    camera.azimuth = 135
    
    frames = []
    print("[Render] Starting off-screen rendering...")
    mujoco.mj_resetData(model, data)
    
    dt = model.opt.timestep
    steps_per_frame = int(1.0 / (fps * dt))
    
    render_frames = min(total_frames, int(duration * fps))
    
    for frame_idx in range(render_frames):
        # 强制设置根节点状态 (以便完美呈现参考运动，不依赖反馈控制)
        if model.nq >= 7:
            data.qpos[0:3] = base_pos[frame_idx]
            # CSV 的四元数是 (x,y,z,w)，MuJoCo 需要 (w,x,y,z)
            q_xyzw = base_rot[frame_idx]
            data.qpos[3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        
        # 设置关节（如果是29自由度，精准匹配）
        if model.nq >= 7 + min(nu, data_nu):
            # 将关节位置硬写入以便纯属渲染，或者作为前馈控制写入 ctrl 
            # 这里为了演示流畅，直接作为PD控制目标写入 ctrl (由于没有设控制刚度，也可以直接修改 qpos)
            data.ctrl[:min(nu, data_nu)] = joint_pos[frame_idx, :min(nu, data_nu)]
            data.qpos[7:7+min(nu, data_nu)] = joint_pos[frame_idx, :min(nu, data_nu)]
            
        mujoco.mj_forward(model, data) # 更新运动学
        
        # 渲染
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
        
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{render_frames} frames...")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    print(f"\n[Done] Video saved: {output_path} ({len(frames)} frames)")
    renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    
    # 强制使用已下载的原生 SDK 文件
    model_path = os.path.join(PROJECT_ROOT, "robot_sdk", "unitree", "unitree_mujoco", "unitree_robots", "g1", "scene_29dof.xml")
    csv_path = os.path.join(PROJECT_ROOT, "robot_sdk", "unitree", "unitree_rl_mjlab", "src", "assets", "motions", "g1", "dance1_subject2.csv")
    output_path = os.path.join(PROJECT_ROOT, "docs", "figures", "g1_open_source_demo.mp4")
    
    if not os.path.exists(model_path):
        print(f"[Error] Cannot find model at {model_path}. Please check directory structure.")
        return
        
    if not os.path.exists(csv_path):
        print(f"[Error] Cannot find MoCap data at {csv_path}. Please check directory structure.")
        return
        
    render_video(model_path, csv_path, output_path, duration=args.duration)

if __name__ == "__main__":
    main()
