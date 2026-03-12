"""
G1WalkEnv — Gymnasium Environment for Unitree G1 with Imitation Learning
========================================================================

包含以下核心功能:
1. 加载 G1 的 29 自由度模型 (来自于 robot_sdk)
2. 读取和插值 MoCap 参考动作 (dance1_subject2.csv)
3. 融合强化学习强化 (Velocity Tracking) 和 模仿学习强化 (Joint Tracking)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class G1WalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.dt = 0.02  # 控制频率 50Hz
        
        # 加载 模型
        model_path = os.path.join(PROJECT_ROOT, "robot_sdk", "unitree", "unitree_mujoco", "unitree_robots", "g1", "scene_29dof.xml")
        csv_path = os.path.join(PROJECT_ROOT, "robot_sdk", "unitree", "unitree_rl_mjlab", "src", "assets", "motions", "g1", "dance1_subject2.csv")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"G1 MJCF not found at: {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.002 # physics freq 500Hz
        self.frame_skip = int(self.dt / self.model.opt.timestep) 
        
        self.data = mujoco.MjData(self.model)
        
        # 行动空间 (直接通过力矩控制 29 个关节)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        
        # 状态空间:
        # [Torso Z, Torso VX, Torso VY] (3)
        # [Joint Positions] (29)
        # [Joint Velocities] (29)
        obs_dim = 3 + self.model.nq - 7 + self.model.nv - 6  # 3 + 29 + 29 = 61
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._load_reference_motion(csv_path)

        # 视频渲染
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 1
            self.camera.distance = 2.5
            self.camera.elevation = -15
            self.camera.azimuth = 135

    def _load_reference_motion(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"[Warning] MoCap reference not found: {csv_path}. Using zero targets.")
            self.ref_base_pos = np.zeros((100, 3))
            self.ref_joint_pos = np.zeros((100, 29))
            self.ref_length = 100
            return
            
        data = np.loadtxt(csv_path, delimiter=",")
        self.ref_base_pos = data[:, 0:3]
        self.ref_joint_pos = data[:, 7:36]
        self.ref_length = data.shape[0]

    def _get_obs(self):
        root_z = self.data.qpos[2]
        root_vx = self.data.qvel[0]
        root_vy = self.data.qvel[1]
        
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]
        
        obs = np.concatenate([
            [root_z, root_vx, root_vy],
            joint_pos,
            joint_vel
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.sim_step = 0
        self.ref_step = 0
        
        # 初始化到一个合理的起始姿态（从 MoCap 获取）
        if self.ref_length > 0:
             self.data.qpos[7:7+29] = self.ref_joint_pos[0]
             
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 动作缩放，由于 G1 力矩较大
        action = np.clip(action, -1.0, 1.0)
        torque_limit = self.model.actuator_ctrlrange[:, 1]
        scaled_action = action * torque_limit
        
        self.data.ctrl[:] = scaled_action
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        self.sim_step += 1
        self.ref_step = min(self.sim_step, self.ref_length - 1)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # 终止条件判定：摔倒
        root_z = self.data.qpos[2]
        terminated = bool(root_z < 0.4) 
        
        # 超过 10 秒
        truncated = bool(self.sim_step >= 500)
        
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self):
        # 1. Imitation Learning Reward (Joint Tracking)
        current_joint_pos = self.data.qpos[7:7+29]
        target_joint_pos = self.ref_joint_pos[self.ref_step]
        
        # 仅追踪前12个关节（腿部）的误差，忽略手臂
        pos_error = np.sum(np.square(current_joint_pos[:12] - target_joint_pos[:12]))
        tracking_reward = np.exp(-5.0 * pos_error)
        
        # 2. Reinforcement Learning Reward (Velocity)
        # 目标速度 0.5m/s (X方向)
        v_target = np.array([0.5, 0.0])
        v_current = self.data.qvel[0:2]
        v_error = np.sum(np.square(v_current - v_target))
        vel_reward = np.exp(-2.0 * v_error)
        
        # 3. Survival
        survival_reward = 1.0
        
        # 4. Energy Penalty 
        energy_penalty = -0.001 * np.sum(np.square(self.data.ctrl))
        
        total_reward = 0.5 * tracking_reward + 0.3 * vel_reward + 0.2 * survival_reward + energy_penalty
        return total_reward

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, self.camera)
            return self.renderer.render()
        return None

    def close(self):
        if self.render_mode == "rgb_array":
            self.renderer.close()
