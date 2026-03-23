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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}  # 100Hz

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.dt = 0.01  # 控制频率 100Hz（任务书要求 ≥100Hz）
        
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
        # [root_z(1), vx,vy,vz(3), roll,pitch(2), wx,wy,wz(3)] = 9
        # [Joint Positions(29), Joint Velocities(29)] = 58
        # Total = 67
        obs_dim = 9 + (self.model.nq - 7) + (self.model.nv - 6)  # 9 + 29 + 29 = 67
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
            print(f"[Warning] MoCap reference not found: {csv_path}.")
            print("[Warning] Falling back to sinusoidal walking reference (avoids zero-target IL collapse).")
            # 用简单正弦波生成伪参考步态，避免纯零目标导致 IL 奖励项完全失效
            # 频率约 1Hz（双脚轮换），幅度 0.1 rad（约 5.7°，合理的弯腿幅度）
            T = 1000  # 参考序列长度
            t = np.linspace(0, 2 * np.pi * 5, T)  # 5个步态周期
            self.ref_joint_pos = np.zeros((T, 29))
            # 左髋俯仰(0)、左膝(3)：相位 0°
            self.ref_joint_pos[:, 0] = 0.1 * np.sin(t)        # 左髋俯仰
            self.ref_joint_pos[:, 3] = -0.15 * np.abs(np.sin(t))  # 左膝（始终弯曲）
            # 右髋俯仰(6)、右膝(9)：相位 180°（交替迈步）
            self.ref_joint_pos[:, 6] = 0.1 * np.sin(t + np.pi) # 右髋俯仰
            self.ref_joint_pos[:, 9] = -0.15 * np.abs(np.sin(t + np.pi))  # 右膝
            self.ref_base_pos = np.zeros((T, 3))
            self.ref_length = T
            return
            
        data = np.loadtxt(csv_path, delimiter=",")
        self.ref_base_pos = data[:, 0:3]
        self.ref_joint_pos = data[:, 7:36]
        self.ref_length = data.shape[0]

    def _get_obs(self):
        # === 根部状态（9维）===
        root_z = self.data.qpos[2:3]             # 质心高度 (1)
        root_vel = self.data.qvel[0:3]            # vx, vy, vz (3)
        root_angvel = self.data.qvel[3:6]         # wx, wy, wz (3) ← 新增！
        
        # 姿态角：四元数 → roll, pitch ← 新增！这是平衡的关键感知
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        orientation = np.array([roll, pitch])     # (2)
        
        # === 关节状态（58维）===
        joint_pos = self.data.qpos[7:]            # (29)
        joint_vel = self.data.qvel[6:]            # (29)
        
        obs = np.concatenate([
            root_z, root_vel, orientation, root_angvel,
            joint_pos, joint_vel
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
        # 只使用 30% 最大力矩，防止暴力输出导致失稳
        scaled_action = 0.3 * action * torque_limit
        
        self.data.ctrl[:] = scaled_action
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        self.sim_step += 1
        # 参考轨迹循环播放（mod 防止越界）
        self.ref_step = self.sim_step % self.ref_length
        
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # 终止条件判定：摔倒
        # G1 站立高度约 0.78m，弯腿约 0.6m，低于 0.55m 视为已摔倒
        root_z = self.data.qpos[2]
        terminated = bool(root_z < 0.55)
        
        # 摔倒惩罚：让策略学到"摔倒是非常昂贵的"
        if terminated:
            reward -= 50.0
        
        # 每个 Episode 最多 1000步 × 0.01s = 10s（训练用短Episode，评测用evaluate_100m_walk.py）
        truncated = bool(self.sim_step >= 1000)
        
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self):
        # 1. Imitation Learning Reward (Joint Tracking)
        current_joint_pos = self.data.qpos[7:7+29]
        target_joint_pos = self.ref_joint_pos[self.ref_step]
        pos_error = np.sum(np.square(current_joint_pos[:12] - target_joint_pos[:12]))
        tracking_reward = np.exp(-5.0 * pos_error)
        
        # 2. Velocity Tracking Reward
        vx = self.data.qvel[0]
        vel_reward = np.exp(-2.0 * (vx - 0.5)**2)
        
        # 3. Upright Posture Reward（高度接近 0.78m）
        root_z = self.data.qpos[2]
        height_reward = np.exp(-10.0 * (root_z - 0.78) ** 2)
        survival_reward = 1.0 + height_reward  # 最大2.0
        
        # 4. 姿态保持奖励 ← 新增！直接奖励保持直立
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        # 直立时 roll≈0, pitch≈0 → orientation_reward≈1.0
        orientation_reward = np.exp(-5.0 * (roll**2 + pitch**2))
        
        # 5. 角速度惩罚 ← 新增！惩罚快速旋转（摔倒前兆）
        angvel = self.data.qvel[3:6]
        angvel_penalty = -0.1 * np.sum(np.square(angvel))
        
        # 6. Alive Bonus
        alive_bonus = 0.5
        
        # 7. Energy Penalty（力矩已缩至30%，系数调回 0.00005）
        energy_penalty = -0.00005 * np.sum(np.square(self.data.ctrl))
        
        # 权重：存活+姿态(60%) >> 跟踪(10%) + 速度(10%) + 姿态角(20%)
        total_reward = (
            0.1 * tracking_reward
            + 0.1 * vel_reward
            + 0.3 * survival_reward
            + 0.2 * orientation_reward
            + alive_bonus
            + angvel_penalty
            + energy_penalty
        )
        return total_reward

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, self.camera)
            return self.renderer.render()
        return None

    def close(self):
        if self.render_mode == "rgb_array":
            self.renderer.close()
