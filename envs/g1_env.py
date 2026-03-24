"""
G1WalkEnv V5 — Gymnasium Environment for Unitree G1 Bipedal Walking
====================================================================
核心改进 (v5):
1. 仅控制 12 个腿部关节（冻结上肢），大幅降低探索空间
2. PD 位置控制替代原始力矩控制，更稳定更接近真实部署
3. 基于人类生物力学数据的步态参考（非舞蹈！）
4. 速度跟踪为第一权重的奖励函数
5. 完整的平衡感知观测（roll/pitch/角速度）
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class G1WalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # G1 腿部关节索引（MuJoCo actuator 顺序的前 12 个）
    NUM_LEG_JOINTS = 12
    
    # PD 控制器参数
    KP = 50.0   # 位置增益（刚度）
    KD = 2.0    # 速度增益（阻尼）
    ACTION_SCALE = 0.5  # 动作 → 关节角偏移量的缩放 (rad)

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.dt = 0.01  # 控制频率 100Hz

        # 加载 MuJoCo 模型
        model_path = os.path.join(
            PROJECT_ROOT, "robot_sdk", "unitree", "unitree_mujoco",
            "unitree_robots", "g1", "scene_29dof.xml"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"G1 MJCF not found: {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = 0.002  # 物理仿真 500Hz
        self.frame_skip = int(self.dt / self.model.opt.timestep)
        self.data = mujoco.MjData(self.model)

        # 动作空间: 仅 12 个腿部关节的角度偏移量
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.NUM_LEG_JOINTS,), dtype=np.float32
        )
        
        # 观测空间 (39维):
        # [root_z(1), vx,vy,vz(3), roll,pitch(2), wx,wy,wz(3)] = 9 (根部状态)
        # [leg_joint_pos(12), leg_joint_vel(12)] = 24 (腿部关节)
        # [ref_phase_sin, ref_phase_cos(2)] = 2 (步态相位)
        # [last_action(12)] = 12 (上一步动作，用于平滑惩罚) -- 不放obs里，内部存
        # 注意：这里不再包含上肢关节（反正不控制）
        # 但为了兼容检查，保留一个可配置的 obs_dim
        self.obs_dim = 9 + 12 + 12 + 2  # = 35 -- 不含 ref_target
        # 加上参考目标 12 维
        self.obs_dim += 12  # = 47
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # 步态参考数据
        self._load_reference_motion()

        # 默认关节角度（站立姿态，所有关节为 0）
        self.default_leg_pos = np.zeros(self.NUM_LEG_JOINTS)
        
        # 上一步的动作（用于平滑惩罚）
        self.last_action = np.zeros(self.NUM_LEG_JOINTS)

        # 渲染器
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 1
            self.camera.distance = 2.5
            self.camera.elevation = -15
            self.camera.azimuth = 135

    def _load_reference_motion(self):
        """加载步态参考数据（仅12个腿部关节）"""
        ref_path = os.path.join(PROJECT_ROOT, "data", "cmu_walking_reference.npz")
        
        if os.path.exists(ref_path):
            data = np.load(ref_path)
            self.ref_joint_pos = data["joint_positions"]  # (T, 12)
            self.ref_base_pos = data["base_positions"]     # (T, 3)
            self.ref_length = self.ref_joint_pos.shape[0]
            self.ref_cycle_time = float(data.get("cycle_time", 1.0))
            print(f"[G1WalkEnv] ✅ 加载步态参考: {ref_path} ({self.ref_length} 帧)")
        else:
            print(f"[G1WalkEnv] ⚠️ 步态参考未找到: {ref_path}")
            print("[G1WalkEnv] 请先运行: python scripts/retarget_cmu_to_g1.py")
            # 生成简单的后备数据
            T = 1000
            t = np.linspace(0, 2 * np.pi * 10, T)
            self.ref_joint_pos = np.zeros((T, 12))
            self.ref_joint_pos[:, 0] = 0.3 * np.sin(t)        # 左髋俯仰
            self.ref_joint_pos[:, 3] = 0.15 + 0.4 * np.maximum(0, np.sin(t))  # 左膝
            self.ref_joint_pos[:, 6] = 0.3 * np.sin(t + np.pi) # 右髋俯仰
            self.ref_joint_pos[:, 9] = 0.15 + 0.4 * np.maximum(0, np.sin(t + np.pi))  # 右膝
            self.ref_base_pos = np.zeros((T, 3))
            self.ref_length = T
            self.ref_cycle_time = 1.0

    def _get_obs(self):
        # === 根部状态 (9维) ===
        root_z = self.data.qpos[2:3]              # 质心高度 (1)
        root_vel = self.data.qvel[0:3]             # vx, vy, vz (3)
        root_angvel = self.data.qvel[3:6]          # wx, wy, wz (3)
        
        # 姿态角: 四元数 → roll, pitch
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        orientation = np.array([roll, pitch])      # (2)
        
        # === 腿部关节状态 (24维) ===
        leg_pos = self.data.qpos[7:7+12]           # (12)
        leg_vel = self.data.qvel[6:6+12]           # (12)
        
        # === 步态相位 (2维) ===
        phase = (self.sim_step % (self.ref_cycle_time * 100)) / (self.ref_cycle_time * 100)
        phase_obs = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        
        # === 参考目标 (12维) ===
        ref_target = self.ref_joint_pos[self.ref_step]
        
        obs = np.concatenate([
            root_z, root_vel, orientation, root_angvel,   # 9
            leg_pos, leg_vel,                              # 24
            phase_obs,                                     # 2
            ref_target,                                    # 12
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.sim_step = 0
        self.ref_step = 0
        self.last_action = np.zeros(self.NUM_LEG_JOINTS)
        
        # 初始化为参考动作的第一帧
        if self.ref_length > 0:
            self.data.qpos[7:7+12] = self.ref_joint_pos[0]
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # === PD 位置控制 ===
        # action ∈ [-1, 1] → 目标角度偏移 ∈ [-0.5, 0.5] rad
        target_pos = self.default_leg_pos + action * self.ACTION_SCALE
        
        # 当前腿部关节状态
        current_pos = self.data.qpos[7:7+12]
        current_vel = self.data.qvel[6:6+12]
        
        # PD 力矩计算
        torque = self.KP * (target_pos - current_pos) - self.KD * current_vel
        
        # 对力矩进行裁剪（不超过执行器限制）
        ctrl_limit = self.model.actuator_ctrlrange[:12, 1]  # 前 12 个执行器
        torque = np.clip(torque, -ctrl_limit, ctrl_limit)
        
        # 仅设置前 12 个执行器（腿部），其余锁定为 0
        self.data.ctrl[:12] = torque
        self.data.ctrl[12:] = 0.0  # 上肢锁定
        
        # 物理仿真推进
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self.sim_step += 1
        self.ref_step = self.sim_step % self.ref_length
        
        # 计算奖励
        obs = self._get_obs()
        reward = self._compute_reward(action)
        
        # 终止条件
        root_z = self.data.qpos[2]
        terminated = bool(root_z < 0.50)  # 低于 0.50m 视为摔倒
        
        # 摔倒惩罚
        if terminated:
            reward -= 50.0
        
        # Episode 最大 1000 步 = 10 秒
        truncated = bool(self.sim_step >= 1000)
        
        # 记录动作用于平滑惩罚
        self.last_action = action.copy()
        
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, action):
        # =====================================================
        # 1. 速度跟踪奖励 (35%) — 第一权重！鼓励前进
        # =====================================================
        vx = self.data.qvel[0]
        vel_reward = np.exp(-4.0 * (vx - 0.5) ** 2)
        
        # =====================================================
        # 2. 步态模仿奖励 (25%) — 基于真实人类步态数据
        # =====================================================
        current_leg_pos = self.data.qpos[7:7+12]
        target_leg_pos = self.ref_joint_pos[self.ref_step]
        imitation_error = np.sum(np.square(current_leg_pos - target_leg_pos))
        imitation_reward = np.exp(-3.0 * imitation_error)
        
        # =====================================================
        # 3. 质心高度奖励 (15%) — 保持站立高度
        # =====================================================
        root_z = self.data.qpos[2]
        height_reward = np.exp(-10.0 * (root_z - 0.78) ** 2)
        
        # =====================================================
        # 4. 直立姿态奖励 (10%) — roll/pitch 接近 0
        # =====================================================
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
        orientation_reward = np.exp(-5.0 * (roll**2 + pitch**2))
        
        # =====================================================
        # 5. 存活奖励 (15%) — 固定每步奖励
        # =====================================================
        alive_bonus = 1.0
        
        # =====================================================
        # 惩罚项
        # =====================================================
        # 动作平滑度惩罚（防止抖动）
        action_diff = action - self.last_action
        smooth_penalty = -0.01 * np.sum(np.square(action_diff))
        
        # 角速度惩罚（防止旋转失稳）
        angvel = self.data.qvel[3:6]
        angvel_penalty = -0.05 * np.sum(np.square(angvel))
        
        # 能耗惩罚（PD 输出的力矩）
        energy_penalty = -0.00005 * np.sum(np.square(self.data.ctrl[:12]))
        
        # 横向速度惩罚（鼓励直线行走）
        vy = self.data.qvel[1]
        lateral_penalty = -0.5 * vy ** 2
        
        # =====================================================
        # 总奖励
        # =====================================================
        total_reward = (
            0.35 * vel_reward
            + 0.25 * imitation_reward
            + 0.15 * height_reward
            + 0.10 * orientation_reward
            + 0.15 * alive_bonus
            + smooth_penalty
            + angvel_penalty
            + energy_penalty
            + lateral_penalty
        )
        return total_reward

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data, self.camera)
            return self.renderer.render()
        return None

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()
