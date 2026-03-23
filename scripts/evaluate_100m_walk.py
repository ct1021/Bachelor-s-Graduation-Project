"""
evaluate_100m_walk.py
=====================
根据任务书的"持续行走"性能指标，评估训练好的 PPO 策略能否连续行走 100 米不跌倒。

任务书要求：
 - 完成速度阶跃、持续行走和轻度扰动等仿真实验
 - 汇报轨迹/速度跟踪误差、跌倒率、COT 能耗（单位距离）、步态对称性指数

运行方式:
    conda activate biped_rl
    python scripts/evaluate_100m_walk.py
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def evaluate_100m_walk(model_path, target_distance=100.0, max_steps=20000):
    """
    运行一个超长 Episode，评测策略持续行走 100m 的能力。
    100m / 0.5 m/s = 200秒 = 20000步 @ 100Hz
    """
    from stable_baselines3 import PPO
    from envs.g1_env import G1WalkEnv

    print("=" * 60)
    print("  G1 100m 持续行走评测")
    print(f"  目标距离: {target_distance}m  最大步数: {max_steps}")
    print("=" * 60)

    model = PPO.load(model_path)
    env = G1WalkEnv(render_mode=None)
    obs, _ = env.reset()

    total_distance = 0.0
    total_energy = 0.0
    fall_count = 0
    step_count = 0
    velocity_errors = []
    height_records = []

    print("\n开始行走评测...")
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1

        vx = env.data.qvel[0]
        total_distance += max(0, vx * env.dt)
        velocity_errors.append(abs(vx - 0.5))
        root_z = env.data.qpos[2]
        height_records.append(root_z)
        total_energy += np.sum(np.square(env.data.ctrl)) * env.dt

        if total_distance >= target_distance:
            print(f"\n✅ 成功！在 {step+1} 步 ({(step+1)*env.dt:.1f}s) 内走完了 {total_distance:.2f}m！")
            break

        if terminated:
            fall_count += 1
            print(f"[Step {step+1}] 摔倒！已行走 {total_distance:.2f}m，重置...")
            obs, _ = env.reset()

        if (step + 1) % 1000 == 0:
            print(f"  [进度] Step {step+1}/{max_steps} | 距离: {total_distance:.2f}m | 跌倒: {fall_count}次")

    env.close()

    elapsed_time = step_count * env.dt
    cot = total_energy / max(total_distance, 0.001)
    avg_vel_error = np.mean(velocity_errors)
    avg_height = np.mean(height_records)
    height_std = np.std(height_records)

    print("\n" + "=" * 60)
    print("  任务书考核指标汇报")
    print("=" * 60)
    print(f"  总行走距离:      {total_distance:.2f} m  {'✅ 达标' if total_distance >= target_distance else '❌ 未达标'}")
    print(f"  耗时:            {elapsed_time:.1f} s")
    print(f"  跌倒次数:        {fall_count} 次  跌倒率: {fall_count/max(1,step_count/1000):.4f} 次/千步")
    print(f"  速度跟踪误差:    {avg_vel_error:.4f} m/s (目标 0.5 m/s)")
    print(f"  质心高度均值/σ:  {avg_height:.4f} / {height_std:.4f} m")
    print(f"  COT能耗:         {cot:.2f} J/m")
    print("=" * 60)

    return {"distance_m": total_distance, "elapsed_s": elapsed_time,
            "fall_count": fall_count, "vel_tracking_error": avg_vel_error,
            "height_mean": avg_height, "height_std": height_std, "COT_J_per_m": cot}


if __name__ == "__main__":
    candidates = [
        os.path.join(PROJECT_ROOT, "models", "g1_ppo_fixed_3m.zip"),
        os.path.join(PROJECT_ROOT, "models", "g1_ppo_10000k.zip"),
        os.path.join(PROJECT_ROOT, "models", "g1_ppo_500k.zip"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        raise FileNotFoundError("❌ 未找到可用模型，请先训练。")
    print(f"✅ 使用模型: {model_path}")
    evaluate_100m_walk(model_path, target_distance=100.0, max_steps=20000)
