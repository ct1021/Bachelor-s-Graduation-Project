import gymnasium as gym
from stable_baselines3 import PPO
import os

# 1. 创建环境 (使用内置的 CartPole 作为双足平衡的简化模型)
env = gym.make("CartPole-v1", render_mode="human")

# 2. 实例化 PPO 算法
# 修改第 11 行左右
# 加入 tensorboard_log="./logs/" 这样系统会自动生成日志文件 [cite: 12]
model = PPO("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log="./logs/")
# 3. 开始训练
print("🚀 开始平衡训练...")
model.learn(total_timesteps=10000)

# 4. 保存模型 (符合 GitHub 规范，建议建立 models 文件夹)
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/ppo_cartpole_balance")
print("✅ 模型已保存至 models/ 文件夹")

# 5. 测试训练效果
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()