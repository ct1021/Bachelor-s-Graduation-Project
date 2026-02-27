# -*- coding: utf-8 -*-
"""
test_load_model.py -- Model Self-Check / Integration Test

Usage:
    conda activate biped_rl
    cd "E:\\Microsoft VS Code\\programs\\gradually-work"
    python scripts/test_load_model.py

This script verifies all core modules can load and run correctly:
    [OK] Test 1: GaitPriorNetwork forward pass
    [OK] Test 2: HybridLoss computation
    [OK] Test 3: PhysicsRewardWrapper environment wrapping
    [OK] Test 4: SB3 saved model loading
"""

import sys
import os
import traceback

# 确保 scripts/ 所在的项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_gait_prior_network():
    """Test 1: Verify GaitPriorNetwork forward pass and distribution generation."""
    import torch
    from scripts.gait_prior_model import GaitPriorNetwork

    OBS_DIM, ACT_DIM, BATCH = 24, 6, 16
    model = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)

    # 前向传播
    dummy_obs = torch.randn(BATCH, OBS_DIM)
    mean, log_std = model(dummy_obs)
    assert mean.shape == (BATCH, ACT_DIM), f"均值 shape 错误: {mean.shape}"
    assert log_std.shape == (BATCH, ACT_DIM), f"log_std shape 错误: {log_std.shape}"

    # 概率分布
    dist = model.get_distribution(dummy_obs)
    action = dist.sample()
    assert action.shape == (BATCH, ACT_DIM), f"采样动作 shape 错误: {action.shape}"

    # 保存 & 加载
    test_path = os.path.join(PROJECT_ROOT, "models", "_test_prior.pt")
    model.save(test_path)
    model2 = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)
    model2.load(test_path)

    # 清理测试文件
    if os.path.exists(test_path):
        os.remove(test_path)

    return True


def test_hybrid_loss():
    """Test 2: Verify HybridLoss hybrid loss computation."""
    import torch
    from scripts.gait_prior_model import GaitPriorNetwork
    from scripts.hybrid_loss import HybridLoss

    OBS_DIM, ACT_DIM, BATCH = 24, 6, 32

    prior = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)
    online = GaitPriorNetwork(obs_dim=OBS_DIM, action_dim=ACT_DIM)

    criterion = HybridLoss(prior_model=prior, lambda_kl=0.1)

    obs = torch.randn(BATCH, OBS_DIM)
    expert_actions = torch.randn(BATCH, ACT_DIM) * 0.3

    policy_dist = online.get_distribution(obs)
    pred_actions = policy_dist.mean

    losses = criterion(policy_dist, pred_actions, expert_actions, obs)

    assert "total_loss" in losses, "缺少 total_loss"
    assert "bc_loss" in losses, "缺少 bc_loss"
    assert "kl_loss" in losses, "缺少 kl_loss"
    assert losses["total_loss"].requires_grad, "total_loss 不可求导"

    # 测试反向传播
    losses["total_loss"].backward()

    return True


def test_physics_reward():
    """Test 3: Verify PhysicsRewardWrapper environment wrapping."""
    import gymnasium as gym
    from scripts.physics_reward import PhysicsRewardWrapper

    env = gym.make("CartPole-v1")
    wrapped = PhysicsRewardWrapper(
        env,
        target_velocity=0.5,
        robot_mass=10.0,
    )

    obs, info = wrapped.reset()
    assert obs is not None, "reset 返回空 obs"

    # 执行若干步
    for _ in range(10):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert isinstance(reward, float), f"奖励类型错误: {type(reward)}"
        assert "reward/total" in info, "info 中缺少奖励分量"
        if terminated or truncated:
            obs, info = wrapped.reset()

    wrapped.close()
    return True


def test_sb3_model_load():
    """Test 4: Verify SB3 saved model loading."""
    from stable_baselines3 import PPO

    model_path = os.path.join(PROJECT_ROOT, "models", "ppo_cartpole_balance.zip")
    if not os.path.exists(model_path):
        print("  [SKIP] saved model not found")
        return True  # 不阻塞其他测试

    model = PPO.load(model_path)
    assert model is not None, "模型加载返回 None"
    assert model.policy is not None, "model missing policy"
    print(f"  Loaded model, policy: {model.policy.__class__.__name__}")

    return True


# ======================== 主测试入口 ========================
def main():
    tests = [
        ("1. GaitPriorNetwork forward pass", test_gait_prior_network),
        ("2. HybridLoss hybrid loss", test_hybrid_loss),
        ("3. PhysicsRewardWrapper env wrapping", test_physics_reward),
        ("4. SB3 saved model loading", test_sb3_model_load),
    ]

    print("=" * 60)
    print("[TEST] Biped RL Project - Module Self-Check")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n> {name} ...", end=" ")
            result = test_fn()
            if result:
                print("[PASS]")
                passed += 1
            else:
                print("[FAIL]")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Result: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print("WARNING: Some tests failed. Check errors above.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
