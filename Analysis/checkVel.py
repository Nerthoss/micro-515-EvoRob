import numpy as np
import gymnasium as gym
from final_project_train import FinalWorld

x_best = np.load("results/test02_200gen/98/x_best.npy")

env = gym.make("FlatEnv-v0", max_episode_steps=500)
obs, _ = env.reset(seed=42)
world = FinalWorld(env)
world.geno2pheno(x_best)

velocities = []
total_reward = 0

for _ in range(500):
    action = world.controller.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    velocities.append(info["x_velocity"])
    total_reward += reward
    if terminated or truncated:
        break

print(f"Mean x_velocity: {np.mean(velocities):.3f} m/s")
print(f"Total reward: {total_reward:.2f}")
print(f"Episodes ended early: {terminated}")