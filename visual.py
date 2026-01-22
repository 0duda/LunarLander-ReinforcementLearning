"""
Visualization Script for the Lunar Lander Agent.

This script loads a trained PPO model and the custom Lunar Lander environment,
then runs a specified number of episodes with visual rendering enabled ("human" mode).
It allows for the visual inspection of the agent's performance and behavior.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from soft_precision_landing_no_wind import LunarLander # adjust according to the desired agent to act in the environment

# Load environment with rendering
env = LunarLander(render_mode="human", continuous=True)

# Load the model
model_path = "models/PrecisionLanding/ppo_precision_landing" # adjust according to the desired agent to act in the environment
model = PPO.load(model_path, env=env)

episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    score = 0
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward        
    print(f"Episode {ep+1}: Score = {score:.2f}")

env.close()