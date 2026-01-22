"""
Training Script for Custom Lunar Lander Agent using PPO.

This script manages the training process for the Lunar Lander environment.
It sets up the environment, model, and directories, performs the training
using the PPO algorithm, and saves the resulting model.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# Import the custom class
from soft_precision_penalties import LunarLander # adjust according to the desired agent to act in the environment

# 1. Create directories to store models and logs
models_dir = "models/Soft+Precision+Penalties" # adjust according to the desired agent to act in the environment
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 2. Instantiate the environment
env = LunarLander(render_mode=None, continuous=True)

# 3. Define the Model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

print("-----------------------------------------")
print("STARTING TRAINING...")
print("-----------------------------------------")

# 4. Train the Agent
TIMESTEPS = 500000
model.learn(total_timesteps=TIMESTEPS, tb_log_name="soft_precision_penalties")

# 5. Save the trained Agent
model_path = f"{models_dir}/ppo_soft_precision_penalties"
model.save(model_path)

print("-----------------------------------------")
print(f"TRAINING FINISHED. Model saved at: {model_path}.zip")
print("-----------------------------------------")