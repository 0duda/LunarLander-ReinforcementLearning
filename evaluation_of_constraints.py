"""
Evaluation Script for Lunar Lander with Constraints.

This script loads a pre-trained PPO agent and evaluates its performance
in a custom Lunar Lander environment (specifically one with random initial
forces, though others can be substituted). It calculates metrics related to
success rate, spatial precision (distance to center), and impact quality
(vertical velocity).

The results are printed to the console in a tabular format.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from constraint_random_force import LunarLander  # adjust according to the desired agent to act in the environment

# Initialize the environment
# We use the custom LunarLander class imported above.
env = LunarLander(
    render_mode=None,
    continuous=True
)

# Load the trained model
model = PPO.load(
    "models/Baseline/ppo_baseline_1m", # adjust according to the desired agent to act in the environment
    env=env
)

episodes = 1000

# Global counters
successful = 0

# Spatial precision
center_landings = 0
flag_landings = 0

# Impact quality
perfect_landings = 0
good_landings = 0
acceptable_landings = 0

impact_velocities = []

# Main evaluation loop
for ep in range(episodes):
    obs, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

    # Episode ended → landed
    # Check if the episode ended successfully (landed within bounds and upright)
    if terminated and reward > 0:
        successful += 1

        x_dist = abs(obs[0])
        # Retrieve the vertical velocity at the moment of impact from the environment wrapper
        impact_vy = env.unwrapped.last_air_vy

        impact_velocities.append(impact_vy)

        # Spatial precision 
        if x_dist < 0.05:
            center_landings += 1
        elif x_dist < 0.30:
            flag_landings += 1

        # Impact quality 
        if impact_vy < 0.03:
            perfect_landings += 1
        elif impact_vy < 0.10:
            good_landings += 1
        elif impact_vy < 0.20:
            acceptable_landings += 1

    print(f"Episode {ep + 1} completed")

env.close()

# Final statistics
avg_impact_vy = np.mean(impact_velocities) if impact_velocities else 0.0

# ===================== TABLE OUTPUT =====================

print("\n===== RESULTS =====\n")

print(f"Episodes evaluated: {episodes}")
print(f"Successful landings: {successful}\n")

print("Spatial precision:")
print("+------------------------------+-------+")
print("| Category                     | Total |")
print("+------------------------------+-------+")
print(f"| Center (< 0.05)              | {center_landings:5d} |")
print(f"| Flags (0.05 – 0.30)          | {flag_landings:5d} |")
print("+------------------------------+-------+\n")

print("Impact quality (|vy|):")
print("+------------------------------+-------+")
print("| Category                     | Total |")
print("+------------------------------+-------+")
print(f"| Perfect (< 0.03)             | {perfect_landings:5d} |")
print(f"| Good (0.03 – 0.10)           | {good_landings:5d} |")
print(f"| Acceptable (0.10 – 0.20)     | {acceptable_landings:5d} |")
print("+------------------------------+-------+\n")

print(f"Average impact velocity (|vy|): {avg_impact_vy:.4f}")