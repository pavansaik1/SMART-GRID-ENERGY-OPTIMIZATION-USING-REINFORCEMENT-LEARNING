
from smart_grid_env import SmartGridBatteryEnv

env = SmartGridBatteryEnv()

obs, info = env.reset(seed=42)

print("Initial state:", obs)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
print()

# Test: Hold, Charge, Discharge, Hold, Charge
actions = [0, 1, 2, 0, 1]

for action in actions:
    obs, reward, terminated, truncated, info = env.step(action)

    print("Action:", action)
    print("Next state:", obs)
    print("Reward:", round(reward, 4))
    print("Grid import:", round(info["grid_import"], 3))
    print("Battery:", round(info["battery_soc"], 2))
    print("-" * 30)