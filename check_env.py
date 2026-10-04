
from stable_baselines3.common.env_checker import check_env
from smart_grid_env import SmartGridBatteryEnv

env = SmartGridBatteryEnv()

check_env(env, warn=True)

print("Environment passed the checker!")