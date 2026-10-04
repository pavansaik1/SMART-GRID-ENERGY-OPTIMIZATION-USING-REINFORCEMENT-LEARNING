
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from smart_grid_env import SmartGridBatteryEnv


# Create folders for saving results
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# Create and wrap the environment
env = SmartGridBatteryEnv()
env = Monitor(env)

# Create the A2C agent
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.001,
    n_steps=24,
    gamma=0.99,
    verbose=1,
    seed=42
)

# Train the agent
print("Starting A2C training...")

model.learn(
    total_timesteps=50000,
    progress_bar=False
)

# Save the trained model
model.save("models/smart_grid_a2c_time")

print("Training complete!")
print("Model saved in models/smart_grid_a2c_time.zip")

env.close()