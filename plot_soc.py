import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from smart_grid_env import SmartGridBatteryEnv

env = SmartGridBatteryEnv()
model = A2C.load("models/smart_grid_a2c_time")

obs, info = env.reset(seed=42)

soc_values = [env.battery_soc]
grid_values = []

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(
        int(action)
    )

    soc_values.append(info["battery_soc"])
    grid_values.append(info["grid_import"])

    if terminated or truncated:
        break

hours = list(range(len(soc_values)))

plt.figure(figsize=(12, 5))
plt.plot(hours, soc_values, label="Battery SoC")
plt.axhline(
    y=5,
    linestyle="--",
    label="Target SoC (5 kWh)"
)

plt.title("A2C Battery State of Charge Over 30 Days")
plt.xlabel("Hour")
plt.ylabel("Battery SoC (kWh)")
plt.ylim(0, 10)
plt.legend()
plt.tight_layout()
plt.savefig("results/a2c_soc.png", dpi=300)
plt.show()

print("Final SoC:", soc_values[-1], "kWh")
plt.figure(figsize=(12, 5))
plt.plot(range(72), grid_values[:72])

plt.title("A2C Grid Electricity Import (First 3 Days)")
plt.xlabel("Hour")
plt.ylabel("Grid import (kWh)")
plt.tight_layout()
plt.savefig("results/a2c_grid_import.png", dpi=300)
plt.show()