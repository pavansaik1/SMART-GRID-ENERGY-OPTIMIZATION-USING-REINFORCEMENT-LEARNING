
import numpy as np
from stable_baselines3 import A2C
from smart_grid_env import SmartGridBatteryEnv

# Load the trained model
model = A2C.load("models/smart_grid_a2c_time")
env = SmartGridBatteryEnv()

print("Environment shape:", env.observation_space.shape)
print("Model shape:", model.observation_space.shape)
obs, _ = env.reset(seed=42)

for i in range(10):
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor)
    probs = distribution.distribution.probs.detach().cpu().numpy()[0]

    print(
        f"Step {i}: "
        f"Hold={probs[0]:.3f}, "
        f"Charge={probs[1]:.3f}, "
        f"Discharge={probs[2]:.3f}"
    )

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))

    if terminated or truncated:
        break

env.close()


def evaluate_policy(env, policy, seed):
    obs, info = env.reset(seed=seed)

    total_cost = 0.0
    action_counts = [0, 0, 0]

    while True:
        action = policy(obs)
        action_counts[action] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_cost += info["cost"]

        if terminated or truncated:
            break

    final_soc = info["battery_soc"]
    return total_cost, final_soc, action_counts


# A2C policy
def a2c_policy(obs):
    action, _ = model.predict(obs, deterministic=False)
    return int(action)


# Random policy
rng = np.random.default_rng(42)

def random_policy(obs):
    return int(rng.integers(0, 3))


# Rule-based policy
def rule_policy(obs):
    solar, demand, price, soc, hour = obs

    surplus = solar - demand
    deficit = demand - solar

    if surplus >= 1.0 and soc < 9:
        return 1

    if price >= 0.30 and deficit > 0 and soc > 1:
        return 2

    return 0


# Run 10 episodes per strategy
results = {
    "A2C": [],
    "Random": [],
    "Rule-based": []
}

for seed in range(10):
    for name, policy in [
        ("A2C", a2c_policy),
        ("Random", random_policy),
        ("Rule-based", rule_policy)
    ]:
        env = SmartGridBatteryEnv()
        cost, final_soc, actions = evaluate_policy(env, policy, seed)
        results[name].append((cost, final_soc, actions))
        env.close()


# Print average results
print("\n===== 10-EPISODE EVALUATION =====")

for name, episodes in results.items():
    costs = [x[0] for x in episodes]
    socs = [x[1] for x in episodes]

    print(f"\n{name}")
    print(f"Average cost: {np.mean(costs):.2f}")
    print(f"Minimum cost: {np.min(costs):.2f}")
    print(f"Maximum cost: {np.max(costs):.2f}")
    print(f"Average final SOC: {np.mean(socs):.2f} kWh")


