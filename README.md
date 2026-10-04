# Smart Grid Energy Optimization using Reinforcement Learning

A reinforcement learning project that uses an **Advantage Actor-Critic (A2C)** agent to control a battery in a simulated smart-grid environment. The goal is to reduce electricity imported from the grid by intelligently charging and discharging the battery based on solar generation, household demand, electricity price, battery state of charge, and time of day.

---

## Project Overview

Solar energy is not always available when electricity demand is high. A battery can store excess solar energy and use it later when solar generation is insufficient.

In this project, a reinforcement learning agent learns how to control the battery.

At every hourly timestep, the agent chooses one of three actions:

- **Hold** – Do nothing
- **Charge** – Charge the battery using available solar surplus
- **Discharge** – Discharge the battery to reduce grid electricity usage

The trained A2C agent is compared with:

1. Random control
2. Rule-based control

---

## Objectives

- Simulate a simple smart-grid energy management system.
- Model solar generation, household demand, electricity prices, and battery storage.
- Train an A2C reinforcement learning agent.
- Reduce electricity imported from the grid.
- Compare the learned policy with baseline strategies.
- Visualize battery state of charge and grid electricity usage.

---

## Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Gymnasium**
- **Stable-Baselines3**
- **A2C (Advantage Actor-Critic)**
- **Matplotlib**

---

## Dataset

The project uses a synthetic dataset containing **720 hourly records**, representing **30 days** of simulated smart-grid operation.

The dataset contains:

| Column | Description |
|---|---|
| `solar_kw` | Solar power generation |
| `demand_kw` | Household electricity demand |
| `price_per_kwh` | Electricity price |

The dataset is stored in:

```text
data/smart_grid_data.csv
