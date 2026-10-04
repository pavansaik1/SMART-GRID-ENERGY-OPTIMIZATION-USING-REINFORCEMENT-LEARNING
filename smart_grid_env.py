
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path


class SmartGridBatteryEnv(gym.Env):

    def __init__(self, csv_path=None):
        super().__init__()

        # Load the dataset
        if csv_path is None:
            csv_path = Path(__file__).parent / "data" / "smart_grid_data.csv"

        self.df = pd.read_csv(csv_path)

        required = [
            "solar_kw",
            "demand_kw",
            "price_per_kwh"
        ]
        if not all(col in self.df.columns for col in required):
            raise ValueError("CSV is missing required columns")

        self.solar_gen = self.df["solar_kw"].to_numpy()
        self.demand = self.df["demand_kw"].to_numpy()
        self.prices = self.df["price_per_kwh"].to_numpy()

        self.timesteps = len(self.df)

        # Actions: 0=Hold, 1=Charge, 2=Discharge
        self.action_space = spaces.Discrete(3)

        # State: solar, demand, price, battery charge
        # State: solar, demand, price, battery charge, hour
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([8, 6, 0.5, 10, 23], dtype=np.float32),
            dtype=np.float32
        )

        self.battery_capacity = 10.0  # kWh
        self.charge_rate = 2.0        # kWh per 1-hour step

        self.current_step = 0
        self.battery_soc = 5.0

    def _get_obs(self):
        i = self.current_step
        hour = i % 24

        return np.array([
            self.solar_gen[i],
            self.demand[i],
            self.prices[i],
            self.battery_soc,
            hour
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.battery_soc = 5.0

        return self._get_obs(), {}

    
    def step(self, action):
        i = self.current_step

        solar = self.solar_gen[i]
        demand = self.demand[i]
        price = self.prices[i]

        # Remaining demand after solar supplies the house
        deficit = max(0.0, demand - solar)
        surplus = max(0.0, solar - demand)

        if action == 1:  # Charge using solar surplus only
            charge = min(
                self.charge_rate,
                self.battery_capacity - self.battery_soc,
                surplus
            )

            self.battery_soc += charge
            grid_import = deficit

        elif action == 2:  # Discharge
            # Discharge only when the house needs extra energy
            discharge = min(
                self.charge_rate,
                self.battery_soc,
                deficit
            )

            self.battery_soc -= discharge
            grid_import = deficit - discharge

        else:  # Hold
            charge = 0.0
            discharge = 0.0
            grid_import = deficit

        cost = grid_import * price
        reward = -float(cost)

        self.current_step += 1
        terminated = self.current_step >= self.timesteps

        if terminated:
            target_soc = 5.0
            penalty_per_kwh = 10.0

            terminal_penalty = (
                abs(self.battery_soc - target_soc) * penalty_per_kwh
            )

            reward -= float(terminal_penalty)
            next_obs = np.zeros(5, dtype=np.float32)
        else:
            next_obs = self._get_obs()

        info = {
            "grid_import": grid_import,
            "cost": cost,
            "battery_soc": self.battery_soc
        }

        return next_obs, reward, terminated, False, info