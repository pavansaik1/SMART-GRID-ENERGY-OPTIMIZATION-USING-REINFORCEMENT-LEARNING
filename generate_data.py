import numpy as np
import pandas as pd

# Reproducible synthetic data
rng = np.random.default_rng(42)

hours = 24 * 30  # 30 days of hourly data
time = np.arange(hours)
hour_of_day = time % 24

# Solar: daytime production, zero at night
daylight = np.maximum(
    0,
    np.sin(np.pi * (hour_of_day - 6) / 12)
)
solar = daylight * 6.0
solar += rng.normal(0, 0.25, hours)
solar = np.clip(solar, 0, 8)

# Demand: base usage plus morning/evening peaks
morning_peak = np.exp(-0.5 * ((hour_of_day - 8) / 2) ** 2)
evening_peak = np.exp(-0.5 * ((hour_of_day - 19) / 3) ** 2)

demand = (
    1.5
    + 1.5 * morning_peak
    + 2.5 * evening_peak
    + rng.normal(0, 0.35, hours)
)
demand = np.clip(demand, 0.5, 6)

# Electricity tariff: higher during peak hours
peak_price = (
    ((hour_of_day >= 17) & (hour_of_day <= 21))
)
prices = 0.15 + 0.20 * peak_price
prices += rng.normal(0, 0.015, hours)
prices = np.clip(prices, 0.08, 0.50)

df = pd.DataFrame({
    "hour": time,
    "solar_kw": solar,
    "demand_kw": demand,
    "price_per_kwh": prices
})

df.to_csv("data/smart_grid_data.csv", index=False)

print(df.head())
print("Dataset shape:", df.shape)