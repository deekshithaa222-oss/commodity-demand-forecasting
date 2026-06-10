"""Generate synthetic commodity demand data with price, weather, and macro drivers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMMODITIES = {
    "wheat": {"base": 1850, "price": 7.2, "elasticity": -95, "season_peak": 255},
    "corn": {"base": 2400, "price": 5.1, "elasticity": -135, "season_peak": 230},
    "soybeans": {"base": 1725, "price": 13.4, "elasticity": -70, "season_peak": 250},
    "copper": {"base": 1180, "price": 4.0, "elasticity": -185, "season_peak": 80},
    "lithium": {"base": 720, "price": 23.5, "elasticity": -14, "season_peak": 105},
}


def build_calendar(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="D")
    calendar = pd.DataFrame({"date": dates})
    day_of_year = calendar["date"].dt.dayofyear
    calendar["month"] = calendar["date"].dt.month
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["is_weekend"] = calendar["day_of_week"].isin([5, 6]).astype(int)
    calendar["holiday_flag"] = (
        ((calendar["month"] == 1) & (calendar["date"].dt.day <= 3))
        | ((calendar["month"] == 7) & (calendar["date"].dt.day.between(1, 7)))
        | ((calendar["month"] == 12) & (calendar["date"].dt.day >= 24))
    ).astype(int)
    calendar["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    calendar["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return calendar


def generate(seed: int = 42, start: str = "2019-01-01", years: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = years * 365
    calendar = build_calendar(start, periods)
    rows = []

    macro_index = 100 + np.cumsum(rng.normal(0.01, 0.12, periods))
    freight_index = 90 + np.cumsum(rng.normal(0.015, 0.18, periods))

    for commodity, cfg in COMMODITIES.items():
        price = cfg["price"] + np.cumsum(rng.normal(0.0, cfg["price"] * 0.004, periods))
        price += 0.4 * np.sin(2 * np.pi * np.arange(periods) / 180)
        rainfall = rng.gamma(shape=2.0, scale=2.2, size=periods)
        temperature = 62 + 18 * calendar["season_sin"].to_numpy() + rng.normal(0, 4, periods)
        supply_disruption = rng.binomial(1, 0.018, periods)
        capacity_constraint = rng.binomial(1, 0.026, periods)
        seasonality = 210 * np.cos(2 * np.pi * (calendar["date"].dt.dayofyear - cfg["season_peak"]) / 365.25)
        trend = np.linspace(0, 110, periods)
        holiday_effect = -95 * calendar["holiday_flag"].to_numpy()
        weekend_effect = -55 * calendar["is_weekend"].to_numpy()
        macro_effect = 9.5 * (macro_index - macro_index.mean())
        freight_effect = -4.8 * (freight_index - freight_index.mean())
        weather_effect = -2.2 * np.maximum(temperature - 84, 0) + 3.7 * np.minimum(rainfall, 7)
        disruption_effect = -210 * supply_disruption
        capacity_effect = -160 * capacity_constraint
        price_effect = cfg["elasticity"] * (price - cfg["price"])
        noise = rng.normal(0, cfg["base"] * 0.045, periods)

        demand = (
            cfg["base"]
            + trend
            + seasonality
            + holiday_effect
            + weekend_effect
            + macro_effect
            + freight_effect
            + weather_effect
            + disruption_effect
            + capacity_effect
            + price_effect
            + noise
        )

        commodity_df = calendar.copy()
        commodity_df["commodity"] = commodity
        commodity_df["price"] = np.round(price, 3)
        commodity_df["temperature"] = np.round(temperature, 2)
        commodity_df["rainfall"] = np.round(rainfall, 2)
        commodity_df["macro_index"] = np.round(macro_index, 3)
        commodity_df["freight_index"] = np.round(freight_index, 3)
        commodity_df["supply_disruption"] = supply_disruption
        commodity_df["capacity_constraint"] = capacity_constraint
        commodity_df["demand"] = np.maximum(80, np.round(demand, 2))
        rows.append(commodity_df)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic commodity demand data.")
    parser.add_argument("--output", default="data/commodity_demand.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    df = generate(seed=args.seed, years=args.years)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")
    print(f"Commodities: {', '.join(sorted(df['commodity'].unique()))}")


if __name__ == "__main__":
    main()
