"""Shared feature engineering utilities for commodity demand forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_FEATURES = [
    "price",
    "temperature",
    "rainfall",
    "macro_index",
    "freight_index",
    "supply_disruption",
    "capacity_constraint",
    "month",
    "day_of_week",
    "is_weekend",
    "holiday_flag",
    "season_sin",
    "season_cos",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_30",
    "price_lag_7",
    "rolling_demand_7",
    "rolling_demand_30",
]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["commodity", "date"])
    grouped = df.groupby("commodity", group_keys=False)
    for lag in [1, 7, 30]:
        df[f"demand_lag_{lag}"] = grouped["demand"].shift(lag)
    df["price_lag_7"] = grouped["price"].shift(7)
    df["rolling_demand_7"] = grouped["demand"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["rolling_demand_30"] = grouped["demand"].shift(1).rolling(30).mean().reset_index(level=0, drop=True)
    return df.dropna().reset_index(drop=True)


def load_feature_frame(path: str | Path = "data/commodity_demand.csv") -> pd.DataFrame:
    return add_lag_features(pd.read_csv(path))


def time_split(df: pd.DataFrame, test_days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test
