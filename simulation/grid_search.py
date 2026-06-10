"""Simulate price and capacity scenarios to find recommended commodity actions."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.features import BASE_FEATURES, load_feature_frame, time_split
from models.xgboost_model import build_model


OUTPUT_PATH = Path("outputs/scenario_recommendations.csv")
MODEL_FEATURES = BASE_FEATURES + ["commodity"]


def build_scenarios(latest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    price_changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
    capacity_flags = [0, 1]
    for _, row in latest.iterrows():
        for price_delta, capacity in product(price_changes, capacity_flags):
            scenario = row.copy()
            scenario["scenario_price_delta"] = price_delta
            scenario["scenario_capacity_constraint"] = capacity
            scenario["price"] = row["price"] * (1 + price_delta)
            scenario["capacity_constraint"] = capacity
            rows.append(scenario)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_feature_frame()
    train, _ = time_split(df, test_days=90)
    model = build_model()
    model.fit(train[MODEL_FEATURES], train["demand"])

    latest = df.sort_values("date").groupby("commodity").tail(1).copy()
    scenarios = build_scenarios(latest)
    scenarios["forecast_demand"] = model.predict(scenarios[MODEL_FEATURES])
    scenarios["forecast_revenue"] = scenarios["forecast_demand"] * scenarios["price"]

    recommendations = (
        scenarios.sort_values(["commodity", "forecast_revenue"], ascending=[True, False])
        .groupby("commodity")
        .head(3)
        .loc[
            :,
            [
                "commodity",
                "scenario_price_delta",
                "scenario_capacity_constraint",
                "price",
                "forecast_demand",
                "forecast_revenue",
            ],
        ]
        .reset_index(drop=True)
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(OUTPUT_PATH, index=False)
    print("Top scenario recommendations:")
    print(recommendations.to_string(index=False, formatters={"forecast_demand": "{:,.0f}".format, "forecast_revenue": "${:,.0f}".format}))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
