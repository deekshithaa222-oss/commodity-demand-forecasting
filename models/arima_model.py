"""SARIMAX baseline demand forecast for each commodity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_PATH = Path("data/commodity_demand.csv")
OUTPUT_PATH = Path("outputs/arima_forecast.csv")


def fit_commodity(series: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    series = series.sort_values("date").set_index("date")
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    exog_cols = ["price", "macro_index", "freight_index", "holiday_flag", "supply_disruption"]
    model = SARIMAX(
        train["demand"],
        exog=train[exog_cols],
        order=(2, 1, 2),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=horizon, exog=test[exog_cols]).predicted_mean
    return pd.DataFrame(
        {
            "date": test.index,
            "commodity": test["commodity"].values,
            "actual": test["demand"].values,
            "forecast": np.maximum(0, forecast.values),
            "model": "SARIMAX",
        }
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = [fit_commodity(part) for _, part in df.groupby("commodity")]
    output = pd.concat(forecasts, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    mae = mean_absolute_error(output["actual"], output["forecast"])
    mape = mean_absolute_percentage_error(output["actual"], output["forecast"])
    print(f"SARIMAX MAE: {mae:,.2f}")
    print(f"SARIMAX MAPE: {mape:.3%}")
    print(f"Wrote forecast to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
