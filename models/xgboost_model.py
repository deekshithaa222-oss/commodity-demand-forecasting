"""Gradient-boosted demand forecasting with seasonal, price, and lag features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

try:
    from models.features import BASE_FEATURES, load_feature_frame, time_split
except ModuleNotFoundError:
    from features import BASE_FEATURES, load_feature_frame, time_split


OUTPUT_PATH = Path("outputs/xgboost_forecast.csv")
MODEL_FEATURES = BASE_FEATURES + ["commodity"]


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("commodity", OneHotEncoder(handle_unknown="ignore"), ["commodity"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.88,
        colsample_bytree=0.86,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    df = load_feature_frame()
    train, test = time_split(df, test_days=90)
    model = build_model()
    model.fit(train[MODEL_FEATURES], train["demand"])
    predictions = model.predict(test[MODEL_FEATURES])

    output = test[["date", "commodity", "demand"]].copy()
    output["forecast"] = predictions
    output = output.rename(columns={"demand": "actual"})
    output["model"] = "XGBoost"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    mae = mean_absolute_error(output["actual"], output["forecast"])
    mape = mean_absolute_percentage_error(output["actual"], output["forecast"])
    print(f"XGBoost MAE: {mae:,.2f}")
    print(f"XGBoost MAPE: {mape:.3%}")
    print(f"Wrote forecast to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
