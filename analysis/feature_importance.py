"""Rank demand drivers from the XGBoost commodity forecast model."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.features import BASE_FEATURES, load_feature_frame, time_split
from models.xgboost_model import build_model


OUTPUT_CSV = Path("outputs/feature_importance.csv")
OUTPUT_PNG = Path("outputs/feature_importance.png")
MODEL_FEATURES = BASE_FEATURES + ["commodity"]


def main() -> None:
    df = load_feature_frame()
    train, _ = time_split(df, test_days=90)
    model = build_model()
    model.fit(train[MODEL_FEATURES], train["demand"])

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(OUTPUT_CSV, index=False)

    top = importance_df.head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#2f6f73")
    plt.title("Top Demand Forecast Features")
    plt.xlabel("XGBoost feature importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=160)

    print("Top 10 demand drivers:")
    print(importance_df.head(10).to_string(index=False, formatters={"importance": "{:.4f}".format}))
    print(f"Wrote {OUTPUT_CSV} and {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
