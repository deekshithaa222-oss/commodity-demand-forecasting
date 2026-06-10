# commodity-demand-forecasting

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Forecasting](https://img.shields.io/badge/Forecasting-SARIMAX%20%7C%20XGBoost-green)
![Demand Planning](https://img.shields.io/badge/Demand-Planning-orange)
![Scenario Analysis](https://img.shields.io/badge/Scenario-Price%20%7C%20Capacity-lightgrey)

Forecast commodity demand using price, weather, macroeconomic, and supply-chain drivers. The project compares a statistical SARIMAX baseline with a machine-learning XGBoost model, explains demand drivers, and simulates price/capacity scenarios for operational planning.

## Business Problem

Commodity planners need reliable demand forecasts for inventory, sourcing, and pricing decisions. Demand moves with seasonality, price elasticity, weather, holidays, macro conditions, freight pressure, supply disruptions, and capacity constraints.

This project answers:

- What is the expected demand over the next planning window?
- Which demand drivers matter most?
- How do price changes and capacity constraints affect forecasted demand and revenue?
- Which scenario should a planner recommend for each commodity?

## Project Structure

```text
data/generate_data.py
  -> data/commodity_demand.csv

models/arima_model.py
  -> outputs/arima_forecast.csv

models/xgboost_model.py
  -> outputs/xgboost_forecast.csv

analysis/feature_importance.py
  -> outputs/feature_importance.csv
  -> outputs/feature_importance.png

simulation/grid_search.py
  -> outputs/scenario_recommendations.csv

notebooks/forecasting_pipeline.ipynb
  -> end-to-end walkthrough
```

## Quickstart

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate data and run the full workflow:

```bash
python data/generate_data.py
python models/arima_model.py
python models/xgboost_model.py
python analysis/feature_importance.py
python simulation/grid_search.py
```

Note: this project targets Python 3.10. The pinned scientific stack may try to build from source on newer unsupported Python versions.

## Modeling Approach

| Component | Method | Purpose |
| --- | --- | --- |
| Synthetic data | Daily commodity panel for wheat, corn, soybeans, copper, and lithium | Creates realistic seasonal, price, weather, and supply-chain demand signals. |
| SARIMAX | Seasonal ARIMA with exogenous variables | Strong interpretable baseline for time-series forecasting. |
| XGBoost | Tree-based regression with lag and rolling features | Captures nonlinear relationships and interactions. |
| Feature importance | XGBoost importance ranking | Explains which drivers matter most to the forecast. |
| Scenario grid | Price and capacity simulation | Converts forecasts into planning recommendations. |

## Example Results

Results from the default synthetic dataset seed:

| Model | Metric | Result | Business meaning |
| --- | --- | ---: | --- |
| SARIMAX | MAE | 125.98 | Baseline average absolute forecast error. |
| SARIMAX | MAPE | 10.147% | Baseline time-series percentage error. |
| XGBoost | MAE | 61.42 | Lower absolute forecast error from nonlinear lag, price, and supply features. |
| XGBoost | MAPE | 4.203% | About 58.6% lower MAPE than SARIMAX. |

Top demand drivers from the XGBoost model:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `commodity_lithium` | 0.6331 |
| 2 | `demand_lag_1` | 0.1553 |
| 3 | `demand_lag_7` | 0.1436 |
| 4 | `demand_lag_30` | 0.0136 |
| 5 | `rolling_demand_7` | 0.0098 |

Best scenario-grid recommendation by commodity:

| Commodity | Price Change | Capacity Constraint | Forecast Demand | Forecast Revenue |
| --- | ---: | ---: | ---: | ---: |
| copper | +10% | 0 | 1,257 | $4,871 |
| corn | +10% | 0 | 1,875 | $16,804 |
| lithium | +10% | 0 | 627 | $14,284 |
| soybeans | +10% | 0 | 1,326 | $24,372 |
| wheat | +10% | 0 | 1,757 | $12,709 |

## Business Interpretation

The strongest drivers are expected to include lagged demand, rolling demand, seasonality, price, macro index, freight index, and capacity flags. A planner can use the forecast outputs to decide where to increase safety stock, where to protect capacity, and where price moves may improve revenue without creating excess demand risk.

## Resume Talking Points

- Built an end-to-end commodity demand forecasting pipeline with synthetic but realistic market data.
- Compared statistical and machine-learning forecasting methods.
- Engineered lag, rolling-window, holiday, weather, macro, and price features.
- Added model explainability and scenario planning for business recommendations.
- Designed the repo so scripts can run independently or as a full notebook workflow.
