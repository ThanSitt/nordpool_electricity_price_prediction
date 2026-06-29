# Nordpool Electricity Price Prediction

Predict Finland electricity spot prices using weather data and engineered features.

## Project Overview

This project builds XGBoost models to forecast Nordpool day-ahead electricity prices. It spans two resolutions (hourly and 15-minute) and four model versions, demonstrating the impact of feature engineering and temporal granularity on prediction accuracy.

### Data Sources

| Source | Location | Resolution | Columns |
|--------|----------|-----------|---------|
| Electricity prices | `data/originalData/electricPrices/` | 15-min (also hourly) | `timestamp`, `price` |
| Temperature | `data/originalData/Temperature/` | 10-min raw → 15-min resampled | Helsinki-Vantaa airport |
| Wind speed & direction | `data/originalData/WindDirection&Speed/` | 10-min raw → 15-min resampled | Oulu Vihreäsaari harbour |

### Pipeline Overview

```
Raw data (10-min)  →  Resample to 15-min  →  Merge  →  Feature Engineering  →  XGBoost Model
```

## Model Versions

| Version | Resolution | Features | RMSE | R² | Description |
|---------|-----------|----------|------|-----|-------------|
| V1      | Hourly    | Weather only (temp, wind) | 46.34 | 0.107 | Baseline — weather-only hourly model |
| V1.5    | 15-min    | Weather only (temp, wind, direction) | 45.78 | 0.125 | 15-min baseline — weather-only |
| V2      | Hourly    | Full engineered (lags, rolling, calendar, holiday) | 14.62 | 0.911 | Hourly with feature engineering |
| **V2.5** | **15-min** | **Full engineered** | **8.22** | **0.972** | **Best model** |

### Key Insight

Feature engineering is the primary driver of accuracy, not resolution alone:

- **V1 → V1.5:** Increased resolution only (hourly → 15-min) with same weather features → R² 0.107 → 0.125 (negligible)
- **V1 → V2:** Added engineered features at same hourly resolution → R² 0.107 → 0.911 (game-changer)
- **V2 → V2.5:** Both higher resolution AND engineered features → R² 0.911 → 0.972 (best)

## Project Structure

```
nordpool_electricity_price_prediction/
│
├── data/
│   ├── originalData/             # Raw source files
│   │   ├── electricPrices/       # Price data (15min + hourly)
│   │   ├── Temperature/          # Temperature CSV files + 15-min resampling notebook
│   │   └── WindDirection&Speed/  # Wind data + 15-min resampling notebook
│   └── convertData/              # Processed datasets and feature engineering notebooks
│       ├── V1.5_15min_Dataset.csv       # Merged price + temp + wind (15-min, no features)
│       ├── V1.5_15min_Dataset.ipynb     # Dataset creation notebook
│       ├── V2.5_15min_feature_engineering.ipynb  # Feature engineering notebook
│       ├── V2.5_15min_features.csv     # Feature-engineered 15-min data
│       └── ... (other converted files)
│
├── models/                       # Training notebooks
│   ├── modelV1.ipynb             # Hourly weather-only baseline
│   ├── modelV1.5.ipynb           # 15-min weather-only baseline
│   ├── modelV2.ipynb             # Hourly with engineered features
│   └── modelV2.5.ipynb           # 15-min with engineered features (best)
│
├── notebooks/
│   └── LearningNotes_CQL/        # Learning guides and project plans
│
├── reports/                      # Saved outputs (predictions, feature lists)
│
└── README.md                     # This file
```

## Notebook Execution Order

### Hourly Pipeline
1. `data/convertData/03_data_cleaning_and_alignment.ipynb` — Merge price + weather → `finland_electricity_predict_dataset.csv`
2. `data/convertData/feature_engineering.ipynb` — Feature engineering → `finland_electricity_features_v2.csv`
3. `models/modelV1.ipynb` → Train/evaluate V1
4. `models/modelV2.ipynb` → Train/evaluate V2

### 15-Minute Pipeline
1. `data/originalData/Temperature/15min_Temperature.ipynb` — 10-min → 15-min temperature → `temperature_15min.csv`
2. `data/originalData/WindDirection&Speed/15min_Wind.ipynb` — 10-min → 15-min wind (with sin/cos) → `wind_15min.csv`
3. `data/convertData/V1.5_15min_Dataset.ipynb` — Merge price + temp + wind → `V1.5_15min_Dataset.csv`
4. `data/convertData/V2.5_15min_feature_engineering.ipynb` — Feature engineering → `V2.5_15min_features.csv`
5. `models/modelV1.5.ipynb` → Train/evaluate V1.5
6. `models/modelV2.5.ipynb` → Train/evaluate V2.5

### Recommended Execution Order for Beginners

To understand the learning curve, run in this order:

```
V1 (hourly, weather only)  →  V2 (hourly, engineered)  →  V1.5 (15-min, weather)  →  V2.5 (15-min, engineered)
```

## Features

All engineered features are created in `V2.5_15min_feature_engineering.ipynb` (15-min) and `feature_engineering.ipynb` (hourly):

| Category | Features | Purpose |
|----------|----------|---------|
| **Temporal** | `hour`, `minute`, `day_of_week`, `month`, `season`, `time_of_day` | Capture daily/weekly/seasonal patterns |
| **Cyclic** | `hour_sin/cos`, `month_sin/cos`, `day_of_week_sin/cos` | Preserve circular continuity |
| **Holiday** | `is_holiday`, `is_non_working` | Flag Finnish public holidays |
| **Lag** | `price_lag_1` ... `price_lag_672` (15-min steps) | Autoregressive price history |
| **Rolling** | `price_rolling_mean_1h/24h/7d`, `price_rolling_std_24h` | Recent trends and volatility |
| **Weather-derived** | `HDD`, `wind_power_proxy`, `temp_lag_4/96` | Energy supply/demand proxies |

## Dependencies

- Python 3.12+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- holidays
- joblib

Install with: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn holidays joblib`

## Results Summary

### Comparison Table

```
Model       Resolution   Features          RMSE     R²      Improvement
──────      ──────────   ────────          ────     ──      ───────────
V1          Hourly       Weather only      46.34    0.107   Baseline
V1.5        15-min       Weather only      45.78    0.125   +1.2% RMSE
V2          Hourly       Engineered        14.62    0.911   -68.5% RMSE
V2.5        15-min       Engineered         8.22    0.972   -82.3% RMSE vs V1
```

### What This Means

- **V2.5 is production-ready** with an average error of ~3 EUR/MWh
- **Feature engineering** (lags, rolling, calendar) is essential — without it, even more data doesn't help
- **Higher resolution** (15-min vs hourly) adds another 44% RMSE reduction on top of engineered features

## Future Work

- Hyperparameter tuning (grid search or Optuna) to push R² beyond 0.98
- Time-series cross-validation for stability checks
- SHAP feature importance analysis for interpretability
- Real-time prediction script (`predict.py`)
- External features: EU carbon prices, fuel costs, grid load
- Ensemble: combine hourly + 15-min predictions
