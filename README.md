# Nordpool Electricity Price Prediction

Predict Finland (FI) Nord Pool day-ahead electricity prices using weather data and machine learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Four Model Versions](#four-model-versions)
- [Project Structure](#project-structure)
- [XGBoost vs LightGBM](#xgboost-vs-lightgbm)
- [Live Prediction Automation](#live-prediction-automation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

This project predicts Finnish electricity spot prices using temperature, wind speed, and wind direction data. It covers two resolutions (hourly and 15-minute) and four model versions, showing how feature engineering and temporal granularity affect prediction accuracy.

The pipeline has two main flows:

```
Historical: Raw CSV → Data Cleaning → Feature Engineering → Train → Save Model
Live:       APIs → Build Features → Load Saved Model → Predict 7 Days → Save CSV
```

---

## Four Model Versions

| Model    | Resolution | Features                                           | RMSE     | R²        | Description                          |
| -------- | ---------- | -------------------------------------------------- | -------- | --------- | ------------------------------------ |
| V1       | Hourly     | Weather only (temp, wind)                          | 46.34    | 0.107     | Baseline — weather-only hourly model |
| V1.5     | 15-min     | Weather only (temp, wind, direction)               | 45.78    | 0.125     | 15-min baseline — weather only       |
| V2       | Hourly     | Full engineered (lags, rolling, calendar, holiday) | 14.62    | 0.911     | Hourly + feature engineering         |
| **V2.5** | **15-min** | **Full engineered**                                | **8.22** | **0.972** | **Best model — highest accuracy**    |

### Key Findings

1. **Higher resolution alone (V1 → V1.5) barely helps** — R² 0.107 → 0.125. More rows of weak features don't help.
2. **Feature engineering (V1 → V2) is the real breakthrough** — R² 0.107 → 0.911. Lag features, rolling statistics, and calendar features capture the autoregressive nature of electricity prices.
3. **Both combined (V2 → V2.5) gives the best result** — R² 0.911 → 0.972, RMSE drops from 14.62 to 8.22.

---

## Project Structure

```
nordpool_electricity_price_prediction/
│
├── README.md                        ← This file
├── environment.yml                  ← Conda environment config
├── requirements.txt                 ← pip dependencies
│
├── data/
│   ├── originalData/
│   │   ├── electricPrices/          ← Price CSV files
│   │   ├── Temperature/             ← Temperature CSVs + 15-min resampling notebook
│   │   └── WindDirection&Speed/     ← Wind CSVs + 15-min resampling notebook
│   └── convertData/                 ← Processed datasets
│       ├── V1.5_15min_Dataset.csv   ← Merged 15-min data (no features)
│       └── V2.5_15min_features.csv  ← Feature-engineered 15-min data
│
├── xgboost_models/                  ← XGBoost training notebooks
│   ├── modelV1.ipynb                ← V1: hourly, weather only
│   ├── modelV1.5.ipynb              ← V1.5: 15-min, weather only
│   ├── modelV2.ipynb                ← V2: hourly, engineered features
│   └── modelV2.5.ipynb              ← V2.5: 15-min, engineered features (best)
│
├── lightgbm_models/                 ← LightGBM training notebooks
│   ├── modelV2.ipynb                ← V2: hourly + engineered
│   └── modelV2.5.ipynb              ← V2.5: 15-min + engineered
│
├── models/
│   └── saved/                       ← Trained model files (.pkl) used by live predictor
│       ├── xgboost_v1.pkl
│       ├── xgboost_v1_5.pkl
│       ├── xgboost_v2.pkl
│       ├── xgboost_v2_5.pkl
│       ├── lightgbm_v2.pkl
│       └── lightgbm_v2_5.pkl
│
├── src/                             ← Live prediction source code
│   ├── config.py                    ← Configuration (paths, API URLs)
│   ├── features.py                  ← Feature engineering (mirrors training notebooks)
│   ├── fetch_live.py                ← Fetch live prices + weather from APIs
│   ├── predict_system.py            ← Main program: load models, forecast, save
│   └── utils.py                     ← Utility functions
│
├── predictions/                     ← Daily forecast output CSVs
│
├── tests/                           ← Unit tests
│   ├── test_features.py
│   ├── test_fetch_live.py
│   └── test_predict_system.py
│
├── notebooks/
│   └── LearningNotes_CQL/           ← Learning notes (Chinese / English mixed)
│
└── .github/workflows/               ← GitHub Actions auto-run config
```

---

## XGBoost vs LightGBM

### XGBoost (`xgboost_models/` folder)

- **Tree growth:** level-wise (grows all nodes at the same level simultaneously)
- **Pros:** More conservative, less overfitting, beginner-friendly
- **Cons:** Slower on large datasets
- **Used with:** Default hyperparameters

### LightGBM (`lightgbm_models/` folder)

- **Tree growth:** leaf-wise (splits only the leaf with highest loss)
- **Pros:** Faster training, lower memory usage, can be more accurate with tuning
- **Cons:** Can overfit on small data, needs careful parameter tuning
- **Used with:** Optuna automatic hyperparameter search

### In this project

Both algorithms are trained on the same data with the same features. The LightGBM versions use Optuna tuning and perform slightly better. All 6 saved models are loaded and run by the live predictor every day.

---

## Live Prediction Automation

### How the daily forecast works

```
Start: python src/predict_system.py
  │
  ├─ 1. fetch_live.py
  │     ├─ Gets FI day-ahead prices from Elering API (free, no API key)
  │     └─ Gets weather forecast from FMI + Open-Meteo (free)
  │
  ├─ 2. features.py
  │     ├─ Builds time features (hour, minute, day_of_week, season...)
  │     ├─ Builds lag features (price_lag_1, price_lag_96...)
  │     ├─ Builds rolling features (rolling_mean, rolling_std...)
  │     └─ Builds holiday flags
  │
  ├─ 3. predict_system.py
  │     ├─ Loads all 6 models from models/saved/
  │     ├─ Recursively forecasts 7 days for each model
  │     ├─ Saves one CSV per model into predictions/
  │     └─ Fills in actual prices when they become available
  │
  └─ 4. (Optional) GitHub Actions
        └─ .github/workflows/ runs daily at UTC 11:00 automatically
```

### What is recursive forecasting?

Since the model needs "yesterday's price" to predict today, but future prices don't exist yet:

1. Predict hour 1 using real historical prices
2. Use the predicted hour 1 as "history" to predict hour 2
3. Repeat for all 7 days

This is called **recursive forecasting**.

### What is saved in a .pkl file?

Each `.pkl` file contains a Python dictionary with:

- `model` — the trained XGBoost or LightGBM object
- `feature_cols` — list of feature column names used during training
- `step_min` — the model's native resolution (60 = hourly, 15 = 15-min)

```python
# Save a model
meta = {
    'model': trained_model,
    'feature_cols': feature_columns_list,
    'step_min': 15,
}
joblib.dump(meta, 'models/saved/xgboost_v2_5.pkl')

# Load a model for prediction
meta = joblib.load('models/saved/xgboost_v2_5.pkl')
model = meta['model']
feature_cols = meta['feature_cols']
```

### Prediction output format

Each CSV in `predictions/` contains:

- `run_date` — date the forecast was run
- `target_datetime` — the predicted time slot
- `predicted_price` — forecasted price (EUR/MWh)
- `actual_price` — real price (filled in when available)
- `abs_error` — absolute error `|actual - predicted|`

---

## How to Run

### 1. Install environment

```powershell
conda env create -f environment.yml
conda activate nordpool
```

### 2. Run tests

```powershell
python -m unittest discover -s tests -v
```

### 3. Run live prediction

```powershell
python src/predict_system.py
```

Forecasts will be saved in `predictions/`.

### 4. Train models locally

Open and run the notebooks:

- `xgboost_models/modelV1.ipynb` → trains V1
- `xgboost_models/modelV2.ipynb` → trains V2
- `xgboost_models/modelV2.5.ipynb` → trains V2.5 (best)
- `lightgbm_models/modelV2.5.ipynb` → trains LightGBM V2.5

After training, the model is saved to `models/saved/`.

---

## Dependencies

- Python 3.11+
- pandas, numpy
- scikit-learn, xgboost, lightgbm, optuna
- matplotlib, seaborn, holidays
- joblib, requests

See `environment.yml` or `requirements.txt` for exact versions.
