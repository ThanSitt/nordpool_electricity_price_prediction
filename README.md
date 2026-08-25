# Nordpool Electricity Price Prediction

Predict Finland (FI) Nord Pool day-ahead electricity prices using weather data and machine learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Model Versions](#model-versions)
- [Project Structure](#project-structure)
- [XGBoost vs LightGBM](#xgboost-vs-lightgbm)
- [V2.5.2 & V2.5.1 Experiments](#v252--v251-experiments)
- [Live Prediction Automation](#live-prediction-automation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

This project predicts Finnish electricity spot prices using temperature, wind speed, wind direction, cross-border grid flows, and nuclear output data. It covers two resolutions (hourly and 15-minute) and a family of XGBoost/LightGBM models (V1 → V2.5.3 → V3/V4 experiments), showing how feature engineering, hyperparameter tuning, and supply-side data affect prediction accuracy.

The pipeline has two main flows:

```
Historical: Raw CSV → Data Cleaning → Feature Engineering → Train → Save Model
Live:       APIs → Build Features → Load Saved Model → Predict 7 Days → Save CSV
```

---

## Model Versions

| Model             | Features                                           | Test MAE   | RMSE       | R²         | Description                           |
| ----------------- | -------------------------------------------------- | ---------- | ---------- | ---------- | ------------------------------------- |
| V1                | Weather only (temp, wind)                          | 33.13      | 46.34      | 0.107      | Hourly weather-only baseline          |
| V1.5              | Weather only (temp, wind, direction)               | 32.19      | 45.78      | 0.125      | 15-min weather-only baseline          |
| V2                | Full engineered (lags, rolling, calendar, holiday) | 7.22       | 14.62      | 0.911      | Hourly + feature engineering          |
| V2.5              | Full engineered (49)                               | 2.82       | 8.22       | 0.972      | 15-min + engineered (default XGBoost) |
| **V2.5.3**        | Full engineered (49), **Optuna-tuned**             | **2.7236** | **8.1642** | **0.9722** | **Best production XGBoost**           |
| V3 (XGBoost)      | + grid flows (62), **default params**              | 2.847      | 8.368      | 0.971      | Temporary regression — grid hurt      |
| V3.1 (XGBoost)    | + grid flows (62), **Optuna-tuned**                | 2.7152     | 8.0699     | 0.9728     | Grid helps under tuned model          |
| **V4 (XGBoost)**  | + grid + nuclear (68), **Optuna-tuned**            | 2.6993     | 8.0321     | 0.9731     | Best XGBoost — experiment only        |
| **LightGBM V3.1** | + grid + nuclear (68), **V2.5 params**             | **2.6390** | **7.8957** | **0.9740** | **Best model overall — live**         |

### Key Findings

1. **Higher resolution alone (V1 → V1.5) barely helps** — R² 0.107 → 0.125. More rows of weak features don't help.
2. **Feature engineering (V1 → V2) is the real breakthrough** — R² 0.107 → 0.911. Lag features, rolling statistics, and calendar features capture the autoregressive nature of electricity prices.
3. **Both combined (V2 → V2.5) gives the best result** — R² 0.911 → 0.972, RMSE drops from 14.62 to 8.22.
4. **Then tuning beats more features** — Optuna tuning (V2.5.3) beat adding grid features alone: V3 (default params) was a temporary **regression** (MAE 2.847), but re-testing the same grid features under the tuned model (V3.1) **helped** (MAE 2.7152). Adding real supply-side data (nuclear) gives V4, the best XGBoost so far (MAE 2.6993), but it is **not** in the live pipeline yet.

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
│   │   ├── WindDirection&Speed/     ← Wind CSVs + 15-min resampling notebook
│   │   ├── GridTransmission/        ← Fingrid cross-border flows → grid_transmission_15min.csv
│   │   └── Nuclear/                 ← Fingrid nuclear output → nuclear_measured_15min.csv
│   └── convertData/                 ← Processed datasets
│       ├── V1.5_15min_Dataset.csv   ← Merged 15-min data (no features)
│       ├── V2.5_15min_features.csv  ← Feature-engineered 15-min data
│       ├── V3_15min_features.csv    ← V2.5 + grid features (V3)
│       ├── V3.1_15min_features.csv  ← V2.5 + grid + nuclear (V4 / shared dataset)
│       ├── feature_high_volatility.ipynb   ← High-volatility classifier + feature (V2.5.1)
│       └── V2.5.1_15min_Risk_Enhanced_Dataset.csv  ← V2.5 + high_volatility_prob feature
│
├── xgboost_models/                  ← XGBoost training notebooks
│   ├── modelV1.ipynb                ← V1: hourly, weather only
│   ├── modelV1.5.ipynb              ← V1.5: 15-min, weather only
│   ├── modelV2.ipynb                ← V2: hourly, engineered features
│   ├── modelV2.5.ipynb              ← V2.5: 15-min, engineered features
│   ├── modelV2.5.2.ipynb            ← V2.5.2: fair XGBoost vs LightGBM (Optuna-tuned)
│   ├── modelV2.5.1.ipynb            ← V2.5.1: controlled test of the risk feature
│   ├── modelV2.5.3.ipynb            ← V2.5.3: Optuna-tuned XGBoost (best production)
│   ├── modelV2.5.1.1.ipynb          ← V2.5.1.1: risk re-test under tuned model
│   ├── modelV3.ipynb                ← V3: grid features (tuned)
│   ├── modelV3.1.ipynb              ← V3.1: grid re-test under tuned model
│   ├── modelV3.1_live.ipynb         ← V3.1_live: live-safe grid lags (rejected)
│   └── modelV4.ipynb                ← V4: grid + nuclear (best, experiment)
│
├── lightgbm_models/                 ← LightGBM training notebooks
│   ├── modelV2.ipynb                ← V2: hourly + engineered
│   ├── modelV2.5.ipynb              ← V2.5: 15-min + engineered
│   └── modelV3.1.ipynb              ← V3.1: grid + nuclear (best, live)
│
├── models/
│   ├── saved/                       ← Live predictor models (.pkl) — 10 models
│   │   ├── xgboost_v1/v1_5/v2/v2_5/v2_5_2/v2_5_3.pkl
│   │   └── lightgbm_v2/v2_5/v2_5_2/v3_1.pkl
│   └── experiments/                 ← Not live (grid/nuclear not in src/): xgboost_v3/v3_1/v4.pkl
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
├── docs/
│   ├── Project-Learning-Notes.md    ← Deep-dive project notes (16 sections)
│   └── LearningNotes_CQL/           ← 17 bilingual learning guides (01–17)
│
└── .github/workflows/               ← GitHub Actions auto-run config
```

---

## XGBoost vs LightGBM

### XGBoost (`xgboost_models/` folder)

- **Tree growth:** level-wise (grows all nodes at the same level simultaneously)
- **Pros:** More conservative, less overfitting, beginner-friendly
- **Cons:** Slower on large datasets
- **Used with:** Default hyperparameters for V1–V2.5; Optuna-tuned for V2.5.3+

### LightGBM (`lightgbm_models/` folder)

- **Tree growth:** leaf-wise (splits only the leaf with highest loss)
- **Pros:** Faster training, lower memory usage, can be more accurate with tuning
- **Cons:** Can overfit on small data, needs careful parameter tuning
- **Used with:** Optuna automatic hyperparameter search

### In this project

Both algorithms are trained on the same data with the same features. The LightGBM versions use Optuna tuning and perform slightly better. The **best model overall is LightGBM V3.1** (grid + nuclear, MAE 2.6390) — the first live model using Fingrid grid/nuclear features. All **10 saved models** in `models/saved/` are loaded and run by the live predictor every day. XGBoost V3/V4 remain in `models/experiments/`.

---

## V2.5.2 & V2.5.1 Experiments (2026-08)

Two follow-up experiments were added after the original four model versions. They answer two separate questions:

1. **V2.5.2 — Fair comparison:** which algorithm is really better, once both are tuned fairly?
2. **V2.5.1 — Feature test:** does adding a "high-volatility probability" feature actually help the price model?

### V2.5.2 — Fair Comparison: XGBoost vs LightGBM (both Optuna-tuned)

The original XGBoost models used near-default hyperparameters while the LightGBM models were Optuna-tuned, which made the comparison **unfair** (an undertrained XGBoost vs. a carefully tuned LightGBM).

**Method** (`xgboost_models/modelV2.5.2.ipynb`) — everything identical except the algorithm:

| Control variable | Setting                                                             |
| ---------------- | ------------------------------------------------------------------- |
| Data             | `V2.5_15min_features.csv` (105,216 rows)                            |
| Loss function    | MAE (`reg:absoluteerror` for XGBoost, `regression_l1` for LightGBM) |
| Search           | Optuna, 10 trials × 5-fold `TimeSeriesSplit`                        |
| Trees            | 2000 for both                                                       |
| Split            | chronological 80/20                                                 |

**Results:**

| Model           | CV MAE | Test MAE   | Test RMSE  | Test R²    |
| --------------- | ------ | ---------- | ---------- | ---------- |
| XGBoost V2.5.2  | 2.9952 | 2.7652     | 8.2342     | 0.9717     |
| LightGBM V2.5.2 | 2.8851 | **2.7167** | **8.0958** | **0.9727** |

**Conclusion:** under a fair setup, LightGBM is slightly better on all metrics (~1.8% lower MAE), but the two are much closer than the original comparison suggested. Tuning XGBoost clearly helped it (MAE dropped from ~2.82 to 2.77).

### V2.5.1 — High-Volatility Probability Feature Experiment

**Idea:** train a "storm warning" classifier (XGBClassifier) that takes only **weather + time** and outputs `high_volatility_prob` — a 0–1 probability that "this 15-minute moment will be highly volatile" (defined as the top 15% of a 6-hour rolling price std). Then add this probability as a new feature to the price model.

**Feature building** (`data/convertData/feature_high_volatility.ipynb`):

- Label: `is_high_volatility = (6h rolling std of price >= 85th percentile)` → 15% positive
- Classifier inputs: `temp`, `wind_speed`, `wind_direction_deg`, `temp_lag_4`, `hour`, `day_of_week`, `month`
- Test accuracy 0.84, but recall for high volatility only **0.24** (class imbalance — the model misses most spikes)

**Step 1 — Signal check (histogram):** does the feature carry information?

| Group                     | Mean probability | Median |
| ------------------------- | ---------------- | ------ |
| Stable (label=0)          | 0.113            | 0.070  |
| High volatility (label=1) | 0.341            | 0.295  |

The two distributions separate → the feature **carries signal**, but weakly (overlapping tails).

**Step 2 — Controlled test** (`xgboost_models/modelV2.5.1.ipynb`): same data, same chronological 80/20 split, same Optuna-tuned hyperparameters; the **only** difference is whether `high_volatility_prob` is included as a feature. (`price_roll_std_6h` and `is_high_volatility` were excluded from both versions — they are answers derived from price and would leak information.)

| Model    | Without risk feature | With risk feature | MAE change | Verdict |
| -------- | -------------------- | ----------------- | ---------- | ------- |
| XGBoost  | 2.7555               | 2.7957            | +0.0402    | worse   |
| LightGBM | 2.7165               | 2.7426            | +0.0261    | worse   |

**Conclusion:** adding `high_volatility_prob` made both models slightly **worse** (MAE up ~1%). In its current form this feature should **not** be added to V2.5.2. The likely cause is the weak classifier (recall 0.24) feeding "weak signal + noise" into models that already have 49 strong features.

**Lesson:** this honest negative result is valuable — a feature can pass the signal check (histogram) yet still fail to help a model. Always validate features with a controlled experiment before adding them.

---

## Live Prediction Automation

### How the daily forecast works

```
Start: python src/predict_system.py
  │
  ├─ 1. fetch_live.py
  │     ├─ Gets FI day-ahead prices from Elering API (free, no API key)
  │     ├─ Gets weather forecast from FMI + Open-Meteo (free)
  │     └─ Gets grid flows + nuclear output from Fingrid (FINGRID_API_KEY)
  │
  ├─ 2. features.py
  │     ├─ Builds time features (hour, minute, day_of_week, season...)
  │     ├─ Builds lag features (price_lag_1, price_lag_96...)
  │     ├─ Builds rolling features (rolling_mean, rolling_std...)
  │     └─ Builds holiday flags
  │
  ├─ 3. predict_system.py
  │     ├─ Loads all 10 models from models/saved/
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
- `xgboost_models/modelV2.5.ipynb` → trains V2.5
- `xgboost_models/modelV2.5.3.ipynb` → trains V2.5.3 (best production, Optuna-tuned)
- `xgboost_models/modelV4.ipynb` → trains V4 (grid + nuclear, best experiment)
- `lightgbm_models/modelV2.5.ipynb` → trains LightGBM V2.5
- `lightgbm_models/modelV3.1.ipynb` → trains LightGBM V3.1 (best model, live)

After training, the model is saved to `models/saved/`. The live run also needs a `FINGRID_API_KEY` environment variable (set as a GitHub secret) for the Fingrid grid + nuclear fetchers.

---

## Dependencies

- Python 3.11+
- pandas, numpy
- scikit-learn, xgboost, lightgbm, optuna
- matplotlib, seaborn, holidays
- joblib, requests

See `environment.yml` or `requirements.txt` for exact versions.
