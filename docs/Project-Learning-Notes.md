# Project Learning Notes — Nordpool Electricity Price Prediction

> **Project**: Predict Finland (FI) Nord Pool day-ahead electricity spot prices (EUR/MWh) using weather data + engineered features + gradient-boosted trees.
> **Repo**: `https://github.com/ThanSitt/nordpool_electricity_price_prediction`
> **Last updated**: 2026-08-25
>
> These notes are a deep, code-grounded companion to the `README.md`. The README is the quick reference; this document explains _why_ each piece exists, how the system works end-to-end, what has been tried, and what is currently broken or unexplored.

---

## 1. Project Overview

This project predicts **Finland (FI)** Nord Pool day-ahead electricity spot prices at **two time resolutions** (hourly and 15-minute) using **two gradient-boosting algorithms** (XGBoost and LightGBM). It has evolved through **four core model versions** (V1 → V1.5 → V2 → V2.5), several controlled follow-up experiments (V2.5.1, V2.5.2, **V2.5.3 tuned XGBoost**), and supply-side extensions: **V3** (cross-border grid) and **V4** (grid + nuclear power).

**Team split (2026-08)**: a partner tunes **LightGBM**, this user tunes **XGBoost** — they share one dataset `V3.1_15min_features.csv` (V2.5 + grid + nuclear). Version numbers differ per person: partner's LightGBM "V3" = the shared dataset; the user's XGBoost **V3 = grid**, **V4 = grid + nuclear**.

The system has **two independent flows** that share the same feature definition:

```
Historical flow (notebooks):  Raw CSV → Cleaning → Feature Engineering → Train → Save .pkl
Live flow (src/):            Public APIs → Build Features → Load .pkl → Recursive 7-day forecast → CSV → GitHub Actions
```

**The single most important learning in the project** (verified repeatedly): _feature engineering matters far more than temporal resolution._

- Increasing resolution without adding features (V1 → V1.5): R² 0.107 → 0.125 (almost no gain).
- Adding engineered features at the same resolution (V1 → V2): R² 0.107 → 0.911 (breakthrough).
- Doing both (V2 → V2.5): R² 0.911 → 0.972, RMSE 14.62 → 8.22 (best model).

---

## 2. Product Purpose

- **What it predicts**: the FI Nord Pool day-ahead spot price for every 15-minute slot over the next 7 days (672 slots), in EUR/MWh.
- **Why it matters**: day-ahead prices drive energy trading, hedging, and consumption planning. A usable forecast (MAE ≈ 2.8 EUR/MWh offline) is well within practical decision tolerance.
- **How it is delivered**: `src/predict_system.py` runs daily and writes one CSV per model into `predictions/`. A GitHub Actions workflow runs it automatically every day at 11:00 UTC (≈13:00/14:00 Finland time, after Nord Pool publishes tomorrow's prices ~12:00 EET) and commits the updated forecasts.
- **Educational goal**: the repository is deliberately built as a _learning_ project. `docs/LearningNotes_CQL/` contains 17 bilingual (English/Chinese) learning guides (01–17) that document every concept: data cleaning, feature engineering, model training, comparison, automation, visualization, the `src/` folder, why features can fail, the nuclear/V4 work, the model warehouse (saved vs experiments), and the EDA + forecast visualization walkthrough. The `.pkl` "model bundle" concept, recursive forecasting, and time-series evaluation are all explained there for a beginner audience.

---

## 3. System Architecture

### 3.1 High-level picture

```mermaid
flowchart LR
    subgraph TRAIN["Training (notebooks, one-time)"]
        RAW[Raw CSVs<br/>price / temp / wind / grid] --> CLEAN[Cleaning + alignment<br/>Europe/Helsinki, 15-min]
        CLEAN --> FEAT[Feature engineering<br/>lags · rolling · calendar · weather]
        FEAT --> FIT[Fit XGBoost / LightGBM]
        FIT --> PKL[models/saved/*.pkl<br/>{model, feature_cols, step_min}]
    end

    subgraph LIVE["Live prediction (daily, automated)"]
        API[Elering NPS prices<br/>FMI weather<br/>Open-Meteo long-range] --> FETCH[fetch_live.py]
        FETCH --> BUILD[features.py<br/>PriceBuffer + WeatherBuffer]
        BUILD --> REC[recursive 7-day forecast<br/>predict_system.py]
        PKL --> REC
        REC --> CSV[predictions/*_forecasts.csv<br/>+ back-filled actuals + abs_error]
        CSV --> GHA[GitHub Actions<br/>daily_forecast.yml · cron 0 11 * * *]
    end
```

### 3.2 Live prediction flow (module by module)

| Order | Module                                 | Responsibility                                                                                                                                                                                                             |
| ----- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `src/fetch_live.py`                    | Fetch FI day-ahead prices (Elering NPS, 15-min, **no API key**) + weather from **FMI** (observations + ~54 h HIRLAM short-range) + **Open-Meteo** (long-range, 10 forecast days).                                          |
| 2     | `src/features.py`                      | `build_features()` + `PriceBuffer` + `WeatherBuffer`. Mirrors the training feature pipeline exactly so predictions are valid on live data. One function covers all 4 model versions via each model's saved `feature_cols`. |
| 3     | `src/predict_system.py`                | Main program: loads every `*.pkl` in `models/saved/`, recursively forecasts 7 days at each model's native `step_min` resolution, back-fills available actuals, and writes one CSV per model.                               |
| 4     | `.github/workflows/daily_forecast.yml` | Automation: installs pinned deps → runs unit tests → runs `predict_system.py` → commits `predictions/` changes.                                                                                                            |

### 3.3 Design decisions worth noting

- **Timezone discipline**: everything is converted to `Europe/Helsinki`. Raw CSVs have mixed offsets (+02:00 winter / +03:00 summer); the training notebooks fix this by parsing with `utc=True` then converting to Helsinki time.
- **`step_min` contract**: each `.pkl` declares its native resolution (`60` = hourly, `15` = 15-min). Hourly models average 15-min actuals for both history and evaluation; 15-min models use the raw quarter-hour series. This is tested in `tests/test_predict_system.py`.
- **Forecast start**: predictions always start at the **next local midnight** (never halfway through the current delivery day), because Nord Pool has usually already published today's price.
- **No silent data fabrication**: `fetch_weather` raises an error if the long-range weather source is incomplete rather than forward-filling a stale 54-hour forecast for the remaining days.
- **`config.py` contains no credentials** — all live sources are public APIs, so a fresh clone can run without a secrets file.

---

## 4. Technology Stack

### 4.1 Languages & runtime

- **Python 3.11** (pinned in `environment.yml`; GitHub Actions uses `actions/setup-python` 3.11)
- Windows PowerShell for local dev; Linux (ubuntu-latest) for CI

### 4.2 Dependencies (pinned in `requirements.txt`)

Versions are **pinned deliberately**: the saved `joblib` model bundles must deserialize consistently between training and the daily live run.

| Area                      | Packages                                                                    |
| ------------------------- | --------------------------------------------------------------------------- |
| Data                      | `pandas==3.0.3`, `numpy==2.4.6`                                             |
| ML                        | `scikit-learn==1.9.0`, `xgboost==3.2.0`, `lightgbm==4.6.0`, `optuna==4.6.0` |
| Serialization             | `joblib==1.5.3`                                                             |
| Calendar                  | `holidays==0.98` (Finnish holidays)                                         |
| Networking                | `requests==2.34.2`                                                          |
| Visualization (notebooks) | `matplotlib==3.10.9`, `seaborn==0.13.2`, `plotly==6.9.0`, `kaleido==1.3.0`  |
| Notebooks                 | `jupyterlab==4.5.8`, `ipykernel==7.2.0`                                     |

### 4.3 Environment

- `environment.yml` → conda env `nordpool` (python 3.11 + pip installs `requirements.txt`).
- A local `.venv` is also present for development.

### 4.4 Automation

- **GitHub Actions** (`.github/workflows/daily_forecast.yml`): cron `0 11 * * *` UTC, plus manual `workflow_dispatch`. Runs offline tests before the forecast so a broken pipeline never commits garbage.

---

## 5. Data Sources

### 5.1 Historical (training) data — `data/originalData/`

| Source                  | Files                                                                                                                                         | Resolution                 | Notes                                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Electricity prices      | `electricPrices/electricity_prices_helsinki.csv` (hourly), `electricPrices/15min_2023_2025.csv` (15-min)                                      | hourly / 15-min            | Nord Pool FI spot price                                                                                                                                        |
| Temperature             | `Temperature/` — Helsinki-Vantaa airport raw 10-min CSVs (6-month chunks 2023–2025) → `temperature_15min.csv`, `Temperature_hourly_clean.csv` | 10-min raw → 15-min/hourly | via `15min_Temperature.ipynb`                                                                                                                                  |
| Wind                    | `WindDirection&Speed/` — Oulu Vihreäsaari harbour raw 10-min CSVs → `wind_15min.csv`, `hourly_wind_speed.csv`                                 | 10-min raw → 15-min/hourly | via `15min_Wind.ipynb`                                                                                                                                         |
| Grid transmission (V3)  | `GridTransmission/grid_transmission_15min.csv`                                                                                                | 15-min                     | Fingrid Open Data: `fi_ee`, `fi_no`, `fi_se_north`, `fi_se_central` (MW net flow; positive = FI exports)                                                       |
| **Nuclear (V3.1, new)** | `Nuclear/nuclear_measured_15min.csv` (105,216 rows)                                                                                           | 15-min                     | Fingrid "Nuclear power production — real-time data" (dataset 188) via `15min_NucleatData.ipynb`; **measured** output → only lag/rolling features are live-safe |

> Historical anomaly worth remembering: an earlier version fetched **Fingrid dataset 105** as the price source — that dataset is actually _down-regulation bid volume (MW)_, which produced a long run of zeros. The project switched to **Elering's NPS endpoint** for the real FI area price. (This is documented in `fetch_live.py`.)

### 5.2 Live data sources (no credentials needed)

| Source                                     | What it provides                                                                                                 | Endpoint                                     |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Elering NPS**                            | FI Nord Pool day-ahead prices, 15-min resolution                                                                 | `https://dashboard.elering.ee/api/nps/price` |
| **FMI** (Finnish Meteorological Institute) | Observations (`place=Helsinki`/`place=Oulu`) + HIRLAM short-range forecast (~54 h, `latlon=` grid interpolation) | `https://opendata.fmi.fi/wfs`                |
| **Open-Meteo**                             | Long-range hourly forecast (10 forecast days) for the remaining days                                             | `https://api.open-meteo.com/v1/forecast`     |
| **Fingrid (V3.1, new)**                    | Cross-border grid flows (datasets 55/57/60/61) + nuclear output (dataset 188), 15-min — **requires `FINGRID_API_KEY` env var** | `https://data.fingrid.fi/api/datasets/...` |

**Station pairing is deliberate**: temperature comes from Helsinki-Vantaa airport and wind from Oulu — the **same stations used in training** — so live features stay consistent with training distributions.

---

## 6. Data Pipeline

The pipeline is a sequence of notebooks that each produce the next dataset. Each notebook is self-documenting.

### 6.1 Notebook → dataset chain

| Notebook                               | Output                                                       | Description                                                                         |
| -------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `V1_data_cleaning_and_alignment.ipynb` | `V1_finland_electricity_predict_dataset.csv` (26,304 rows)   | Hourly: clean + align price/temp/wind                                               |
| `V1.5_15min_Dataset.ipynb`             | `V1.5_15min_Dataset.csv` (105,216 rows)                      | 15-min merged base (no features)                                                    |
| `V2_feature_engineering.ipynb`         | `V2_finland_electricity_features.csv` (26,304 rows, 43 cols) | Hourly + engineered features                                                        |
| `V2.5_15min_feature_engineering.ipynb` | `V2.5_15min_features.csv` (105,216 rows, 51 cols)            | 15-min + engineered features (**used by training**)                                 |
| `feature_high_volatility.ipynb`        | `V2.5.1_15min_Risk_Enhanced_Dataset.csv` (105,193 rows)      | Adds `high_volatility_prob` (experiment, rejected)                                  |
| `V3_15min_feature_engineering.ipynb`   | `V3_15min_features.csv` (105,216 rows, 64 cols)              | V2.5 + cross-border grid flows (trains XGBoost V3)                                  |
| `V3.1_15min_feature_engineering.ipynb` | `V3.1_15min_features.csv` (105,216 rows, 70 cols)            | **Shared dataset**: V2.5 + grid + nuclear (trains XGBoost V4 / partner LightGBM V3) |

### 6.2 Data cleaning & alignment steps (from `V1_data_cleaning_and_alignment.ipynb`)

1. **Parse timestamps** — read as UTC, convert to `Europe/Helsinki`.
2. **Resample** all 10-min weather to the target resolution (15-min or hourly).
3. **Merge** price + temp + wind on the common datetime index (left joins).
4. **Handle gaps** — inspect NaNs, forward/back-fill only where justified.
5. **Rename** columns to a simple, consistent style (e.g. `temp`, `wind_speed`, `wind_direction_deg`).
6. **Sort** by time, drop duplicates.
7. **Save** a clean intermediate before feature engineering.

### 6.3 V3 / V3.1 merge specifics

- **V3 (grid)**: left-joins V2.5 onto `grid_transmission_15min.csv` on `datetime`; back-fills 8 leading NaN rows (2023-01-01 00:00–01:45 EET, before UTC midnight). Only **lagged** grid features are kept (`lag_96` = 24 h, `lag_672` = 7 d) — the current-period flow is not available at inference time.
- **V3.1 (grid + nuclear)**: starts from V3 and left-joins `nuclear_measured_15min.csv` (`nuclear_power_mw`, Fingrid dataset 188). Because nuclear is **measured** (realized) output, only lag/rolling derivatives are created (`nuclear_lag_96/672`, `nuclear_rolling_mean_24h/7d`, `nuclear_change_1d`). Output: `V3.1_15min_features.csv` (70 cols) — the shared dataset with the partner's LightGBM "V3".

---

## 7. Feature Engineering

All feature engineering is centralized in `src/features.py` (`build_features()`), which mirrors the training notebooks exactly. The full V2.5 feature set (49 features) is built for every timestamp; each model selects its own subset via its saved `feature_cols`.

### 7.1 Feature categories

| Category                      | Features                                                                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Temporal**                  | `hour`, `minute`, `day_of_week`, `day_of_month`, `month`, `week_of_year`, `quarter`, `year`, `season`, `time_of_day`                                     |
| **Cyclic (sin/cos)**          | `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos`, `week_of_year_sin/cos`                                                                           |
| **Calendar flags**            | `is_weekend`, `is_peak_hour`, `is_night_hour`, `is_holiday`, `is_non_working` (Finnish holidays via `holidays.Finland()`)                                |
| **Weather raw**               | `temp`, `wind_speed`, `wind_direction_deg`                                                                                                               |
| **Weather derived**           | `wind_dir_sin/cos` (directional continuity), `HDD = max(0, 17 − temp)` (heating-degree-day proxy), `wind_power_proxy = wind³` (wind-power physics proxy) |
| **Weather lags/rolling**      | `temp_lag_4/96/24h/168h`, `temp_rolling_mean_1h/24h`                                                                                                     |
| **Price lags (hourly, V2)**   | `price_lag_1h/2h/3h/6h/12h/24h/48h/168h`                                                                                                                 |
| **Price lags (15-min, V2.5)** | `price_lag_1/2/4/8/16/32/96/672` (15 min → 7 days)                                                                                                       |
| **Price rolling**             | `price_rolling_mean_1h/6h/24h/7d`, `price_rolling_std_1h/24h`, `price_rolling_min/max_24h`                                                               |
| **Grid flows (V3)**           | `fi_ee`, `fi_no`, `fi_se_north`, `fi_se_central`, `fi_se_total`, `fi_total_net`, `fi_se_abs` + `{fi_total_net, fi_se_total, fi_ee}_lag_96/672`           |
| **Nuclear (V3.1)**            | `nuclear_power_mw`, `nuclear_lag_96`, `nuclear_lag_672`, `nuclear_rolling_mean_24h`, `nuclear_rolling_mean_7d`, `nuclear_change_1d`                      |
| **Risk (V2.5.1, rejected)**   | `high_volatility_prob`                                                                                                                                   |

### 7.2 Why wind direction is encoded as sin/cos

Raw degrees are a bad feature: 359° and 1° are numerically far apart but physically nearly identical. `wind_dir_sin`/`wind_dir_cos` preserve angular continuity (this was one of the earliest explicit lessons in the learning guides).

### 7.3 Why lags and rolling stats dominate

Electricity prices are strongly **autoregressive** — today's price is the single best predictor of tomorrow's. Lag features (`price_lag_*`) and rolling statistics (`price_rolling_*`) let tree models exploit this, which is exactly why V2 jumped from R² 0.107 → 0.911.

### 7.4 Leakage guardrails (learned the hard way)

- In the V2.5.1 experiment, `price_roll_std_6h` and `is_high_volatility` were **excluded** from both the baseline and enhanced feature sets — they are answers derived from price and would leak information.
- Time features are always derived **from the target timestamp only**; rolling/lag values use **preceding windows excluding the current timestamp** (the `PriceBuffer`/`WeatherBuffer` helpers explicitly exclude `ts`).

---

## 8. Machine Learning Pipeline

### 8.1 Algorithms

|                 | XGBoost                                           | LightGBM                                       |
| --------------- | ------------------------------------------------- | ---------------------------------------------- |
| Tree growth     | level-wise (balanced)                             | leaf-wise (loss-driven)                        |
| Pros            | Conservative, less overfitting, beginner-friendly | Faster, lower memory, more accurate when tuned |
| Cons            | Slower on large data                              | Overfits small data; needs tuning              |
| In this project | `xgboost_models/`; default-ish params for V1–V2.5 | `lightgbm_models/`; Optuna-tuned               |

### 8.2 Training setup

- **Data**: `V2.5_15min_features.csv` (105,216 rows) for the best models; hourly files for V1/V2.
- **Target**: `price`; features = all other columns except `datetime`/`price`.
- **Split**: chronological 80/20 with `train_test_split(..., shuffle=False)` — **never shuffle** time-series data (future must not leak into training).
- **XGBoost V2.5 hyperparameters**: `objective='reg:squarederror'`, `learning_rate=0.1`, `n_estimators=100`, `max_depth=6`, `random_state=42` (identical to V1/V2 for a fair comparison).
- **V2.5.2 (fair comparison)**: both algorithms trained with Optuna — MAE loss, 10 trials × 5-fold `TimeSeriesSplit`, 2000 trees each.
- **Metrics**: MAE, MSE, RMSE, R², adjusted R² (`sklearn.metrics`).

### 8.3 Model artifact format (the `.pkl` contract)

```python
meta = {
    'model': trained_model,      # XGBoost or LightGBM object
    'feature_cols': [...],       # exact feature column order used at training
    'step_min': 15,              # 15 = 15-min, 60 = hourly
}
joblib.dump(meta, 'models/saved/xgboost_v2_5.pkl')

meta = joblib.load('models/saved/xgboost_v2_5.pkl')
model = meta['model']; feature_cols = meta['feature_cols']; step_min = meta['step_min']
```

This bundle is what decouples _training_ (notebook, one-time) from _prediction_ (script, daily).

### 8.4 Recursive forecasting (why it's needed and what it costs)

Because the model needs `price_lag_*` features, but future prices don't exist yet:

1. Predict the first slot using real historical prices.
2. Feed that prediction back into the price buffer as "history".
3. Repeat for all 672 slots.

```python
for i in range(FORECAST_HOURS * (60 // step_min)):
    features = build_features(timestamp, price_buf, wx_buf)
    prediction = model.predict(row)[0]
    price_buf.add(timestamp, prediction)   # feed prediction back in
```

**Cost**: errors compound. Every predicted value feeds the next prediction, so small errors can grow over the 7-day horizon. This is a major reason live recursive MAE is much higher than offline one-step test MAE (see §14).

### 8.5 Evaluation & back-filling

- Each `predictions/*.csv` accumulates rows across runs (`run_date` + `target_datetime` as unique key, `keep='last'` on duplicates).
- Once a forecast slot's actual price is published, `fill_actuals()` back-fills `actual_price` and `abs_error = |actual − predicted|`.
- Hourly models resample the 15-min actuals to hourly **mean**; 15-min models use raw quarter-hour values (covered by unit tests).

---

## 9. Model Evolution

### 9.1 Version lineage

```
V1 (hourly, weather-only)          V1.5 (15-min, weather-only)
        │                                   │
        └─────────── feature engineering ───┘
                         │
                V2 (hourly, engineered) ──→ V2.5 (15-min, engineered)
                         │                              │
                 (fair tuning)                  (risk feature test)
                         │                              │
                 V2.5.2 (Optuna both)          V2.5.1 (high_volatility_prob)
                         │
                 V2.5.3 (XGBoost, Optuna 30) ★ best production XGBoost
                         │
        ┌────────────────┴─────────────────────────┐
   V3 = + grid (default)        V3.1 = + grid (Optuna) → V4 = + grid + nuclear (Optuna)
   (regression, MAE 2.847)      (helps, MAE 2.7152)    ★ best overall (MAE 2.6993)
        (V3_15min_features.csv) (V3.1_15min_features.csv, shared with partner)
```

### 9.2 Per-version summary

| Model            | Features        | Train rows | Test MAE   | RMSE       | R²         | Note                                   |
| ---------------- | --------------- | ---------- | ---------- | ---------- | ---------- | -------------------------------------- |
| **V1**           | 2 (weather)     | ~21k       | 33.13      | 46.34      | 0.107      | Hourly baseline                        |
| **V1.5**         | 5 (weather)     | ~84k       | 32.19      | 45.78      | 0.125      | Resolution alone ≈ no gain             |
| **V2**           | 41 (engineered) | ~21k       | 7.22       | 14.62      | 0.911      | Feature engineering breakthrough       |
| **V2.5**         | 49 (engineered) | ~84k       | 2.82       | 8.22       | 0.972      | Default XGBoost                        |
| **V2.5.3**       | 49 (engineered) | ~84k       | 2.7236     | 8.1642     | 0.9722     | **XGBoost + Optuna 30** (production)   |
| **V3** (XGBoost)   | 62 (+13 grid)   | ~84k       | 2.847      | 8.368      | 0.9708     | Default params — temporary regression  |
| **V3.1** (XGBoost) | 62 (+13 grid)   | ~84k       | 2.7152     | 8.0699     | 0.9728     | Optuna re-test — grid helps            |
| **V4** (XGBoost)   | 68 (+6 nuclear) | ~84k       | **2.6993** | **8.0321** | **0.9731** | **Best XGBoost so far** (grid+nuclear) |
| **V3.1** (LightGBM) | 68 (+19 grid+nuclear) | ~84k | **2.6390** | **7.8957** | **0.9740** | **Best model overall — live** (V2.5 params) |

### 9.3 Why V2.5 wins

1. **Feature engineering** (the primary driver): lags/rolling/calendar capture autoregression.
2. **4× more training rows** (~84k vs ~21k) → better generalization.
3. **Finer resolution**: sees intra-hour dynamics hourly aggregation smooths away.
4. **Finer lag granularity**: lags at 1, 2, 4, 8, 16, 32, 96, 672 steps capture 15-min → 7-day dependencies.

### 9.4 Error profile

- RMSE (8.22) ≈ 3× MAE (2.82) → some large errors remain, typically during **extreme price spikes** (actual prices range 0–150 EUR/MWh; std ≈ 36).
- The RMSE/MAE gap is smaller than V2's (14.62 vs 7.22), so V2.5 handles volatile periods better.

---

## 10. Experiment History

| #   | Experiment                | Notebook                                                | Setup                                                  | Result                                                | Verdict                                             |
| --- | ------------------------- | ------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------- |
| 1   | V1 baseline               | `xgboost_models/modelV1.ipynb`                          | Hourly, weather only                                   | MAE 33.13 / RMSE 46.34 / R² 0.107                     | Baseline                                            |
| 2   | V1.5 resolution test      | `xgboost_models/modelV1.5.ipynb`                        | 15-min, weather only                                   | MAE 32.19 / R² 0.125                                  | Resolution alone ≈ no gain                          |
| 3   | V2 feature engineering    | `xgboost_models/modelV2.ipynb`                          | Hourly, engineered                                     | MAE 7.22 / R² 0.911                                   | Breakthrough                                        |
| 4   | V2.5 best model           | `xgboost_models/modelV2.5.ipynb`                        | 15-min, engineered                                     | MAE 2.82 / RMSE 8.22 / R² 0.972                       | **Best**                                            |
| 5   | LightGBM V2 & V2.5        | `lightgbm_models/modelV2*.ipynb`                        | Same data, Optuna-tuned LightGBM                       | Slightly better than XGBoost                          | LightGBM tuned ≈ wins                               |
| 6   | V2.5.2 fair comparison    | `xgboost_models/modelV2.5.2.ipynb`                      | Both Optuna-tuned, MAE loss, 2000 trees, 10×5-fold TSS | XGB MAE 2.7652; **LGBM MAE 2.7167**                   | LightGBM ~1.8% better; much closer than before      |
| 7   | V2.5.1 risk feature       | `xgboost_models/modelV2.5.1.ipynb`                      | ± `high_volatility_prob`, same data/split/params       | Both models got ~1% **worse**                         | Feature rejected                                    |
| 8   | V3 grid features (default) | `xgboost_models/modelV3.ipynb`                          | V2.5 + 13 grid, **default params**                     | MAE 2.847 (vs V2.5 2.82) — slightly worse            | Negative result; needs tuning                        |
| 9   | **V2.5.3 XGBoost Optuna** | `xgboost_models/modelV2.5.3.ipynb`                      | MAE loss, 30 trials × 5-fold TS-CV, 2000 trees         | MAE 2.7236 / RMSE 8.1642 / R² 0.9722                  | **Tuning > new features** — best production XGBoost |
| 10  | V2.5.1.1 risk re-test     | `xgboost_models/modelV2.5.1.1.ipynb`                    | risk feature under TUNED model                         | +0.0045 (still worse, smaller)                        | Risk feature robustly rejected                      |
| 11  | V3.1 grid re-test         | `xgboost_models/modelV3.1.ipynb`                        | grid under TUNED model (V2.5.3 params)                 | MAE 2.7152 (Δ −0.0084, helps)                        | **Tune first, then test features**                  |
| 12  | V3.1_live (grid lags)     | `xgboost_models/modelV3.1_live.ipynb`                   | lag-only grid (55), live-feasible                      | +0.0180 (hurts)                                       | Grid NOT deployable (train/serve gap)               |
| 13  | Nuclear + V3.1 dataset    | `data/convertData/V3.1_15min_feature_engineering.ipynb` | V3 + 6 nuclear features (shared dataset)               | `V3.1_15min_features.csv` (70 cols)                   | Shared with partner (his LightGBM "V3")             |
| 14  | **XGBoost V4**            | `xgboost_models/modelV4.ipynb`                          | V3 + nuclear, tuned params                             | **MAE 2.6993 / RMSE 8.0321 / R² 0.9731** (Δ−0.0159)   | **Best XGBoost so far** — nuclear helps             |
| 15  | **LightGBM V3.1**         | `lightgbm_models/modelV3.1.ipynb`                       | V3.1 dataset (68 feats), Optuna 30×5 then **V2.5 params** | CV 2.8485 → test **2.6390 / 7.8957 / 0.9740** | Optuna overfit; V2.5 regularization transferred — **new best, now live** |
| 16  | **Grid+nuclear live**     | `src/features.py`, `src/fetch_live.py`, `src/predict_system.py` | GridBuffer/NuclearBuffer + fetch_grid/fetch_nuclear (Fingrid) | `lightgbm_v3_1.pkl` runs daily with real grid/nuclear | Train/serve gap closed (2026-08-25)                  |

### 10.1 V2.5.2 details (fair XGBoost vs LightGBM)

| Control | Setting                                      |
| ------- | -------------------------------------------- |
| Data    | `V2.5_15min_features.csv` (105,216 rows)     |
| Loss    | MAE (`reg:absoluteerror` / `regression_l1`)  |
| Search  | Optuna, 10 trials × 5-fold `TimeSeriesSplit` |
| Trees   | 2000 for both                                |
| Split   | chronological 80/20                          |

Results: **LightGBM MAE 2.7167 / RMSE 8.0958 / R² 0.9727** vs **XGBoost MAE 2.7652 / RMSE 8.2342 / R² 0.9717**. Under a fair setup the two are very close; tuning XGBoost clearly helped it (MAE ~2.82 → 2.77).

### 10.2 V2.5.1 details (high-volatility probability feature)

- **Feature builder**: `data/convertData/feature_high_volatility.ipynb` — an XGBClassifier takes weather + time only and outputs `high_volatility_prob` (0–1). Label: `is_high_volatility = (6h rolling std of price ≥ 85th percentile)` (≈15% positive). Test accuracy 0.84 but **recall for high volatility only 0.24** (class imbalance).
- **Signal check (histogram)**: stable moments mean prob 0.113 vs high-volatility moments 0.341 → distributions separate, so the feature _carries signal_, but weakly (overlapping tails).
- **Controlled test**: same data/split/hyperparams; only difference is the extra feature → both models got slightly **worse** (MAE +1%).
- **Lesson**: a feature can pass the "signal" check yet fail the "does it help the model?" check. Always validate candidate features with a controlled experiment. This honest negative result saved future debugging time.

---

## 11. Important Files

### 11.1 Source code (`src/`)

| File                | Purpose                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `config.py`         | Runtime config: paths, API URLs, `FORECAST_HOURS=168`, `PRICE_HISTORY_HOURS=200`, **Fingrid dataset IDs + `GRID_HISTORY_HOURS`**, timezone. No hardcoded credentials. |
| `features.py`       | `build_features()` + `PriceBuffer` + `WeatherBuffer` + **`GridBuffer` + `NuclearBuffer`** — the canonical feature pipeline (now includes V3.1 grid/nuclear features). |
| `fetch_live.py`     | `fetch_prices()` (Elering) + `fetch_weather()` (FMI + Open-Meteo) + **`fetch_grid()` + `fetch_nuclear()`** (Fingrid, uses `FINGRID_API_KEY`). |
| `predict_system.py` | Main program: `load_models()`, `run_forecast()`, `fill_actuals()`, `save_csv()`, accuracy summary. Fetches grid + nuclear and passes buffers to every model. |
| `utils.py`          | Currently **empty** (placeholder).                                                                                  |
| `test_fmi.py`       | Diagnostic script for FMI WFS endpoint variants (latlon, place, fmisid, bbox, timestep).                            |

### 11.2 Tests (`tests/`)

| File                     | Covers                                                                                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_features.py`       | WeatherBuffer rolling windows exclude current timestamp.                                                                                                  |
| `test_fetch_live.py`     | Elering FI prices at quarter-hour resolution; rejects all-zero suspicious responses.                                                                      |
| `test_predict_system.py` | `fill_actuals` resolution matching (15-min raw / hourly mean); native 15-min forecast keeps 4 steps/hour; hourly model uses hourly mean not last quarter. |

### 11.3 Model artifacts

`models/saved/` — **10 pkls**, all consumed by the live predictor:

`xgboost_v1.pkl`, `xgboost_v1_5.pkl`, `xgboost_v2.pkl`, `xgboost_v2_5.pkl`, `xgboost_v2_5_2.pkl`, **`xgboost_v2_5_3.pkl`** (Optuna-tuned), `lightgbm_v2.pkl`, `lightgbm_v2_5.pkl`, `lightgbm_v2_5_2.pkl`, **`lightgbm_v3_1.pkl`** (grid + nuclear — the only live model using those features).

`models/experiments/` — XGBoost grid/nuclear experiments, **not live**: `xgboost_v3.pkl`, `xgboost_v3_1.pkl`, `xgboost_v4.pkl`. (Grid/nuclear features ARE now in `src/features.py`; these XGBoost variants simply were never promoted.)

### 11.4 Predictions (`predictions/`)

One CSV per model (`<name>_forecasts.csv`) with columns: `run_date`, `target_datetime`, `predicted_price`, `actual_price`, `abs_error`.

### 11.5 Notebooks

- Data: `data/convertData/*.ipynb` (cleaning, feature engineering, risk feature).
- Models: `xgboost_models/modelV*.ipynb`, `lightgbm_models/modelV2*.ipynb`, **`lightgbm_models/modelV3.1.ipynb`** (best model).
- Visualization: `data_visualization/forecast_visualization.ipynb` (interactive Plotly, **no PNG saving** since 2026-08-25), `data_visualization/eda_1.1.ipynb`, `docs/LearningNotes_CQL/model_visualization.ipynb`.

### 11.6 Docs & automation

- `README.md` — quick reference.
- `docs/Project-Learning-Notes.md` — **this file**.
- `docs/LearningNotes_CQL/*.md` — 17 bilingual learning guides (01–17): roadmap, training, cleaning, feature engineering, V1.5 plan, comparison+automation, high-volatility classifier, training→automation mental model, V2.5.1 experiment, visualization, `src/` folder, why features fail + Optuna, grid revert record, nuclear/V4, models/ folder (saved vs experiments), EDA + forecast visualization walkthrough.
- `.github/workflows/daily_forecast.yml` — daily automation.

---

## 12. Important Code Flow

### 12.1 `predict_system.py` — main run

```text
now = Helsinki now
predict_start = next local midnight ; predict_end = start + 168h − 15min
history_start = now − 200h ; price_fetch_end = predict_start + 24h
  ↓
fetch_prices(history_start, price_fetch_end)        # Elering, FI 15-min
fetch_weather(history_start, weather_end)           # FMI + Open-Meteo hourly
fetch_grid(history_start, now)                      # Fingrid cross-border flows (FINGRID_API_KEY)
fetch_nuclear(history_start, now)                   # Fingrid nuclear output (FINGRID_API_KEY)
  ↓
for each *.pkl in models/saved:
    frame = fill_actuals(load_csv(name), actuals, step_min)   # back-fill past
    frame = frame[run_date != today]                          # idempotent retry
    price_buf = PriceBuffer(history at step_min resolution)
    predictions = run_forecast(model, feature_cols, step_min, predict_start, ...)
    append new rows → dedupe(keep='last') → sort → save CSV
  ↓
print accuracy summary (MAE over evaluated rows per model)
```

### 12.2 Feature building for one timestamp (`features.py::build_features`)

1. Localize to Helsinki; derive time/calendar/cyclic features.
2. `wx.get(dt)` → weather + HDD + wind power proxy + wind sin/cos.
3. Weather rolling/lags (preceding windows only).
4. Price lags (hourly steps for V2, 15-min steps for V2.5) from `PriceBuffer`.
5. Price rolling stats (mean/std/min/max over 1h/6h/24h/7d).
6. Return the flat dict; caller selects `feature_cols`.

### 12.3 Live weather acquisition (`fetch_live.py::fetch_weather`)

```text
if end <= now:            → FMI observations only (last 168h window)
elif start >= now:        → FMI HIRLAM forecast (6h-aligned 54h window)
else:                     → concat observations + HIRLAM forecast
if end > now:             → also fetch Open-Meteo long-range (10 days)
long_range_needed = end > hirlam_end
if long_range_needed and result has NaNs after hirlam_end:
    raise RuntimeError('Long-range weather coverage is incomplete')   # never fabricate
result = ffill().bfill()  # only for small source gaps now
```

### 12.4 GitHub Actions flow

```yaml
schedule: cron '0 11 * * *' (UTC)  # ≈ 13:00/14:00 Finland, after price publication
→ setup-python 3.11 → pip install -r requirements.txt
→ python -m unittest discover -s tests -v          # gate
→ python src/predict_system.py                      # forecast + save CSVs
→ git add predictions/ && (commit "forecast: YYYY-MM-DD daily update" if changed) && push
```

---

## 13. Current Project Status

- **10 trained models** live in `models/saved/` — including `xgboost_v2_5_3.pkl` (Optuna-tuned XGBoost) and the new **`lightgbm_v3_1.pkl`**.
- **Best model overall**: **LightGBM V3.1** (grid + nuclear, MAE 2.6390 / RMSE 7.8957 / R² 0.9740) — the first and only live model using grid/nuclear features; it now runs every day.
- **Grid + nuclear are LIVE** (2026-08-25): `src/` now fetches Fingrid flows + nuclear via `fetch_grid`/`fetch_nuclear` and builds them with `GridBuffer`/`NuclearBuffer`; the workflow injects `FINGRID_API_KEY` as a secret.
- **XGBoost experiments** in `models/experiments/`: `xgboost_v3.pkl`, `xgboost_v3_1.pkl`, `xgboost_v4.pkl` (V4 = best XGBoost, MAE 2.6993 — not promoted).
- **Daily forecasts are running and committing automatically** (git history shows daily updates through 2026-08-24).
- **Best production XGBoost**: V2.5.3 (MAE 2.7236 / RMSE 8.1642 / R² 0.9722).
- **Nuclear data done**: `data/originalData/Nuclear/nuclear_measured_15min.csv` (105,216 rows, Fingrid dataset 188).
- **Grid → src integration REVERTED** (2026-08-14): records in `docs/LearningNotes_CQL/14_grid_src_integration_reverted.md` + `grid_src_integration.patch`.
- **Unit tests** pass (7) and are enforced in CI before every forecast run.
- **Team split**: partner = LightGBM (shared dataset = his "V3"); this user = XGBoost only (V3 = grid, V4 = grid + nuclear).

---

## 14. Known Problems

### 14.1 ⚠ Live recursive accuracy is far worse than offline test accuracy (verified)

This is the most important open issue.

- Offline one-step test MAE (V2.5) = **2.82** EUR/MWh.
- **Live recursive MAE** (as of the 2026-08-15 run, all 4,056 evaluated rows for `xgboost_v2_5`) ≈ **28.3 overall** — still ~10× the offline test MAE (2.82). Per-model live MAE: `lightgbm_v2_5` **19.58** · `lightgbm_v2` 20.85 · `xgboost_v2_5_2` 22.01 · `xgboost_v2` 26.70 · `xgboost_v1_5` 27.86 · `xgboost_v1` 28.31 · `xgboost_v2_5` 28.32. (LightGBM leads live too, mirroring its offline edge.)
- The model **regresses toward the mean**: predicted means cluster around **44–53 EUR/MWh** while actual means swing **9 – 59 EUR/MWh** (actual range 0–150, std ≈ 36).

Likely causes:

1. **Recursive error accumulation** — predicted values feed back into lag/rolling features and compound over 672 steps.
2. **Price-regime drift** — models were trained on 2023–2025; live forecasts are for Aug 2026, where prices were unusually low then spiky.
3. **Extreme spikes** — the model under-shoots spikes (can't predict jumps) and over-shoots calm periods.

> Actionable: treat offline metrics as an upper bound; evaluate live forecasts continuously, consider shorter horizons, re-training cadence, and drift monitoring (see §15).

### 14.2 V2.5.1 risk feature rejected

`high_volatility_prob` made both XGBoost and LightGBM slightly worse (MAE +1%) despite passing the histogram signal check. Not added to the production feature set.

### 14.3 ⚠ Hardcoded API key in a notebook (security)

`data/originalData/GridTransmission/15min_GridTransmission.ipynb` **still contains a hardcoded Fingrid API key** in a code cell (verified at line 52, marked "TEMPORARY — do not commit with real key"). **Treat it as compromised and revoke it.** The newer `data/originalData/Nuclear/15min_NucleatData.ipynb` shows the correct pattern — it reads `FINGRID_API_KEY` from a gitignored `.env` file and raises a clear error otherwise. (`config.py` correctly contains no credentials.)

### 14.4 Historical bug: wrong price source

Earlier price data came from Fingrid dataset 105 (down-regulation bid volume, MW), producing long sequences of zeros. Fixed by switching to Elering's NPS endpoint for real FI prices. The code now also guards against all-zero responses at runtime.

### 14.5 `.gitignore` allowlist fragility

`*.pkl` is ignored with per-model exceptions in `.gitignore`. New models not explicitly allowlisted are **not** committed to GitHub, so GitHub Actions silently won't run them. (V2.5.2 was exactly this trap; `lightgbm_v2_5_2.pkl` was added to the allowlist, but any future model needs a manual `.gitignore` edit.)

### 14.6 FMI quirks

- The observation endpoint does **not** reliably resolve arbitrary lat/lon → uses `place=Helsinki`/`place=Oulu`; forecast endpoint uses `latlon=` grid interpolation.
- FMI short-range forecast only covers ~54 h, which is why Open-Meteo is needed for the rest of the 7-day horizon.

### 14.7 Minor

- `src/utils.py` is an empty placeholder.
- V3 grid sign convention (positive = exports) is stated but should be re-verified against the Fingrid catalog before trusting the model.
- The 8-row NaN back-fill at the start of `grid_transmission_15min.csv` is a one-off hack (documented, negligible impact).

### 14.8 Grid & nuclear features — now LIVE (resolved 2026-08-25)

The train/serve gap for grid/nuclear features was **closed**: `src/features.py` now has `GridBuffer` + `NuclearBuffer`, `src/fetch_live.py` adds `fetch_grid()`/`fetch_nuclear()` (Fingrid), and `predict_system.py` feeds them to every model. `lightgbm_v3_1.pkl` (which needs those features) is the live proof.

Remaining caveats:
- **Live fallback is forward-fill, not NaN**: both buffers fall back to the last observed value for forecast-period steps (grid flows and nuclear output are assumed stable within a day). This is a deliberate approximation; verify it holds during unusual grid events.
- **`FINGRID_API_KEY` is now a required live dependency** (GitHub secret `FINGRID_API_KEY`; set it or grid/nuclear features degrade to NaN → LightGBM V3.1 forecasts degrade).
- XGBoost V3/V4 remain experiments; the live grid/nuclear path was only validated through the LightGBM V3.1 model.

---

## 15. Future Improvements

### Model & validation

- **Time-series cross-validation** (e.g. rolling-origin / multiple `TimeSeriesSplit` folds) instead of a single 80/20 split, to confirm stability across seasons (V2.5.3 already uses 5-fold TS-CV for Optuna).
- **Feature importance analysis** on V2.5.3/V4 to confirm which features drive predictions (likely `price_lag_672` and `price_rolling_mean_7d`).
- **Propagate tuning to the whole fleet** — V2.5.3 is Optuna-tuned; the other saved models (V1/V1.5/V2, LightGBM V2) are not yet.

### Features & experiments

- **Promote XGBoost V4 (MAE 2.6993) or keep LightGBM V3.1 (MAE 2.6390) as the single production model** — grid/nuclear features are now live, so promotion is feasible.
- **Validate the forward-fill fallback** in `GridBuffer`/`NuclearBuffer` during high-volatility grid periods; consider raising instead of silently carrying the last value.
- **Monitor LightGBM V3.1's live error** once it accumulates evaluated rows — it's the first model exercising the new live Fingrid path.
- **Strengthen the high-volatility classifier** (class imbalance via `scale_pos_weight`, more features) then re-run the V2.5.1 test.
- Apply the two-level feature validation (signal check → controlled model test) to any new candidate feature.

### Live system

- **Shorten/decay forecast confidence**: quantify how error grows with horizon and possibly report confidence bands (quantile regression / conformal prediction).
- **Rolling retraining / drift monitoring**: retrain periodically on fresh data; monitor live MAE and price-regime drift.
- **Extend to more price areas** or to a probabilistic (e.g. quantile) output.
- **Secure secrets**: remove the hardcoded Fingrid key; document an env-var workflow.

### Ops

- Add a workflow step that fails/notifies when live MAE exceeds a threshold (drift alert).
- Re-visit `.gitignore` model allowlist strategy (e.g. commit a manifest of tracked pkls).

---

## 16. Learning Notes

The following lessons are captured in depth in `docs/LearningNotes_CQL/` (bilingual guides 01–17). Highlights:

1. **Baseline before models** — always build a naive/linear baseline to know if the real model beats a simple guess.
2. **Time-series evaluation** — split chronologically, never shuffle; test on the most recent period the model hasn't seen.
3. **Feature engineering > resolution** — more rows of weak features don't help; lags/rolling/calendar features are the real driver (R² 0.107 → 0.911 from features vs 0.107 → 0.125 from resolution).
4. **Wind direction needs angular encoding** (sin/cos), because 359° ≈ 1°.
5. **Data leakage awareness** — never include features derived from the target (e.g. rolling std of price as a model input) in training.
6. **The `.pkl` mental model** — training produces a _model file_ (model + feature_cols + step_min); prediction just loads and calls `predict()`. The notebook is the school; the pkl is the frozen brain; GitHub Actions is the alarm clock.
7. **Automation is config, not code** — `.yml` is a YAML description; the only real command is `python src/predict_system.py`.
8. **Controlled experiments give honest answers** — the V2.5.1 "failure" (feature made things worse) is as valuable as any win; a feature can carry signal yet still hurt a model.
9. **Fair comparisons matter** — V2.5.2 showed that tuning XGBoost closed most of the gap with Optuna-tuned LightGBM; the original comparison was unfair (default vs tuned).
10. **Live ≠ offline** — recursive forecasting compounds error, and price regimes drift; always validate on live rolling predictions, not just the held-out test set.
11. **Tune first, then test features** (guide 12) — Optuna tuning improved XGBoost more than adding 13 grid features ever did; the V3/V3.1 experiments flipped from "worse" to "better" after tuning.
12. **Train/serve gap** (guides 13–15) — the part of a feature that helps in training (current-period flows) is often NOT available at live forecast time; only lag/forward-known features are deployable.
13. **Real supply-side signals help; noise features don't** (guide 15) — nuclear power (real, stable) improved XGBoost V4 to MAE 2.6993, while the weak-classifier "high_volatility_prob" did not.
14. **Branch awareness in a team project** — the same repo evolves on parallel branches (`chenqi` XGBoost vs partner LightGBM vs `main`); always check `git log`/`git branch` before trusting a metric, a saved-model path, or a forecast CSV. (This caught the V3-live-on-`main` vs V3-experiment-on-`chenqi` divergence.)
15. **Reuse proven regularization when adding features** (LightGBM V3.1, 2026-08) — Optuna re-tuning on the new 68-feature set overfit (CV 2.8485 → test 2.6788); carrying over the V2.5 params (heavy `reg_lambda`) beat both and set a new best (2.6390). When you add features, don't assume retuning helps — the old, well-regularized config may transfer better.
