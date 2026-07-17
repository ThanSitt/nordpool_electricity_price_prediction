# Finland Nord Pool Electricity Price Prediction

This project forecasts Finland (`FI`) Nord Pool day-ahead electricity prices
with hourly and 15-minute XGBoost/LightGBM models. It includes the historical
research pipeline, six saved models, a reproducible live predictor, and a daily
GitHub Actions workflow.

## What it predicts

| Model | Resolution | Inputs |
| --- | --- | --- |
| V1 / V1.5 | 60 / 15 min | Weather-only baselines |
| V2 | 60 min | Calendar, weather, price lags and rolling statistics |
| V2.5 | 15 min | Full engineered feature set |

The LightGBM V2.5 training notebook reports a held-out MAE of 2.64 EUR/MWh and
R² of 0.9738. This is an offline historical result, not a promise of live
performance; live errors are recorded separately after delivery prices exist.

## Data flow

```text
Historical CSVs → resample/align → feature engineering → train → models/saved
Elering FI prices + FMI/Open-Meteo weather → src/predict_system.py → predictions
```

Historical raw data covers Finnish spot prices, Helsinki-Vantaa temperature and
Oulu wind. The notebooks produce hourly and 15-minute aligned datasets, then
feature-engineered V2/V2.5 datasets. All notebooks are valid Jupyter JSON and
can be opened directly.

## Run locally

Use Python 3.11 so local inference matches GitHub Actions:

```powershell
conda env create -f environment.yml
conda activate nordpool
python -m unittest discover -s tests -v
python src/predict_system.py
```

No API key is required. Live FI day-ahead prices are read from Elering's public
NPS endpoint. FMI provides observations and short-range weather; Open-Meteo
provides the remaining seven-day weather horizon. A missing long-range forecast
causes the run to fail rather than silently repeating stale weather.

The predictor starts at the next Helsinki delivery day. It writes hourly rows
for hourly models and 15-minute rows for 15-minute models. A later daily run
fills `actual_price` and `abs_error` at the same resolution. Re-running on the
same day replaces that run's rows instead of duplicating them.

## Automate with GitHub Actions

The committed workflow in `.github/workflows/daily_forecast.yml` runs at 11:00
UTC every day (13:00 EET / 14:00 EEST), executes offline tests, runs the
predictor, and commits changed CSVs. It needs only the repository's standard
`contents: write` GitHub Actions permission; there are no secrets to configure.

## Retrain models

Run the notebooks in this order:

1. Resample weather: `data/originalData/Temperature/15min_Temperature.ipynb`
   and `data/originalData/WindDirection&Speed/15min_Wind.ipynb`.
2. Build aligned data: `V1_data_cleaning_and_alignment.ipynb` or
   `V1.5_15min_Dataset.ipynb`.
3. Create features: `V2_feature_engineering.ipynb` or
   `V2.5_15min_feature_engineering.ipynb`.
4. Train and save a model in the corresponding `xgboost_models/` or
   `lightgbm_models/` notebook.

Before publishing a retrained model, update its feature contract
(`feature_cols`, `step_min`) and run the test suite plus one local forecast.
