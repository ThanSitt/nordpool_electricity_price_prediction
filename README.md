# Finland Nord Pool Electricity Price Prediction

This repository predicts Finland (`FI`) Nord Pool day-ahead electricity prices.
It contains the full research pipeline, cleaned and aligned datasets, feature
engineering notebooks, saved XGBoost and LightGBM model bundles, a live
forecasting script, and automated daily forecast delivery.

## Project overview

The codebase is organized around two linked flows:

```text
Raw CSVs and notebooks → cleaning/alignment → feature engineering → training → models/saved
Live price and weather APIs → src/predict_system.py → predictions
```

The historical pipeline builds both hourly and 15-minute datasets. The live
predictor then uses the saved model metadata to run each model at its native
resolution.

## What is included

| Area | Contents |
| --- | --- |
| Raw data | Finnish spot prices, Helsinki-Vantaa temperature, Oulu wind data |
| Cleaned data | Hourly and 15-minute aligned CSVs in `data/convertData/` |
| Feature engineering | V2 hourly and V2.5 15-minute feature notebooks and outputs |
| Models | Six saved bundles in `models/saved/` |
| Live prediction | `src/predict_system.py`, `src/fetch_live.py`, `src/features.py` |
| Tests | Offline regression tests in `tests/` |
| Automation | Daily GitHub Actions forecast workflow |

Current saved model bundles:

- `models/saved/xgboost_v1.pkl`
- `models/saved/xgboost_v1_5.pkl`
- `models/saved/xgboost_v2.pkl`
- `models/saved/xgboost_v2_5.pkl`
- `models/saved/lightgbm_v2.pkl`
- `models/saved/lightgbm_v2_5.pkl`

## Data pipeline

The historical notebooks are grouped by stage:

1. Raw price and weather exploration lives in `data/originalData/`.
2. Weather resampling notebooks create 15-minute aligned inputs.
3. Alignment notebooks build the model-ready datasets for V1 and V1.5.
4. Feature engineering notebooks create V2 and V2.5 feature tables.
5. Training notebooks in `xgboost_models/` and `lightgbm_models/` describe the
   final model versions saved under `models/saved/`.

Key notebooks include:

- `data/originalData/electricPrices/price_data.ipynb`
- `data/originalData/Temperature/15min_Temperature.ipynb`
- `data/originalData/WindDirection&Speed/15min_Wind.ipynb`
- `data/convertData/V1_data_cleaning_and_alignment.ipynb`
- `data/convertData/V1.5_15min_Dataset.ipynb`
- `data/convertData/V2_feature_engineering.ipynb`
- `data/convertData/V2.5_15min_feature_engineering.ipynb`
- `xgboost_models/modelV1.ipynb`
- `xgboost_models/modelV1.5.ipynb`
- `xgboost_models/modelV2.ipynb`
- `xgboost_models/modelV2.5.ipynb`
- `lightgbm_models/modelV2.ipynb`
- `lightgbm_models/modelV2.5.ipynb`

## Live prediction

`src/predict_system.py` performs the production-style forecast run:

- fetches recent FI market prices and weather history
- builds a seven-day recursive forecast at each model's native resolution
- writes one CSV per model into `predictions/`
- back-fills `actual_price` and `abs_error` when realized prices become available
- replaces the current day's rows on re-run instead of duplicating them

The live fetch layer uses public sources only:

- FI day-ahead prices from Elering's public NPS endpoint
- FMI observations and short-range weather
- Open-Meteo long-range weather for the remaining forecast horizon

The run fails if the long-range weather coverage is incomplete, which prevents
quietly reusing stale weather data for the tail of the seven-day horizon.

## Environment

Use the conda environment defined in `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate nordpool
```

The environment targets Python 3.11 and installs the pinned dependencies from
`requirements.txt`. No API key is required for the current pipeline.

## Run locally

```powershell
python -m unittest discover -s tests -v
python src/predict_system.py
```

The predictor starts from the next Helsinki delivery day. Hourly models produce
hourly forecast rows, and 15-minute models produce 15-minute rows.

## Tests

The test suite focuses on the most fragile parts of the pipeline:

- feature buffer behavior and rolling-window calculations
- live price fetching and zero-price guardrails
- forecast evaluation at the correct native resolution

Run the suite with:

```powershell
python -m unittest discover -s tests -v
```

## GitHub Actions

The workflow in `.github/workflows/daily_forecast.yml` runs every day at 11:00
UTC, installs dependencies, runs the offline tests, executes the predictor, and
commits updated files under `predictions/`.

It only needs the repository's standard `contents: write` permission.

## Retraining notes

When retraining or adding a new model, keep the saved bundle contract aligned
with the live predictor:

- `feature_cols` must match the features produced by `src/features.py`
- `step_min` must match the model's native output resolution
- price history must stay in chronological order; do not shuffle time series

The `src/features.py` pipeline is shared by all live models. Wind direction is
encoded with sine and cosine values rather than as a plain linear degree.

## Reports

Generated report artifacts, if any, should be added to `reports/` only when
they are non-empty and ready to keep. The placeholder `report1.pdf` has already
been removed.
