"""
src/predict_system.py — daily electricity price prediction system

Run once a day (ideally after 13:00 Helsinki time, when Nordpool day-ahead
prices for tomorrow have been published):

    conda activate nordpool
    python src/predict_system.py

On each run:
  1. Fetches last 200 hours of actual prices from Fingrid (lag buffer)
  2. Fetches actual + forecast weather from FMI
  3. Generates 7-day ahead hourly forecasts from all saved models
  4. Appends new predictions to predictions/forecasts.csv
  5. Back-fills actual_price / abs_error for past predictions that are now available

Setup (one-time):
  - Add your Fingrid API key to src/config.py
  - Run each training notebook, then run the Save Model cell at the bottom
    to export models to models/saved/*.pkl
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd

import config
import fetch_live
import features as feat_lib

FORECAST_HOURS = 7 * 24   # 168 hours ahead
CSV_COLS = ['run_date', 'target_datetime', 'model',
            'predicted_price', 'actual_price', 'abs_error']


# ── model loading ──────────────────────────────────────────────────────────────

def load_models() -> dict:
    models = {}
    for pkl in sorted(config.SAVED_MODELS_DIR.glob('*.pkl')):
        try:
            meta = joblib.load(pkl)
            models[pkl.stem] = meta
            print(f'  loaded {pkl.stem}  '
                  f'({len(meta["feature_cols"])} features, {meta["step_min"]}min)')
        except Exception as e:
            print(f'  [skip] {pkl.name}: {e}')
    return models


# ── forecasting ────────────────────────────────────────────────────────────────

def run_forecast(model, feature_cols: list[str], step_min: int,
                 start_dt: pd.Timestamp,
                 price_buf: feat_lib.PriceBuffer,
                 wx_buf: feat_lib.WeatherBuffer) -> list[tuple]:
    """
    Recursive multi-step forecast for FORECAST_HOURS hours.

    Each predicted price is immediately added to the price buffer so that
    subsequent steps use it for their lag features — exactly how the model
    would behave in a real deployment.
    """
    results = []
    step_td = pd.Timedelta(minutes=step_min)
    n_steps = FORECAST_HOURS * (60 // step_min)

    for i in range(n_steps):
        dt = start_dt + i * step_td
        feat = feat_lib.build_features(dt, price_buf, wx_buf)
        row  = pd.DataFrame([{c: feat.get(c, np.nan) for c in feature_cols}])[feature_cols]
        pred = float(model.predict(row)[0])
        price_buf.add(dt, pred)
        results.append((dt, pred))

    return results


def to_hourly(preds: list[tuple]) -> list[tuple]:
    """Average 15-min predictions into hourly slots."""
    df = pd.DataFrame(preds, columns=['dt', 'price'])
    df['dt_hr'] = df['dt'].apply(lambda x: x.replace(minute=0))
    return list(df.groupby('dt_hr')['price'].mean().items())


# ── CSV management ─────────────────────────────────────────────────────────────

def load_csv() -> pd.DataFrame:
    if config.FORECAST_CSV.exists():
        df = pd.read_csv(config.FORECAST_CSV,
                         parse_dates=['run_date', 'target_datetime'])
        for col in ('run_date', 'target_datetime'):
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(config.HELSINKI)
            else:
                df[col] = df[col].dt.tz_convert(config.HELSINKI)
        return df
    return pd.DataFrame(columns=CSV_COLS)


def save_csv(df: pd.DataFrame):
    config.FORECAST_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(config.FORECAST_CSV, index=False)


def fill_actuals(df: pd.DataFrame, actual_prices: pd.Series) -> pd.DataFrame:
    """Fill in actual_price and abs_error for past predictions now available."""
    price_map = {}
    for ts, v in actual_prices.items():
        key = ts.tz_convert(config.HELSINKI).replace(minute=0, second=0, microsecond=0)
        price_map[key] = float(v)

    mask = df['actual_price'].isna()
    for idx in df[mask].index:
        tgt = pd.Timestamp(df.at[idx, 'target_datetime'])
        if tgt.tzinfo is None:
            tgt = tgt.tz_localize(config.HELSINKI)
        tgt = tgt.tz_convert(config.HELSINKI).replace(minute=0, second=0, microsecond=0)
        if tgt in price_map:
            actual = price_map[tgt]
            df.at[idx, 'actual_price'] = actual
            df.at[idx, 'abs_error']    = abs(actual - df.at[idx, 'predicted_price'])
    return df


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    now           = pd.Timestamp.now(tz=config.HELSINKI)
    predict_start = (now + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    print(f'\nRun time       : {now.strftime("%Y-%m-%d %H:%M %Z")}')
    print(f'Forecast start : {predict_start.strftime("%Y-%m-%d %H:%M")}')
    print(f'Forecast end   : {(predict_start + pd.Timedelta(hours=FORECAST_HOURS - 1)).strftime("%Y-%m-%d %H:%M")}')

    # need 200h of price history to cover price_lag_168h (7d) + rolling buffers
    hist_start = now - pd.Timedelta(hours=200)
    print(f'\nFetching prices {hist_start.strftime("%Y-%m-%d")} → now ...')
    actual_prices = fetch_live.fetch_prices(hist_start, now)
    print(f'  got {len(actual_prices)} hourly price records')

    wx_end = predict_start + pd.Timedelta(hours=FORECAST_HOURS)
    print(f'Fetching weather {hist_start.strftime("%Y-%m-%d")} → {wx_end.strftime("%Y-%m-%d")} ...')
    weather_df = fetch_live.fetch_weather(hist_start, wx_end)
    print(f'  got {len(weather_df)} hourly weather records')

    # build shared buffers
    price_buf = feat_lib.PriceBuffer(actual_prices)
    wx_buf    = feat_lib.WeatherBuffer(weather_df)

    # load forecast CSV and back-fill any newly available actual prices
    df = load_csv()
    df = fill_actuals(df, actual_prices)

    # load models
    print('\nLoading models ...')
    models = load_models()
    if not models:
        print('\nNo models found in', config.SAVED_MODELS_DIR)
        print('Open each training notebook, run the Save Model cell at the')
        print('bottom, then re-run this script.')
        save_csv(df)
        return

    # generate forecasts — each model gets its own copy of the price buffer
    # so recursive predictions from model A don't contaminate model B
    print('\nGenerating 7-day ahead forecasts ...')
    new_rows = []

    for model_name, meta in models.items():
        buf_copy     = feat_lib.PriceBuffer()
        buf_copy._d  = dict(price_buf._d)

        raw_preds    = run_forecast(meta['model'], meta['feature_cols'],
                                    meta['step_min'], predict_start,
                                    buf_copy, wx_buf)
        hourly_preds = to_hourly(raw_preds) if meta['step_min'] == 15 else raw_preds

        for dt, pred in hourly_preds:
            new_rows.append({
                'run_date':        now,
                'target_datetime': dt,
                'model':           model_name,
                'predicted_price': round(pred, 4),
                'actual_price':    np.nan,
                'abs_error':       np.nan,
            })
        print(f'  {model_name}: done ({len(hourly_preds)} hours)')

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=CSV_COLS)],
                       ignore_index=True)

    save_csv(df)
    print(f'\nSaved → {config.FORECAST_CSV}  ({len(df):,} total rows)')

    # accuracy summary for already-evaluated predictions
    done = df.dropna(subset=['actual_price'])
    if not done.empty:
        print('\nAccuracy summary (evaluated predictions):')
        summary = (done.groupby('model')['abs_error']
                   .agg(MAE='mean', predictions='count').round(4))
        print(summary.to_string())
    else:
        print('\nNo evaluated predictions yet (check back after the first week).')


if __name__ == '__main__':
    main()
