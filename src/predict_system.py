"""Daily Finland electricity-price forecasting system.

Run after Nord Pool has published the next delivery day's prices:

    python src/predict_system.py

Each run fetches a 15-minute FI price history and weather forecasts, creates a
seven-day recursive forecast at every model's native resolution, back-fills
available actuals, and writes one CSV per model.
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

FORECAST_HOURS = config.FORECAST_HOURS
CSV_COLS = ['run_date', 'target_datetime', 'predicted_price', 'actual_price', 'abs_error']


def load_models() -> dict:
    """Load all model bundles, each of which declares its feature contract."""
    models = {}
    for pkl in sorted(config.SAVED_MODELS_DIR.glob('*.pkl')):
        try:
            meta = joblib.load(pkl)
            models[pkl.stem] = meta
            print(f'  loaded {pkl.stem} ({len(meta["feature_cols"])} features, {meta["step_min"]} min)')
        except Exception as exc:
            print(f'  [skip] {pkl.name}: {exc}')
    return models


def run_forecast(model, feature_cols: list[str], step_min: int,
                 start_dt: pd.Timestamp, price_buf: feat_lib.PriceBuffer,
                 wx_buf: feat_lib.WeatherBuffer) -> list[tuple[pd.Timestamp, float]]:
    """Recursively produce seven days of predictions at ``step_min`` resolution."""
    step = pd.Timedelta(minutes=step_min)
    results: list[tuple[pd.Timestamp, float]] = []
    for i in range(FORECAST_HOURS * (60 // step_min)):
        timestamp = start_dt + i * step
        features = feat_lib.build_features(timestamp, price_buf, wx_buf)
        row = pd.DataFrame([{column: features.get(column, np.nan) for column in feature_cols}])[feature_cols]
        prediction = float(model.predict(row)[0])
        price_buf.add(timestamp, prediction)
        results.append((timestamp, prediction))
    return results


def csv_path(model_name: str) -> Path:
    return config.PREDICTIONS_DIR / f'{model_name}_forecasts.csv'


def load_csv(model_name: str) -> pd.DataFrame:
    path = csv_path(model_name)
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLS)
    frame = pd.read_csv(path, parse_dates=['run_date', 'target_datetime'])
    for column in ('run_date', 'target_datetime'):
        if frame[column].dt.tz is None:
            frame[column] = frame[column].dt.tz_localize('UTC').dt.tz_convert(config.HELSINKI)
        else:
            frame[column] = frame[column].dt.tz_convert(config.HELSINKI)
    return frame


def save_csv(frame: pd.DataFrame, model_name: str) -> None:
    config.PREDICTIONS_DIR.mkdir(exist_ok=True)
    frame.to_csv(csv_path(model_name), index=False)


def fill_actuals(frame: pd.DataFrame, actual_prices: pd.Series, step_min: int) -> pd.DataFrame:
    """Back-fill actual prices at the same resolution as the saved forecast."""
    if frame.empty:
        return frame
    if step_min == 60:
        evaluation_prices = actual_prices.resample('1h').mean()
    elif step_min == 15:
        evaluation_prices = actual_prices
    else:
        raise ValueError(f'Unsupported model step: {step_min} minutes')

    price_map = {
        timestamp.tz_convert(config.HELSINKI).floor(f'{step_min}min'): float(value)
        for timestamp, value in evaluation_prices.items()
    }
    for idx in frame[frame['actual_price'].isna()].index:
        target = pd.Timestamp(frame.at[idx, 'target_datetime'])
        if target.tzinfo is None:
            target = target.tz_localize(config.HELSINKI)
        target = target.tz_convert(config.HELSINKI).floor(f'{step_min}min')
        if target in price_map:
            actual = price_map[target]
            frame.at[idx, 'actual_price'] = actual
            frame.at[idx, 'abs_error'] = abs(actual - frame.at[idx, 'predicted_price'])
    return frame


def _history_for_model(actual_prices: pd.Series, step_min: int) -> pd.Series:
    """Return a price history at the resolution used during model training."""
    if step_min == 15:
        return actual_prices
    if step_min == 60:
        return actual_prices.resample('1h').mean()
    raise ValueError(f'Unsupported model step: {step_min} minutes')


def main() -> None:
    now = pd.Timestamp.now(tz=config.HELSINKI)
    # A day-ahead run begins at the next local delivery day, never halfway
    # through a day whose market price may already be published.
    predict_start = now.normalize() + pd.Timedelta(days=1)
    predict_end = predict_start + pd.Timedelta(hours=FORECAST_HOURS) - pd.Timedelta(minutes=15)

    print(f'\nRun time       : {now:%Y-%m-%d %H:%M %Z}')
    print(f'Forecast start : {predict_start:%Y-%m-%d %H:%M}')
    print(f'Forecast end   : {predict_end:%Y-%m-%d %H:%M}')

    history_start = now - pd.Timedelta(hours=config.PRICE_HISTORY_HOURS)
    # Nord Pool publishes tomorrow's prices ~12:00 EET, workflow runs ~14:00 EEST.
    # Fetching through end of the next delivery day ensures the price buffer has
    # valid lag features for the first forecast steps instead of NaN.
    price_fetch_end = predict_start + pd.Timedelta(hours=24)
    print(f'\nFetching FI prices {history_start:%Y-%m-%d} → {price_fetch_end:%Y-%m-%d %H:%M} ...')
    actual_prices = fetch_live.fetch_prices(history_start, price_fetch_end)
    print(f'  got {len(actual_prices)} price records')

    weather_end = predict_start + pd.Timedelta(hours=FORECAST_HOURS)
    print(f'Fetching weather {history_start:%Y-%m-%d} → {weather_end:%Y-%m-%d} ...')
    weather_df = fetch_live.fetch_weather(history_start, weather_end)
    print(f'  got {len(weather_df)} hourly weather records')
    weather_buf = feat_lib.WeatherBuffer(weather_df)

    print('\nLoading models ...')
    models = load_models()
    if not models:
        raise RuntimeError(f'No readable models in {config.SAVED_MODELS_DIR}')

    print('\nGenerating seven-day forecasts ...')
    run_day = now.normalize()
    for model_name, meta in models.items():
        step_min = int(meta['step_min'])
        frame = fill_actuals(load_csv(model_name), actual_prices, step_min)
        # A manual retry replaces this day's rows rather than duplicating them.
        run_dates = pd.to_datetime(frame['run_date'], errors='coerce')
        frame = frame[run_dates.dt.normalize() != run_day]

        price_buf = feat_lib.PriceBuffer(_history_for_model(actual_prices, step_min))
        predictions = run_forecast(meta['model'], meta['feature_cols'], step_min,
                                   predict_start, price_buf, weather_buf)
        new_rows = pd.DataFrame([
            {
                'run_date': now,
                'target_datetime': timestamp,
                'predicted_price': round(prediction, 4),
                'actual_price': np.nan,
                'abs_error': np.nan,
            }
            for timestamp, prediction in predictions
        ], columns=CSV_COLS)
        frame = pd.concat([frame, new_rows], ignore_index=True)
        frame = frame.drop_duplicates(subset=['run_date', 'target_datetime'], keep='last')
        frame = frame.sort_values(['run_date', 'target_datetime']).reset_index(drop=True)
        save_csv(frame, model_name)
        print(f'  {model_name}: {len(predictions)} {step_min}-minute steps → {csv_path(model_name).name}')

    print('\nAccuracy summary (evaluated predictions):')
    for model_name in models:
        evaluated = load_csv(model_name).dropna(subset=['actual_price'])
        if evaluated.empty:
            print(f'  {model_name}: no evaluated predictions yet')
        else:
            print(f'  {model_name}: MAE={evaluated["abs_error"].mean():.4f} ({len(evaluated)} evaluated)')


if __name__ == '__main__':
    main()
