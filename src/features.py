"""
src/features.py — feature engineering for live predictions

Mirrors the exact feature pipeline used during training so that model
predictions are valid on live data.
"""

from __future__ import annotations
from math import sin, cos, pi

import holidays
import numpy as np
import pandas as pd

_FI       = holidays.Finland()
_HELSINKI = 'Europe/Helsinki'


# ── helpers ────────────────────────────────────────────────────────────────────

def _season(m: int) -> int:
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[m]


def _cyc(val, period):
    return sin(2 * pi * val / period), cos(2 * pi * val / period)


# ── price buffer ───────────────────────────────────────────────────────────────

class PriceBuffer:
    """
    Rolling store of electricity prices at 15-minute resolution.

    Holds actual historical prices and fills forward with model predictions
    as each forecast step is generated. Supports both hourly (V2) and
    15-min (V2.5) lag/rolling computations.
    """

    def __init__(self, series: pd.Series | None = None):
        self._d: dict[pd.Timestamp, float] = {}
        if series is not None:
            for ts, v in series.items():
                if not np.isnan(v):
                    self._d[self._snap(ts)] = float(v)

    @staticmethod
    def _snap(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(_HELSINKI)
        else:
            ts = ts.tz_convert(_HELSINKI)
        m = (ts.minute // 15) * 15
        return ts.replace(minute=m, second=0, microsecond=0)

    def add(self, ts, price: float):
        self._d[self._snap(ts)] = price

    def lag_hours(self, ts, h: int) -> float:
        return self._d.get(self._snap(ts) - pd.Timedelta(hours=h), np.nan)

    def lag_steps(self, ts, steps: int) -> float:
        return self._d.get(self._snap(ts) - pd.Timedelta(minutes=15 * steps), np.nan)

    def rolling_hours(self, ts, h: int, fn) -> float:
        key = self._snap(ts)
        vals = [self._d.get(key - pd.Timedelta(minutes=15 * i), np.nan)
                for i in range(1, h * 4 + 1)]
        clean = [v for v in vals if not np.isnan(v)]
        return float(fn(clean)) if clean else np.nan

    def rolling_steps(self, ts, steps: int, fn) -> float:
        key = self._snap(ts)
        vals = [self._d.get(key - pd.Timedelta(minutes=15 * i), np.nan)
                for i in range(1, steps + 1)]
        clean = [v for v in vals if not np.isnan(v)]
        return float(fn(clean)) if clean else np.nan


# ── weather buffer ─────────────────────────────────────────────────────────────

class WeatherBuffer:
    """Hourly weather lookup with forward-fill for future timestamps."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()
        if not self._df.index.tz:
            self._df.index = self._df.index.tz_localize(_HELSINKI)
        else:
            self._df.index = self._df.index.tz_convert(_HELSINKI)

    def get(self, ts) -> dict:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(_HELSINKI)
        ts_hr = ts.replace(minute=0, second=0, microsecond=0)

        if self._df.empty:
            temp, wind, wdir = np.nan, np.nan, 180.0
            return {
                'temp': temp, 'wind_speed': wind, 'wind_direction_deg': wdir,
                'wind_dir_sin': 0.0, 'wind_dir_cos': -1.0,
                'HDD': np.nan, 'wind_power_proxy': np.nan,
            }

        if ts_hr in self._df.index:
            row = self._df.loc[ts_hr]
        else:
            earlier = self._df.loc[:ts_hr]
            row = earlier.iloc[-1] if not earlier.empty else self._df.iloc[0]

        temp = float(row.get('temp', 0.0))
        wind = float(row.get('wind_speed', 0.0))
        wdir = float(row.get('wind_direction_deg', 180.0))
        return {
            'temp':              temp,
            'wind_speed':        wind,
            'wind_direction_deg': wdir,
            'wind_dir_sin':      sin(wdir * pi / 180),
            'wind_dir_cos':      cos(wdir * pi / 180),
            'HDD':               max(0.0, 17.0 - temp),
            'wind_power_proxy':  wind ** 3,
        }


# ── feature builder ────────────────────────────────────────────────────────────

def build_features(dt: pd.Timestamp,
                   buf: PriceBuffer,
                   wx: WeatherBuffer) -> dict:
    """
    Build a flat dict containing every possible feature for timestamp dt.

    Each model selects its own subset using its saved feature_cols list,
    so one function covers V1 / V1.5 / V2 / V2.5 for both XGBoost and LightGBM.
    """
    dt = pd.Timestamp(dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize(_HELSINKI)

    woy     = int(dt.strftime('%V'))
    h_s, h_c = _cyc(dt.hour, 24)
    d_s, d_c = _cyc(dt.weekday(), 7)
    m_s, m_c = _cyc(dt.month, 12)
    w_s, w_c = _cyc(woy, 52)
    is_hol  = int(dt.date() in _FI)
    is_wknd = int(dt.weekday() >= 5)

    f: dict = {
        # temporal
        'hour':          dt.hour,
        'minute':        dt.minute,
        'day_of_week':   dt.weekday(),
        'day_of_month':  dt.day,
        'month':         dt.month,
        'week_of_year':  woy,
        'quarter':       (dt.month - 1) // 3 + 1,
        'year':          dt.year,
        'season':        _season(dt.month),
        'time_of_day':   dt.hour * 4 + dt.minute // 15,
        'is_weekend':    is_wknd,
        'is_peak_hour':  int(dt.hour in (7, 8, 9, 17, 18, 19, 20)),
        'is_night_hour': int(dt.hour >= 23 or dt.hour <= 6),
        'is_holiday':    is_hol,
        'is_non_working': int(is_hol or is_wknd),
        # cyclic encodings
        'hour_sin': h_s, 'hour_cos': h_c,
        'day_of_week_sin': d_s, 'day_of_week_cos': d_c,
        'month_sin': m_s, 'month_cos': m_c,
        'week_of_year_sin': w_s, 'week_of_year_cos': w_c,
    }

    # weather (current + derived)
    w = wx.get(dt)
    f.update(w)
    f['air_temp_mean']   = w['temp']       # V1 renamed columns
    f['wind_speed_mean'] = w['wind_speed']

    # temperature lags / rolling (use WeatherBuffer for accuracy)
    f['temp_rolling_mean_24h'] = wx.get(dt - pd.Timedelta(hours=12))['temp']  # midpoint proxy
    f['temp_rolling_mean_1h']  = wx.get(dt - pd.Timedelta(minutes=30))['temp']
    f['temp_lag_24h']  = wx.get(dt - pd.Timedelta(hours=24))['temp']
    f['temp_lag_168h'] = wx.get(dt - pd.Timedelta(hours=168))['temp']
    f['temp_lag_4']    = wx.get(dt - pd.Timedelta(minutes=60))['temp']   # 4×15min = 1h
    f['temp_lag_96']   = wx.get(dt - pd.Timedelta(hours=24))['temp']     # 96×15min = 24h

    # price lags — hourly steps (V2)
    f['price_lag_1h']   = buf.lag_hours(dt, 1)
    f['price_lag_2h']   = buf.lag_hours(dt, 2)
    f['price_lag_3h']   = buf.lag_hours(dt, 3)
    f['price_lag_6h']   = buf.lag_hours(dt, 6)
    f['price_lag_12h']  = buf.lag_hours(dt, 12)
    f['price_lag_24h']  = buf.lag_hours(dt, 24)
    f['price_lag_48h']  = buf.lag_hours(dt, 48)
    f['price_lag_168h'] = buf.lag_hours(dt, 168)

    # price rolling — hourly (V2)
    f['price_rolling_mean_24h']  = buf.rolling_hours(dt, 24, np.mean)
    f['price_rolling_std_24h']   = buf.rolling_hours(dt, 24, np.std)
    f['price_rolling_min_24h']   = buf.rolling_hours(dt, 24, np.min)
    f['price_rolling_max_24h']   = buf.rolling_hours(dt, 24, np.max)
    f['price_rolling_mean_168h'] = buf.rolling_hours(dt, 168, np.mean)

    # price lags — 15-min steps (V2.5)
    f['price_lag_1']   = buf.lag_steps(dt, 1)    # 15 min
    f['price_lag_2']   = buf.lag_steps(dt, 2)    # 30 min
    f['price_lag_4']   = buf.lag_steps(dt, 4)    # 1 h
    f['price_lag_8']   = buf.lag_steps(dt, 8)    # 2 h
    f['price_lag_16']  = buf.lag_steps(dt, 16)   # 4 h
    f['price_lag_32']  = buf.lag_steps(dt, 32)   # 8 h
    f['price_lag_96']  = buf.lag_steps(dt, 96)   # 24 h
    f['price_lag_672'] = buf.lag_steps(dt, 672)  # 7 d

    # price rolling — 15-min steps (V2.5)
    f['price_rolling_mean_1h']  = buf.rolling_steps(dt, 4,   np.mean)
    f['price_rolling_std_1h']   = buf.rolling_steps(dt, 4,   np.std)
    f['price_rolling_mean_6h']  = buf.rolling_steps(dt, 24,  np.mean)
    f['price_rolling_mean_7d']  = buf.rolling_steps(dt, 672, np.mean)
    # price_rolling_mean/std/min/max_24h shared with hourly V2 (already set above)

    return f
