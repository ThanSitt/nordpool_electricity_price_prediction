"""
src/fetch_live.py — live data fetchers

fetch_prices(start, end)  → pd.Series  (hourly EUR/MWh from Fingrid)
fetch_weather(start, end) → pd.DataFrame (hourly temp/wind from FMI)
"""

from __future__ import annotations
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

import config

_TIMEOUT  = 60
_HELSINKI = config.HELSINKI

# Helsinki-Vantaa Airport (temperature source, same as training data)
_TEMP_LATLON = '60.3172,24.9633'
# Oulu area (wind source, same as training data)
_WIND_LATLON = '65.0126,25.4647'

_WX_COLS = ['temp', 'wind_speed', 'wind_direction_deg']


# ── Fingrid (electricity prices) ───────────────────────────────────────────────

def fetch_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Finland day-ahead electricity prices from Fingrid dataset 105.
    Returns hourly pd.Series indexed by Helsinki-timezone timestamps.
    """
    url     = 'https://data.fingrid.fi/api/datasets/105/data'
    headers = {'x-api-key': config.FINGRID_API_KEY}

    start_utc = start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    end_utc   = end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')

    records: dict[pd.Timestamp, float] = {}
    page = 1
    while True:
        resp = requests.get(url, headers=headers, params={
            'startTime': start_utc, 'endTime': end_utc,
            'format': 'json', 'page': page, 'pageSize': 10000, 'locale': 'en',
        }, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        for row in body['data']:
            dt = (pd.Timestamp(row['startTime'])
                  .tz_convert(_HELSINKI)
                  .replace(second=0, microsecond=0))
            records[dt] = float(row['value'])
        if page >= body['pagination']['lastPage']:
            break
        page += 1

    return pd.Series(records, name='price').sort_index()


# ── FMI (weather) ──────────────────────────────────────────────────────────────

def _fmi_request(storedquery: str, latlon: str, params: str,
                 start: pd.Timestamp, end: pd.Timestamp) -> str:
    """HTTP GET to FMI WFS with 3 retries on transient errors."""
    url_params = {
        'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
        'storedquery_id': storedquery,
        'latlon': latlon,
        'parameters': params,
        'starttime': start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'endtime':   end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'timestep': '60',
    }
    for attempt in range(3):
        try:
            r = requests.get('https://opendata.fmi.fi/wfs',
                             params=url_params, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f'  FMI request failed (attempt {attempt+1}/3): {e} — retrying...')
            time.sleep(5)
    return ''  # unreachable but keeps type checker happy


def _parse_fmi(xml_text: str) -> pd.DataFrame:
    """Parse FMI WFS XML into a wide DataFrame (one column per parameter)."""
    if not xml_text:
        return pd.DataFrame()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()

    # FMI sometimes wraps errors in a 200 ExceptionReport — detect and skip
    for elem in root.iter():
        if elem.tag.endswith('}ExceptionReport') or elem.tag == 'ExceptionReport':
            print(f'  FMI returned ExceptionReport: {xml_text[:300]}')
            return pd.DataFrame()

    rows: list[dict] = []
    for elem in root.iter():
        if not elem.tag.endswith('}BsWfsElement') and elem.tag != 'BsWfsElement':
            continue
        r: dict[str, str] = {}
        for child in elem:
            local = child.tag.split('}')[-1]
            if local in ('Time', 'ParameterName', 'ParameterValue'):
                r[local] = child.text or ''
        if len(r) < 3 or r.get('ParameterValue', 'NaN') in ('NaN', ''):
            continue
        try:
            val = float(r['ParameterValue'])
        except ValueError:
            continue
        dt = (pd.Timestamp(r['Time'])
              .tz_convert(_HELSINKI)
              .replace(second=0, microsecond=0))
        rows.append({'datetime': dt, 'param': r['ParameterName'].lower(), 'value': val})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index='datetime', columns='param', values='value', aggfunc='first')


def _fetch_block(start: pd.Timestamp, end: pd.Timestamp,
                 forecast: bool) -> pd.DataFrame:
    """One block of weather (observed or forecast) as standardised DataFrame."""
    sq = ('fmi::forecast::hirlam::surface::point::simple'
          if forecast else 'fmi::observations::weather::simple')
    params = 'temperature,windspeedms,winddirection'

    try:
        temp_df = _parse_fmi(_fmi_request(sq, _TEMP_LATLON, params, start, end))
    except Exception as e:
        print(f'  WARNING: temp fetch failed: {e}')
        temp_df = pd.DataFrame()

    try:
        wind_df = _parse_fmi(_fmi_request(sq, _WIND_LATLON, params, start, end))
    except Exception as e:
        print(f'  WARNING: wind fetch failed: {e}')
        wind_df = pd.DataFrame()

    # build result index from whichever source has data
    if not temp_df.empty:
        idx = temp_df.index
    elif not wind_df.empty:
        idx = wind_df.index
    else:
        return pd.DataFrame(columns=_WX_COLS)

    result = pd.DataFrame(index=idx, dtype=float)
    result['temp'] = temp_df['temperature'].reindex(idx) if (not temp_df.empty and 'temperature' in temp_df.columns) else np.nan

    if not wind_df.empty:
        result['wind_speed']         = wind_df['windspeedms'].reindex(idx)   if 'windspeedms'   in wind_df.columns else np.nan
        result['wind_direction_deg'] = wind_df['winddirection'].reindex(idx) if 'winddirection' in wind_df.columns else np.nan
    elif not temp_df.empty:
        result['wind_speed']         = temp_df['windspeedms'].reindex(idx)   if 'windspeedms'   in temp_df.columns else np.nan
        result['wind_direction_deg'] = temp_df['winddirection'].reindex(idx) if 'winddirection' in temp_df.columns else np.nan
    else:
        result['wind_speed']         = np.nan
        result['wind_direction_deg'] = np.nan

    # guarantee all 3 columns exist before selecting
    for col in _WX_COLS:
        if col not in result.columns:
            result[col] = np.nan

    return result[_WX_COLS].dropna(how='all')


def fetch_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Hourly weather DataFrame with columns [temp, wind_speed, wind_direction_deg].

    Uses FMI observations for past data and HIRLAM forecast for future.
    HIRLAM covers ~54 hours ahead; beyond that the last value is held constant.
    Returns a fully-indexed DataFrame (ffill+bfill) — never crashes.
    """
    now = pd.Timestamp.now(tz=_HELSINKI)
    all_hours = pd.date_range(start, end, freq='1h', tz=_HELSINKI)

    _OBS_LIMIT = pd.Timedelta(hours=168)  # FMI observations max window

    # HIRLAM forecasts are published at 00/06/12/18 UTC — snap start to last boundary
    hirlam_start = now.tz_convert('UTC').replace(minute=0, second=0, microsecond=0)
    hirlam_start = hirlam_start - pd.Timedelta(hours=hirlam_start.hour % 6)
    hirlam_start = hirlam_start.tz_convert(_HELSINKI)

    try:
        if end <= now:
            obs_start = max(start, end - _OBS_LIMIT)
            df = _fetch_block(obs_start, end, forecast=False)
        elif start >= now:
            hirlam_end = min(end, hirlam_start + pd.Timedelta(hours=54))
            df = _fetch_block(hirlam_start, hirlam_end, forecast=True)
        else:
            obs_start = max(start, now - _OBS_LIMIT)
            obs   = _fetch_block(obs_start, now, forecast=False)
            fcast = _fetch_block(hirlam_start, min(end, hirlam_start + pd.Timedelta(hours=54)), forecast=True)
            df    = pd.concat([obs, fcast]).sort_index()
            df    = df[~df.index.duplicated(keep='last')]
    except Exception as e:
        print(f'  WARNING: weather fetch failed entirely: {e} — using NaN weather')
        df = pd.DataFrame(columns=_WX_COLS)

    result = df.reindex(all_hours).ffill().bfill()
    if result.empty or result.isna().all().all():
        print('  WARNING: no weather data available — all weather features will be NaN')
        result = pd.DataFrame(np.nan, index=all_hours, columns=_WX_COLS)
    return result
