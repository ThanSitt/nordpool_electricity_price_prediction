"""
src/fetch_live.py — live data fetchers

fetch_prices(start, end)  → pd.Series  (hourly EUR/MWh from Fingrid)
fetch_weather(start, end) → pd.DataFrame (hourly temp/wind from FMI)
"""

from __future__ import annotations
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
    r = requests.get('https://opendata.fmi.fi/wfs', params={
        'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
        'storedquery_id': storedquery,
        'latlon': latlon,
        'parameters': params,
        'starttime': start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'endtime':   end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'timestep': '60',
    }, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.text


def _parse_fmi(xml_text: str) -> pd.DataFrame:
    """Parse FMI WFS XML into a wide DataFrame (one column per parameter)."""
    root = ET.fromstring(xml_text)
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

    temp_df = _parse_fmi(_fmi_request(sq, _TEMP_LATLON, params, start, end))
    wind_df = _parse_fmi(_fmi_request(sq, _WIND_LATLON, params, start, end))

    # column name aliases (FMI uses same lowercase names for obs and forecast)
    rename = {'temperature': 'temp', 'windspeedms': 'wind_speed',
              'winddirection': 'wind_direction_deg'}

    result = pd.DataFrame(index=temp_df.index if not temp_df.empty else pd.DatetimeIndex([]))
    if not temp_df.empty:
        result['temp'] = temp_df.get('temperature', np.nan)
    if not wind_df.empty:
        result['wind_speed']         = wind_df.get('windspeedms', np.nan).reindex(result.index)
        result['wind_direction_deg'] = wind_df.get('winddirection', np.nan).reindex(result.index)
    elif not temp_df.empty:
        result['wind_speed']         = temp_df.get('windspeedms', np.nan)
        result['wind_direction_deg'] = temp_df.get('winddirection', np.nan)

    return result[['temp', 'wind_speed', 'wind_direction_deg']].dropna(how='all')


def fetch_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Hourly weather DataFrame with columns [temp, wind_speed, wind_direction_deg].

    Uses FMI observations for past data and HIRLAM forecast for future.
    HIRLAM covers ~54 hours ahead; beyond that the last value is held constant.
    """
    now = pd.Timestamp.now(tz=_HELSINKI)
    all_hours = pd.date_range(start, end, freq='1h', tz=_HELSINKI)

    _OBS_LIMIT = pd.Timedelta(hours=168)  # FMI observations max window

    if end <= now:
        obs_start = max(start, end - _OBS_LIMIT)
        df = _fetch_block(obs_start, end, forecast=False)
    elif start >= now:
        hirlam_end = min(end, now + pd.Timedelta(hours=54))
        df = _fetch_block(start, hirlam_end, forecast=True)
    else:
        obs_start = max(start, now - _OBS_LIMIT)
        obs   = _fetch_block(obs_start, now, forecast=False)
        fcast = _fetch_block(now, min(end, now + pd.Timedelta(hours=54)), forecast=True)
        df    = pd.concat([obs, fcast]).sort_index()
        df    = df[~df.index.duplicated(keep='last')]

    return df.reindex(all_hours).ffill().bfill()
