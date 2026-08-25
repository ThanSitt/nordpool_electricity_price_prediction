"""Live market and weather data fetchers.

``fetch_prices`` returns Finland (FI) Nord Pool day-ahead prices at their
published resolution.  ``fetch_weather`` combines FMI observations/short-range
forecast with an Open-Meteo long-range forecast so a seven-day run never
silently reuses a 54-hour weather forecast for the remaining days.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

import config

_TIMEOUT = 60
_HELSINKI = config.HELSINKI

# Forecast uses grid interpolation. Observations use place= because FMI's
# observation endpoint does not reliably resolve arbitrary lat/lon requests.
_TEMP_LATLON = '60.3172,24.9633'  # Helsinki-Vantaa Airport
_WIND_LATLON = '65.0126,25.4647'  # Oulu area
_TEMP_PLACE = 'Helsinki'
_WIND_PLACE = 'Oulu'

_TEMP_COORDS = (60.3172, 24.9633)
_WIND_COORDS = (65.0126, 25.4647)
_WX_COLS = ['temp', 'wind_speed', 'wind_direction_deg']


def _to_helsinki(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(_HELSINKI)
    return ts.tz_convert(_HELSINKI)


# --- Nord Pool FI day-ahead prices via Elering's public NPS endpoint ---------

def fetch_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return FI Nord Pool day-ahead prices indexed in Helsinki time.

    The previous implementation requested Fingrid dataset 105.  That dataset
    is down-regulation bid volume (MW), not an electricity price, which is why
    it produced a long sequence of zeros.  Elering's NPS endpoint exposes the
    FI area price directly and includes quarter-hour values where available.
    """
    start = _to_helsinki(start)
    end = _to_helsinki(end)
    if end <= start:
        raise ValueError('end must be later than start')

    response = requests.get(
        config.ELERING_PRICE_URL,
        params={
            'start': start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end': end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    rows = body.get('data', {}).get('fi', [])
    if not body.get('success') or not rows:
        raise RuntimeError('No FI Nord Pool prices returned by Elering')

    records: dict[pd.Timestamp, float] = {}
    for row in rows:
        try:
            timestamp = pd.to_datetime(row['timestamp'], unit='s', utc=True).tz_convert(_HELSINKI)
            price = float(row['price'])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= timestamp <= end:
            records[timestamp] = price

    series = pd.Series(records, name='price', dtype=float).sort_index()
    if series.empty:
        raise RuntimeError('Elering returned no FI price records in the requested interval')
    if len(series) >= 24 and series.eq(0).all():
        raise RuntimeError('All returned FI prices are zero; refusing to evaluate forecasts with invalid market data')
    return series


# --- FMI weather ---------------------------------------------------------------

def _fmi_request(storedquery: str, location: dict, params: str,
                 start: pd.Timestamp, end: pd.Timestamp,
                 timestep: int | None = None) -> str:
    """HTTP GET to FMI WFS with three retries for transient failures."""
    url_params = {
        'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
        'storedquery_id': storedquery,
        **location,
        'parameters': params,
        'starttime': start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'endtime': end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    if timestep is not None:
        url_params['timestep'] = str(timestep)
    for attempt in range(3):
        try:
            response = requests.get('https://opendata.fmi.fi/wfs', params=url_params, timeout=_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            print(f'  FMI request failed (attempt {attempt + 1}/3): {exc}; retrying...')
            time.sleep(5)
    raise AssertionError('unreachable')


def _parse_fmi(xml_text: str) -> pd.DataFrame:
    """Parse FMI WFS XML into a wide DataFrame (one column per parameter)."""
    if not xml_text:
        return pd.DataFrame()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame()

    if any(elem.tag.endswith('}ExceptionReport') or elem.tag == 'ExceptionReport' for elem in root.iter()):
        print(f'  FMI returned ExceptionReport: {xml_text[:300]}')
        return pd.DataFrame()

    rows: list[dict] = []
    for elem in root.iter():
        if not elem.tag.endswith('}BsWfsElement') and elem.tag != 'BsWfsElement':
            continue
        record: dict[str, str] = {}
        for child in elem:
            local = child.tag.split('}')[-1]
            if local in ('Time', 'ParameterName', 'ParameterValue'):
                record[local] = child.text or ''
        if len(record) < 3 or record.get('ParameterValue') in ('NaN', ''):
            continue
        try:
            value = float(record['ParameterValue'])
            timestamp = pd.Timestamp(record['Time']).tz_convert(_HELSINKI).replace(second=0, microsecond=0)
        except (TypeError, ValueError):
            continue
        rows.append({'datetime': timestamp, 'param': record['ParameterName'].lower(), 'value': value})

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index='datetime', columns='param', values='value', aggfunc='first')


def _fetch_fmi_block(start: pd.Timestamp, end: pd.Timestamp, forecast: bool) -> pd.DataFrame:
    if forecast:
        storedquery = 'fmi::forecast::edited::weather::scandinavia::point::simple'
        params, timestep = 'temperature,windspeedms,winddirection', 60
        temp_loc, wind_loc = {'latlon': _TEMP_LATLON}, {'latlon': _WIND_LATLON}
        temp_col, wind_col, direction_col = 'temperature', 'windspeedms', 'winddirection'
    else:
        storedquery = 'fmi::observations::weather::simple'
        params, timestep = 't2m,ws_10min,wd_10min', None
        temp_loc, wind_loc = {'place': _TEMP_PLACE}, {'place': _WIND_PLACE}
        temp_col, wind_col, direction_col = 't2m', 'ws_10min', 'wd_10min'

    try:
        temp_df = _parse_fmi(_fmi_request(storedquery, temp_loc, params, start, end, timestep))
    except Exception as exc:
        print(f'  WARNING: temperature fetch failed: {exc}')
        temp_df = pd.DataFrame()
    try:
        wind_df = _parse_fmi(_fmi_request(storedquery, wind_loc, params, start, end, timestep))
    except Exception as exc:
        print(f'  WARNING: wind fetch failed: {exc}')
        wind_df = pd.DataFrame()

    if temp_df.empty and wind_df.empty:
        return pd.DataFrame(columns=_WX_COLS)
    index = temp_df.index if not temp_df.empty else wind_df.index
    result = pd.DataFrame(index=index, dtype=float)
    result['temp'] = temp_df[temp_col].reindex(index) if temp_col in temp_df else np.nan
    wind_source = wind_df if wind_col in wind_df else temp_df
    result['wind_speed'] = wind_source[wind_col].reindex(index) if wind_col in wind_source else np.nan
    result['wind_direction_deg'] = wind_source[direction_col].reindex(index) if direction_col in wind_source else np.nan
    return result[_WX_COLS].dropna(how='all')


# --- Long-range weather --------------------------------------------------------

def _fetch_open_meteo_point(latitude: float, longitude: float) -> pd.DataFrame:
    response = requests.get(
        config.OPEN_METEO_FORECAST_URL,
        params={
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m,wind_speed_10m,wind_direction_10m',
            'wind_speed_unit': 'ms',
            'timezone': _HELSINKI,
            # A run launched in the afternoon forecasts from next midnight;
            # request extra calendar days so the full seven delivery days are
            # covered even near the end of today's API window.
            'forecast_days': 10,
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    hourly = response.json().get('hourly', {})
    times = hourly.get('time', [])
    if not times:
        return pd.DataFrame()
    index = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize(_HELSINKI, ambiguous='NaT', nonexistent='shift_forward')
    result = pd.DataFrame(index=index)
    result['temp'] = hourly.get('temperature_2m', np.nan)
    result['wind_speed'] = hourly.get('wind_speed_10m', np.nan)
    result['wind_direction_deg'] = hourly.get('wind_direction_10m', np.nan)
    return result[~result.index.isna()]


def _fetch_open_meteo_weather() -> pd.DataFrame:
    """Use Helsinki temperature and Oulu wind, matching the training stations."""
    temp = _fetch_open_meteo_point(*_TEMP_COORDS)
    wind = _fetch_open_meteo_point(*_WIND_COORDS)
    if temp.empty and wind.empty:
        return pd.DataFrame(columns=_WX_COLS)
    index = temp.index if not temp.empty else wind.index
    result = pd.DataFrame(index=index)
    result['temp'] = temp['temp'].reindex(index) if 'temp' in temp else np.nan
    result['wind_speed'] = wind['wind_speed'].reindex(index) if 'wind_speed' in wind else np.nan
    result['wind_direction_deg'] = wind['wind_direction_deg'].reindex(index) if 'wind_direction_deg' in wind else np.nan
    return result[_WX_COLS]


# --- Fingrid (grid transmission + nuclear) ------------------------------------

def _fetch_fingrid_pages(dataset_id: int, start: pd.Timestamp,
                         end: pd.Timestamp, api_key: str) -> pd.DataFrame:
    """Fetch all pages from one Fingrid dataset with 429 backoff."""
    url = config.FINGRID_API_URL.format(dataset_id=dataset_id)
    headers = {'x-api-key': api_key}
    start_utc = start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    end_utc   = end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    all_rows: list = []
    page = 1
    while True:
        for attempt in range(5):
            resp = requests.get(url, headers=headers, params={
                'startTime': start_utc, 'endTime': end_utc,
                'format': 'json', 'pageSize': 10000, 'page': page,
            }, timeout=_TIMEOUT)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f'    Fingrid 429 — waiting {wait}s (attempt {attempt+1}/5) ...')
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f'Fingrid dataset {dataset_id}: still 429 after 5 retries')
        rows = resp.json().get('data', [])
        all_rows.extend(rows)
        if len(rows) < 10000:
            break
        page += 1
        time.sleep(2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df['datetime'] = pd.to_datetime(df['startTime'], utc=True).dt.tz_convert(_HELSINKI)
    return df.set_index('datetime')[['value']].sort_index()


def fetch_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch cross-border power flows from Fingrid for the given window.

    Returns a DataFrame with columns fi_ee, fi_no, fi_se_north, fi_se_central
    (plus derived fi_se_total, fi_total_net, fi_se_abs) indexed in Helsinki time
    at 15-minute resolution.  Returns an empty DataFrame if the API key is
    missing or all fetches fail — callers must handle gracefully.
    """
    api_key = os.environ.get('FINGRID_API_KEY')
    if not api_key:
        print('  WARNING: FINGRID_API_KEY not set — grid features will be NaN')
        return pd.DataFrame()
    start, end = _to_helsinki(start), _to_helsinki(end)
    frames: dict[str, pd.Series] = {}
    for i, (name, ds_id) in enumerate(config.FINGRID_GRID_DATASETS.items()):
        if i > 0:
            time.sleep(10)
        try:
            df = _fetch_fingrid_pages(ds_id, start, end, api_key)
            if not df.empty:
                frames[name] = df['value'].rename(name)
        except Exception as exc:
            print(f'  WARNING: Fingrid grid fetch {name} (dataset {ds_id}): {exc}')
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames.values(), axis=1).sort_index()
    idx_15 = pd.date_range(start, end, freq='15min', tz=_HELSINKI)
    combined = combined.reindex(idx_15).ffill().bfill()
    # Derived features
    combined['fi_se_total']  = combined['fi_se_north'] + combined['fi_se_central']
    combined['fi_total_net'] = combined[list(config.FINGRID_GRID_DATASETS)].sum(axis=1)
    combined['fi_se_abs']    = combined['fi_se_total'].abs()
    return combined


def fetch_nuclear(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch nuclear power production from Fingrid for the given window.

    Returns a DataFrame with a single column nuclear_power_mw in Helsinki time
    at 15-minute resolution (resampled from the ~3-min real-time feed).
    Returns an empty DataFrame if the API key is missing or the fetch fails.
    """
    api_key = os.environ.get('FINGRID_API_KEY')
    if not api_key:
        print('  WARNING: FINGRID_API_KEY not set — nuclear features will be NaN')
        return pd.DataFrame()
    start, end = _to_helsinki(start), _to_helsinki(end)
    try:
        df = _fetch_fingrid_pages(config.FINGRID_NUCLEAR_DATASET, start, end, api_key)
    except Exception as exc:
        print(f'  WARNING: Fingrid nuclear fetch failed: {exc}')
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    # Resample ~3-min real-time feed to 15-min mean (matches training resolution)
    nuclear_15 = df['value'].resample('15min').mean()
    idx_15 = pd.date_range(start, end, freq='15min', tz=_HELSINKI)
    nuclear_15 = nuclear_15.reindex(idx_15).ffill().bfill()
    return nuclear_15.rename('nuclear_power_mw').to_frame()


def fetch_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return hourly weather without fabricating a seven-day forecast.

    FMI observations cover history and its HIRLAM product supplies short-range
    data.  Open-Meteo supplies the remaining horizon.  If the long-range source
    is unavailable for a requested seven-day run, this function fails instead
    of silently forward-filling the last weather observation.
    """
    start, end = _to_helsinki(start), _to_helsinki(end)
    now = pd.Timestamp.now(tz=_HELSINKI)
    start_hour = start.floor('h')
    all_hours = pd.date_range(start_hour, end.ceil('h'), freq='1h', tz=_HELSINKI)
    observation_limit = pd.Timedelta(hours=168)

    hirlam_start = now.tz_convert('UTC').floor('h')
    hirlam_start -= pd.Timedelta(hours=hirlam_start.hour % 6)
    hirlam_start = hirlam_start.tz_convert(_HELSINKI)
    hirlam_end = hirlam_start + pd.Timedelta(hours=54)

    try:
        if end <= now:
            fmi = _fetch_fmi_block(max(start, end - observation_limit), end, forecast=False)
        elif start >= now:
            fmi = _fetch_fmi_block(hirlam_start, min(end, hirlam_end), forecast=True)
        else:
            observations = _fetch_fmi_block(max(start, now - observation_limit), now, forecast=False)
            forecast = _fetch_fmi_block(hirlam_start, min(end, hirlam_end), forecast=True)
            fmi = pd.concat([observations, forecast]).sort_index()
            fmi = fmi[~fmi.index.duplicated(keep='last')]
    except Exception as exc:
        print(f'  WARNING: FMI weather fetch failed: {exc}')
        fmi = pd.DataFrame(columns=_WX_COLS)

    long_range_needed = end > hirlam_end
    open_meteo = pd.DataFrame(columns=_WX_COLS)
    if end > now:
        try:
            open_meteo = _fetch_open_meteo_weather()
        except Exception as exc:
            if long_range_needed:
                raise RuntimeError(f'Long-range weather forecast unavailable: {exc}') from exc
            print(f'  WARNING: Open-Meteo forecast unavailable: {exc}')

    result = fmi.reindex(all_hours).combine_first(open_meteo.reindex(all_hours))
    # Check coverage before filling small source gaps.  Checking afterwards
    # would make a stale FMI value look like a valid long-range forecast.
    if long_range_needed:
        long_range_rows = result.index > hirlam_end
        if result.loc[long_range_rows].isna().any().any():
            raise RuntimeError('Long-range weather coverage is incomplete')
    result = result.ffill().bfill()
    if result.empty or result.isna().all().all():
        raise RuntimeError('No weather data available for forecast features')
    return result[_WX_COLS]
