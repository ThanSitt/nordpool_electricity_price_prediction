"""
src/test_fmi.py — diagnostic script to see exactly what FMI returns.
Run: python src/test_fmi.py
"""
import requests
import xml.etree.ElementTree as ET

BASE = 'https://opendata.fmi.fi/wfs'

def _request(sq, latlon, params, start, end, timestep=None, place=None, **kwargs):
    url_params = {
        'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
        'storedquery_id': sq,
        'starttime': start,
        'endtime': end,
    }
    if params is not None:
        url_params['parameters'] = params
    if latlon is not None:
        url_params['latlon'] = latlon
    if place is not None:
        url_params['place'] = place
    if timestep is not None:
        url_params['timestep'] = str(timestep)
    if 'maxlocations' in kwargs:
        url_params['maxlocations'] = str(kwargs['maxlocations'])
    if 'fmisid' in kwargs:
        url_params['fmisid'] = str(kwargs['fmisid'])
    if 'bbox' in kwargs:
        url_params['bbox'] = kwargs['bbox']
    r = requests.get(BASE, params=url_params, timeout=30)
    return r.status_code, r.text


def _count_elements(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return f'ParseError: {e}'
    bswfs = [e for e in root.iter() if e.tag.endswith('}BsWfsElement') or e.tag == 'BsWfsElement']
    nans = 0
    vals = 0
    for elem in bswfs:
        for child in elem:
            local = child.tag.split('}')[-1]
            if local == 'ParameterValue':
                if child.text in ('NaN', '', None):
                    nans += 1
                else:
                    vals += 1
    exception = any(e.tag.endswith('}ExceptionReport') or e.tag == 'ExceptionReport' for e in root.iter())
    return f'{len(bswfs)} BsWfsElements  |  {vals} valid values  |  {nans} NaN values  |  ExceptionReport={exception}'


def test(label, sq, latlon, params, start, end, timestep=None, place=None, **kwargs):
    print(f'\n--- {label} ---')
    print(f'    latlon={latlon}  params={params}  timestep={timestep}  extra={kwargs}')
    status, text = _request(sq, latlon, params, start, end, timestep, place, **kwargs)
    print(f'    HTTP {status}  |  body length {len(text)}')
    summary = _count_elements(text)
    print(f'    {summary}')
    if '0 BsWfsElements' in summary:
        print(f'    RAW: {text[:800]}')


if __name__ == '__main__':
    # ── window size tests ──────────────────────────────────────────────────────
    test('OBS place=Helsinki  2h window',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-07-05T22:00:00Z', '2026-07-06T00:00:00Z',
         place='Helsinki')

    test('OBS place=Helsinki  24h window',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-07-05T12:00:00Z', '2026-07-06T12:00:00Z',
         place='Helsinki')

    test('OBS place=Helsinki  168h (7-day) window',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-06-29T12:00:00Z', '2026-07-06T12:00:00Z',
         place='Helsinki')

    test('OBS place=Oulu  24h window',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-07-05T12:00:00Z', '2026-07-06T12:00:00Z',
         place='Oulu')

    # ── old tests (A–F) ───────────────────────────────────────────────────────
    # A: add maxlocations=1 (nearest station lookup)
    test('OBS latlon + maxlocations=1',
         'fmi::observations::weather::simple',
         '60.3172,24.9633', 't2m,ws_10min,wd_10min',
         '2026-07-05T20:00:00Z', '2026-07-06T00:00:00Z',
         maxlocations=1)

    # B: maxlocations=1 + no params
    test('OBS latlon + maxlocations=1  no params',
         'fmi::observations::weather::simple',
         '60.3172,24.9633', None,
         '2026-07-05T20:00:00Z', '2026-07-06T00:00:00Z',
         maxlocations=1)

    # C: fmisid=100968 (Helsinki-Vantaa Airport FMI station)
    test('OBS fmisid=100968 (Helsinki-Vantaa)',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-07-05T20:00:00Z', '2026-07-06T00:00:00Z',
         fmisid=100968)

    # D: fmisid=100968 + no params (see what's available)
    test('OBS fmisid=100968  no params',
         'fmi::observations::weather::simple',
         None, None,
         '2026-07-05T20:00:00Z', '2026-07-06T00:00:00Z',
         fmisid=100968)

    # E: bbox around Helsinki (lon_min,lat_min,lon_max,lat_max)
    test('OBS bbox Helsinki area',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-07-05T22:00:00Z', '2026-07-06T00:00:00Z',
         bbox='24.5,60.1,25.5,60.6')

    # F: place=Helsinki (city name in English)
    test('OBS place=Helsinki',
         'fmi::observations::weather::simple',
         None, 't2m,ws_10min,wd_10min',
         '2026-07-05T22:00:00Z', '2026-07-06T00:00:00Z',
         place='Helsinki')

    # forecast — timestep=60 works fine
    test('FCAST  Helsinki-Vantaa (edited scandinavia)',
         'fmi::forecast::edited::weather::scandinavia::point::simple',
         '60.3172,24.9633', 'temperature,windspeedms,winddirection',
         '2026-07-06T00:00:00Z', '2026-07-06T12:00:00Z',
         timestep=60)

    test('FCAST  Oulu (edited scandinavia)',
         'fmi::forecast::edited::weather::scandinavia::point::simple',
         '65.0126,25.4647', 'temperature,windspeedms,winddirection',
         '2026-07-06T00:00:00Z', '2026-07-06T12:00:00Z',
         timestep=60)
