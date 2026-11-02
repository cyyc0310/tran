"""Fixed pre-experiment ERA5 climate references, never future observed weather."""
from __future__ import annotations

import hashlib
from http.client import IncompleteRead, RemoteDisconnected
import json
from pathlib import Path
import tempfile
import time
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

YEARS = (2016, 2017, 2018, 2019, 2020)
FIELDS = ('temperature_2m', 'wind_speed_100m', 'shortwave_radiation')
UNITS = dict(time='unixtime', temperature_2m='°C', wind_speed_100m='m/s',
             shortwave_radiation='W/m²')
API = 'https://archive-api.open-meteo.com/v1/archive'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(latitude: float, longitude: float, year: int) -> dict:
    """One UTC day of padding on either side supports all local monthly bins."""
    if year not in YEARS:
        raise ValueError('only fixed 2016–2020 reference years allowed')
    if not np.isfinite([latitude, longitude]).all() or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('invalid coordinates')
    return dict(latitude=latitude, longitude=longitude,
                start_date=f'{year-1}-12-31', end_date=f'{year+1}-01-02',
                hourly=','.join(FIELDS), models='era5', timezone='UTC',
                timeformat='unixtime', wind_speed_unit='ms', temperature_unit='celsius',
                elevation='nan', cell_selection='nearest')


def decode(payload: bytes, params: dict) -> pd.DataFrame:
    """Reject missing hours, units or values before any monthly aggregation."""
    if params != parameters(params['latitude'], params['longitude'], int(params['start_date'][:4])+1):
        raise ValueError('request differs from fixed ERA5 contract')
    record = json.loads(payload)
    if record.get('utc_offset_seconds') != 0 or record.get('hourly_units') != UNITS:
        raise ValueError('weather clock or units differ')
    point = np.asarray([record.get('latitude'), record.get('longitude')], float)
    if not np.isfinite(point).all() or np.max(np.abs(point-[params['latitude'], params['longitude']])) > .26:
        raise ValueError('returned grid point differs from requested point')
    expected = pd.date_range(params['start_date'], pd.Timestamp(params['end_date'])+pd.Timedelta(days=1),
                             freq='h', inclusive='left', tz='UTC')
    frame = pd.DataFrame(record['hourly'])
    epoch = frame['time'].to_numpy(float)
    if not np.array_equal(epoch, expected.asi8//10**9):
        raise ValueError('weather timestamps are missing, duplicated or unordered')
    values = frame[list(FIELDS)].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 1:] < 0).any():
        raise ValueError('weather values missing, negative wind or negative solar')
    frame['instant_at'] = expected
    frame['solar_interval_start'] = expected-pd.Timedelta(hours=1)
    return frame


def acquire(directory: Path, params: dict) -> tuple[pd.DataFrame, dict]:
    """Resume only verified immutable checkpoints; retry network failures only."""
    if directory.exists():
        receipt = json.loads((directory/'receipt.json').read_text())
        required = {'parameters', 'url', 'retrieved_at', 'sha256', 'headers', 'returned_point',
                    'units', 'hours', 'eligible_for_issued_public', 'available_at', 'vintage', 'support'}
        if (not required <= receipt.keys() or receipt['parameters'] != params
                or receipt['url'] != API+'?'+urlencode(params)
                or receipt['eligible_for_issued_public'] is not False or receipt['available_at'] is not None
                or receipt['units'] != UNITS
                or receipt['vintage'] != 'revised archive, original public release not certified'
                or receipt['support'] != 'T/wind instantaneous; solar preceding hour mean'
                or digest(directory/'response.json') != receipt['sha256']):
            raise ValueError('cached reference checkpoint differs')
        payload = (directory/'response.json').read_bytes()
        frame = decode(payload, params); record = json.loads(payload)
        if (receipt['hours'] != len(frame) or receipt['returned_point'] !=
                {k: record[k] for k in ('latitude', 'longitude', 'elevation')}
                or pd.Timestamp(receipt['retrieved_at']).tzinfo is None):
            raise ValueError('cached reference receipt differs from payload')
        return frame, receipt
    url = API+'?'+urlencode(params)
    for attempt in range(3):
        try:
            with urlopen(url, timeout=40) as response:
                payload = response.read(2_000_001)
                headers = dict(response.headers.items())
            break
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(2*(attempt+1))
        except (URLError, TimeoutError, IncompleteRead, RemoteDisconnected, ConnectionResetError, ssl.SSLError):
            if attempt == 2:
                raise
            time.sleep(2*(attempt+1))
    if len(payload) > 2_000_000:
        raise ValueError('annual single-point response unexpectedly large')
    frame = decode(payload, params)
    data = json.loads(payload)
    receipt = dict(parameters=params, url=url, retrieved_at=pd.Timestamp.now(tz='UTC').isoformat(),
                   sha256=hashlib.sha256(payload).hexdigest(), headers=headers,
                   returned_point={k: data[k] for k in ('latitude', 'longitude', 'elevation')},
                   units=data['hourly_units'], hours=len(frame),
                   eligible_for_issued_public=False, available_at=None,
                   vintage='revised archive, original public release not certified',
                   support='T/wind instantaneous; solar preceding hour mean')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-', dir=directory.parent) as temp:
        stage = Path(temp)/'complete'; stage.mkdir()
        (stage/'response.json').write_bytes(payload)
        (stage/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        stage.rename(directory)
    return frame, receipt


def monthly_resources(frame: pd.DataFrame, year: int, zone: str) -> pd.DataFrame:
    """Validate each local month's absolute hourly support, including DST."""
    if year not in YEARS:
        raise ValueError('reference year outside fixed budget')
    rows = []
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1, tz=zone)
        end = start+pd.offsets.MonthBegin(1)
        expected = pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC')
        instantaneous = frame.loc[frame.instant_at.ge(start) & frame.instant_at.lt(end)]
        solar = frame.loc[frame.solar_interval_start.ge(start) & frame.solar_interval_start.lt(end)]
        if (not np.array_equal(pd.DatetimeIndex(instantaneous.instant_at).asi8, expected.asi8)
                or not np.array_equal(pd.DatetimeIndex(solar.solar_interval_start).asi8, expected.asi8)):
            raise ValueError('incomplete absolute hourly coverage in a local reference month')
        temperature = instantaneous.temperature_2m.to_numpy(float)
        wind = instantaneous.wind_speed_100m.to_numpy(float)
        shortwave = solar.shortwave_radiation.to_numpy(float)
        if not np.isfinite(np.r_[temperature, wind, shortwave]).all() or (wind < 0).any() or (shortwave < 0).any():
            raise ValueError('invalid monthly weather')
        rows.append(dict(year=year, month=month, hours=len(expected),
                         temperature_c=float(temperature.mean()),
                         heating=float(np.maximum(18.-temperature, 0.).mean()),
                         cooling=float(np.maximum(temperature-22., 0.).mean()),
                         wind_resource=float((np.minimum(wind, 25.)**3).mean()),
                         solar_resource=float(shortwave.mean())))
    return pd.DataFrame(rows)


def climatology(annual: pd.DataFrame) -> pd.DataFrame:
    """Each of five years receives equal weight, including leap Februaries."""
    if annual.duplicated(['region', 'year', 'month']).any():
        raise ValueError('duplicate reference year/month')
    for _, group in annual.groupby('region'):
        if set(zip(group.year, group.month)) != {(y, m) for y in YEARS for m in range(1, 13)}:
            raise ValueError('all five years and twelve months required per point')
    columns = ['temperature_c', 'heating', 'cooling', 'wind_resource', 'solar_resource']
    if not len(annual) or not np.isfinite(annual[columns].to_numpy()).all():
        raise ValueError('empty or invalid reference')
    result = annual.groupby(['region', 'month'])[columns].mean().reset_index()
    result['reference_years'] = len(YEARS)
    result['eligible_for_issued_public'] = False
    return result
