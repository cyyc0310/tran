import json

import numpy as np
import pandas as pd
import pytest

from transcif.data.weather_reference import API, YEARS, UNITS, parameters, decode, monthly_resources, climatology, acquire


def fixture(year=2020):
    params = parameters(51., 10., year)
    clock = pd.date_range(params['start_date'], pd.Timestamp(params['end_date'])+pd.Timedelta(days=1),
                          inclusive='left', freq='h', tz='UTC')
    record = dict(latitude=51., longitude=10., elevation=100., utc_offset_seconds=0,
                  hourly_units=UNITS.copy(), hourly=dict(time=(clock.asi8//10**9).tolist(),
                  temperature_2m=np.full(len(clock), 10.).tolist(),
                  wind_speed_100m=np.resize([0., 30.], len(clock)).tolist(),
                  shortwave_radiation=np.arange(len(clock), dtype=float).tolist()))
    return params, record


def test_reference_rejects_future_year_and_unit_or_hour_corruption():
    with pytest.raises(ValueError, match='2016'):
        parameters(51., 10., 2022)
    params, record = fixture()
    record['hourly_units'] = {**UNITS, 'wind_speed_100m': 'km/h'}
    with pytest.raises(ValueError, match='units'):
        decode(json.dumps(record).encode(), params)
    record['hourly_units'] = UNITS
    record['hourly']['time'][10] = record['hourly']['time'][9]
    with pytest.raises(ValueError, match='timestamps'):
        decode(json.dumps(record).encode(), params)


def test_full_local_months_dst_leap_and_preceding_hour_solar():
    params, record = fixture(); frame = decode(json.dumps(record).encode(), params)
    rows = monthly_resources(frame, 2020, 'Europe/Berlin').set_index('month')
    assert rows.loc[2, 'hours'] == 29*24
    assert rows.loc[3, 'hours'] == 31*24-1
    assert rows.loc[10, 'hours'] == 31*24+1
    assert rows.loc[1, 'wind_resource'] == 25.**3/2
    assert rows.loc[1, 'heating'] == 8.
    start = pd.Timestamp('2020-01-01', tz='Europe/Berlin')
    end = pd.Timestamp('2020-02-01', tz='Europe/Berlin')
    expected = frame.loc[frame.solar_interval_start.ge(start)&frame.solar_interval_start.lt(end)]
    assert rows.loc[1, 'solar_resource'] == expected.shortwave_radiation.mean()
    instant = frame.loc[frame.instant_at.ge(start)&frame.instant_at.lt(end)]
    assert rows.loc[1, 'solar_resource'] != instant.shortwave_radiation.mean()
    with pytest.raises(ValueError, match='coverage'):
        monthly_resources(frame.drop(frame.index[500]), 2020, 'Europe/Berlin')


def test_climatology_equal_years_not_equal_hours_and_complete_contract():
    annual = pd.DataFrame([dict(region='R', year=y, month=m, hours=1 if y==2016 else 2,
        temperature_c=y-2016, heating=0., cooling=0., wind_resource=1., solar_resource=1.)
        for y in YEARS for m in range(1,13)])
    result = climatology(annual)
    assert result.temperature_c.eq(2.).all() and len(result)==12
    assert not result.eligible_for_issued_public.any()
    with pytest.raises(ValueError, match='five years'):
        climatology(annual.iloc[1:])
    with pytest.raises(ValueError, match='duplicate'):
        climatology(pd.concat([annual,annual.iloc[:1]]))


def test_cached_reference_hash_and_contract_are_checked(tmp_path):
    import hashlib
    from urllib.parse import urlencode
    params, record = fixture(); payload=json.dumps(record).encode()
    folder=tmp_path/'complete'; folder.mkdir()
    (folder/'response.json').write_bytes(payload)
    receipt=dict(parameters=params, sha256=hashlib.sha256(payload).hexdigest(),
        url=API+'?'+urlencode(params), retrieved_at='2026-10-31T00:00:00Z', headers={},
        returned_point={k:record[k] for k in ('latitude','longitude','elevation')},
        units=UNITS, hours=len(record['hourly']['time']), eligible_for_issued_public=False,
        available_at=None, vintage='revised archive, original public release not certified',
        support='T/wind instantaneous; solar preceding hour mean')
    (folder/'receipt.json').write_text(json.dumps(receipt))
    frame,_=acquire(folder,params)
    assert len(frame)>8760
    (folder/'response.json').write_bytes(payload+b' ')
    with pytest.raises(ValueError, match='checkpoint'):
        acquire(folder,params)
    (folder/'response.json').write_bytes(payload)
    receipt['eligible_for_issued_public']=True
    (folder/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='checkpoint'):
        acquire(folder,params)


def test_response_read_transport_failure_retries_without_bad_checkpoint(tmp_path, monkeypatch):
    from http.client import IncompleteRead
    import transcif.data.weather_reference as module
    params,record=fixture(); payload=json.dumps(record).encode(); calls=[]
    class Response:
        headers={}
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def read(self,limit):
            calls.append(limit)
            if len(calls)<3: raise IncompleteRead(b'partial',100)
            return payload
    monkeypatch.setattr(module,'urlopen',lambda *args,**kwargs:Response())
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    frame,receipt=acquire(tmp_path/'complete',params)
    assert len(calls)==3 and len(frame)>8760
    assert receipt['eligible_for_issued_public'] is False
    acquire(tmp_path/'complete',params)
    assert len(calls)==3
