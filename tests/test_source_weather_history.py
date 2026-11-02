import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transcif.data.weather_reference import UNITS
from transcif.research import multicountry as mc


def producer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('prepare_source_weather_history')


def test_source_weather_decodes_interval_support_and_rejects_wrong_period_units(monkeypatch):
    m=producer(monkeypatch);params=m.source_parameters(51.,10.)
    clock=pd.date_range('2021-12-31','2022-10-03',freq='h',inclusive='left',tz='UTC')
    record=dict(latitude=51.,longitude=10.,elevation=100.,utc_offset_seconds=0,hourly_units=UNITS.copy(),
        hourly=dict(time=(clock.asi8//10**9).tolist(),temperature_2m=np.full(len(clock),10.).tolist(),
            wind_speed_100m=np.full(len(clock),5.).tolist(),shortwave_radiation=np.arange(len(clock)).tolist()))
    frame=m.decode_source(json.dumps(record).encode(),params)
    assert len(frame)==len(clock)-1
    assert frame.shortwave_w_per_m2.iloc[0]==1. and frame.temperature_c.iloc[0]==10.
    with pytest.raises(ValueError,match='contract'):
        m.decode_source(json.dumps(record).encode(),{**params,'end_date':'2023-09-30'})
    record['hourly_units']={**UNITS,'wind_speed_100m':'km/h'}
    with pytest.raises(ValueError,match='units'):
        m.decode_source(json.dumps(record).encode(),params)


def test_expanded_source_cohort_keeps_only_available_history_and_binds_metadata(monkeypatch):
    m=producer(monkeypatch)
    clock=pd.Series(pd.date_range('2022-01-01',periods=744,freq='h').append(pd.date_range('2022-03-15',periods=24,freq='h')))
    raw=pd.DataFrame(np.ones((len(clock),len(mc.FINE))),columns=mc.FINE)
    raw['clock']=clock;raw['time']=clock.dt.tz_localize('UTC');raw['country']='S'
    raw['expected_hours']=clock.dt.days_in_month*24
    weather=pd.DataFrame(dict(valid_at=clock.iloc[-24:].dt.tz_localize('UTC'),temperature_c=10.,wind_speed_ms=5.,shortwave_w_per_m2=100.))
    cohort,audit=m.source_cohort({'R':raw},{'R':weather})
    assert len(cohort)==24 and cohort.latest_allowed_period.eq('2022-01').all()
    assert cohort.origin.eq('2022-03-15 00:00:00+00:00').all()
    assert cohort.weather_kind.eq('source_historical_reanalysis').all()
    assert not {'coal','gas','oil','other'}&set(cohort.columns)
    np.testing.assert_allclose(cohort.wind,1./9.)  # Public monthly share, not hourly MW.
    assert sum(row['eligible'] for row in audit)==1
    assert len([row for row in audit if not row['eligible']])==31


def test_cached_source_receipt_cannot_be_certified_true(tmp_path,monkeypatch):
    m=producer(monkeypatch);folder=tmp_path/'R';folder.mkdir()
    (folder/'response.json').write_text('{}')
    params=m.source_parameters(51.,10.)
    (folder/'receipt.json').write_text(json.dumps(dict(parameters=params,url=m.API+'?'+m.urlencode(params),
        role='source_observed_weather_training',eligible_for_issued_public=True,available_at=None)))
    with pytest.raises(ValueError,match='checkpoint'):m.acquire_source(folder,params)
