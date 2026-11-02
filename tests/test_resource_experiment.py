import importlib
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from transcif.research import multicountry as mc


def test_expanded_source_rows_bind_country_clock_and_scope(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    validate=importlib.import_module('run_resource_experiment').validate_observed_source_rows
    clock=pd.Series(pd.date_range('2022-03-15',periods=24,freq='h'))
    frame=pd.DataFrame(dict(clock=clock,valid_at=clock.dt.tz_localize('UTC'),country='S',region='R',
                            split='train',label_row=range(24)))
    full_clock=pd.Series(pd.date_range('2022-01-01','2022-04-01',freq='h',inclusive='left'))
    source=pd.DataFrame(np.ones((len(full_clock),len(mc.FINE))),columns=mc.FINE)
    source['clock']=full_clock;source['time']=full_clock.dt.tz_localize('UTC');source['country']='S'
    source['expected_hours']=full_clock.dt.days_in_month*24
    raw={'R':source};frame['label_row']=pd.Index(source.clock).get_indexer(frame.clock)
    expected=mc.public_features(mc.public_months(source),frame.clock)
    for name in mc.HISTORY:frame[name]=expected[name].to_numpy()
    frame['forecast_origin_local']='2022-03-15'
    frame['latest_allowed_period']='2022-01';frame['origin']='2022-03-15T00:00:00Z'
    validate(frame,raw)
    forged=frame.copy();forged['country']='T'
    with pytest.raises(ValueError,match='canonical'):validate(forged,raw)
    shifted=frame.copy();shifted['valid_at']+=pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match='canonical'):validate(shifted,raw)
    with pytest.raises(ValueError,match='unique'):validate(pd.concat([frame,frame.iloc[:1]]),raw)
    future=frame.copy();future['clock']+=pd.DateOffset(months=7)
    with pytest.raises(ValueError,match='September'):validate(future,raw)
    invalid=frame.copy();invalid['split']='calibration'
    with pytest.raises(ValueError,match='source history'):validate(invalid,raw)
    future_origin=frame.copy();future_origin['forecast_origin_local']='2023-03-15'
    with pytest.raises(ValueError,match='local day'):validate(future_origin,raw)
    future_month=frame.copy();future_month['latest_allowed_period']='2022-03'
    with pytest.raises(ValueError,match='t-2'):validate(future_month,raw)
    future_values=frame.copy();future_values['thermal']+=.01
    with pytest.raises(ValueError,match='past-only'):validate(future_values,raw)
