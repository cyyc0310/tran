import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_resource_controls import reference_fixture
from transcif.research import resource_controls_daily as daily
from transcif.research import coarse_trajectories as ct
from transcif.research import multicountry as mc


def test_three_paired_references_match_frozen_provider_and_label_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_hourly_transfer_baselines')
    f,c,fuel,power,climate=reference_fixture()
    model=daily.fit(f,c,power,climate,allowed_countries=('S',));bank=ct.fit(f,fuel)
    query=f.copy();query['clock']+=pd.DateOffset(years=1)
    actual=runner.paired_recomposition(bank,model,query,climate)
    expected,_=daily.predict_methods(bank,model,query,climate)
    for name in actual:
        old={'paired_joint':'joint_original','paired_hgb_no_weather':'hgb_no_weather','paired_hgb_weather':'hgb_weather'}[name]
        np.testing.assert_array_equal(actual[name],expected[old])
    raw=pd.DataFrame(fuel,columns=mc.FINE)
    raw['time']=f.clock.dt.tz_localize('UTC');raw['clock']=f.clock;raw['country']='S'
    frame=f.copy();frame['valid_at']=raw.time;frame['label_row']=np.arange(len(frame))
    np.testing.assert_array_equal(runner.canonical_fuel(frame,{'R':raw}),fuel)
    frame['country']='TARGET'
    with pytest.raises(ValueError,match='identity'):runner.canonical_fuel(frame,{'R':raw})
