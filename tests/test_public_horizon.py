import numpy as np
import pandas as pd
import pytest

from transcif.research.multicountry import GROUPS,HISTORY,public_features
from transcif.research.public_horizon import public_horizon_features


def table():
    return pd.DataFrame(np.ones((24,6)),columns=GROUPS,index=pd.period_range('2022-01',periods=24,freq='M'))


def test_horizon_crossing_month_cannot_admit_new_valid_month_statistics():
    months=table();months.loc['2023-03']=[80.,1.,1.,1.,1.,1.]
    origin=pd.Timestamp('2023-04-30T12:00')
    clock=pd.Series(pd.date_range(origin,periods=24,freq='h'))
    result=public_horizon_features(months,clock,origin)
    assert result.thermal.eq(1/6).all()
    assert result.latest_allowed_period.eq('2023-02').all()
    assert result.period.nunique()==2 and result.cal4.nunique()==2
    months.loc[months.index>pd.Period('2023-02')]=[1.,200.,1.,1.,1.,1.]
    after=public_horizon_features(months,clock,origin)
    pd.testing.assert_frame_equal(result[list(HISTORY)],after[list(HISTORY)])


def test_same_month_horizon_matches_existing_reference_exactly():
    months=table();origin=pd.Timestamp('2023-04-15T00:00')
    clock=pd.Series(pd.date_range(origin,periods=24,freq='h'))
    expected=public_features(months,clock)
    result=public_horizon_features(months,clock,origin)
    pd.testing.assert_frame_equal(expected[list(HISTORY)],result[list(HISTORY)],check_exact=True)


def test_horizon_rejects_missing_anchor_or_mixed_clock_conventions():
    months=table();clock=pd.Series(pd.date_range('2023-04-15',periods=24,freq='h'))
    with pytest.raises(ValueError,match='origin-lag2'):
        public_horizon_features(months.drop(pd.Period('2023-02')),clock,pd.Timestamp('2023-04-15'))
    with pytest.raises(ValueError,match='local clocks'):
        public_horizon_features(months,clock,pd.Timestamp('2023-04-15T00:00Z'))
