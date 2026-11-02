import numpy as np
import pandas as pd

from transcif.research.multicountry import GROUPS,HISTORY,METHODS,fit,predict,public_features,weights


def test_public_history_ignores_future_and_preserves_nonthermal_remainder():
    index=pd.period_range('2022-01',periods=24,freq='M')
    months=pd.DataFrame(np.ones((24,6)),index=index,columns=GROUPS)
    clock=pd.Series(pd.date_range('2023-05-01',periods=24,freq='h'))
    before=public_features(months,clock)
    assert before.other_renewable.eq(1/6).all()
    months.loc[months.index>pd.Period('2023-03')]=[1000.,2.,3.,4.,5.,6.]
    after=public_features(months,clock)
    pd.testing.assert_frame_equal(before[list(HISTORY)],after[list(HISTORY)])


def test_country_then_region_weighting_does_not_follow_country_size():
    frame=pd.DataFrame({'country':['US']*6+['GB']*2,'region':['A']*2+['B']*4+['GB']*2})
    w=weights(frame)
    assert np.isclose(w[:6].sum(),.5) and np.isclose(w[6:].sum(),.5)
    assert np.isclose(w[:2].sum(),.25) and np.isclose(w[2:6].sum(),.25)


def test_prediction_consumes_public_features_without_hourly_fuel_fields():
    months=pd.DataFrame(np.ones((24,6)),index=pd.period_range('2022-01',periods=24,freq='M'),columns=GROUPS)
    clock=pd.Series(pd.date_range('2023-05-01',periods=240,freq='h'))
    public=public_features(months,clock)
    training=public.assign(country='source',region='source')
    fuel=np.ones((240,9))
    coarse=np.tile([4.,1.,1.,1.,1.,1.],(240,1))
    model=fit(training,fuel,coarse)
    result=predict(model,public)
    assert set(result)==set(METHODS)
    for p in result.values():
        assert len(p)==240 and np.isfinite(p).all()
