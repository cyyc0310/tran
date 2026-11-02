import numpy as np
import pandas as pd

from tests.test_resource_controls import reference_fixture
from transcif.research import resource_controls_daily as daily
from transcif.research import resource_controls as controls
from transcif.research import coarse_trajectories as ct


def test_daily_models_ignore_roundoff_in_analytically_zero_harmonics():
    template,coarse,fuel,past,climate=reference_fixture()
    frames=[];labels=[]
    for day in range(12):
        f=template.iloc[:24].copy();f['clock']+=pd.Timedelta(days=day)
        f['forecast_origin_local']=str(f.clock.iloc[0]);f['origin']=str(f.clock.iloc[0])
        f['cal2']=np.sin(f.clock.dt.dayofweek*2*np.pi/7)
        f['cal3']=np.cos(f.clock.dt.dayofweek*2*np.pi/7)
        f['temperature_c']+=day;f['wind_speed_ms']+=day/10.
        c=coarse[:24].copy();c[:,3]=1.+day/2.;c[:,4]=1.+day/4.
        frames.append(f);labels.append(c)
    f=pd.concat(frames,ignore_index=True);c=np.vstack(labels)
    power=np.repeat(past[:1],12,axis=0);fuel=np.tile(fuel[:24],(12,1))
    saved=f.copy(deep=True);noisy=f.copy()
    rng=np.random.default_rng(416)
    noisy.loc[:,list(daily.ZERO_DAILY)]+=rng.normal(0.,1e-13,(len(f),len(daily.ZERO_DAILY)))
    before=daily.fit(f,c,power,climate,allowed_countries=('S',))
    after=daily.fit(noisy,c,power,climate,allowed_countries=('S',))
    pd.testing.assert_frame_equal(daily.daily_public(f),daily.daily_public(noisy))
    pd.testing.assert_frame_equal(f,saved)
    # This is a nonconstant regression problem with enough days for HGB splits.
    assert any(np.count_nonzero(tree[0].nodes['is_leaf'])>1 for tree in before['models']['hgb_W'][0]._predictors)
    query=f.copy();query['clock']+=pd.DateOffset(years=1)
    design,_,_=daily.day_inputs(query,climate)
    for learner in ('ridge','rf','hgb'):
        for kind in ('B','W'):
            np.testing.assert_array_equal(controls.predict_amplitudes(before,design,learner,kind),
                                          controls.predict_amplitudes(after,design,learner,kind))
    bank=ct.fit(f,fuel);actual,_=daily.predict_methods(bank,before,query,climate)
    expected,_=ct.predict(bank,query)
    np.testing.assert_array_equal(actual['joint_original'],expected['joint'])
    # Caller features remain intact for any separate hourly prediction model.
    assert not np.array_equal(f[list(daily.ZERO_DAILY)],np.zeros((len(f),len(daily.ZERO_DAILY))))
