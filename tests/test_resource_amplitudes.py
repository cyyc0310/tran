import numpy as np
import pandas as pd
import pytest

from transcif.research import coarse_trajectories as ct
from transcif.research import resource_amplitudes as amp
from transcif.research.multicountry import GROUPS, public_features


def fixture():
    frames=[]
    for m in (5,6,8):
        clock=pd.Series(pd.date_range(f'2022-{m:02d}-15',periods=24,freq='h'))
        monthly=pd.DataFrame(np.ones((12,6)),columns=GROUPS,index=pd.period_range('2021-10',periods=12,freq='M'))
        f=public_features(monthly,clock)
        f['country'],f['region'],f['split']='S','R','train'
        f['origin']=str(clock.iloc[0]);f['forecast_origin_local']=str(clock.iloc[0])
        f['latest_allowed_period']=str(clock.iloc[0].to_period('M')-2)
        f['temperature_c']=20.;f['wind_speed_ms']=np.linspace(1,10,24)
        f['shortwave_w_per_m2']=np.maximum(0,np.sin(np.arange(24)*np.pi/12))*100
        frames.append(f)
    f=pd.concat(frames,ignore_index=True)
    coarse=np.tile(np.array([4.,1.,1.,1.,1.,1.]),(len(f),1))
    fuel=np.ones((len(f),9));past=np.ones((3,6))*10
    climate=pd.DataFrame([dict(region='R',month=m,temperature_c=10.+m,
        wind_resource=100.*m,solar_resource=10.*m,reference_years=5,
        eligible_for_issued_public=False) for m in range(1,13)])
    return f,coarse,fuel,past,climate


def test_scale_identity_support_and_only_wind_solar_change():
    q=np.arange(2*24*6,dtype=np.float32).reshape(2,24,6)/100
    support=np.ones((2,6),bool);support[0,3]=False;q[0,:,3]=1.
    same=amp.scale_paths(q,support,np.ones(2))
    assert same.dtype==q.dtype and same.tobytes()==q.tobytes()
    result=amp.scale_paths(q,support,np.array([2.,.5]))
    np.testing.assert_array_equal(result[0,:,3],q[0,:,3])
    np.testing.assert_allclose(result[1,:,3],q[1,:,3]*2)
    np.testing.assert_array_equal(result[:,:,[0,1,2,5]],q[:,:,[0,1,2,5]])
    with pytest.raises(ValueError,match='valid'):
        amp.scale_paths(q,support,np.array([np.nan,1.]))


def test_seasonal_and_weather_ratio_are_distinct_and_future_month_forbidden():
    f,_,_,_,climate=fixture();d,season,weather=amp.day_inputs(f,climate)
    assert season[0,0]==pytest.approx(5/3)
    expected=(np.minimum(f.wind_speed_ms.iloc[:24],25)**3).mean()/300
    assert weather[0,0]==pytest.approx(expected)
    assert d.reference_period.iloc[0]=='2022-03'
    f.loc[:23,'latest_allowed_period']='2022-04'
    with pytest.raises(ValueError,match='two months'):
        amp.day_inputs(f,climate)


def test_source_fit_guards_and_target_label_columns_ignored():
    f,c,_,past,climate=fixture();model=amp.fit(f,c,past,climate,allowed_countries=('S',))
    d,_,_=amp.day_inputs(f,climate);before=amp.head_amplitudes(model,d,'W')
    f['label_proxy_cif']=-99999.;f['target_cif']=1e10
    changed,_,_=amp.day_inputs(f,climate)
    np.testing.assert_array_equal(before,amp.head_amplitudes(model,changed,'W'))
    with pytest.raises(ValueError,match='source-2022'):
        amp.fit(f,c,past,climate,allowed_countries=('T',))
    f['clock']+=pd.DateOffset(years=1)
    with pytest.raises(ValueError,match='source-2022'):
        amp.fit(f,c,past,climate,allowed_countries=('S',))


def test_past_power_guard_rejects_different_composition():
    f,c,_,past,climate=fixture();past[0,0]*=2
    with pytest.raises(ValueError,match='visible monthly'):
        amp.fit(f,c,past,climate,allowed_countries=('S',))


def test_prediction_baseline_identity_reordering_and_zero_climate_fallback():
    f,c,fuel,past,climate=fixture();bank=ct.fit(f,fuel)
    fitted=amp.fit(f,c,past,climate,allowed_countries=('S',))
    query=f.copy();query['clock']+=pd.DateOffset(years=1)
    query['forecast_origin_local']=query.clock.astype(str)
    for ix in amp.day_groups(query):
        query.loc[ix,'forecast_origin_local']=str(query.clock.iloc[ix[0]])
    values,_=amp.predict_methods(bank,fitted,query,climate)
    original,_=ct.predict(bank,query)
    np.testing.assert_array_equal(values['joint_original'],original['joint'])
    reversed_values,_=amp.predict_methods(bank,fitted,query.iloc[::-1],climate)
    for name in values:np.testing.assert_array_equal(values[name],reversed_values[name][::-1])
    zero=climate.copy();zero[['wind_resource','solar_resource']]=0.
    fallback,_=amp.predict_methods(bank,fitted,query,zero)
    for name in fallback:np.testing.assert_array_equal(fallback[name],original['joint'])
    missing=climate.loc[climate.month.ne(3)]
    absent,_=amp.predict_methods(bank,fitted,query.iloc[:24],missing)
    for name in absent:np.testing.assert_array_equal(absent[name],original['joint'][:24])


def test_runner_past_month_power_uses_absolute_energy_hours_and_origin_cutoff(monkeypatch):
    import importlib
    from pathlib import Path
    from transcif.research.multicountry import FINE
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    past_power=importlib.import_module('run_resource_amplitudes').past_power
    f,_,_,_,_=fixture();train=f.iloc[:24].copy()
    raw=pd.DataFrame(np.full((31*24,len(FINE)),2.),columns=FINE)
    raw['clock']=pd.date_range('2022-03-01',periods=len(raw),freq='h')
    raw['expected_hours']=31*24
    values,audit=past_power(train,{'R':raw})
    np.testing.assert_array_equal(values,np.array([[8.,2.,2.,2.,2.,2.]]))
    np.testing.assert_array_equal(audit[0]['energy'],values[0]*744)
    assert audit[0]['reference_period']=='2022-03' and audit[0]['expected_month_hours']==744
    train['latest_allowed_period']='2022-04'
    with pytest.raises(ValueError,match='cutoff'):
        past_power(train,{'R':raw})
