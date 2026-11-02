import numpy as np
import pandas as pd
import pytest

from tests.test_resource_controls import reference_fixture
from transcif.research import hourly_transfer_baselines as baseline


def example():
    f,_,_,_,climate=reference_fixture()
    f.loc[:23,'clock']-=pd.DateOffset(months=2)
    for ix in baseline.resources.day_groups(f):
        first=f.clock.iloc[ix[0]]
        f.loc[ix,'forecast_origin_local']=str(first)
        f.loc[ix,'origin']=str(first)
        f.loc[ix,'latest_allowed_period']=str(first.to_period('M')-2)
    f['cal2']=np.sin(f.clock.dt.dayofweek*2*np.pi/7);f['cal3']=np.cos(f.clock.dt.dayofweek*2*np.pi/7)
    f['cal4']=np.sin(f.clock.dt.month*2*np.pi/12);f['cal5']=np.cos(f.clock.dt.month*2*np.pi/12)
    f['capacity_factor']=600.
    return f,climate


def test_public_design_budget_label_isolation_and_hourly_calendar():
    f,c=example();original=baseline.design(f,c,weather=False,capacity=False)
    f['target_cif']=1e20;f['label_generation_coal']=-1e8;f['capacity_factor']=900.
    f['wind_speed_ms']*=2.;f['shortwave_w_per_m2']*=3.
    np.testing.assert_array_equal(original,baseline.design(f,c,weather=False,capacity=False))
    without_forecast=f.drop(columns=['temperature_c','wind_speed_ms','shortwave_w_per_m2'])
    np.testing.assert_array_equal(original,baseline.design(without_forecast,c,weather=False,capacity=False))
    assert np.unique(original[:24,0]).size>10  # Hour harmonics are valid hourly inputs.
    assert baseline.design(f,c,weather=True,capacity=True).shape[1]>original.shape[1]
    a=baseline.design(f,c,weather=True,capacity=True)
    f['capacity_factor']=500.;b=baseline.design(f,c,weather=True,capacity=True)
    np.testing.assert_array_equal(a[:,:-1],b[:,:-1]);assert not np.array_equal(a[:,-1],b[:,-1])


def test_source_oof_never_trains_held_block_and_source_guards(monkeypatch):
    f,c=example();x=baseline.design(f,c,weather=True,capacity=False)
    monkeypatch.setattr(baseline,'OPTIONS',dict(max_iter=3,max_leaf_nodes=3,min_samples_leaf=2,
        learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=416))
    y=200.+np.arange(len(f))
    model,first=baseline.fit(f,y,x,allowed_countries=('S',))
    changed=y.copy();changed[:24]+=100.
    _,second=baseline.fit(f,changed,x,allowed_countries=('S',))
    np.testing.assert_array_equal(first[:24],second[:24])
    for fold in model['audit']['folds']:
        assert not set(fold['proper_months'])&set(fold['held_months'])
    with pytest.raises(ValueError,match='source-2022'):
        baseline.fit(f,y,x,allowed_countries=('T',))
    f['clock']+=pd.DateOffset(years=1)
    with pytest.raises(ValueError,match='source-2022'):
        baseline.fit(f,y,x,allowed_countries=('S',))


def test_signed_cqr_can_shrink_and_returns_ordered_nonempty_intervals():
    p=dict(point=np.array([4.,4.]),scale=np.array([2.,2.]),lower=np.array([0.,0.]),
           median=np.array([5.,5.]),upper=np.array([10.,10.]))
    q=baseline.calibrate(p,np.array([5.,5.]),np.array([.5,.5]))
    assert q['cqr_signed']==-5. and q['cqr_nonnegative']==0.
    result=baseline.intervals(p,q)
    np.testing.assert_array_equal(result['cqr_signed']['lower'],[5.,5.])
    np.testing.assert_array_equal(result['cqr_signed']['upper'],[5.,5.])
    assert np.all(result['cqr_nonnegative']['upper']-result['cqr_nonnegative']['lower']==10.)
    for prediction in result.values():
        assert np.all(prediction['lower']<=prediction['median']) and np.all(prediction['median']<=prediction['upper'])
    np.testing.assert_array_equal(result['global']['median'],p['point'])
    np.testing.assert_array_equal(result['cqr_signed']['median'],p['median'])
    asymmetric={**p,'lower':np.array([0.,0.]),'median':np.array([2.,2.]),'upper':np.array([100.,100.])}
    repaired=baseline.intervals(asymmetric,{**q,'cqr_signed':-10.})['cqr_signed']
    # Unrepaired [10,90] is nonempty but excludes q50=2; declared envelope adds it.
    np.testing.assert_array_equal(repaired['lower'],[2.,2.]);np.testing.assert_array_equal(repaired['upper'],[90.,90.])


def test_wis_uses_distribution_median_and_matches_three_quantile_pinball():
    y=np.array([0.,10.,30.]);q=np.array([[1.,5.,9.],[1.,5.,9.],[1.,5.,9.]])
    point=np.array([4.,4.,4.])
    scores=baseline.score_intervals(y,point,q[:,1],q[:,0],q[:,2])
    alpha=np.array([.05,.5,.95]);residual=y[:,None]-q
    pinball=np.maximum(alpha*residual,(alpha-1)*residual).mean()
    assert scores['wis90']==pytest.approx(2*pinball)
    assert scores['mae']!=scores['median_mae']


def test_permille_response_conversion_applies_to_point_scale_and_quantiles():
    class Constant:
        def __init__(self,value):self.value=value
        def predict(self,x):return np.full(len(x),self.value)
    fitted=dict(mean=Constant(300.),scale=Constant(20.),quantiles=[Constant(v) for v in [250.,300.,350.]])
    x=np.zeros((2,3));p=baseline.forecast(fitted,x,np.array([.4,.6]))
    for key,expected in [('point',[120.,180.]),('scale',[8.,12.]),('lower',[100.,150.]),
                         ('median',[120.,180.]),('upper',[140.,210.])]:
        np.testing.assert_allclose(p[key],expected)
    with pytest.raises(ValueError,match='conversion'):
        baseline.forecast(fitted,x,np.array([1.1,1.]))
