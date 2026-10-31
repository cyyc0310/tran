import numpy as np
import pandas as pd
from transcif.research.structural_v3 import zero_mean, delayed


def test_shape_projection_is_zero_mean():
    x=np.random.default_rng(1).normal(size=(19,24))
    assert np.allclose(zero_mean(x).mean(axis=1),0,atol=1e-14)
    assert np.allclose((zero_mean(x)*np.exp(-1e9/90)).mean(axis=1),0)


def test_delay_never_mutates_or_advances_release():
    frame=pd.DataFrame({'available_at':['2022-01-31T00:00:00Z']})
    out=delayed(frame,30)
    assert frame.available_at.iloc[0]=='2022-01-31T00:00:00Z'
    assert out.available_at.iloc[0]==pd.Timestamp('2022-03-02T00:00:00Z')


def model_fixture():
    from transcif.research.structural_v3 import StructuralModel
    def fit(outputs):
        return np.zeros(14),np.ones(14),np.zeros((15,outputs))
    return StructuralModel(np.tile(np.arange(24),(12,1)),np.tile(np.ones(8)/8,(12,1)),
        fit(1),fit(24),np.zeros((2,14)),np.zeros((2,24)),np.zeros(2),('source',),30)


def history_fixture():
    from transcif.research.eia_proxy import FUELS
    row=dict(region='target',period_start='2022-01-01T00:00:00Z',period_end='2022-02-01T00:00:00Z',
             available_at='2022-03-01T00:00:00Z',coverage=1,total_mwh=100)
    row.update({f'fuelshare_{f}':float(i==0) for i,f in enumerate(FUELS)})
    return pd.DataFrame([row])


def test_future_monthly_prefix_cannot_change_forecast():
    model=model_fixture(); history=history_fixture()
    origin=pd.DatetimeIndex(['2022-04-01T00:00:00Z'])
    before=model.predict(origin,'target',history)
    future=history.copy(); future['available_at']='2023-01-01T00:00:00Z'
    future['fuelshare_col']=float('nan')
    from transcif.research.eia_proxy import FUELS
    for fuel in FUELS:
        future[f'fuelshare_{fuel}']=float('nan')
    future['period_start']='2022-05-01T00:00:00Z';future['period_end']='2022-06-01T00:00:00Z'
    after=model.predict(origin,'target',pd.concat([history,future]))
    for name in before:
        assert np.array_equal(before[name],after[name])


def test_level_persists_at_extreme_age():
    model=model_fixture()
    predictions=model.predict(pd.DatetimeIndex(['2050-04-01T00:00:00Z']),'target',history_fixture())
    assert np.isclose(predictions['ridge_shape'].mean(),1000)
    assert model.source_regions==('source',)


def test_policy_inner_training_excludes_held_source(monkeypatch):
    import transcif.research.structural_v3 as module
    calls=[]
    days=pd.date_range('2022-07-01',periods=2,tz='UTC')
    monkeypatch.setattr(module,'daily_profiles',lambda frame:(days,np.ones((2,24))))
    class Dummy:
        def __init__(self,regions):self.source_regions=tuple(regions)
        def predict(self,origins,region,monthly):
            assert region not in self.source_regions
            return {name:np.zeros((len(origins),24)) for name in module.POINTS}
    def fit(frames,monthly,cutoff,delay):
        calls.append((set(frames),cutoff))
        return Dummy(frames)
    monkeypatch.setattr(module,'fit_model',fit)
    winner,banks,policy=module.source_policy({'a':None,'b':None,'c':None},pd.DataFrame())
    assert len(calls)==3
    assert all(len(regions)==2 and cutoff.startswith('2022-07-01') for regions,cutoff in calls)
    assert policy['source_regions']==['a','b','c']
    assert winner in module.POINTS


def test_runner_seals_before_validation_label_read():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'scripts/run_structural_search_v3.py').read_text()
    assert source.index("folder/'prediction_seal.json'") < source.index("daily_profiles(read_year(args.data/f'{target}_hourly.csv',args.validation_year))")
