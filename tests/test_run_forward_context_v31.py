"""Synthetic forecast-input, label-support and execution-boundary regressions."""
import numpy as np
import pandas as pd
import pytest
from scripts import run_forward_context_v31 as r


def label(region='BPAT',date='2025-01-01',zero=False):
    start=pd.Timestamp(date,tz='UTC');end=start+pd.offsets.MonthBegin(1)
    mix={f:float(f==('hydro' if zero else 'gas')) for f in r.model.parent.FUELS}
    return dict(region=region,period_start=start.isoformat(),period_end=end.isoformat(),
        available_at=(end+pd.Timedelta(days=30)).isoformat(),coverage='1',total_mwh='10000',
        **{f'fuelshare_{f}':str(v) for f,v in mix.items()})


def test_forecast_reader_only_origins_and_mg_draws(monkeypatch):
    origins=pd.date_range('2025-01-01','2025-06-30',tz='UTC').as_unit('ns').asi8
    draws=np.full((181,128,5),.2)
    class Archive:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            if key=='origins':return origins
            if key=='MG_draws':return draws
            pytest.fail('Forbidden hourly or M8 field accessed: '+key)
    monkeypatch.setattr(r.np,'load',lambda *a,**kw:Archive())
    selected=r.forecast_inputs('/synthetic')
    assert selected.shape==(6,128,5) and np.array_equal(selected,draws[[0,31,59,90,120,151]])


def test_archive_grid_stays_nanoseconds_across_pandas_resolutions(tmp_path):
    expected=np.array([pd.Timestamp(f'2025-{m:02d}-01',tz='UTC').value for m in range(1,7)])
    np.testing.assert_array_equal(r.DATES.asi8,expected)
    dates=pd.date_range('2025-01-01','2025-06-30',tz='UTC')
    draws=np.full((181,128,5),.2)
    path=tmp_path/'forecast.npz'
    np.savez(path,origins=dates.as_unit('ns').asi8,MG_draws=draws)
    np.testing.assert_array_equal(r.forecast_inputs(str(path)),draws[[0,31,59,90,120,151]])
    np.savez(path,origins=dates.as_unit('us').asi8,MG_draws=draws)
    with pytest.raises(ValueError,match='archived MG daily forecast grid'):
        r.forecast_inputs(str(path))


def test_no_label_access_before_complete_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path);monkeypatch.setattr(r,'check',lambda:{})
    def missing():raise ValueError('Incomplete forecast seal')
    monkeypatch.setattr(r,'predictions',missing)
    monkeypatch.setattr(r.common,'monthly_rows',lambda _:pytest.fail('Current labels before seal'))
    with pytest.raises(ValueError,match='Incomplete forecast seal'):r.score()


def seal(tmp_path):
    return dict(freeze_sha256='fixed',complete_grid=True,forward_forecast=True,oracle_current_group=False,
        hourly_values_read=False,new_training_labels_used=False,target_values_opened=False,unseen_values_opened=False,
        archives=7,arms=6,origins_per_source=6,head_refits=42,
        entries=[dict(region=s,path=str(tmp_path/'predictions'/f'{s}.npz'),audit_path=str(tmp_path/'predictions'/f'{s}.json')) for s in r.model.SOURCES])


@pytest.mark.parametrize('key,value',[('oracle_current_group',True),('forward_forecast',False),('hourly_values_read',0),
    ('new_training_labels_used',True),('arms',10),('head_refits',70)])
def test_bad_seal_rejected_before_archives(tmp_path,monkeypatch,key,value):
    monkeypatch.setattr(r,'OUT',tmp_path);s=seal(tmp_path);s[key]=value
    monkeypatch.setattr(r,'read',lambda _:s);monkeypatch.setattr(r,'sha',lambda _:'fixed')
    with pytest.raises(ValueError,match='forecast-only seal'):r.predictions()


@pytest.mark.parametrize('defect',['extra_file','extra_directory','missing','symlink'])
def test_exact_prediction_inventory_precedes_labels(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path);folder=tmp_path/'predictions';folder.mkdir()
    for source in r.model.SOURCES:
        for ext in ['npz','json']:(folder/f'{source}.{ext}').touch()
    if defect=='extra_file':(folder/'orphan.npz').touch()
    elif defect=='extra_directory':(folder/'orphan').mkdir()
    elif defect=='missing':(folder/'BPAT.npz').unlink()
    else:
        target=tmp_path/'foreign';target.touch();(folder/'BPAT.npz').unlink();(folder/'BPAT.npz').symlink_to(target)
    monkeypatch.setattr(r,'read',lambda _:seal(tmp_path));monkeypatch.setattr(r,'sha',lambda _:'fixed')
    monkeypatch.setattr(r,'verify',lambda _:pytest.fail('Inventory should precede arrays'))
    monkeypatch.setattr(r,'check',lambda:{})
    monkeypatch.setattr(r.common,'monthly_rows',lambda _:pytest.fail('Inventory should precede labels'))
    with pytest.raises(ValueError,match='fourteen'):r.score()


@pytest.mark.parametrize('key,value',[('forecast_paths',{}),('head_source_paths',{}),('input_hashes',{}),
    ('code_hashes',{}),('oracle_current_group',True),('forward_forecast',False),('hourly_values_read',0),('score_cutoff','2026')])
def test_contract_not_self_authorized(monkeypatch,key,value):
    expected=dict(forecast_paths={'a':'b'},head_source_paths={'c':'d'},input_hashes={'e':'h'},code_hashes={'x':'h'},
        oracle_current_group=False,forward_forecast=True,hourly_values_read=False,score_cutoff='2025-09-01')
    frozen=dict(created_utc='fixed',**expected)
    monkeypatch.setattr(r,'read',lambda _:frozen);monkeypatch.setattr(r,'expected_contract',lambda:expected)
    assert r.check()==frozen
    frozen[key]=value
    with pytest.raises(ValueError,match='forecast-only contract'):r.check()


def test_support_quality_and_release_gates_precede_hidden_floats():
    zero=label(zero=True);late=label(date='2025-02-01');late['available_at']='2025-10-01T00:00:00Z'
    low=label(date='2025-03-01');low.update(coverage='.89',total_mwh='POISON')
    empty=label(date='2025-04-01');empty['total_mwh']='0'
    for row in [late,low,empty]:
        row.update({column:'HIDDEN_POISON' for column in r.model.parent.SHARES})
    late.update(coverage='POISON',total_mwh='POISON')
    labels,support=r.scoring_labels([zero,late,low,empty])
    assert len(support)==42 and len(labels)==1
    assert labels['BPAT/2025-01']['cif']==0 and not labels['BPAT/2025-01']['ef_defined']
    assert np.isnan(labels['BPAT/2025-01']['ef'])
    statuses=support.set_index('key').status
    assert statuses['BPAT/2025-02']=='not_released' and statuses['BPAT/2025-03']=='insufficient_coverage'
    assert statuses['BPAT/2025-04']=='nonpositive_total' and statuses['BPAT/2025-05']=='missing'


def test_signal_requires_both_baselines_and_never_authorizes_hourly():
    data=pd.DataFrame([dict(region=s,arm=a,**{k:1. for k in r.METRICS},cif_rmse=1.,ef_rmse=1.,scored_months=6,internal_EF_months=6)
        for a in r.model.ARMS for s in r.model.SOURCES])
    data.loc[data.arm.eq('Size-Ridge'),'cif_abs_error']=.8
    _,choice=r.summarize(data)
    assert choice['size_signal_models']==['Ridge'] and choice['substantial_forward_size_signal']
    assert choice['hourly_stage_authorized'] is choice['new_holdout_authorized'] is choice['mechanism_breakthrough'] is False
    data.loc[data.arm.eq('Trend-Ridge'),'cif_abs_error']=.7
    _,choice=r.summarize(data)
    assert choice['size_signal_models']==[] and not choice['substantial_forward_size_signal']


def test_zero_pos_primary_retained_secondary_undefined(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path);folder=tmp_path/'predictions';folder.mkdir()
    (tmp_path/'freeze.json').write_text('{}');(tmp_path/'prediction_seal.json').write_text('{}')
    entries=[];rows=[]
    for source in r.model.SOURCES:
        path=folder/f'{source}.npz';keys=np.array([f'{source}/{at:%Y-%m}' for at in r.DATES])
        draws=np.tile([.6,0,.4,0,0],(6,128,1))
        np.savez(path,keys=keys,MG_draws=draws,CIF_point=np.full((6,6),240.),EF_point=np.full((6,6),400.),projection_l1=np.zeros((6,6,128)))
        entries.append(dict(region=source,path=str(path)))
        rows += [label(source,at.strftime('%Y-%m-%d'),zero=source=='BPAT' and at.month==1) for at in r.DATES]
    monkeypatch.setattr(r,'check',lambda:{})
    monkeypatch.setattr(r,'predictions',lambda:dict(entries=entries))
    monkeypatch.setattr(r.common,'monthly_rows',lambda _:rows)
    r.score()
    case=pd.read_csv(tmp_path/'case_metrics.csv');macro=pd.read_csv(tmp_path/'macro_metrics.csv')
    zero=case[case.key.eq('BPAT/2025-01')]
    assert len(case)==252 and len(zero)==6 and zero.cif_abs_error.eq(240).all()
    assert not zero.ef_defined.any() and zero[list(r.SECONDARY)].isna().all().all()
    assert macro.scored_region_months.eq(42).all() and macro.internal_EF_region_months.eq(41).all()
    assert len(pd.read_csv(tmp_path/'label_support.csv'))==42


def test_grid_and_preserve_attempt(tmp_path,monkeypatch):
    assert len(r.matched_pairs())==len(set(r.matched_pairs()))==9
    assert ('Size-Ridge','Trend-Ridge') in r.matched_pairs()
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'expected_contract',lambda:pytest.fail('Existing attempt must remain intact'))
    with pytest.raises(FileExistsError):r.prepare()
