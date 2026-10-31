import pandas as pd
import pytest

from scripts import run_group_regression_v27 as r


def test_no_source_truth_before_complete_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'check',lambda:dict(source=str(tmp_path)))
    def missing():
        raise ValueError('no complete seal')
    monkeypatch.setattr(r,'predictions',missing)
    monkeypatch.setattr(r.prior.base,'source_days',lambda *a:pytest.fail('Truth before seal'))
    with pytest.raises(ValueError,match='no complete seal'):r.score()


@pytest.mark.parametrize('defect',['missing_region','flag','foreign_path','arm_count'])
def test_incomplete_seal_rejected_before_archives(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path)
    seal=dict(freeze_sha256='fixed',complete_grid=True,archives=7,raw_arms=7,
        prediction_reads_new_fine=False,target_values_opened=False,unseen_values_opened=False,
        entries=[dict(region=s,path=str(tmp_path/'predictions'/f'{s}.npz'),
                      audit_path=str(tmp_path/'predictions'/f'{s}.json')) for s in r.monthly.SOURCES])
    if defect=='missing_region':seal['entries'].pop()
    elif defect=='flag':seal['target_values_opened']=True
    elif defect=='arm_count':seal['raw_arms']=4
    else:seal['entries'][0]['path']=str(tmp_path/'other.npz')
    monkeypatch.setattr(r,'read',lambda _:seal)
    monkeypatch.setattr(r,'sha',lambda _:'fixed')
    monkeypatch.setattr(r,'verify',lambda _:pytest.fail('Invalid seal reached archive hash'))
    with pytest.raises(ValueError):r.predictions()


@pytest.mark.parametrize('key,value',[('source','/foreign'),('input_hashes',{}),
    ('code_hashes',{}),('monthly_prediction_paths',{}),('samples',1),
    ('china_category_mapping_verified',True),('annual_values_consumed',True)])
def test_contract_reconstructed_not_self_authorized(monkeypatch,key,value):
    expected=dict(source='/fixed',input_hashes={'a':'hash'},code_hashes={'b':'hash'},
        monthly_prediction_paths={'BPAT':'/fixed/pred'},samples=128,
        china_category_mapping_verified=False,annual_values_consumed=False)
    frozen=dict(created_utc='fixed',**expected)
    monkeypatch.setattr(r,'expected_contract',lambda:expected)
    monkeypatch.setattr(r,'read',lambda _:frozen)
    assert r.check()==frozen
    frozen[key]=value
    with pytest.raises(ValueError,match='contract changed'):r.check()


def test_selection_does_not_promote_failed_configuration():
    rows=[]
    for arm in ['MG-recompose_conditional','Ridge-residual_recompose']:
        for region in r.monthly.SOURCES:
            row=dict(region=region,arm=arm,factor=1.,**{k:1. for k in r.METRICS})
            row['empirical_coverage90']=.8
            rows.append(row)
    macro,choices=r.prior.summarize(pd.DataFrame(rows))
    assert not macro.qualified.any()
    assert all(not x['qualified'] for x in choices['choices'].values())
    assert choices['new_holdout_authorized'] is False
