import pandas as pd
import pytest

from scripts import run_annual_context_v26 as r


@pytest.mark.parametrize('stage', ['prepare', 'predict', 'score'])
def test_paused_formal_stages_refuse_before_any_io(monkeypatch, stage):
    for name in ['expected_contract', 'check', 'predictions', 'read', 'write', 'sha']:
        monkeypatch.setattr(r, name, lambda *a, **k: pytest.fail('Paused draft reached formal I/O'))
    monkeypatch.setattr(r.prior.base, 'source_days', lambda *a: pytest.fail('Paused draft read source truth'))
    with pytest.raises(RuntimeError, match='DRAFT/PAUSED'):
        getattr(r, stage)()


@pytest.mark.parametrize('defect',['missing_region','flag','foreign_path','arm_count'])
def test_incomplete_seal_rejected_before_archives(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path)
    seal=dict(freeze_sha256='fixed',complete_grid=True,archives=7,raw_arms=5,
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
    ('code_hashes',{}),('monthly_prediction_paths',{}),('annual_path','/foreign'),
    ('china_annual_fields_verified',True),('conditional_annual_information_value',False)])
def test_contract_reconstructed_not_self_authorized(monkeypatch,key,value):
    expected=dict(source='/fixed',input_hashes={'a':'hash'},code_hashes={'b':'hash'},
        monthly_prediction_paths={'BPAT':'/fixed/pred'},annual_path='/fixed/annual',
        china_annual_fields_verified=False,conditional_annual_information_value=True)
    frozen=dict(created_utc='fixed',**expected)
    monkeypatch.setattr(r,'expected_contract',lambda:expected)
    monkeypatch.setattr(r,'read',lambda _:frozen)
    assert r.check()==frozen
    frozen[key]=value
    with pytest.raises(ValueError,match='contract changed'):r.check()


def test_selection_does_not_promote_failed_configuration():
    rows=[]
    for arm in ['MG-recompose_conditional','AY-fixed_recompose']:
        for region in r.monthly.SOURCES:
            row=dict(region=region,arm=arm,factor=1.,**{k:1. for k in r.METRICS})
            row['empirical_coverage90']=.8
            rows.append(row)
    macro,choices=r.prior.summarize(pd.DataFrame(rows))
    assert not macro.qualified.any()
    assert all(not x['qualified'] for x in choices['choices'].values())
    assert choices['new_holdout_authorized'] is False
