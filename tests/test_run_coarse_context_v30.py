"""Synthetic contracts for coarse-only source diagnostic stages."""
import numpy as np
import pandas as pd
import pytest
from scripts import run_coarse_context_v30 as r


def test_score_labels_not_reached_without_complete_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path);monkeypatch.setattr(r,'check',lambda:{})
    def missing():raise ValueError('Incomplete seal')
    monkeypatch.setattr(r,'predictions',missing)
    monkeypatch.setattr(r,'monthly_rows',lambda *_:pytest.fail('Rows before complete seal'))
    monkeypatch.setattr(r.model,'response',lambda *_:pytest.fail('Label before complete seal'))
    with pytest.raises(ValueError,match='Incomplete seal'):r.score()


def valid_seal(tmp_path):
    return dict(freeze_sha256='fixed',complete_grid=True,archives=7,arms=10,fit_count=70,eligible_cases=7,
        oracle_current_group=True,forward_forecast=False,hourly_values_read=False,
        target_values_opened=False,unseen_values_opened=False,
        entries=[dict(region=s,path=str(tmp_path/'predictions'/f'{s}.npz'),
            audit_path=str(tmp_path/'predictions'/f'{s}.json'),rows=1,query_keys=[s+'/2024-01']) for s in r.model.SOURCES])


@pytest.mark.parametrize('key,value',[('oracle_current_group',False),('forward_forecast',True),
    ('hourly_values_read',True),('unseen_values_opened',0),('fit_count',10),('arms',11)])
def test_invalid_seal_rejected_before_files(tmp_path,monkeypatch,key,value):
    monkeypatch.setattr(r,'OUT',tmp_path);seal=valid_seal(tmp_path);seal[key]=value
    monkeypatch.setattr(r,'read',lambda _:seal);monkeypatch.setattr(r,'sha',lambda _:'fixed')
    with pytest.raises(ValueError,match='conditional seven-source'):r.predictions()


@pytest.mark.parametrize('defect',['extra_file','extra_directory','missing_file','symlink'])
def test_inventory_closed_before_label_read(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path);folder=tmp_path/'predictions';folder.mkdir()
    for s in r.model.SOURCES:
        for ext in ['npz','json']:(folder/f'{s}.{ext}').touch()
    if defect=='extra_file':(folder/'extra.npz').touch()
    elif defect=='extra_directory':(folder/'extra').mkdir()
    elif defect=='missing_file':(folder/'BPAT.npz').unlink()
    else:
        target=tmp_path/'foreign';target.touch()
        (folder/'BPAT.npz').unlink();(folder/'BPAT.npz').symlink_to(target)
    monkeypatch.setattr(r,'read',lambda _:valid_seal(tmp_path));monkeypatch.setattr(r,'sha',lambda _:'fixed')
    monkeypatch.setattr(r,'check',lambda:{})
    monkeypatch.setattr(r,'verify',lambda _:pytest.fail('Bad inventory reached arrays'))
    monkeypatch.setattr(r,'monthly_rows',lambda _:pytest.fail('Bad inventory reached labels'))
    with pytest.raises(ValueError,match='fourteen'):r.score()


@pytest.mark.parametrize('key,value',[('monthly_path','/foreign'),('input_hashes',{}),('code_hashes',{}),
    ('history_rows',1),('fit_count',10),('oracle_current_group',False),('forward_forecast',True),('hourly_values_read',0)])
def test_contract_cannot_self_authorize(monkeypatch,key,value):
    expected=dict(monthly_path='/fixed',input_hashes={'a':'hash'},code_hashes={'b':'hash'},history_rows=3,
        fit_count=70,oracle_current_group=True,forward_forecast=False,hourly_values_read=False)
    frozen=dict(created_utc='fixed',**expected)
    monkeypatch.setattr(r,'read',lambda _:frozen);monkeypatch.setattr(r,'expected_contract',lambda:expected)
    assert r.check()==frozen
    frozen[key]=value
    with pytest.raises(ValueError,match='contract changed'):r.check()


def regional_fixture():
    return pd.DataFrame([dict(region=source,arm=arm,**{m:1. for m in r.METRICS},ef_rmse=float(i+1))
        for arm in r.model.ARMS for i,source in enumerate(r.model.SOURCES)])


def test_signal_is_balanced_coarse_diagnostic_never_hourly_authorization():
    data=regional_fixture();data.loc[data.arm.eq('History-ET'),'ef_abs_error']=.8
    macro,choice=r.summarize(data)
    item=next(s for s in choice['comparisons'] if s['candidate']=='History-ET' and s['reference']=='Season-ET')
    assert item['signal'] and item['improved_regions']==7 and choice['substantial_context_signal']
    assert choice['hourly_stage_authorized'] is choice['new_holdout_authorized'] is choice['mechanism_breakthrough'] is False
    assert choice['oracle_current_group'] is True and choice['forward_forecast'] is False
    assert np.all(macro.ef_rmse.eq(4.))


def test_miso_erco_tradeoff_rejects_context_signal():
    data=regional_fixture();data.loc[data.arm.eq('History-ET'),'ef_abs_error']=.8
    data.loc[data.arm.eq('History-ET')&data.region.eq('ERCO'),'ef_abs_error']=1.01
    _,choice=r.summarize(data)
    item=next(s for s in choice['comparisons'] if s['candidate']=='History-ET' and s['reference']=='Season-ET')
    assert item['improved_regions']==6 and not item['signal']


def test_fixed_comparisons_and_preserve_attempt(tmp_path,monkeypatch):
    pairs=r.matched_pairs();assert len(pairs)==len(set(pairs))==19
    assert all(a in r.model.ARMS and b in r.model.ARMS for a,b in pairs)
    assert ('Size-ET','Trend-ET') in pairs and ('History-Ridge','G5-Ridge') in pairs
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'expected_contract',lambda:pytest.fail('Must preserve existing attempt first'))
    with pytest.raises(FileExistsError):r.prepare()
