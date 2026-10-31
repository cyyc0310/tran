import csv

import pandas as pd
import pytest

from scripts import run_monthly_transfer_v25 as r


def test_monthly_future_numeric_poison_is_never_parsed(tmp_path):
    meta=['region','period_start','period_end','available_at']
    path=tmp_path/'monthly.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=meta+list(r.model.M8_COLUMNS));writer.writeheader()
        for year,value in [(2024,'0.125'),(2026,'FUTURE_POISON')]:
            writer.writerow(dict(region='BPAT',period_start=f'{year}-01-01T00:00:00Z',
                period_end=f'{year}-02-01T00:00:00Z',available_at=f'{year}-03-02T00:00:00Z',
                **{k:value for k in r.model.M8_COLUMNS}))
    frame=r.monthly_prefix(path)
    assert len(frame)==1
    assert frame.period_start.iloc[0].year==2024
    assert frame[list(r.model.M8_COLUMNS)].iloc[0].sum()==1


def test_no_score_truth_before_full_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'check',lambda:dict(source=str(tmp_path)))
    def fail_seal():
        raise ValueError('missing complete seal')
    monkeypatch.setattr(r,'predictions',fail_seal)
    monkeypatch.setattr(r.base,'source_days',lambda *a:pytest.fail('Source labels parsed before seal'))
    with pytest.raises(ValueError,match='missing complete seal'):
        r.score()


@pytest.mark.parametrize('defect',['missing_region','false_flag','foreign_path'])
def test_incomplete_or_foreign_prediction_seal_fails(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path)
    seal=dict(freeze_sha256='fixed',complete_grid=True,archives=7,raw_arms=9,
        prediction_reads_new_fine=False,target_values_opened=False,unseen_values_opened=False,
        entries=[dict(region=s,path=str(tmp_path/'predictions'/f'{s}.npz'),
                      audit_path=str(tmp_path/'predictions'/f'{s}.json')) for s in r.model.SOURCES])
    if defect=='missing_region':seal['entries'].pop()
    elif defect=='false_flag':seal['target_values_opened']=True
    else:seal['entries'][0]['path']=str(tmp_path/'elsewhere.npz')
    monkeypatch.setattr(r,'read',lambda _:seal)
    monkeypatch.setattr(r,'sha',lambda _:'fixed')
    monkeypatch.setattr(r,'verify',lambda _:pytest.fail('Invalid seal reached file hashes'))
    with pytest.raises(ValueError):r.predictions()


def test_failed_selection_stays_failed():
    rows=[]
    for arm in r.model.ARMS:
        for factor in r.FACTORS:
            for region in r.model.SOURCES:
                row=dict(region=region,arm=arm,factor=factor,**{k:1. for k in r.METRICS})
                row['empirical_coverage90']=.8
                rows.append(row)
    macro,choices=r.summarize(pd.DataFrame(rows))
    assert not macro.qualified.any()
    assert all(not x['qualified'] for x in choices['choices'].values())
    assert choices['new_holdout_authorized'] is False


@pytest.mark.parametrize('field,value',[('source','/unregistered/source'),('code_hashes',{}),
    ('input_hashes',{}),('query_paths',{}),('banks_by_outer',{}),('samples',1),('seed',0),
    ('target_values_opened',True)])
def test_frozen_contract_must_equal_fixed_parent(monkeypatch,field,value):
    expected=dict(source='/fixed/source',code_hashes={'code':'hash'},input_hashes={'input':'hash'},
        query_paths={'BPAT':'fixed'},banks_by_outer={'BPAT':['a','b']},samples=128,seed=17023,
        target_values_opened=False)
    frozen=dict(created_utc='2026-10-26T00:00:00Z',**expected)
    monkeypatch.setattr(r,'expected_contract',lambda:expected)
    monkeypatch.setattr(r,'read',lambda _:frozen)
    assert r.check()==frozen
    frozen[field]=value
    with pytest.raises(ValueError,match='contract changed'):r.check()
