"""Synthetic execution-contract regressions; no formal inputs are read."""
import pandas as pd
import pytest
from scripts import run_conditional_law_v29 as r


def test_truth_reader_not_reached_without_complete_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'check',lambda:dict(source=str(tmp_path)))
    def missing():raise ValueError('no complete seal')
    monkeypatch.setattr(r,'predictions',missing)
    monkeypatch.setattr(r.prior.base,'source_days',lambda *a:pytest.fail('Truth before seal'))
    with pytest.raises(ValueError,match='no complete seal'):r.score()


@pytest.mark.parametrize('defect',['missing_region','flag','foreign_path','arm_count','false_bool'])
def test_incomplete_seal_rejected_before_archive_reads(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path)
    seal=dict(freeze_sha256='fixed',complete_grid=True,archives=7,raw_arms=10,
        prediction_reads_new_fine=False,target_values_opened=False,unseen_values_opened=False,
        entries=[dict(region=s,path=str(tmp_path/'predictions'/f'{s}.npz'),
                      audit_path=str(tmp_path/'predictions'/f'{s}.json')) for s in r.monthly.SOURCES])
    if defect=='missing_region':seal['entries'].pop()
    elif defect=='flag':seal['target_values_opened']=True
    elif defect=='arm_count':seal['raw_arms']=21
    elif defect=='false_bool':seal['prediction_reads_new_fine']=0
    else:seal['entries'][0]['path']=str(tmp_path/'other.npz')
    monkeypatch.setattr(r,'read',lambda _:seal)
    monkeypatch.setattr(r,'sha',lambda _:'fixed')
    monkeypatch.setattr(r,'verify',lambda _:pytest.fail('Invalid seal reached archive hash'))
    with pytest.raises(ValueError):r.predictions()


@pytest.mark.parametrize('key,value',[('source','/foreign'),('input_hashes',{}),('code_hashes',{}),
    ('monthly_prediction_paths',{}),('background_prediction_paths',{}),('monthly_path','/foreign'),
    ('fixed_deployment_cutoff','2026-01-01'),('samples',1),('supervision_window','LR24'),
    ('scale_floor_squared',.1),('forest_models_expected',7),('china_category_mapping_verified',True),
    ('annual_values_consumed',0)])
def test_contract_not_self_authorized(monkeypatch,key,value):
    expected=dict(source='/fixed',input_hashes={'a':'hash'},code_hashes={'b':'hash'},
        monthly_prediction_paths={'BPAT':'/fixed/pred'},background_prediction_paths={'BPAT':'/fixed/old'},
        monthly_path='/fixed/monthly',fixed_deployment_cutoff='2025-01-01T00:00:00+00:00',samples=128,
        supervision_window='L24',scale_floor_squared=1e-12,forest_models_expected=49,
        china_category_mapping_verified=False,annual_values_consumed=False)
    frozen=dict(created_utc='fixed',**expected)
    monkeypatch.setattr(r,'expected_contract',lambda:expected)
    monkeypatch.setattr(r,'read',lambda _:frozen)
    assert r.check()==frozen
    frozen[key]=value
    with pytest.raises(ValueError,match='contract changed'):r.check()


def test_failed_selection_never_authorizes_holdout():
    rows=[]
    for arm in ['Leaf-recompose','Scale-anchor']:
        for region in r.monthly.SOURCES:
            row=dict(region=region,arm=arm,factor=1.,**{k:1. for k in r.METRICS})
            row['empirical_coverage90']=.8;rows.append(row)
    macro,choice=r.prior.summarize(pd.DataFrame(rows))
    assert not macro.qualified.any()
    assert all(not x['qualified'] for x in choice['choices'].values())
    assert choice['new_holdout_authorized'] is False


def test_fixed_comparisons_include_simple_anchor_and_paired_shuffle():
    pairs=r.matched_pairs()
    assert len(pairs)==len(set(pairs))==16
    assert all(a in r.model.ARMS and b in r.model.ARMS for a,b in pairs)
    for family in ('Leaf','Gaussian','Scale'):
        assert (family+'-anchor','L24-ET-residual_anchor') in pairs
    assert ('Scale-shuffle_recompose','Scale-recompose') in pairs


def test_formal_prepare_preserves_existing_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'OUT',tmp_path)
    monkeypatch.setattr(r,'expected_contract',lambda:pytest.fail('Existing attempt must be rejected first'))
    with pytest.raises(FileExistsError):r.prepare()


@pytest.mark.parametrize('defect',['extra_file','extra_directory','missing_file','symlink'])
def test_prediction_inventory_closed_before_truth(tmp_path,monkeypatch,defect):
    monkeypatch.setattr(r,'OUT',tmp_path)
    folder=tmp_path/'predictions';folder.mkdir()
    entries=[]
    for source in r.monthly.SOURCES:
        paths=[folder/f'{source}.{suffix}' for suffix in ('npz','json')]
        for path in paths:path.touch()
        entries.append(dict(region=source,path=str(paths[0]),audit_path=str(paths[1])))
    if defect=='extra_file':(folder/'orphan.npz').touch()
    elif defect=='extra_directory':(folder/'orphan').mkdir()
    elif defect=='missing_file':(folder/'BPAT.npz').unlink()
    else:
        target=tmp_path/'foreign';target.touch()
        (folder/'BPAT.npz').unlink();(folder/'BPAT.npz').symlink_to(target)
    seal=dict(freeze_sha256='fixed',complete_grid=True,archives=7,raw_arms=10,
        prediction_reads_new_fine=False,target_values_opened=False,unseen_values_opened=False,entries=entries)
    monkeypatch.setattr(r,'read',lambda _:seal)
    monkeypatch.setattr(r,'sha',lambda _:'fixed')
    monkeypatch.setattr(r,'check',lambda:dict(source=str(tmp_path)))
    monkeypatch.setattr(r,'verify',lambda _:pytest.fail('Incomplete inventory reached arrays'))
    monkeypatch.setattr(r.prior.base,'source_days',lambda *a:pytest.fail('Truth before inventory'))
    with pytest.raises(ValueError,match='fourteen'):r.score()
