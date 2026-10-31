"""Boundary and mechanism checks for the V11 source-only runner."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts import run_weather_recomposition_v11 as r


def test_allowed_sources_rejects_unseen_and_excludes_outer():
    assert 'CISO' not in r.allowed_sources('CISO')
    assert set(r.allowed_sources()) == set(r.SOURCES)
    with pytest.raises(ValueError, match='registered source'):
        r.allowed_sources('AZPS')


def test_grid_has_weather_matched_arms_and_excludes_diagnostics_from_primary():
    grid = r.configurations()
    assert len(grid) == 12
    assert {c['model'] for c in grid if c['primary_eligible']} == {'donor','global_total','hetero_total'}
    for mode in ('donor','donor_no_shape_risk','global_total','hetero_total'):
        assert {c['budget'] for c in grid if c['model'] == mode} == {'B','W'}


def test_scoring_never_reads_truth_without_prediction_seal(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('evaluation labels opened')
    monkeypatch.setattr(r.v10, '_truth_lookup', forbidden)
    with pytest.raises(ValueError, match='seal required'):
        r.score_phase(tmp_path, {}, 'target', {})


def test_bank_rejects_target_source_and_wrong_cohort():
    with pytest.raises(ValueError, match='permitted source'):
        r._bank({}, ['AZPS','CISO'])


def _selection_table():
    rows=[]
    for region in r.SOURCES:
        for c in r.configurations():
            for factor in r.FACTORS:
                metrics = {m: 1. for m in r.METRICS}
                metrics['empirical_coverage90'] = .925
                metrics['crps'] = 20 + abs(factor-1)
                if c['model'].endswith('Brisk') or c['model']=='donor_no_shape_risk':
                    metrics['crps'] = 0
                rows.append(dict(stage='source', region=region, semester='H1',lag=7,
                                 budget=c['budget'], model=c['model'],factor=factor,**metrics))
    return pd.DataFrame(rows)


def test_selection_canonical_csv_and_diagnostic_exclusion(tmp_path):
    table = _selection_table()
    path=tmp_path/'table.csv'; table.to_csv(path,index=False)
    choice=r.selection(pd.read_csv(path))
    assert choice['primary']['model'] in {'donor','global_total','hetero_total'}
    assert choice['baseline']['model'] in {'donor','global_total','hetero_total'}
    assert choice['primary']['factor'] == 1
    assert 'W--global_Brisk' not in choice['selections']
    assert choice==r.selection(pd.read_csv(path))
    with pytest.raises(ValueError, match='Incomplete'):
        r.selection(table.iloc[:-1])


def test_selection_coverage_gate_beats_crps():
    table=_selection_table()
    bad=table.model.eq('global_total') & table.budget.eq('W')
    table.loc[bad, 'crps']=0
    table.loc[bad, 'empirical_coverage90']=.8
    assert r.selection(table)['primary']['model'] != 'global_total'


def test_shape_energy_score_matches_pairwise_definition():
    rng=np.random.default_rng(7)
    paths=rng.normal(size=(2,5,24)); paths-=paths.mean(2,keepdims=True)
    truth=rng.normal(size=(2,24)); truth-=truth.mean(1,keepdims=True)
    expected=np.linalg.norm(paths-truth[:,None],axis=2).mean(1)/np.sqrt(24)-.5*np.linalg.norm(paths[:,:,None]-paths[:,None,:],axis=3).mean((1,2))/np.sqrt(24)
    np.testing.assert_allclose(r.shape_energy_score(paths,truth),expected,atol=1e-9)


def test_level_shape_scores_separate_constant_offset():
    paths=np.zeros((2,128,24)); truth=np.full((2,24),10.)
    metrics=r._extra_metrics(paths.mean(1),paths,truth,paths.mean(1))
    np.testing.assert_allclose(metrics['level_crps'],10.)
    np.testing.assert_allclose(metrics['shape_crps'],0.)
    np.testing.assert_allclose(metrics['centered_shape_energy_score'],0.)


def test_shape_cache_requests_nested_donor_predictions():
    outer=tuple(sorted(r.SOURCES[:-1])); queries={outer:[r.SOURCES[-1]]}
    cache=r.ShapeCache({},[outer],queries)
    for cal in outer:
        inner=tuple(x for x in outer if x != cal)
        assert (cal,'calibration') in cache.requests[inner]
        for donor in inner:
            smaller=tuple(x for x in inner if x != donor)
            assert (donor,'proper') in cache.requests[smaller]
            assert cal not in smaller and r.SOURCES[-1] not in smaller


def test_total_risk_predictor_ignores_coarse_particles(monkeypatch):
    # Substitute deterministic shape/bank/scale pieces and make coarse draws
    # extreme. Total-risk outputs must be exactly unchanged, donor arms change.
    days=pd.date_range('2025-01-01',periods=2,tz='UTC')
    class Bank:
        source_regions_=r.SOURCES[:3]
        days_=pd.date_range('2024-01-25',periods=3,tz='UTC')
        q_=np.ones((3,24,8)); fuel_support_=np.ones((3,8),bool)
        raw_cif_=np.full((3,24),100.); donor_mix_=np.full((3,8),1/8)
    class Shapes:
        def get(self,subset,budget,region,split):
            return np.ones((2 if split=='evaluation' else 1,24,8))
    class Scale:
        def predict(self,x): return np.ones((len(x),2))
    records={('NYIS','evaluation'):dict(days=days,features={'B':np.ones((2,25)),'W':np.ones((2,99))})}
    cal=dict(regions=np.array(r.SOURCES[:3]),errors=np.arange(72.).reshape(3,24),
             model=Scale(),scales=np.ones((3,2)),audit=[])
    monkeypatch.setattr(r,'_bank',lambda *a:Bank())
    monkeypatch.setattr(r,'_indices',lambda *a:(np.zeros((2,128),int),np.zeros((2,128),int)))
    monkeypatch.setattr(r,'_donor_predictions',lambda *a:np.ones((3,24,8)))
    monkeypatch.setattr(r,'center_paths',lambda *a:np.full((2,24),200.))
    mix=np.full((2,8),1/8); draws=np.broadcast_to(mix[:,None],(2,128,8)).copy()
    first=r._predict_region({},records,Shapes(),r.SOURCES[:3],'NYIS',mix,draws,{'B':cal,'W':cal})[0]
    draws[:]=0; draws[:,:,0]=1
    second=r._predict_region({},records,Shapes(),r.SOURCES[:3],'NYIS',mix,draws,{'B':cal,'W':cal})[0]
    for budget in ('B','W'):
        for mode in ('global_total','hetero_total'):
            np.testing.assert_array_equal(first[budget,mode]['raw'],second[budget,mode]['raw'])
    assert not np.array_equal(first['B','donor']['raw'],second['B','donor']['raw'])
