"""Synthetic source scope and prediction-seal checks; no research data access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scripts import run_ordinary_controls_v43 as run


def training(outer='BPAT'):
    extra = ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'WACM', 'SEC', 'SCEG')
    keys, start, end, release = [], [], [], []
    for source in (*run.SOURCES, *extra):
        n = 24 if source in ('FPL', 'BANC') else 26
        for at in pd.date_range('2022-05-01', periods=n, freq='MS', tz='UTC'):
            stop = at+pd.offsets.MonthBegin(1)
            keys.append(f'{source}/{at:%Y-%m}')
            start.append(at.value); end.append(stop.value); release.append((stop+pd.Timedelta(days=30)).value)
    cohort = dict(keys=keys, period_start_ns=start, period_end_ns=end, available_ns=release)
    indices = [i for i, key in enumerate(keys) if key.split('/')[0] != outer]
    audit = dict(region=outer, outer=outer, training_keys=[keys[i] for i in indices],
        training_regions=[keys[i].split('/')[0] for i in indices], training_q=np.tile([.1, .6, .1, .2], (len(indices), 1)))
    for field in ('period_start_ns', 'period_end_ns', 'available_ns'):
        audit['training_'+field] = [cohort[field][i] for i in indices]
    return cohort, audit


@pytest.mark.parametrize('outer,n,old_n', [('BPAT', 386, 154), ('FPL', 388, 156)])
def test_exact_source_cohort_and_old_q_projection(outer, n, old_n):
    cohort, audit = training(outer)
    old, expanded = run.training_q(audit, cohort, outer)
    assert old.shape == (old_n, 4) and expanded.shape == (n, 4)
    np.testing.assert_array_equal(old, expanded[:old_n])


def test_source_metadata_checked_before_q_conversion():
    cohort, audit = training()
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('q accessed before source exclusion')
    audit['training_q'] = Poison()
    audit['training_regions'][0] = 'BPAT'
    with pytest.raises(ValueError, match='before q values'):
        run.training_q(audit, cohort, 'BPAT')


def test_dates_are_checked_before_q_values():
    cohort, audit = training()
    audit['training_period_start_ns'][0] += 1
    with pytest.raises(ValueError, match='dates changed'):
        run.training_q(audit, cohort, 'BPAT')


def test_query_capacity_cannot_depend_on_chosen_particle(tmp_path):
    source = 'BPAT'
    mg = np.zeros((6, 128, 5)); mg[..., :2] = .5
    keys = np.array([f'{source}/{at:%Y-%m}' for at in run.DATES])
    parent = tmp_path / 'parent.npz'; old = tmp_path / 'old.npz'
    np.savez(parent, arms=np.array(run.model.PARENT_ARMS), keys=keys, origins=run.DATES.asi8,
        MG_draws=mg, EF_draws=np.full((12, 6, 128), 400.), CIF_draws=np.full((12, 6, 128), 200.))
    raw = np.zeros((6, 128, 36)); raw[..., :5] = mg; raw[..., 31:35] = [.0, .9, .1, .0]
    raw[0, 1, 32:34] = [.8, .2]
    np.savez(old, arms=np.array(['Capacity-OP', *[f'other{i}' for i in range(10)]]), keys=keys,
        origins=run.DATES.asi8, MG_draws=mg, EF_draws=np.full((11, 6, 128), 440.), query_feature_SizeCap=raw)
    with pytest.raises(ValueError, match='particle-constant'):
        run.archive(dict(path=str(parent), original_path=str(old)), source)


def sealed(tmp_path, monkeypatch, bad_grid=False):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    folder = tmp_path / 'predictions'; folder.mkdir()
    run.write(tmp_path / 'freeze.json', {})
    freeze = run.sha(tmp_path / 'freeze.json')
    run.write(tmp_path / 'prediction_started.json', dict(freeze_sha256=freeze))
    (tmp_path / 'route_table.csv').write_text('region,key,arm,right_weight\n')
    (tmp_path / 'training_limits.csv').write_text('region,old_cases,expanded_cases\n')
    entries = []
    for source in run.SOURCES:
        path = folder / (source+'.npz')
        keys = np.array([f'{source}/{at:%Y-%m}' for at in run.DATES])
        if bad_grid and source == run.SOURCES[-1]:
            keys[0] = 'CLOSED/2025-01'
        mg = np.zeros((6, 128, 5)); mg[..., :2] = .5
        np.savez_compressed(path, arms=np.array(run.model.ARMS), keys=keys, origins=run.DATES.asi8,
            MG_draws=mg, EF_draws=np.full((35, 6, 128), 500.), CIF_draws=np.full((35, 6, 128), 250.),
            weights=np.full((35, 6), .5))
        run.write(path.with_suffix('.json'), {})
        entries.append(dict(region=source, path=str(path), sha256=run.sha(path),
            audit_path=str(path.with_suffix('.json')), audit_sha256=run.sha(path.with_suffix('.json'))))
    run.write(tmp_path / 'prediction_seal.json', dict(freeze_sha256=freeze,
        prediction_started_sha256=run.sha(tmp_path / 'prediction_started.json'),
        route_table_sha256=run.sha(tmp_path / 'route_table.csv'),
        training_limits_sha256=run.sha(tmp_path / 'training_limits.csv'), entries=entries))


def test_full_grid_is_validated_before_score_label_access(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch, bad_grid=True)
    monkeypatch.setattr(run, 'check', lambda: {})
    def forbidden(*args):
        raise AssertionError('Score labels read before complete prediction validation')
    monkeypatch.setattr(run.common, 'monthly_rows', forbidden)
    with pytest.raises(ValueError, match='prediction grid'):
        run.score()


def test_routes_are_bound_to_prediction_seal(tmp_path, monkeypatch):
    sealed(tmp_path, monkeypatch)
    assert len(run.predictions()['entries']) == 7
    (tmp_path / 'route_table.csv').write_text('changed\n')
    with pytest.raises(ValueError, match='prediction/routes seal'):
        run.predictions()


def test_score_attempt_cannot_be_reentered_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    run.write(tmp_path / 'score_started.json', {})
    def forbidden():
        raise AssertionError('Score attempt restarted')
    monkeypatch.setattr(run, 'check', forbidden)
    with pytest.raises(FileExistsError, match='score attempt'):
        run.score()
