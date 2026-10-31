"""Synthetic checks for exclusion, shared forecasts, gain loss and stage seals."""
from __future__ import annotations

import csv
import numpy as np
import pandas as pd
import pytest
from scripts import run_gain_direction_v44 as run


def source_fixture(outer='BPAT'):
    keys, starts, ends, releases = [], [], [], []
    extra = ('SWPP', 'AVA', 'BANC', 'JEA', 'SC', 'NWMT', 'WACM', 'SEC', 'SCEG')
    for source in (*run.SOURCES, *extra):
        n = 24 if source in ('FPL', 'BANC') else 26
        for at in pd.date_range('2022-05-01', periods=n, freq='MS', tz='UTC'):
            stop = at+pd.offsets.MonthBegin(1)
            keys.append(f'{source}/{at:%Y-%m}')
            starts.append(at.value); ends.append(stop.value); releases.append((stop+pd.Timedelta(days=30)).value)
    cohort = dict(keys=keys, period_start_ns=starts, period_end_ns=ends, available_ns=releases)
    ids = [i for i, key in enumerate(keys) if key.split('/')[0] != outer]
    x = np.zeros((len(ids), 31)); x[:, :2] = .5; x[:, -1] = np.log(1000)
    audit = dict(region=outer, outer=outer, training_keys=[keys[i] for i in ids],
        training_regions=[keys[i].split('/')[0] for i in ids], training_Size=x,
        training_q=np.tile([.1, .6, .1, .2], (len(ids), 1)), training_y=np.tile([.2, .5, .1, .2], (len(ids), 1)),
        training_pos_capacity=np.full(len(ids), 1000.))
    for field in ('period_start_ns', 'period_end_ns', 'available_ns'):
        audit['training_'+field] = [cohort[field][i] for i in ids]
    return cohort, audit


@pytest.mark.parametrize('outer,n', [('BPAT', 386), ('FPL', 388)])
def test_source_data_exact_outer_pool(outer, n):
    cohort, audit = source_fixture(outer)
    result = run.source_data(audit, cohort, outer)
    assert result['y'].shape == (n, 4) and len(set(result['regions'])) == 15
    assert outer not in result['regions']


def test_metadata_precedes_fine_values():
    cohort, audit = source_fixture()
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Fine response read before outer exclusion')
    audit['training_y'] = Poison(); audit['training_regions'][0] = 'BPAT'
    with pytest.raises(ValueError, match='before fine values'):
        run.source_data(audit, cohort, 'BPAT')


def test_original_dates_cannot_be_silently_changed():
    cohort, audit = source_fixture()
    audit['training_available_ns'][0] += 1
    with pytest.raises(ValueError, match='source dates'):
        run.source_data(audit, cohort, 'BPAT')


def test_monthly_source_file_ownership(tmp_path):
    paths = {}
    for group in ('legacy', 'v4', 'v5'):
        p = tmp_path / (group+'.csv'); paths[group] = str(p)
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['region', 'period_start']); w.writeheader()
            w.writerows(dict(region=s, period_start='not parsed') for s in ('BPAT', 'SWPP', 'SCL', 'PACE'))
    assert [r['region'] for r in run.monthly_rows(paths)] == ['BPAT', 'SWPP', 'SCL']


@pytest.mark.parametrize('truth,action,regret,noninferior', [
    (.3, True, 0., 1.), (.3, False, .3, 0.), (-.2, True, .2, 0.), (-.2, False, 0., 1.),
    (0., True, 0., 1.), (0., False, 0., 1.)])
def test_common_action_regret_and_true_ties(truth, action, regret, noninferior):
    row = run.gain_metrics(.1, truth, action)
    assert row['regret'] == pytest.approx(regret)
    assert row['noninferior_action'] == noninferior
    assert row['true_tie'] == float(truth == 0)
    assert row['gain_sq_error'] == pytest.approx((.1-truth)**2)


def sealed_queries(tmp_path, monkeypatch, *, bad=False):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'teachers_sealed', lambda: {})
    folder = tmp_path / 'predictions'; folder.mkdir()
    run.write(tmp_path / 'freeze.json', {})
    run.write(tmp_path / 'teacher_seal.json', {})
    freeze = run.sha(tmp_path / 'freeze.json'); teacher = run.sha(tmp_path / 'teacher_seal.json')
    run.write(tmp_path / 'prediction_started.json', dict(freeze_sha256=freeze, teacher_seal_sha256=teacher))
    (tmp_path / 'query_routes.csv').write_text('region,key,arm,action\n')
    entries = []
    for source in run.SOURCES:
        p = folder / (source+'.npz')
        keys = np.array([f'{source}/{at:%Y-%m}' for at in run.DATES])
        if bad and source == run.SOURCES[-1]: keys[0] = 'CLOSED/2025-01'
        mg = np.zeros((6, 128, 5)); mg[..., :2] = .5
        np.savez_compressed(p, arms=np.asarray(run.model.ARMS), keys=keys, origins=run.DATES.asi8,
            MG_draws=mg, EF_draws=np.full((30, 6, 128), 500.), CIF_draws=np.full((30, 6, 128), 250.),
            gain_prediction=np.full((30, 6), .1), actions=np.ones((30, 6), bool))
        run.write(p.with_suffix('.json'), {})
        entries.append(dict(region=source, path=str(p), sha256=run.sha(p), audit_path=str(p.with_suffix('.json')),
                            audit_sha256=run.sha(p.with_suffix('.json'))))
    run.write(tmp_path / 'prediction_seal.json', dict(freeze_sha256=freeze, teacher_seal_sha256=teacher,
        prediction_started_sha256=run.sha(tmp_path / 'prediction_started.json'),
        query_routes_sha256=run.sha(tmp_path / 'query_routes.csv'), entries=entries))


def test_all_query_validation_precedes_outer_label_read(tmp_path, monkeypatch):
    sealed_queries(tmp_path, monkeypatch, bad=True)
    monkeypatch.setattr(run, 'check', lambda: {})
    def forbidden(*args):
        raise AssertionError('Outer scoring labels read before prediction validation')
    monkeypatch.setattr(run.prior.common, 'monthly_rows', forbidden)
    with pytest.raises(ValueError, match='query prediction grid'):
        run.score()
    assert not (tmp_path / 'score_started.json').exists()


def test_query_routes_must_match_seal(tmp_path, monkeypatch):
    sealed_queries(tmp_path, monkeypatch)
    assert len(run.predictions()['entries']) == 7
    (tmp_path / 'query_routes.csv').write_text('changed\n')
    with pytest.raises(ValueError, match='query prediction seal'):
        run.predictions()


@pytest.mark.parametrize('stage,marker', [('teachers', 'teacher_started.json'), ('predict', 'prediction_started.json'), ('score', 'score_started.json')])
def test_started_stages_cannot_be_reentered(tmp_path, monkeypatch, stage, marker):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    run.write(tmp_path / marker, {})
    monkeypatch.setattr(run, 'check', lambda: {})
    monkeypatch.setattr(run, 'teachers_sealed', lambda: {})
    with pytest.raises(FileExistsError, match='attempt'):
        getattr(run, stage)()


def test_teacher_indices_cannot_point_to_other_source_mg(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    cohort, _ = source_fixture(); cp = tmp_path / 'cohort.json'; run.write(cp, cohort)
    run.write(tmp_path / 'freeze.json', dict(cohort_path=str(cp)))
    mg = np.zeros((412, 128, 5)); mg[..., :2] = .5
    np.savez_compressed(tmp_path / 'source_MG.npz', draws=mg)
    folder = tmp_path / 'teachers'; folder.mkdir(); entries = []
    for source in run.SOURCES:
        p = folder / (source+'.npz')
        if source == run.SOURCES[0]:
            keys = [k for k in cohort['keys'] if k.split('/')[0] != source]
            np.savez_compressed(p, keys=np.asarray(keys), regions=np.asarray([k.split('/')[0] for k in keys]),
                source_MG_indices=np.zeros(len(keys), int))
        else:
            p.write_bytes(b'not reached')
        run.write(p.with_suffix('.json'), {})
        entries.append(dict(region=source, path=str(p), sha256=run.sha(p), audit_path=str(p.with_suffix('.json')),
                            audit_sha256=run.sha(p.with_suffix('.json'))))
    with pytest.raises(ValueError, match='shared source MG indices'):
        run.validate_entries(dict(entries=entries), 'teachers', teacher=True)


def test_teacher_stage_writes_and_seals_one_outer_with_real_inner_api(tmp_path, monkeypatch):
    cohort, audit = source_fixture()
    cp = tmp_path / 'cohort.json'; ap = tmp_path / 'source.json'
    run.write(cp, cohort); run.write(ap, audit)
    frozen = dict(cohort_path=str(cp), monthly_paths={}, archives={'BPAT': dict(audit_path=str(ap))})
    run.write(tmp_path / 'freeze.json', frozen)
    mg = np.zeros((412, 128, 5)); mg[..., :2] = .5
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'SOURCES', ('BPAT',))
    monkeypatch.setattr(run, 'check', lambda: frozen)
    monkeypatch.setattr(run, 'monthly_rows', lambda paths: [])
    monkeypatch.setattr(run.model, 'source_mg', lambda rows, keys: dict(draws=mg, points=mg[:, 0],
        auditrows=[dict(key=k, prefix_count=3) for k in keys]))
    run.teachers()
    assert len(run.teachers_sealed()['entries']) == 1
    saved = run.read(tmp_path / 'teachers/BPAT.json')
    assert len(saved['inner_folds']) == 15 and saved['fine_head_fit_count'] == 45
    assert sum(fold['fit_count'] for fold in saved['inner_folds']) == 45
